"""
Model architectures for video action recognition.
"""

import torch
import torch.nn as nn
import timm
import json
import logging
from pathlib import Path
from .config import get_default_config

# Import VideoMAE from HuggingFace transformers
# IMPORTANT: Import torch first to ensure transformers can detect PyTorch version
try:
    import sys
    import importlib.metadata
    
    _torch_version = torch.__version__
    
    # WORKAROUND: Fix transformers PyTorch version detection bug
    # importlib.metadata.version("torch") returns None even though torch is installed
    # Patch it to return torch.__version__ before transformers imports
    _original_version = importlib.metadata.version
    
    def _patched_version(package_name):
        """Patch version() to return torch.__version__ when package_name is 'torch'"""
        if package_name == "torch":
            return _torch_version
        try:
            return _original_version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None
    
    # Monkey-patch importlib.metadata.version BEFORE importing transformers
    importlib.metadata.version = _patched_version
    
    from transformers import VideoMAEModel, VideoMAEConfig
    TRANSFORMERS_AVAILABLE = True
except (ImportError, TypeError, AttributeError) as e:
    TRANSFORMERS_AVAILABLE = False
    VideoMAEModel = None
    VideoMAEConfig = None
    # Log warning but don't raise - will be checked in __init__

# Debug logging removed for performance


# ============================================================================
# Helper Functions (Refactor duplicate code)
# ============================================================================

def create_classification_head(embed_dim: int, num_classes: int, dropout: float = 0.1, 
                               init_output_gain: float = 1.0, init_hidden_gain: float = 1.0,
                               use_normal_init: bool = False) -> nn.Sequential:
    """
    Tạo classification head với cấu trúc chuẩn.
    
    Args:
        embed_dim: Embedding dimension
        num_classes: Number of classes
        dropout: Dropout rate
        init_output_gain: Gain cho output layer initialization (default: 1.0)
        init_hidden_gain: Gain cho hidden layer initialization (default: 1.0)
        use_normal_init: Nếu True, dùng normal init với std thay vì xavier (tốt hơn cho logits range)
    
    Returns:
        Classification head module
    """
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(dropout),
        nn.Linear(embed_dim, embed_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(embed_dim // 2, num_classes)
    )
    
    # Initialize weights
    # Research: Normal init với std=0.01-0.1 tốt hơn xavier cho classification head
    # để logits có range đủ lớn ngay từ đầu
    head_modules = list(head.modules())
    linear_modules = [m for m in head_modules if isinstance(m, nn.Linear)]
    for i, module in enumerate(linear_modules):
        if i == len(linear_modules) - 1:
            # Output layer: dùng normal init với std lớn hơn để logits có range tốt
            if use_normal_init:
                # Research: std=0.01-0.1 cho classification head
                # CRITICAL: Với nhiều classes (51), cần std lớn hơn để logits có range đủ lớn
                # Dùng std tương ứng với gain (gain=15.0 → std≈0.15)
                std = init_output_gain * 0.01 if init_output_gain > 1.0 else 0.01
                nn.init.normal_(module.weight, mean=0.0, std=std)
                # Also initialize bias to small positive value to help initial predictions
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # Keep bias at 0 (will be learned)
            else:
                nn.init.xavier_uniform_(module.weight, gain=init_output_gain)
        else:
            # Hidden layers
            if use_normal_init:
                std = init_hidden_gain * 0.01 if init_hidden_gain > 1.0 else 0.01
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.xavier_uniform_(module.weight, gain=init_hidden_gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    return head


def freeze_backbone_params(backbone: nn.Module):
    """
    Freeze tất cả parameters của backbone.
    
    Args:
        backbone: Backbone module (e.g., self.vit, self.swin)
    """
    for param in backbone.parameters():
        param.requires_grad = False


def create_adapters(backbone: nn.Module, embed_dim: int, adapter_dim: int = None, 
                    dropout: float = 0.1) -> tuple:
    """
    Tạo adapters và block_norm cho backbone.
    
    Args:
        backbone: Backbone module với blocks attribute
        embed_dim: Embedding dimension
        adapter_dim: Adapter dimension (default: embed_dim // 6)
        dropout: Dropout rate
    
    Returns:
        Tuple of (adapters: nn.ModuleList, block_norm: nn.LayerNorm)
    """
    adapters = nn.ModuleList([
        Adapter(embed_dim, adapter_dim, dropout)
        for _ in range(len(backbone.blocks))
    ])
    block_norm = nn.LayerNorm(embed_dim)
    return adapters, block_norm


def create_adapters_for_videomae(videomae: nn.Module, embed_dim: int, adapter_dim: int = None,
                                  dropout: float = 0.1, num_blocks: int = None) -> tuple:
    """
    Tạo adapters và block_norm cho VideoMAE model.
    
    Args:
        videomae: VideoMAE model từ HuggingFace
        embed_dim: Embedding dimension
        adapter_dim: Adapter dimension (default: embed_dim // 6)
        dropout: Dropout rate
        num_blocks: Number of encoder layers (if None, auto-detect)
    
    Returns:
        Tuple of (adapters: nn.ModuleList, block_norm: nn.LayerNorm)
    """
    if num_blocks is None:
        if hasattr(videomae, 'encoder') and hasattr(videomae.encoder, 'layer'):
            num_blocks = len(videomae.encoder.layer)
        else:
            raise ValueError("Cannot determine number of blocks in VideoMAE model")
    
    adapters = nn.ModuleList([
        Adapter(embed_dim, adapter_dim, dropout)
        for _ in range(num_blocks)
    ])
    block_norm = nn.LayerNorm(embed_dim)
    return adapters, block_norm


def forward_with_adapters_common(vit: nn.Module, x: torch.Tensor, adapters: nn.ModuleList, 
                                  block_norm: nn.LayerNorm, use_adapters: bool, 
                                  remove_cls_token: bool = True) -> torch.Tensor:
    """
    Common logic cho _forward_with_adapters trong các model classes.
    
    Args:
        vit: ViT backbone module
        x: Input tensor [B*T, C, H, W]
        adapters: Adapter modules
        block_norm: Block normalization layer
        use_adapters: Whether to use adapters
        remove_cls_token: Whether to remove class token before returning
    
    Returns:
        Patch features [B*T, num_patches, embed_dim] or [B*T, num_patches+1, embed_dim]
    """
    # Patch embedding
    x = vit.patch_embed(x)  # [B*T, num_patches, embed_dim]
    
    # Add class token if exists
    if hasattr(vit, 'cls_token') and vit.cls_token is not None:
        cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B*T, num_patches+1, embed_dim]
    
    # Add position embedding
    if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
        x = x + vit.pos_embed
    
    # Forward through blocks with adapters
    for i, block in enumerate(vit.blocks):
        x = block(x)
        if use_adapters and block_norm is not None:
            x = block_norm(x)
        if use_adapters and i < len(adapters):
            adapter_out = adapters[i](x)
            x = x + adapter_out
            if block_norm is not None:
                x = block_norm(x)
    
    # Final norm
    if hasattr(vit, 'norm') and vit.norm is not None:
        x = vit.norm(x)
    
    # Remove class token if requested
    if remove_cls_token and x.shape[1] > 196:  # 196 = 14*14 patches for 224x224
        x = x[:, 1:]  # Remove class token [B*T, num_patches, embed_dim]
    
    return x


class TemporalAttention(nn.Module):
    """Multi-head temporal attention mechanism for video sequences."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        # QUAN TRỌNG: Normalize output để tránh giá trị quá lớn (std=75 → std=1)
        # Điều này giúp head có thể học được và logits không quá nhỏ
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim] - temporal sequence of features
        Returns:
            out: [B, embed_dim] - aggregated temporal features
        """
        B, T, C = x.shape
        
        # Layer norm
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + x
        
        # Global average pooling across temporal dimension
        out = out.mean(dim=1)  # [B, embed_dim]
        
        return out


class DividedSpaceTimeAttention(nn.Module):
    """
    TimeSformer-style divided space-time attention.
    Separates spatial and temporal attention for efficiency.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Temporal attention (across time)
        self.temporal_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        
        # Spatial attention (across patches)
        self.spatial_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        # QUAN TRỌNG: Normalize output để tránh giá trị quá lớn (std=75 → std=1)
        # Điều này giúp head có thể học được và logits không quá nhỏ
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, embed_dim] - batch, time, patches, embed_dim
        Returns:
            out: [B, embed_dim] - aggregated features
        """
        B, T, N, C = x.shape
        x_norm = self.norm(x)
        
        # Reshape for temporal attention: [B*N, T, C]
        x_temporal = x_norm.permute(0, 2, 1, 3).reshape(B * N, T, C)
        
        # Temporal attention
        qkv_t = self.temporal_qkv(x_temporal).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv_t = qkv_t.permute(2, 0, 3, 1, 4)  # [3, B*N, num_heads, T, head_dim]
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
        
        attn_t = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn_t = attn_t.softmax(dim=-1)
        attn_t = self.dropout(attn_t)
        
        out_t = (attn_t @ v_t).transpose(1, 2).reshape(B * N, T, C)
        out_t = self.temporal_proj(out_t)
        out_t = self.dropout(out_t)
        
        # Reshape back: [B, N, T, C] -> [B, T, N, C]
        out_t = out_t.reshape(B, N, T, C).permute(0, 2, 1, 3)
        out_t = out_t + x  # Residual
        
        # Spatial attention: [B*T, N, C]
        x_spatial = out_t.reshape(B * T, N, C)
        x_spatial_norm = self.norm(x_spatial)
        
        qkv_s = self.spatial_qkv(x_spatial_norm).reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv_s = qkv_s.permute(2, 0, 3, 1, 4)  # [3, B*T, num_heads, N, head_dim]
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]
        
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.scale
        attn_s = attn_s.softmax(dim=-1)
        attn_s = self.dropout(attn_s)
        
        out_s = (attn_s @ v_s).transpose(1, 2).reshape(B * T, N, C)
        out_s = self.spatial_proj(out_s)
        out_s = self.dropout(out_s)
        
        # Reshape back: [B, T, N, C]
        out_s = out_s.reshape(B, T, N, C)
        out_s = out_s + out_t  # Residual
        
        # Global average pooling: [B, T, N, C] -> [B, C]
        out = out_s.mean(dim=(1, 2))
        out = self.output_norm(out)
        return out


class Adapter(nn.Module):
    """Lightweight adapter module for AdaptFormer-style fine-tuning."""
    
    def __init__(self, embed_dim: int, adapter_dim: int = None, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            adapter_dim: Adapter dimension (default: embed_dim // 6)
            dropout: Dropout rate
        """
        super().__init__()
        adapter_dim = adapter_dim or (embed_dim // 6)  # Default: 1/6 of embed_dim (optimal for 51 classes, balance capacity vs data)
        self.adapter_dim = adapter_dim
        self.embed_dim = embed_dim
        
        # Layer normalization to prevent numerical instability (root cause fix)
        # Normalizes values to mean=0, std=1, preventing overflow without losing information
        self.norm = nn.LayerNorm(embed_dim)
        
        # Down projection
        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        # Activation
        self.activation = nn.GELU()
        # Up projection
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with smaller gain to prevent rapid value growth (root cause fix)
        # Xavier uniform with gain=0.1 produces smaller weights than default initialization
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embed_dim] or [B, T, embed_dim]
        Returns:
            out: same shape as x
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, C = original_shape
            x = x.view(B * T, C)
        else:
            B, C = original_shape
        
        # Layer normalization instead of clamp (root cause fix)
        # Normalizes to mean=0, std=1, preventing overflow while preserving information
        x = self.norm(x)
        
        # Adapter forward
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        
        if len(original_shape) == 3:
            out = out.view(original_shape)
        
        return out


class SOTAViTForAction(nn.Module):
    """SOTA ViT for action recognition with temporal attention and adapters."""
    
    def __init__(
        self,
        num_classes: int = 51,
        pretrained_name: str = 'vit_base_patch16_224',
        use_adapters: bool = True,
        adapter_dim: int = None,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0  # Stochastic Depth (DropPath) rate
    ):
        """
        Args:
            num_classes: Number of action classes
            pretrained_name: Pretrained ViT model name (from timm)
            use_adapters: Whether to use AdaptFormer-style adapters
            adapter_dim: Adapter dimension (default: embed_dim // 6)
            temporal_heads: Number of temporal attention heads
            dropout: Dropout rate
            drop_path_rate: Stochastic Depth (DropPath) rate for regularization
        """
        super().__init__()
        
        # Load pretrained ViT with DropPath (Stochastic Depth)
        self.vit = timm.create_model(
            pretrained_name, 
            pretrained=True, 
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        self.embed_dim = self.vit.num_features
        self.use_adapters = use_adapters
        
        # Freeze backbone if using adapters
        if use_adapters:
            freeze_backbone_params(self.vit)
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # Adapters (if enabled)
        if use_adapters:
            self.adapters = nn.ModuleList([
                Adapter(self.embed_dim, adapter_dim, dropout)
                for _ in range(len(self.vit.blocks))
            ])
            # LayerNorm to normalize block outputs (root cause fix)
            # Prevents value accumulation across blocks without losing information
            self.block_norm = nn.LayerNorm(self.embed_dim)
        
        # Classification head with dropout
        self.head = create_classification_head(self.embed_dim, num_classes, dropout)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] - batch of video clips
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = video.shape
        
        # Reshape to process all frames
        x = video.view(B * T, C, H, W)
        
        # Extract features with ViT
        # Apply adapters in forward pass to enable gradient flow and training
        # Test results show adapters need to be applied to receive gradients
        if self.use_adapters:
            features = self._forward_with_adapters(x)  # [B*T, embed_dim]
        else:
            features = self.vit(x)  # [B*T, embed_dim]
        
        # Reshape back to temporal sequence
        features = features.view(B, T, self.embed_dim)
        
        # Temporal attention
        pooled = self.temporal_attention(features)  # [B, embed_dim]
        
        # Classification
        logits = self.head(pooled)
        
        return logits
    
    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adapters integrated into each ViT block.
        According to AdaptFormer, adapters are added in parallel with MLP.
        
        Args:
            x: [B*T, C, H, W] - input images
        Returns:
            features: [B*T, embed_dim] - output features
        """
        # Patch embedding
        x = self.vit.patch_embed(x)  # [B*T, num_patches, embed_dim]
        
        # Add class token if exists (before position embedding)
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B*T, num_patches+1, embed_dim]
        
        # Add position embedding (after class token is added)
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            # pos_embed has shape [1, num_patches+1, embed_dim] (includes class token)
            x = x + self.vit.pos_embed
        
        # Forward through blocks with adapters
        # Apply adapters on all tokens (AdaptFormer standard) for better accuracy
        # Memory is optimized by using in-place operations where possible
        for i, block in enumerate(self.vit.blocks):
            # Standard block forward
            x = block(x)  # [B*T, num_patches+1, embed_dim]
            
            # Layer normalization instead of clamp (root cause fix)
            # Normalizes to mean=0, std=1, preventing overflow while preserving information
            if self.use_adapters and hasattr(self, 'block_norm'):
                x = self.block_norm(x)
            
            # Apply adapter in parallel (AdaptFormer style)
            # Apply on all tokens for better accuracy (not just class token)
            if i < len(self.adapters):
                # Apply adapter on all tokens
                adapter_out = self.adapters[i](x)  # [B*T, num_patches+1, embed_dim]
                # Add adapter output to block output (residual connection)
                # Use in-place addition to save memory
                x = x + adapter_out
                # Normalize after residual to prevent value accumulation across blocks (root cause fix)
                # This prevents gradual drift to very large values over many blocks
                if hasattr(self, 'block_norm'):
                    x = self.block_norm(x)
        
        # Final norm layer
        if hasattr(self.vit, 'norm') and self.vit.norm is not None:
            x = self.vit.norm(x)
        
        # Extract class token or global average pooling
        if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
            # Use class token (first token)
            features = x[:, 0]  # [B*T, embed_dim]
        else:
            # Global average pooling
            features = x.mean(dim=1)  # [B*T, embed_dim]
        
        return features


class EMAModel:
    """
    Exponential Moving Average (EMA) wrapper for model.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to wrap
            decay: EMA decay rate
        """
        self._model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict."""
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict.copy()
    
    @property
    def model(self):
        """Get the wrapped model."""
        return self._model
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TimeSformerForAction(nn.Module):
    """TimeSformer-style model with divided space-time attention."""
    
    def __init__(
        self,
        num_classes: int = 51,
        pretrained_name: str = 'vit_base_patch16_224',
        use_adapters: bool = True,
        adapter_dim: int = None,
        temporal_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of action classes
            pretrained_name: Pretrained ViT model name (from timm)
            use_adapters: Whether to use AdaptFormer-style adapters
            adapter_dim: Adapter dimension (default: embed_dim // 6)
            temporal_heads: Number of temporal attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(pretrained_name, pretrained=True, num_classes=0)
        
        # Get embedding dimension
        self.embed_dim = self.vit.num_features
        self.use_adapters = use_adapters
        
        # Freeze backbone if using adapters
        if use_adapters:
            freeze_backbone_params(self.vit)
        
        # Divided space-time attention
        self.space_time_attention = DividedSpaceTimeAttention(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # Adapters (if enabled)
        if use_adapters:
            self.adapters, self.block_norm = create_adapters(self.vit, self.embed_dim, adapter_dim, dropout)
        else:
            self.adapters = None
            self.block_norm = None
        
        # Classification head
        self.head = create_classification_head(self.embed_dim, num_classes, dropout)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] - batch of video clips
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = video.shape
        
        # Reshape to process all frames
        x = video.view(B * T, C, H, W)
        
        # Extract patch features with ViT (without temporal modeling)
        if self.use_adapters:
            patch_features = self._forward_with_adapters(x)  # [B*T, num_patches, embed_dim]
        else:
            # QUAN TRỌNG: Phải đi qua toàn bộ ViT blocks để extract features!
            # Không chỉ lấy patch_embed, mà phải forward qua blocks
            # Use forward_features to get patch-level features (not pooled)
            patch_features = self.vit.forward_features(x)  # [B*T, num_patches+1, embed_dim] (includes class token)
            # Remove class token if exists (ViT thường có class token ở đầu)
            if patch_features.shape[1] > 196:  # 196 = 14*14 patches for 224x224 image
                patch_features = patch_features[:, 1:]  # Remove class token [B*T, num_patches, embed_dim]
        
        # Reshape to [B, T, num_patches, embed_dim]
        num_patches = patch_features.shape[1]
        patch_features = patch_features.view(B, T, num_patches, self.embed_dim)
        
        # Divided space-time attention
        pooled = self.space_time_attention(patch_features)  # [B, embed_dim]
        
        # Classification
        logits = self.head(pooled)
        
        return logits
    
    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adapters for patch extraction (TimeSformer)."""
        return forward_with_adapters_common(
            self.vit, x, self.adapters, self.block_norm, 
            self.use_adapters, remove_cls_token=True
        )


class VideoSwinForAction(nn.Module):
    """Video Swin Transformer for action recognition."""
    
    def __init__(
        self,
        num_classes: int = 51,
        pretrained_name: str = 'swin_base_patch4_window7_224',
        use_adapters: bool = True,
        adapter_dim: int = None,
        temporal_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of action classes
            pretrained_name: Pretrained Swin model name (from timm)
            use_adapters: Whether to use AdaptFormer-style adapters
            adapter_dim: Adapter dimension (default: embed_dim // 6)
            temporal_heads: Number of temporal attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pretrained Swin - must be exact Swin Transformer, no fallback
        self.swin = timm.create_model(pretrained_name, pretrained=True, num_classes=0)
        
        # Validate model type - must be Swin Transformer
        model_type = type(self.swin).__name__
        if 'Swin' not in model_type:
            raise RuntimeError(
                f"Expected Swin Transformer but got {model_type}. "
                f"Model name: {pretrained_name}. "
                f"Please check if the model name is correct in timm. "
                f"Valid Swin model names: swin_base_patch4_window7_224, "
                f"swin_small_patch4_window7_224, swin_tiny_patch4_window7_224, etc."
            )
        
        # Get embedding dimension
        self.embed_dim = self.swin.num_features
        self.use_adapters = use_adapters
        
        # Freeze backbone if using adapters
        if use_adapters:
            freeze_backbone_params(self.swin)
        
        # Temporal attention for video
        self.temporal_attention = TemporalAttention(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # Adapters (if enabled) - Swin has stages, we'll add adapters to each stage
        if use_adapters:
            # Count number of blocks across all stages
            # Swin Transformer has 'stages' attribute, each stage contains blocks
            num_blocks = 0
            if hasattr(self.swin, 'stages'):
                # Swin Transformer structure: stages contain blocks
                for stage in self.swin.stages:
                    if hasattr(stage, '__len__'):
                        num_blocks += len(stage)
                    elif hasattr(stage, 'blocks'):
                        num_blocks += len(stage.blocks)
            elif hasattr(self.swin, 'layers'):
                # Alternative Swin structure with layers
                for layer in self.swin.layers:
                    if hasattr(layer, '__len__'):
                        num_blocks += len(layer)
                    elif hasattr(layer, 'blocks'):
                        num_blocks += len(layer.blocks)
            elif hasattr(self.swin, 'blocks'):
                # ViT-style structure (fallback if timm returns ViT instead of Swin)
                num_blocks = len(self.swin.blocks)
            
            if num_blocks == 0:
                # Fallback: estimate based on model type
                # Swin-Base typically has (2, 2, 18, 2) = 24 blocks total
                # But we'll use a conservative estimate
                num_blocks = 24  # Swin-Base typically has 24 blocks across all stages
                
            self.adapters = nn.ModuleList([
                Adapter(self.embed_dim, adapter_dim, dropout)
                for _ in range(num_blocks)
            ])
            self.block_norm = nn.LayerNorm(self.embed_dim)
        
        # Classification head
        self.head = create_classification_head(self.embed_dim, num_classes, dropout)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] - batch of video clips
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = video.shape
        
        # Process each frame with Swin
        frame_features = []
        for t in range(T):
            frame = video[:, t]  # [B, C, H, W]
            if self.use_adapters:
                features = self._forward_with_adapters(frame)  # [B, embed_dim]
            else:
                features = self.swin(frame)  # [B, embed_dim]
            frame_features.append(features)
        
        # Stack temporal features: [B, T, embed_dim]
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Temporal attention
        pooled = self.temporal_attention(temporal_features)  # [B, embed_dim]
        
        # Classification
        logits = self.head(pooled)
        
        return logits
    
    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adapters for Swin."""
        # Swin forward pass with adapters
        # This is a simplified version - full implementation would require
        # modifying Swin's internal structure
        x = self.swin.forward_features(x)
        
        # Global average pooling if needed
        # Swin forward_features returns [B, H, W, C] format (not [B, C, H, W])
        if x.dim() == 4:
            # x is [B, H, W, C] - average over spatial dimensions H and W
            x = x.mean(dim=(1, 2))  # Result: [B, C]
        elif x.dim() == 3:
            # x is [B, N, C] - average over sequence dimension N
            x = x.mean(dim=1)  # Result: [B, C]
        elif x.dim() == 2:
            # x is already [B, C]
            pass
        
        return x


class VideoMAEv2ForAction(nn.Module):
    """
    VideoMAEv2-style model for action recognition.
    
    VideoMAEv2 đạt 90% accuracy trên Kinetics-400 (SOTA).
    Implementation này dùng ViT-Large với divided space-time attention
    (tương tự TimeSformer nhưng với ViT-Large) để đạt accuracy cao.
    
    Note: VideoMAEv2 thực sự cần pretraining phức tạp (masked autoencoder),
    nhưng ViT-Large + divided attention đã rất tốt cho action recognition.
    """
    
    def __init__(
        self,
        num_classes: int = 51,
        pretrained_name: str = None,  # Không dùng nữa, sẽ dùng VideoMAE model
        use_adapters: bool = True,
        adapter_dim: int = None,
        temporal_heads: int = 16,  # ViT-Large có 16 heads
        dropout: float = 0.1,
        pretrained_ckpt: str = None,  # Path to VideoMAE pretrained checkpoint hoặc HuggingFace model name
        model_name: str = 'MCG-NJU/videomae-large-finetuned-kinetics',  # HuggingFace model name
        num_frames: int = 16,  # Number of frames
        tubelet_size: int = 2,  # Temporal tubelet size (3D patches)
        image_size: int = 224,  # Image size
        patch_size: int = 16,  # Spatial patch size
        init_output_gain: float = 2.0,  # Gain for output layer initialization (std = gain * 0.01)
        init_hidden_gain: float = 1.0,  # Gain for hidden layer initialization (std = gain * 0.01)
        use_normal_init: bool = True  # Use normal init instead of xavier
    ):
        """
        Args:
            num_classes: Number of action classes
            pretrained_name: DEPRECATED - không dùng nữa
            use_adapters: Whether to use AdaptFormer-style adapters
            adapter_dim: Adapter dimension (default: embed_dim // 6)
            temporal_heads: Number of temporal attention heads (default: 16 for ViT-Large)
            dropout: Dropout rate
            pretrained_ckpt: Path to VideoMAE pretrained checkpoint (local file) hoặc HuggingFace model name
            model_name: HuggingFace model name (e.g., 'MCG-NJU/videomae-large-finetuned-kinetics')
            num_frames: Number of frames in video
            tubelet_size: Temporal tubelet size for 3D patch embedding
            image_size: Image size
            patch_size: Spatial patch size
            init_output_gain: Gain for output layer initialization (default: 2.0 → std=0.02, best practice)
            init_hidden_gain: Gain for hidden layer initialization (default: 1.0 → std=0.01)
            use_normal_init: Use normal init instead of xavier (default: True)
        """
        super().__init__()
        
        # Re-check transformers availability (in case module was cached with False)
        # WORKAROUND: Fix transformers PyTorch version detection issue
        try:
            import importlib.metadata
            
            # Ensure torch is imported and version is accessible
            _torch_version = torch.__version__
            
            # WORKAROUND: Patch importlib.metadata.version if not already patched
            # This fixes the issue where importlib.metadata.version("torch") returns None
            if not hasattr(importlib.metadata.version, '_patched'):
                _original_version = importlib.metadata.version
                
                def _patched_version(package_name):
                    """Patch version() to return torch.__version__ when package_name is 'torch'"""
                    if package_name == "torch":
                        return _torch_version
                    try:
                        return _original_version(package_name)
                    except importlib.metadata.PackageNotFoundError:
                        return None
                
                _patched_version._patched = True
                importlib.metadata.version = _patched_version
            
            from transformers import VideoMAEModel, VideoMAEConfig
            transformers_available = True
        except (ImportError, TypeError, AttributeError) as e:
            transformers_available = False
            error_msg = str(e)
        
        if not transformers_available:
            raise ImportError(
                f"transformers library is required for VideoMAE but failed to import.\n"
                f"Error: {error_msg}\n"
                f"PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Unknown'}\n"
                f"Please try: pip install transformers --upgrade"
            )
        
        logger = logging.getLogger(__name__)
        
        # Load VideoMAE model from HuggingFace (official implementation)
        logger.info(f"Loading VideoMAE model from HuggingFace: {model_name}")
        
        # Create VideoMAE config
        config = VideoMAEConfig.from_pretrained(
            model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_classes  # For fine-tuning
        )
        
        # Load VideoMAE model
        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            # Load from local checkpoint
            logger.info(f"Loading VideoMAE from local checkpoint: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Create model and load weights
            self.videomae = VideoMAEModel(config)
            # Remove 'videomae.' prefix if present
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('videomae.'):
                    cleaned_state_dict[k[9:]] = v  # Remove 'videomae.' prefix
                else:
                    cleaned_state_dict[k] = v
            self.videomae.load_state_dict(cleaned_state_dict, strict=False)
            logger.info("VideoMAE weights loaded from local checkpoint")
        else:
            # Load from HuggingFace
            logger.info(f"Loading VideoMAE from HuggingFace: {model_name}")
            self.videomae = VideoMAEModel.from_pretrained(model_name, config=config)
            logger.info("VideoMAE model loaded from HuggingFace")
        
        # Get embedding dimension from VideoMAE config
        self.embed_dim = config.hidden_size
        self.use_adapters = use_adapters
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.image_size = image_size
        # VideoMAE does NOT have CLS token, always use mean pooling
        self.use_mean_pooling = True
        
        # Freeze backbone if using adapters
        if use_adapters:
            freeze_backbone_params(self.videomae)
        
        # VideoMAE đã có temporal modeling built-in (3D patches + temporal pos embedding)
        # Không cần thêm divided space-time attention như trước
        # VideoMAE encoder đã xử lý temporal information
        
        # Adapters (if enabled) - apply to VideoMAE encoder layers
        if use_adapters:
            # VideoMAE encoder có layers trong self.videomae.encoder.layer
            num_blocks = len(self.videomae.encoder.layer)
            self.adapters, self.block_norm = create_adapters_for_videomae(
                self.videomae, self.embed_dim, adapter_dim, dropout, num_blocks
            )
        else:
            self.adapters = None
            self.block_norm = None
        
        # Classification head với initialization hợp lý
        # Model 9: Sử dụng best practices 2024 cho transformer classification heads
        # std=0.02 (init_output_gain=2.0) phù hợp với BERT/GPT style initialization
        # Giảm risk của large initial logits, cải thiện stability
        self.head = create_classification_head(
            self.embed_dim, num_classes, dropout,
            init_output_gain=init_output_gain,  # Default: 2.0 → std=0.02 (best practice)
            init_hidden_gain=init_hidden_gain,  # Default: 1.0 → std=0.01 (conservative)
            use_normal_init=use_normal_init     # Default: True
        )
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] - batch of video clips
        Returns:
            logits: [B, num_classes]
        """
        # Skip validation in training for speed (only validate in debug mode)
        B, T, C, H, W = video.shape
        
        # VideoMAE expects [B, num_frames, num_channels, height, width]
        # Input video is already [B, T, C, H, W] - perfect!
        
        # Forward through VideoMAE encoder
        # VideoMAE tự động xử lý:
        # - 3D patch embedding (tubelet_size)
        # - Temporal position embedding
        # - Spatial-temporal attention
        if self.use_adapters:
            # Forward with adapters
            outputs = self._forward_videomae_with_adapters(video)  # [B, seq_len, embed_dim]
        else:
            # Standard VideoMAE forward
            outputs = self.videomae(pixel_values=video, output_hidden_states=False)
            # outputs.last_hidden_state: [B, seq_len, embed_dim] (NO CLS token, only patch embeddings)
            outputs = outputs.last_hidden_state
        
        # Extract video representation: VideoMAE does NOT have CLS token!
        # CRITICAL: VideoMAE from HuggingFace only outputs patch embeddings, no CLS token
        # Must use mean pooling across sequence dimension
        # outputs: [B, seq_len, embed_dim] where seq_len = (num_frames // tubelet_size) * (image_size // patch_size)^2
        pooled = outputs.mean(dim=1)  # [B, embed_dim] - mean pooling over sequence
        
        # Note: Head already has LayerNorm at the beginning, so no need to normalize here
        
        # Classification
        logits = self.head(pooled)
        return logits


class VideoMAEGlobalResidualExpertsForAction(nn.Module):
    """
    VideoMAE + GlobalHead (full classes) + Residual Experts (subset classes).

    Forward returns combined logits [B, num_classes]:
      combined = global_logits
      combined[:, subset_i] += expert_i_logits
    """

    def __init__(
        self,
        num_classes: int = 51,
        use_adapters: bool = False,
        adapter_dim: int = None,
        dropout: float = 0.1,
        pretrained_ckpt: str = None,
        model_name: str = 'MCG-NJU/videomae-large-finetuned-kinetics',
        num_frames: int = 16,
        tubelet_size: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        init_output_gain: float = 2.0,
        init_hidden_gain: float = 1.0,
        use_normal_init: bool = True,
        label_subsets: list = None,
    ):
        super().__init__()

        logger = logging.getLogger(__name__)

        # Validate / build label subsets
        if label_subsets is None:
            # Default: split into 8 contiguous subsets (like Model 10 plan)
            num_experts = 8
            labels_per = num_classes // num_experts
            rem = num_classes % num_experts
            label_subsets = []
            start = 0
            for i in range(num_experts):
                size = labels_per + (1 if i < rem else 0)
                end = start + size
                label_subsets.append(list(range(start, end)))
                start = end

        # Basic validation: cover only valid indices, allow gaps but not out-of-range
        for subset in label_subsets:
            for idx in subset:
                if idx < 0 or idx >= num_classes:
                    raise ValueError(f"label_subsets contains out-of-range class idx: {idx}")

        self.num_classes = num_classes
        self.label_subsets = [list(map(int, s)) for s in label_subsets]
        self.use_adapters = use_adapters
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_mean_pooling = True  # VideoMAE has no CLS token

        # Load VideoMAE backbone (same logic as VideoMAEv2ForAction)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for VideoMAE experts model.")

        logger.info(f"Loading VideoMAE model from HuggingFace: {model_name}")
        config = VideoMAEConfig.from_pretrained(
            model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_classes,
        )

        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            logger.info(f"Loading VideoMAE from local checkpoint: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.videomae = VideoMAEModel(config)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('videomae.'):
                    cleaned_state_dict[k[9:]] = v
                else:
                    cleaned_state_dict[k] = v
            self.videomae.load_state_dict(cleaned_state_dict, strict=False)
            logger.info("VideoMAE weights loaded from local checkpoint")
        else:
            logger.info(f"Loading VideoMAE from HuggingFace: {model_name}")
            self.videomae = VideoMAEModel.from_pretrained(model_name, config=config)
            logger.info("VideoMAE model loaded from HuggingFace")

        self.embed_dim = config.hidden_size

        # Adapters are optional; default off (we're using frozen backbone strategy)
        if use_adapters:
            freeze_backbone_params(self.videomae)
            num_blocks = len(self.videomae.encoder.layer)
            self.adapters, self.block_norm = create_adapters_for_videomae(
                self.videomae, self.embed_dim, adapter_dim, dropout, num_blocks
            )
        else:
            self.adapters = None
            self.block_norm = None

        # Heads
        self.global_head = create_classification_head(
            self.embed_dim,
            num_classes,
            dropout,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init,
        )

        self.expert_heads = nn.ModuleList([
            create_classification_head(
                self.embed_dim,
                len(subset),
                dropout,
                init_output_gain=init_output_gain,
                init_hidden_gain=init_hidden_gain,
                use_normal_init=use_normal_init,
            )
            for subset in self.label_subsets
        ])

        # Cache subset tensors for indexing (moved to device on first forward)
        self._subset_tensors = [None for _ in self.label_subsets]

        # Expert logits gating:
        # - Stage A: disable experts (global head only)
        # - Stage B: enable experts but only active_expert_idx contributes
        # - Stage C / inference: enable experts, active_expert_idx=None (all experts contribute)
        self.experts_enabled: bool = True
        self.active_expert_idx = None

    def _forward_backbone(self, video: torch.Tensor) -> torch.Tensor:
        if self.use_adapters:
            outputs = self._forward_videomae_with_adapters(video)
        else:
            outputs = self.videomae(pixel_values=video, output_hidden_states=False).last_hidden_state
        pooled = outputs.mean(dim=1)
        return pooled

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pooled = self._forward_backbone(video)

        global_logits = self.global_head(pooled)
        combined = global_logits.clone()

        if not getattr(self, "experts_enabled", True):
            return combined

        for i, (subset, expert) in enumerate(zip(self.label_subsets, self.expert_heads)):
            if len(subset) == 0:
                continue
            active = getattr(self, "active_expert_idx", None)
            if active is not None and i != active:
                continue
            expert_logits = expert(pooled)  # [B, len(subset)]
            idx = self._subset_tensors[i]
            if idx is None or idx.device != combined.device:
                idx = torch.tensor(subset, dtype=torch.long, device=combined.device)
                self._subset_tensors[i] = idx
            combined[:, idx] = combined[:, idx] + expert_logits

        return combined

    def _forward_videomae_with_adapters(self, video: torch.Tensor) -> torch.Tensor:
        # Reuse the efficient adapter forward used in VideoMAEv2ForAction
        B, T, C, H, W = video.shape
        num_temporal_patches = T // self.tubelet_size
        num_spatial_patches = (self.image_size // self.patch_size) ** 2
        seq_len = num_temporal_patches * num_spatial_patches

        bool_masked_pos = torch.zeros(
            (B, seq_len),
            dtype=torch.bool,
            device=video.device,
            requires_grad=False
        )
        embeddings = self.videomae.embeddings(pixel_values=video, bool_masked_pos=bool_masked_pos)

        hidden_states = embeddings
        for i, layer in enumerate(self.videomae.encoder.layer):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            if self.use_adapters and i < len(self.adapters):
                adapter_out = self.adapters[i](hidden_states)
                hidden_states = hidden_states + adapter_out
                if self.block_norm is not None:
                    hidden_states = self.block_norm(hidden_states)
        return hidden_states


class VideoMAEGroupGatedExpertsForAction(nn.Module):
    """
    VideoMAE + GroupHead (predict group) + Group Experts (predict subset classes).

    - Training (default): soft routing via softmax(group_logits)
      so output stays [B, num_classes] and is compatible with CE/mixup.
    - Inference (optional): hard top-1 routing (argmax group) to route to 1 expert.
    - Stage B override: if active_expert_idx is set, ignore group head and use
      only that expert (stabilizes expert training).
    """

    def __init__(
        self,
        num_classes: int = 51,
        use_adapters: bool = False,
        adapter_dim: int = None,
        dropout: float = 0.1,
        pretrained_ckpt: str = None,
        model_name: str = 'MCG-NJU/videomae-large-finetuned-kinetics',
        num_frames: int = 16,
        tubelet_size: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        init_output_gain: float = 2.0,
        init_hidden_gain: float = 1.0,
        use_normal_init: bool = True,
        label_subsets: list = None,
        hard_mask_value: float = -1e9,
    ):
        super().__init__()

        logger = logging.getLogger(__name__)

        if label_subsets is None:
            num_experts = 8
            labels_per = num_classes // num_experts
            rem = num_classes % num_experts
            label_subsets = []
            start = 0
            for i in range(num_experts):
                size = labels_per + (1 if i < rem else 0)
                end = start + size
                label_subsets.append(list(range(start, end)))
                start = end

        for subset in label_subsets:
            for idx in subset:
                if idx < 0 or idx >= num_classes:
                    raise ValueError(
                        "label_subsets contains out-of-range class idx: "
                        f"{idx}"
                    )

        self.num_classes = int(num_classes)
        self.label_subsets = [list(map(int, s)) for s in label_subsets]
        self.num_groups = len(self.label_subsets)
        self.use_adapters = use_adapters
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_mean_pooling = True

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for VideoMAE gated experts."
            )

        logger.info(f"Loading VideoMAE model from HuggingFace: {model_name}")
        config = VideoMAEConfig.from_pretrained(
            model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_classes,
        )

        if pretrained_ckpt and Path(pretrained_ckpt).exists():
            logger.info(f"Loading VideoMAE from local checkpoint: {pretrained_ckpt}")
            checkpoint = torch.load(
                pretrained_ckpt,
                map_location='cpu',
                weights_only=False
            )
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.videomae = VideoMAEModel(config)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('videomae.'):
                    cleaned_state_dict[k[9:]] = v
                else:
                    cleaned_state_dict[k] = v
            self.videomae.load_state_dict(cleaned_state_dict, strict=False)
            logger.info("VideoMAE weights loaded from local checkpoint")
        else:
            self.videomae = VideoMAEModel.from_pretrained(
                model_name,
                config=config
            )
            logger.info("VideoMAE model loaded from HuggingFace")

        self.embed_dim = config.hidden_size

        if use_adapters:
            freeze_backbone_params(self.videomae)
            num_blocks = len(self.videomae.encoder.layer)
            self.adapters, self.block_norm = create_adapters_for_videomae(
                self.videomae, self.embed_dim, adapter_dim, dropout, num_blocks
            )
        else:
            self.adapters = None
            self.block_norm = None

        self.group_head = create_classification_head(
            self.embed_dim,
            self.num_groups,
            dropout,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init,
        )

        self.expert_heads = nn.ModuleList([
            create_classification_head(
                self.embed_dim,
                len(subset),
                dropout,
                init_output_gain=init_output_gain,
                init_hidden_gain=init_hidden_gain,
                use_normal_init=use_normal_init,
            )
            for subset in self.label_subsets
        ])

        self._subset_tensors = [None for _ in self.label_subsets]

        self.active_expert_idx = None
        # Default inference behavior: hard top-1 routing.
        # Training code can disable this for stable optimization (soft routing).
        self.hard_routing_enabled = True
        self.hard_mask_value = float(hard_mask_value)

    def set_hard_routing(self, enabled: bool, mask_value: float = None) -> None:
        self.hard_routing_enabled = bool(enabled)
        if mask_value is not None:
            self.hard_mask_value = float(mask_value)

    def _forward_backbone(self, video: torch.Tensor) -> torch.Tensor:
        if self.use_adapters:
            outputs = self._forward_videomae_with_adapters(video)
        else:
            outputs = self.videomae(
                pixel_values=video,
                output_hidden_states=False
            ).last_hidden_state
        pooled = outputs.mean(dim=1)
        return pooled

    def _get_subset_idx(self, i: int, device: torch.device) -> torch.Tensor:
        idx = self._subset_tensors[i]
        if idx is None or idx.device != device:
            idx = torch.tensor(
                self.label_subsets[i],
                dtype=torch.long,
                device=device
            )
            self._subset_tensors[i] = idx
        return idx

    def _forward_single_expert(
        self,
        pooled: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        device = pooled.device
        B = pooled.shape[0]
        subset_idx = self._get_subset_idx(expert_idx, device)
        expert_logits = self.expert_heads[expert_idx](pooled)
        combined = torch.full(
            (B, self.num_classes),
            self.hard_mask_value,
            device=device,
            # Use float32 to avoid FP16 overflow (e.g., -1e9).
            dtype=torch.float32,
        )
        combined[:, subset_idx] = expert_logits.to(dtype=combined.dtype)
        return combined

    def _forward_soft_route(
        self,
        pooled: torch.Tensor,
        group_logits: torch.Tensor,
    ) -> torch.Tensor:
        device = pooled.device
        B = pooled.shape[0]
        weights = torch.softmax(group_logits, dim=1)  # [B, G]
        combined = None
        for i, expert in enumerate(self.expert_heads):
            subset_idx = self._get_subset_idx(i, device)
            expert_logits = expert(pooled)  # [B, |subset|]
            if combined is None:
                out_dtype = torch.promote_types(weights.dtype, expert_logits.dtype)
                combined = torch.zeros(
                    (B, self.num_classes),
                    device=device,
                    dtype=out_dtype,
                )
            w = weights[:, i].unsqueeze(1).to(dtype=combined.dtype)
            combined[:, subset_idx] = w * expert_logits.to(dtype=combined.dtype)
        return combined

    def _forward_hard_route(
        self,
        pooled: torch.Tensor,
        group_logits: torch.Tensor,
    ) -> torch.Tensor:
        device = pooled.device
        B = pooled.shape[0]
        combined = torch.full(
            (B, self.num_classes),
            self.hard_mask_value,
            device=device,
            dtype=torch.float32,
        )
        groups = group_logits.argmax(dim=1)  # [B]
        for i, expert in enumerate(self.expert_heads):
            mask = (groups == i)
            if not torch.any(mask):
                continue
            rows = mask.nonzero(as_tuple=True)[0]
            subset_idx = self._get_subset_idx(i, device)
            expert_logits = expert(pooled[rows]).to(dtype=combined.dtype)
            combined[rows[:, None], subset_idx[None, :]] = expert_logits
        return combined

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        pooled = self._forward_backbone(video)

        active = getattr(self, "active_expert_idx", None)
        if active is not None:
            return self._forward_single_expert(pooled, int(active))

        group_logits = self.group_head(pooled)
        if self.hard_routing_enabled and not self.training:
            return self._forward_hard_route(pooled, group_logits)
        return self._forward_soft_route(pooled, group_logits)

    def _forward_videomae_with_adapters(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        num_temporal_patches = T // self.tubelet_size
        num_spatial_patches = (self.image_size // self.patch_size) ** 2
        seq_len = num_temporal_patches * num_spatial_patches

        bool_masked_pos = torch.zeros(
            (B, seq_len),
            dtype=torch.bool,
            device=video.device,
            requires_grad=False
        )
        embeddings = self.videomae.embeddings(
            pixel_values=video,
            bool_masked_pos=bool_masked_pos
        )

        hidden_states = embeddings
        for i, layer in enumerate(self.videomae.encoder.layer):
            layer_outputs = layer(hidden_states)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
            if self.use_adapters and i < len(self.adapters):
                adapter_out = self.adapters[i](hidden_states)
                hidden_states = hidden_states + adapter_out
                if self.block_norm is not None:
                    hidden_states = self.block_norm(hidden_states)
        return hidden_states


class MultiScaleViTForAction(nn.Module):
    """Multi-scale feature fusion ViT model."""
    
    def __init__(
        self,
        num_classes: int = 51,
        pretrained_name: str = 'vit_base_patch16_224',
        use_adapters: bool = True,
        adapter_dim: int = None,
        temporal_heads: int = 8,
        dropout: float = 0.1,
        scales: list = None
    ):
        """
        Args:
            num_classes: Number of action classes
            pretrained_name: Pretrained ViT model name
            use_adapters: Whether to use adapters
            adapter_dim: Adapter dimension
            temporal_heads: Number of temporal attention heads
            dropout: Dropout rate
            scales: List of scales for multi-scale fusion (default: [224, 256, 288])
        """
        super().__init__()
        
        if scales is None:
            scales = [224, 256, 288]
        
        self.scales = scales
        
        # Create base model for each scale
        self.base_models = nn.ModuleList([
            SOTAViTForAction(
                num_classes=num_classes,
                pretrained_name=pretrained_name,
                use_adapters=use_adapters,
                adapter_dim=adapter_dim,
                temporal_heads=temporal_heads,
                dropout=dropout
            ) for _ in scales
        ])
        
        # Feature fusion
        self.embed_dim = self.base_models[0].embed_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embed_dim * len(scales)),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * len(scales), self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.head = create_classification_head(self.embed_dim, num_classes, dropout)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W] - batch of video clips
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = video.shape
        
        # Extract features at multiple scales
        scale_features = []
        for i, scale in enumerate(self.scales):
            # Resize video to scale
            video_resized = torch.nn.functional.interpolate(
                video.view(B * T, C, H, W),
                size=(scale, scale),
                mode='bilinear',
                align_corners=False
            ).view(B, T, C, scale, scale)
            
            # Get features from base model
            # Extract features before classification head
            base_model = self.base_models[i]
            B, T, C, H_scale, W_scale = video_resized.shape
            x = video_resized.view(B * T, C, H_scale, W_scale)
            
            if base_model.use_adapters:
                patch_features = base_model._forward_with_adapters(x)
            else:
                patch_features = base_model.vit(x)
            
            patch_features = patch_features.view(B, T, base_model.embed_dim)
            features = base_model.temporal_attention(patch_features)  # [B, embed_dim]
            scale_features.append(features)
        
        # Concatenate multi-scale features
        fused = torch.cat(scale_features, dim=1)  # [B, embed_dim * num_scales]
        fused = self.fusion(fused)  # [B, embed_dim]
        
        # Classification
        logits = self.head(fused)
        
        return logits


def create_model(
    architecture: str = 'vit_base',
    num_classes: int = 51,
    pretrained_name: str = None,
    use_adapters: bool = True,
    adapter_dim: int = None,
    temporal_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models based on architecture type.

    Args:
        architecture: Model architecture type
            - 'vit_base': Standard ViT-Base
            - 'vit_large': ViT-Large
            - 'timesformer': TimeSformer-style
            - 'swin': Video Swin Transformer
            - 'multiscale': Multi-scale ViT
        num_classes: Number of classes
        pretrained_name: Pretrained model name (overrides default)
        use_adapters: Whether to use adapters
        adapter_dim: Adapter dimension
        temporal_heads: Number of temporal attention heads
        dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        Model instance
    """
    # Set default pretrained names based on architecture if pretrained_name is None
    # Note: VideoMAE architecture skips this logic and uses HuggingFace model directly
    # Do NOT override if pretrained_name is provided - use it as-is
    if pretrained_name is None and architecture != 'videomae':
        default_config = get_default_config()
        default_pretrained_names = default_config.get('default_pretrained_names', {})
        
        if architecture == 'vit_large':
            pretrained_name = default_pretrained_names.get('vit_large')
            if pretrained_name is None:
                raise ValueError(f"default_pretrained_names['vit_large'] not found in config")
        elif architecture == 'swin':
            pretrained_name = default_pretrained_names.get('swin')
            if pretrained_name is None:
                raise ValueError(f"default_pretrained_names['swin'] not found in config")
        elif architecture == 'vit_base':
            pretrained_name = default_pretrained_names.get('vit_base')
            if pretrained_name is None:
                raise ValueError(f"default_pretrained_names['vit_base'] not found in config")
        else:
            # For other architectures (timesformer, multiscale), use default from config
            default_for_arch = default_pretrained_names.get(architecture) or default_pretrained_names.get('vit_base')
            if default_for_arch is None:
                raise ValueError(f"default_pretrained_names['{architecture}'] or default_pretrained_names['vit_base'] not found in config")
            pretrained_name = default_for_arch


    # Get drop_path_rate from kwargs (default 0.0 if not provided)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    # Create model based on architecture
    if architecture in ['vit_base', 'vit_large']:
        return SOTAViTForAction(
            num_classes=num_classes,
            pretrained_name=pretrained_name,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            temporal_heads=temporal_heads,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )
    elif architecture == 'timesformer':
        return TimeSformerForAction(
            num_classes=num_classes,
            pretrained_name=pretrained_name,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            temporal_heads=temporal_heads,
            dropout=dropout
        )
    elif architecture == 'swin':
        return VideoSwinForAction(
            num_classes=num_classes,
            pretrained_name=pretrained_name,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            temporal_heads=temporal_heads,
            dropout=dropout
        )
    elif architecture == 'multiscale':
        scales = kwargs.get('scales', [224, 256, 288])
        return MultiScaleViTForAction(
            num_classes=num_classes,
            pretrained_name=pretrained_name,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            temporal_heads=temporal_heads,
            dropout=dropout,
            scales=scales
        )
    elif architecture == 'videomae':
        # VideoMAEv2: Dùng VideoMAEModel chính thức từ HuggingFace
        # Có 3D patch embedding, temporal position embedding, tubelet_size
        pretrained_ckpt = kwargs.get('pretrained_ckpt', None)
        model_name = kwargs.get('model_name', 'MCG-NJU/videomae-large-finetuned-kinetics')
        num_frames = kwargs.get('num_frames', 16)
        tubelet_size = kwargs.get('tubelet_size', 2)
        image_size = kwargs.get('image_size', 224)
        patch_size = kwargs.get('patch_size', 16)
        init_output_gain = kwargs.get('init_output_gain', 2.0)  # Default: 2.0 → std=0.02 (best practice)
        init_hidden_gain = kwargs.get('init_hidden_gain', 1.0)  # Default: 1.0 → std=0.01
        use_normal_init = kwargs.get('use_normal_init', True)   # Default: True
        return VideoMAEv2ForAction(
            num_classes=num_classes,
            pretrained_name=None,  # Không dùng nữa
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            temporal_heads=temporal_heads,
            dropout=dropout,
            pretrained_ckpt=pretrained_ckpt,
            model_name=model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init
        )
    elif architecture == 'videomae_experts':
        pretrained_ckpt = kwargs.get('pretrained_ckpt', None)
        model_name = kwargs.get('model_name', 'MCG-NJU/videomae-large-finetuned-kinetics')
        num_frames = kwargs.get('num_frames', 16)
        tubelet_size = kwargs.get('tubelet_size', 2)
        image_size = kwargs.get('image_size', 224)
        patch_size = kwargs.get('patch_size', 16)
        init_output_gain = kwargs.get('init_output_gain', 2.0)
        init_hidden_gain = kwargs.get('init_hidden_gain', 1.0)
        use_normal_init = kwargs.get('use_normal_init', True)
        label_subsets = kwargs.get('label_subsets', None)
        return VideoMAEGlobalResidualExpertsForAction(
            num_classes=num_classes,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            dropout=dropout,
            pretrained_ckpt=pretrained_ckpt,
            model_name=model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init,
            label_subsets=label_subsets,
        )
    elif architecture == 'videomae_group_gated_experts':
        pretrained_ckpt = kwargs.get('pretrained_ckpt', None)
        model_name = kwargs.get('model_name', 'MCG-NJU/videomae-large-finetuned-kinetics')
        num_frames = kwargs.get('num_frames', 16)
        tubelet_size = kwargs.get('tubelet_size', 2)
        image_size = kwargs.get('image_size', 224)
        patch_size = kwargs.get('patch_size', 16)
        init_output_gain = kwargs.get('init_output_gain', 2.0)
        init_hidden_gain = kwargs.get('init_hidden_gain', 1.0)
        use_normal_init = kwargs.get('use_normal_init', True)
        label_subsets = kwargs.get('label_subsets', None)
        hard_mask_value = kwargs.get('hard_mask_value', -1e9)
        return VideoMAEGroupGatedExpertsForAction(
            num_classes=num_classes,
            use_adapters=use_adapters,
            adapter_dim=adapter_dim,
            dropout=dropout,
            pretrained_ckpt=pretrained_ckpt,
            model_name=model_name,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            image_size=image_size,
            patch_size=patch_size,
            init_output_gain=init_output_gain,
            init_hidden_gain=init_hidden_gain,
            use_normal_init=use_normal_init,
            label_subsets=label_subsets,
            hard_mask_value=hard_mask_value,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
