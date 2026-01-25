# SOTA Training Pipeline - Video Action Recognition

Pipeline training SOTA ViT model cho video action recognition với các kỹ thuật nâng cao: EMA, CutMix, Focal Loss, Progressive Resize, Enhanced TTA.

## Cài đặt

```bash
# Cài đặt dependencies
pip install torch torchvision timm tqdm pillow matplotlib numpy pandas
```

## Cấu trúc

```
sota_training/          # Main package
├── config.py          # Configuration
├── models.py          # Model architectures
├── dataset.py         # Dataset classes
├── augmentation.py    # Augmentation functions
├── losses.py          # Loss functions
├── training.py        # Training functions
├── inference.py       # Inference với TTA
├── utils.py           # Utilities
└── main.py            # Main entry point
```

## Sử dụng

### Auto Mode (Khuyến nghị) ⭐

```bash
# Train tự động 7 models với SOTA features và ensemble - CHỈ CẦN 1 LỆNH DUY NHẤT
python train.py
```

**Auto mode sẽ tự động:**
1. Train 5 models với **variation tối ưu** về:
   - **Model 1**: ViT-Large (base teacher, larger capacity)
   - **Model 2**: ViT-Base (student from Model 1)
   - **Model 3**: TimeSformer (student from Model 2, space-time attention)
   - **Model 4**: Swin Transformer (student from Model 3, hierarchical)
   - **Model 5**: Multi-scale ViT (student from Model 4, multi-scale fusion)
   - **Model 6**: ViT-Base, CutMix, CrossEntropy, Progressive Resize
   - **Model 7**: VideoMAEv2 (ViT-Large + divided space-time attention, SOTA - 90% trên Kinetics-400)
2. Mỗi model có **hyperparameters khác nhau** (learning rate, mixup/cutmix alpha, label smoothing)
3. Mỗi model có **seed khác nhau** để tạo diversity
4. Tự động **ensemble** tất cả models với weighted average (theo validation accuracy)
5. Tạo file submission cuối cùng: `submissions/ensemble_submission.csv`

**Lưu ý:** Nếu bạn muốn override và dùng cùng config cho tất cả models:
```bash
python train.py --use-ema --use-cutmix --use-focal-loss --use-progressive-resize
```

### Training Manual

```bash
# Train model đơn giản
python train.py --model-id 1 --seed 42

# Train với tất cả Phase 1 features
python train.py --model-id 1 --seed 42 --use-vit-large --use-ema --use-cutmix --use-focal-loss --use-progressive-resize

# Train nhiều models (không auto ensemble)
python train.py --num-models 3 --use-ema --use-cutmix --no-auto-ensemble

# Resume training
python train.py --model-id 1 --resume checkpoints/sota_vit_model_1_best.pt
```

### Inference

```bash
# Inference với checkpoint
python train.py --model-id 1 --inference-only --checkpoint checkpoints/sota_vit_model_1_best.pt
```

## Các tùy chọn chính

- `--model-id`: Model ID (1-7, Model 7 là VideoMAEv2)
- `--seed`: Random seed
- `--use-vit-large`: Sử dụng ViT-Large thay vì ViT-Base
- `--use-ema`: Enable Exponential Moving Average
- `--use-cutmix`: Enable CutMix augmentation
- `--use-focal-loss`: Enable Focal Loss
- `--use-progressive-resize`: Enable Progressive Resizing
- `--data-dir`: Path to dataset (default: `./kaggle_data/data`)
- `--output-dir`: Output directory (default: `./checkpoints`)
- `--epochs`: Number of epochs (default: 50)
- `--batch-size`: Batch size (default: 12)

## Output

- **Checkpoints**: `checkpoints/sota_vit_model_{id}_best.pt` (cho models 1-6), `checkpoints/videomae_model_7_best.pt` (cho Model 7)
- **Training plots**: `checkpoints/sota_vit_model_{id}_training.png` (cho models 1-6), `checkpoints/videomae_model_7_training.png` (cho Model 7)
- **Submissions**: 
  - `submissions/submission_model_{id}.csv` (individual models)
  - `submissions/ensemble_submission.csv` (final ensemble - **FILE NÀY DÙNG ĐỂ SUBMIT**)
- **Logs**: `logging/training_model_{id}_{timestamp}.log`

## Dataset Structure

```
kaggle_data/data/
├── data_train/        # Training data (class folders)
└── test/              # Test data (video ID folders)
```

## Examples

```bash
# ⭐ KHUYẾN NGHỊ: Train tự động 7 models với SOTA features variation (CHỈ CẦN 1 LỆNH)
python train.py

# Train model đơn lẻ với config mặc định
python train.py --model-id 1 --seed 42

# Train model đơn lẻ với tất cả SOTA features
python train.py --model-id 1 --seed 42 --use-ema --use-cutmix --use-focal-loss --use-progressive-resize

# Train với ViT-Large và tất cả features
python train.py --model-id 2 --seed 123 --use-vit-large --use-ema --use-cutmix --use-focal-loss --use-progressive-resize
```

## Train VideoMAEv2 (Script riêng) ⭐

Script riêng để train VideoMAEv2 với config tối ưu dựa trên research và best practices:

```bash
# Train VideoMAEv2 với config tối ưu mặc định
python train_videomae.py

# Train với custom parameters
python train_videomae.py --batch-size 8 --num-frames 16 --base-lr 1e-4

# Resume training
python train_videomae.py --resume checkpoints/videomae_model_7_best.pt
```

**Config tối ưu (dựa trên research):**
- **Batch size**: 16 (ViT-Large tốn memory, có thể giảm xuống 8 nếu OOM)
- **Num frames**: 16 (tốt cho temporal modeling)
- **Learning rate**: Backbone 1e-4, Head 2e-3 (theo research cho ViT-Large)
- **Weight decay**: 0.05 (moderate regularization)
- **Dropout**: 0.3 (moderate dropout)
- **Drop path rate**: 0.2 (Stochastic Depth cho ViT-Large)
- **Warmup**: 5 epochs
- **Epochs**: 100
- **Augmentation**: TẮT HẾT (để training nhanh nhất)

**VideoMAEv2 advantages:**
- ✓ ViT-Large backbone (307M params) - lớn hơn ViT-Base
- ✓ Divided space-time attention (tốt hơn simple temporal attention)
- ✓ SOTA accuracy: 90% trên Kinetics-400 (cao nhất trong các models)

**Output:**
- Checkpoint: `checkpoints/videomae_model_{id}_best.pt`
- Training plot: `checkpoints/videomae_model_{id}_training.png`
- Submission: `submissions/submission_videomae_model_{id}.csv`

## Train TimeSformer (Script riêng) ⭐

Script riêng để train TimeSformer với config tối ưu dựa trên research và best practices:

```bash
# Train TimeSformer với config tối ưu mặc định
python train_timesformer.py

# Train với custom parameters
python train_timesformer.py --batch-size 64 --num-frames 16 --base-lr 5e-5

# Resume training
python train_timesformer.py --resume checkpoints/timesformer_model_3_best.pt
```

**Config tối ưu (dựa trên research):**
- **Batch size**: 32 (tối ưu cho tốc độ, space-time attention tốn memory)
- **Num frames**: 16 (tốt cho temporal modeling)
- **Learning rate**: Backbone 5e-5, Head 1e-3 (TimeSformer cần LR thấp hơn)
- **Weight decay**: 0.1 (TimeSformer cần regularization mạnh)
- **Dropout**: 0.4 (TimeSformer cần regularization mạnh)
- **Warmup**: 5 epochs
- **Epochs**: 100
- **Augmentation**: TẮT HẾT (để training nhanh nhất)

**TimeSformer advantages:**
- ✓ Divided space-time attention (tốt hơn simple temporal attention)
- ✓ True spatiotemporal modeling (không chỉ xử lý từng frame riêng)
- ✓ Tốt hơn model Swin hiện tại cho temporal reasoning

**Output:**
- Checkpoint: `checkpoints/timesformer_model_{id}_best.pt`
- Training plot: `checkpoints/timesformer_model_{id}_training.png`
- Submission: `submissions/submission_timesformer_model_{id}.csv`

## Train Swin Transformer (Script riêng)

Script riêng để train Swin Transformer với config tối ưu dựa trên research và best practices:

```bash
# Train Swin với config tối ưu mặc định
python train_swin.py

# Train với custom parameters
python train_swin.py --batch-size 64 --num-frames 32 --base-lr 3e-5

# Resume training
python train_swin.py --resume checkpoints/swin_model_4_best.pt
```

**Config tối ưu (dựa trên research):**
- **Batch size**: 64 (tối ưu cho tốc độ, tránh CPU bottleneck)
- **Num frames**: 16 (tối ưu cho tốc độ, có thể tăng lên 32 nếu muốn accuracy cao hơn)
- **Gradient accumulation**: 2 steps (effective batch size ~128)
- **Num workers**: 2 (giảm CPU bottleneck trên Windows)
- **Learning rate**: Backbone 3e-5, Head 3e-4 (tỷ lệ 10x)
- **Drop path rate**: 0.3 (cho Swin-B)
- **Weight decay**: 0.05 (cho Swin-B)
- **Warmup**: 3 epochs
- **Epochs**: 100
- **Augmentation**: TẮT HẾT (để training nhanh nhất)

**Output:**
- Checkpoint: `checkpoints/swin_model_{id}_best.pt`
- Training plot: `checkpoints/swin_model_{id}_training.png`
- Submission: `submissions/submission_swin_model_{id}.csv`

## Notes

- Windows: `num_workers=0` (default) để tránh multiprocessing issues
- Dataset path: `./kaggle_data/data/data_train` và `./kaggle_data/data/test`
- Logs được lưu tự động trong `logging/` directory
- Enhanced TTA: 10 crops + 2 flips cho inference
