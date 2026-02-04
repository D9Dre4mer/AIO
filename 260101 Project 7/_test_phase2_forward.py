"""Quick test: Phase2 forward with expert_training_subsets (shrunk subset)."""
from sota_training.models import VideoMAEAlphaExpertsForAction, create_classification_head
import torch

num_classes = 51
label_subsets_phase1 = [list(range(i * 6, min((i + 1) * 6, 51))) for i in range(8)]
label_subsets_phase1[0] = list(range(0, 7))
phase2_main0 = [0, 1, 2, 3, 4, 5]
reserved = [list(range(6, 13)), list(range(13, 20))]
label_subsets_phase2 = [phase2_main0] + [label_subsets_phase1[i] for i in range(1, 8)] + reserved
model = VideoMAEAlphaExpertsForAction(
    51,
    label_subsets=label_subsets_phase1 + [[0], [0]],
    num_frames=16,
    image_size=224,
    patch_size=16,
    tubelet_size=2,
)
for r in range(2):
    model.expert_heads[8 + r] = create_classification_head(model.embed_dim, len(reserved[r]), 0.1)
model.label_subsets = label_subsets_phase2
model._subset_tensors = [None for _ in label_subsets_phase2]
model.expert_training_subsets = label_subsets_phase1 + reserved
model.num_active_experts = 10
model.inference_single_best_expert = True
x = torch.randn(2, 16, 3, 224, 224)
out = model(x)
print("out shape", out.shape)
print("OK")
