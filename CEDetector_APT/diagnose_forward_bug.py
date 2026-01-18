"""
Diagnose the forward pass bug in CEDetectorBaseline
"""

import torch
from models.cedetector_baseline import CEDetectorBaseline

# Create dummy model
model = CEDetectorBaseline(
    dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env",
    num_query_patches=6
)

# Create dummy batch (B=4 for simplicity)
B = 4
query_images = torch.randn(B, 3, 224, 224)
reference_images = torch.randn(B, 3, 224, 224)
labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)  # 4 binary labels

print(f"Batch size: {B}")
print(f"Query images shape: {query_images.shape}")
print(f"Reference images shape: {reference_images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"\nExpected: {B} logits (one per query-reference pair)")

# Run forward pass
with torch.no_grad():
    outputs = model(query_images, reference_images)
    logits = outputs['logits']

print(f"\nActual logits shape: {logits.shape}")
print(f"Actual logits values: {logits}")

# Show the bug
print(f"\n{'='*60}")
print("BUG ANALYSIS:")
print(f"{'='*60}")
print(f"Query patches created: {B} images × 6 patches/image = {B*6} patches")
print(f"Reference images: {B}")
print(f"\nCurrent buggy logic:")
print(f"  - Compares ALL {B*6} query patches against ALL {B} references")
print(f"  - Creates {B*6} × {B} = {B*6*B} comparisons")
print(f"  - Takes max across {B*6} patches dimension")
print(f"  - Results in {B} scores")
print(f"\nProblem: Each query patch from image i is being compared")
print(f"         with reference images from ALL images (0..{B-1}), not just image i!")
print(f"\nCorrect logic should be:")
print(f"  - Compare patches from query_i ONLY with reference_i")
print(f"  - Create {B} × 6 = {B*6} comparisons (6 per pair)")
print(f"  - Take max across 6 patches per query")
print(f"  - Result in {B} scores (one per pair)")
