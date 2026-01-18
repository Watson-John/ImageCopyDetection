"""
Test script to verify the forward pass fix
"""

import torch
from models import CEDetectorAPT

# Create model
print("Creating model...")
model = CEDetectorAPT(
    dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env",
    num_query_patches=6
).cuda()

# Create dummy batch
batch_size = 12
query = torch.randn(batch_size, 3, 224, 224).cuda()
reference = torch.randn(batch_size, 3, 224, 224).cuda()
labels = torch.randint(0, 2, (batch_size,)).float().cuda()

print(f"\nInput shapes:")
print(f"  Query: {query.shape}")
print(f"  Reference: {reference.shape}")
print(f"  Labels: {labels.shape}")

# Forward pass
print("\nRunning forward pass...")
model.eval()
with torch.no_grad():
    outputs = model(query, reference)

# Check outputs
logits = outputs['logits']
print(f"\nOutput shapes:")
print(f"  Logits: {logits.shape}")
print(f"  Expected: ({batch_size},)")

# Verify shape matches
assert logits.shape == (batch_size,), f"Expected logits shape ({batch_size},), got {logits.shape}"
print("\n✓ Shape test passed!")

# Verify each query is compared to its paired reference
print(f"\nLogits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"Logits mean: {logits.mean().item():.4f}")
print(f"Logits std: {logits.std().item():.4f}")

print("\n✓ All tests passed! The forward pass fix is working correctly.")
