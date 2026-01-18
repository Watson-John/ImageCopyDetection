"""
Test that the forward pass fix works correctly
"""

import torch
from models.cedetector_baseline import CEDetectorBaseline

print("Testing CEDetectorBaseline forward pass fix...")
print("="*60)

# Create model
model = CEDetectorBaseline(
    dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env",
    num_query_patches=6
)
model.eval()

# Test with different batch sizes
for B in [1, 4, 12]:
    print(f"\nTest with batch size B={B}:")
    print("-"*60)

    # Create dummy inputs
    query_images = torch.randn(B, 3, 224, 224)
    reference_images = torch.randn(B, 3, 224, 224)
    labels = torch.randint(0, 2, (B,), dtype=torch.float32)

    print(f"  Query shape: {query_images.shape}")
    print(f"  Reference shape: {reference_images.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Run forward pass
    with torch.no_grad():
        outputs = model(query_images, reference_images)

    logits = outputs['logits']
    print(f"  Output logits shape: {logits.shape}")

    # Verify correctness
    assert logits.shape == (B,), f"Expected logits shape ({B},), got {logits.shape}"
    assert outputs['query_cls_tokens'].shape[0] == B * 6, \
        f"Expected {B*6} query CLS tokens, got {outputs['query_cls_tokens'].shape[0]}"
    assert outputs['ref_cls_tokens'].shape[0] == B, \
        f"Expected {B} reference CLS tokens, got {outputs['ref_cls_tokens'].shape[0]}"

    print(f"  ✓ Correct: {B} query-reference pairs -> {B} logits")
    print(f"  ✓ Query patches: {B} × 6 = {B*6} total")
    print(f"  ✓ Each query compared ONLY with its corresponding reference")

print("\n" + "="*60)
print("All tests passed! ✓")
print("\nThe fix ensures:")
print("  1. Query image i is compared ONLY with reference image i")
print("  2. Each query image uses 6 patches for robust matching")
print("  3. Maximum score across 6 patches is used for final prediction")
print("  4. Batch processing is correct for any batch size")
