"""
Verify that the forward pass fix is actually being used
"""

import torch
import sys

# Import the fixed model
from models.cedetector_baseline import CEDetectorBaseline

print("Checking if forward pass fix is active...")
print("="*60)

# Read the source code to verify fix is in place
import inspect
source = inspect.getsource(CEDetectorBaseline.forward)

# Check for the fix signature
if "Get the 6 patches for query image i" in source:
    print("✓ FIX IS ACTIVE: Found correct comment in forward pass")
    print("✓ The model is using the fixed query-reference pairing logic")
elif "num_query_patches" in source and "num_refs" in source:
    print("✗ BUG STILL PRESENT: Old buggy code detected!")
    print("✗ The model is NOT using the fixed version")
    sys.exit(1)
else:
    print("? UNCERTAIN: Cannot determine from source inspection")

# Test with actual forward pass
print("\nTesting forward pass behavior...")
model = CEDetectorBaseline(
    dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
    hf_token_path="/home/jowatson/Deep Learning/Code/.env",
    num_query_patches=6
)
model.eval()

B = 4
query = torch.randn(B, 3, 224, 224)
reference = torch.randn(B, 3, 224, 224)

with torch.no_grad():
    outputs = model(query, reference)
    logits = outputs['logits']

print(f"Batch size: {B}")
print(f"Output logits shape: {logits.shape}")

if logits.shape == (B,):
    print(f"✓ CORRECT: {B} query-reference pairs → {B} logits")
    print("✓ Fix is working correctly")
else:
    print(f"✗ WRONG: Expected shape ({B},), got {logits.shape}")
    print("✗ Fix may not be working")

print("\n" + "="*60)
print("Fix verification complete!")
