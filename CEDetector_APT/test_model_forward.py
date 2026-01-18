"""
Test script to verify CEDetector Baseline model outputs correct shapes for CEDLoss
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.cedetector_baseline import CEDetectorBaseline
from utils.losses import CEDLoss

def test_model_forward():
    """Test that model outputs have correct shapes for CEDLoss."""

    print("=" * 60)
    print("Testing CEDetector Baseline Model Forward Pass")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = CEDetectorBaseline(
        dinov3_model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token_path="/home/jowatson/Deep Learning/Code/.env",
        freeze_backbone=False,
        num_query_patches=6,
        k_neighbors=10
    )
    model.eval()

    print(f"   Model created with embed_dim: {model.embed_dim}")

    # Create dummy data
    print("\n2. Creating dummy data...")
    batch_size = 4
    query = torch.randn(batch_size, 3, 224, 224)
    reference = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,)).float()

    print(f"   Query shape: {query.shape}")
    print(f"   Reference shape: {reference.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():
        outputs = model(query, reference)

    # Check outputs
    print("\n4. Checking outputs...")
    required_keys = ['logits', 'query_cls_tokens', 'ref_cls_tokens', 'patch_embeddings']

    for key in required_keys:
        if key in outputs:
            shape = outputs[key].shape
            print(f"   ✓ {key}: {shape}")
        else:
            print(f"   ✗ {key}: MISSING!")
            return False

    # Extract components
    logits = outputs['logits']
    query_cls_tokens = outputs['query_cls_tokens']
    ref_cls_tokens = outputs['ref_cls_tokens']
    patch_embeddings = outputs['patch_embeddings']

    # Verify shapes
    print("\n5. Verifying shapes for CEDLoss...")

    # Average query CLS tokens across patches
    num_patches_per_image = query_cls_tokens.size(0) // batch_size
    print(f"   Number of patches per image: {num_patches_per_image}")

    query_cls_tokens_avg = query_cls_tokens.reshape(batch_size, num_patches_per_image, -1).mean(dim=1)
    patch_embeddings_avg = patch_embeddings.reshape(batch_size, num_patches_per_image, -1).mean(dim=1)

    print(f"   Query CLS tokens (averaged): {query_cls_tokens_avg.shape}")
    print(f"   Reference CLS tokens: {ref_cls_tokens.shape}")
    print(f"   Patch embeddings (averaged): {patch_embeddings_avg.shape}")
    print(f"   Logits: {logits.shape}")
    print(f"   Labels: {labels.shape}")

    # Expected shapes
    expected_shapes = {
        'query_cls_tokens_avg': (batch_size, model.embed_dim),
        'ref_cls_tokens': (batch_size, model.embed_dim),
        'patch_embeddings_avg': (batch_size, model.embed_dim),
        'logits': (batch_size,),
        'labels': (batch_size,)
    }

    shapes_correct = True
    for name, expected_shape in expected_shapes.items():
        if name == 'query_cls_tokens_avg':
            actual_shape = query_cls_tokens_avg.shape
        elif name == 'ref_cls_tokens':
            actual_shape = ref_cls_tokens.shape
        elif name == 'patch_embeddings_avg':
            actual_shape = patch_embeddings_avg.shape
        elif name == 'logits':
            actual_shape = logits.shape
        elif name == 'labels':
            actual_shape = labels.shape

        if actual_shape == expected_shape:
            print(f"   ✓ {name}: {actual_shape} (expected: {expected_shape})")
        else:
            print(f"   ✗ {name}: {actual_shape} (expected: {expected_shape})")
            shapes_correct = False

    # Test CEDLoss
    print("\n6. Testing CEDLoss...")
    criterion = CEDLoss()

    try:
        loss_dict = criterion(
            cls_tokens_i=query_cls_tokens_avg,
            cls_tokens_j=ref_cls_tokens,
            patch_embeddings=patch_embeddings_avg,
            labels=labels,
            logits=logits.unsqueeze(1) if logits.dim() == 1 else logits
        )

        print(f"   ✓ CEDLoss computed successfully!")
        print(f"      Total Loss: {loss_dict['total_loss'].item():.4f}")
        print(f"      Contrast Loss: {loss_dict['contrast_loss'].item():.4f}")
        print(f"      SimCLR Loss: {loss_dict['simclr_loss'].item():.4f}")
        print(f"      KL Loss: {loss_dict['kl_loss'].item():.4f}")
        print(f"      MSL Loss: {loss_dict['msl_loss'].item():.4f}")
        print(f"      BCE Loss: {loss_dict['bce_loss'].item():.4f}")

    except Exception as e:
        print(f"   ✗ CEDLoss failed with error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_model_forward()
    sys.exit(0 if success else 1)
