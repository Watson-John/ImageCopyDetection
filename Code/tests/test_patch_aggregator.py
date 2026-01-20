"""
Tests for PatchAggregator.

Verifies:
- Coordinates are in [-1, 1] range
- Token count decreases for uniform images (fewer splits)
- Output embeddings have correct dimension
"""

import pytest
import torch

from adaptive_dino_icd.apt import EntropyScorer, PatchSelector, PatchAggregator


class TestPatchAggregator:
    """Tests for the PatchAggregator class."""

    def test_coords_in_valid_range(self, patch_aggregator, batch_images, entropy_scorer, patch_selector):
        """Verify coordinates are in [-1, 1] range."""
        # Get entropy and patches
        entropy_maps = entropy_scorer(batch_images)
        patches = patch_selector(entropy_maps, image_size=(224, 224))

        # Get tokens and coords
        tokens_list, coords_list = patch_aggregator(batch_images, patches)

        for coords in coords_list:
            if coords.numel() > 0:
                assert coords.min() >= -1.0, f"Coordinates have values < -1: {coords.min()}"
                assert coords.max() <= 1.0, f"Coordinates have values > 1: {coords.max()}"

    def test_token_count_uniform_vs_textured(
        self, patch_aggregator, uniform_image, textured_image, entropy_scorer, patch_selector
    ):
        """Verify token count is lower for uniform images (fewer splits)."""
        # Process uniform image
        uniform_entropy = entropy_scorer(uniform_image)
        uniform_patches = patch_selector(uniform_entropy, image_size=(224, 224))
        uniform_tokens, _ = patch_aggregator(uniform_image, uniform_patches)
        uniform_count = uniform_tokens[0].shape[0]

        # Process textured image
        textured_entropy = entropy_scorer(textured_image)
        textured_patches = patch_selector(textured_entropy, image_size=(224, 224))
        textured_tokens, _ = patch_aggregator(textured_image, textured_patches)
        textured_count = textured_tokens[0].shape[0]

        assert uniform_count <= textured_count, (
            f"Uniform image should have fewer or equal tokens ({uniform_count}) "
            f"compared to textured image ({textured_count})"
        )

    def test_output_embedding_dimension(self, patch_aggregator, batch_images, entropy_scorer, patch_selector):
        """Verify output embeddings have correct dimension."""
        entropy_maps = entropy_scorer(batch_images)
        patches = patch_selector(entropy_maps, image_size=(224, 224))
        tokens_list, _ = patch_aggregator(batch_images, patches)

        for tokens in tokens_list:
            if tokens.numel() > 0:
                assert tokens.shape[-1] == patch_aggregator.embed_dim, (
                    f"Expected embed_dim {patch_aggregator.embed_dim}, "
                    f"got {tokens.shape[-1]}"
                )

    def test_forward_padded_output(self, patch_aggregator, batch_images, entropy_scorer, patch_selector):
        """Test padded forward pass returns correct structure."""
        entropy_maps = entropy_scorer(batch_images)
        patches = patch_selector(entropy_maps, image_size=(224, 224))
        output = patch_aggregator.forward_padded(batch_images, patches)

        B = batch_images.shape[0]

        # Check all required keys
        assert "tokens" in output
        assert "coords" in output
        assert "attention_mask" in output
        assert "token_counts" in output

        # Check batch dimension
        assert output["tokens"].shape[0] == B
        assert output["coords"].shape[0] == B
        assert output["attention_mask"].shape[0] == B
        assert output["token_counts"].shape[0] == B

        # Check embedding dimension
        assert output["tokens"].shape[-1] == patch_aggregator.embed_dim
        assert output["coords"].shape[-1] == 2

    def test_attention_mask_validity(self, patch_aggregator, batch_images, entropy_scorer, patch_selector):
        """Verify attention mask correctly marks valid tokens."""
        entropy_maps = entropy_scorer(batch_images)
        patches = patch_selector(entropy_maps, image_size=(224, 224))
        output = patch_aggregator.forward_padded(batch_images, patches)

        token_counts = output["token_counts"]
        attention_mask = output["attention_mask"]

        for i in range(batch_images.shape[0]):
            count = token_counts[i].item()
            mask = attention_mask[i]

            # First 'count' positions should be True
            assert mask[:count].all(), f"Sample {i}: First {count} positions should be True"

            # Remaining positions should be False
            if count < mask.shape[0]:
                assert not mask[count:].any(), f"Sample {i}: Positions after {count} should be False"

    def test_empty_patches_handling(self, patch_aggregator):
        """Test handling of empty patch list."""
        images = torch.rand(1, 3, 224, 224)
        empty_patches = [[]]  # Empty patch list for one image

        tokens_list, coords_list = patch_aggregator(images, empty_patches)

        assert len(tokens_list) == 1
        assert tokens_list[0].shape[0] == 0
        assert coords_list[0].shape[0] == 0

    def test_deterministic_output(self, patch_aggregator, batch_images, entropy_scorer, patch_selector):
        """Verify output is deterministic."""
        entropy_maps = entropy_scorer(batch_images)
        patches = patch_selector(entropy_maps, image_size=(224, 224))

        tokens_1, coords_1 = patch_aggregator(batch_images, patches)
        tokens_2, coords_2 = patch_aggregator(batch_images, patches)

        for t1, t2 in zip(tokens_1, tokens_2):
            assert torch.allclose(t1, t2, atol=1e-6)

        for c1, c2 in zip(coords_1, coords_2):
            assert torch.allclose(c1, c2, atol=1e-6)


class TestZeroMLP:
    """Tests for the ZeroMLP class."""

    def test_zero_initialization(self):
        """Verify ZeroMLP is zero-initialized."""
        from adaptive_dino_icd.apt.patch_aggregator import ZeroMLP

        mlp = ZeroMLP(in_dim=64, hidden_dim=128, out_dim=64)

        for name, param in mlp.named_parameters():
            assert torch.all(param == 0), f"Parameter {name} is not zero-initialized"

    def test_zero_mlp_output(self):
        """Verify ZeroMLP produces zero output initially."""
        from adaptive_dino_icd.apt.patch_aggregator import ZeroMLP

        mlp = ZeroMLP(in_dim=64, hidden_dim=128, out_dim=64)
        x = torch.randn(4, 64)

        output = mlp(x)

        assert torch.allclose(output, torch.zeros_like(output)), (
            "ZeroMLP should produce zero output with zero-initialized weights"
        )
