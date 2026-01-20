"""
Tests for EntropyScorer.

Verifies:
- Uniform images have lower entropy than textured images
- Entropy values are in valid range [0, 1] after normalization
- Multi-scale outputs have correct shapes
"""

import pytest
import torch


class TestEntropyScorer:
    """Tests for the EntropyScorer class."""

    def test_uniform_vs_textured_entropy(self, entropy_scorer, uniform_image, textured_image):
        """Verify that uniform images have lower entropy than textured images."""
        # Compute entropy for both images
        uniform_entropy = entropy_scorer(uniform_image)
        textured_entropy = entropy_scorer(textured_image)

        # Check at each scale
        for scale in entropy_scorer.scales:
            uniform_mean = uniform_entropy[scale].mean().item()
            textured_mean = textured_entropy[scale].mean().item()

            assert uniform_mean < textured_mean, (
                f"At scale {scale}: uniform entropy ({uniform_mean:.4f}) "
                f"should be less than textured entropy ({textured_mean:.4f})"
            )

    def test_entropy_range(self, entropy_scorer, batch_images):
        """Verify entropy values are in [0, 1] range when normalized."""
        entropy_maps = entropy_scorer(batch_images)

        for scale, entropy_map in entropy_maps.items():
            assert entropy_map.min() >= 0.0, f"Entropy at scale {scale} has values < 0"
            assert entropy_map.max() <= 1.0, f"Entropy at scale {scale} has values > 1"

    def test_output_shapes(self, entropy_scorer, batch_images):
        """Verify multi-scale outputs have correct shapes."""
        B, C, H, W = batch_images.shape
        entropy_maps = entropy_scorer(batch_images)

        for scale in entropy_scorer.scales:
            expected_h = H // scale
            expected_w = W // scale
            expected_shape = (B, expected_h, expected_w)

            assert scale in entropy_maps, f"Missing entropy map for scale {scale}"
            assert entropy_maps[scale].shape == expected_shape, (
                f"Scale {scale}: expected shape {expected_shape}, "
                f"got {entropy_maps[scale].shape}"
            )

    def test_combined_entropy(self, entropy_scorer, batch_images):
        """Test combining entropy maps from multiple scales."""
        entropy_maps = entropy_scorer(batch_images)
        combined = entropy_scorer.get_combined_entropy(entropy_maps)

        # Should have finest scale resolution
        finest_scale = min(entropy_scorer.scales)
        B, _, H, W = batch_images.shape
        expected_shape = (B, H // finest_scale, W // finest_scale)

        assert combined.shape == expected_shape

    def test_deterministic_output(self, entropy_scorer, batch_images):
        """Verify entropy computation is deterministic."""
        entropy_maps_1 = entropy_scorer(batch_images)
        entropy_maps_2 = entropy_scorer(batch_images)

        for scale in entropy_scorer.scales:
            assert torch.allclose(entropy_maps_1[scale], entropy_maps_2[scale]), (
                f"Entropy at scale {scale} is not deterministic"
            )

    def test_single_channel_input(self, entropy_scorer):
        """Test with single-channel (grayscale) input."""
        gray_image = torch.rand(1, 1, 224, 224)
        entropy_maps = entropy_scorer(gray_image)

        # Should still produce valid output
        for scale in entropy_scorer.scales:
            assert scale in entropy_maps
            assert entropy_maps[scale].shape[0] == 1

    def test_negative_input_normalization(self, entropy_scorer):
        """Test that negative inputs (range [-1, 1]) are handled correctly."""
        # Image in [-1, 1] range
        normalized_image = torch.rand(1, 3, 224, 224) * 2 - 1
        entropy_maps = entropy_scorer(normalized_image)

        for scale, entropy_map in entropy_maps.items():
            assert entropy_map.min() >= 0.0
            assert entropy_map.max() <= 1.0
