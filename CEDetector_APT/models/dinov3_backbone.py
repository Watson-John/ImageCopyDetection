"""
DiNOv3 Backbone Wrapper
Loads pretrained DiNOv3 from HuggingFace and provides interface for feature extraction
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import os
from dotenv import load_dotenv


class DiNOv3Backbone(nn.Module):
    """
    Wrapper for DiNOv3 Vision Transformer from HuggingFace.
    Extracts features at multiple scales for copy-edit detection.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        use_auth_token: str = None,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            use_auth_token: HuggingFace authentication token
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        # Load authentication token if provided path
        if use_auth_token and os.path.exists(use_auth_token):
            load_dotenv(use_auth_token)
            use_auth_token = os.getenv('HF_TOKEN')

        # Load DiNOv3 model
        print(f"Loading DiNOv3 model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            token=use_auth_token,
            attn_implementation="eager"  # Required for output_attentions=True
        )

        # Get model config
        self.config = self.model.config
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_hidden_layers

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        print(f"DiNOv3 loaded: embed_dim={self.embed_dim}, layers={self.num_layers}")

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_attentions: bool = True,
        return_all_layers: bool = False
    ) -> dict:
        """
        Forward pass through DiNOv3.

        Args:
            pixel_values: Input images of shape (B, C, H, W)
            return_attentions: Whether to return attention weights
            return_all_layers: Whether to return outputs from all layers

        Returns:
            Dictionary containing:
                - last_hidden_state: Final layer outputs (B, N, D)
                - cls_token: CLS token embeddings (B, D)
                - patch_tokens: Patch token embeddings (B, N-1, D)
                - attentions: Attention weights from last layer if requested
        """
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=return_attentions,
            output_hidden_states=return_all_layers,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state  # (B, N, D) where N = 1 + num_patches

        # Split CLS token and patch tokens
        cls_token = last_hidden_state[:, 0]  # (B, D)
        patch_tokens = last_hidden_state[:, 1:]  # (B, N-1, D)

        result = {
            'last_hidden_state': last_hidden_state,
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
        }

        # Add attention weights if requested
        if return_attentions and outputs.attentions is not None:
            # Get attention from last layer
            last_layer_attention = outputs.attentions[-1]  # (B, num_heads, N, N)
            result['attentions'] = last_layer_attention

            # Extract CLS attention to patches
            cls_attention = last_layer_attention[:, :, 0, 1:]  # (B, num_heads, N-1)
            # Average across heads
            cls_attention = cls_attention.mean(dim=1)  # (B, N-1)
            result['cls_attention'] = cls_attention

        # Add all hidden states if requested
        if return_all_layers and outputs.hidden_states is not None:
            result['hidden_states'] = outputs.hidden_states

        return result

    def get_patch_embeddings(
        self,
        pixel_values: torch.Tensor
    ) -> tuple:
        """
        Get patch embeddings and CLS token.

        Args:
            pixel_values: Input images (B, C, H, W)

        Returns:
            Tuple of (cls_token, patch_tokens, attentions)
        """
        outputs = self.forward(pixel_values, return_attentions=True)
        return outputs['cls_token'], outputs['patch_tokens'], outputs.get('cls_attention')
