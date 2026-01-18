"""
Baseline CEDetector (without APT)
Pure implementation of the CED paper without Adaptive Patch Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .dinov3_backbone import DiNOv3Backbone
from .feature_aggregation import CEDFeatureAggregation
from .copy_edit_classifier import CopyEditClassifier


class CEDetectorBaseline(nn.Module):
    """
    Baseline CEDetector following the original CED paper.
    Uses fixed patch extraction (no APT adaptive sizing).
    """

    def __init__(
        self,
        # DiNOv3 params
        dinov3_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token_path: str = None,
        freeze_backbone: bool = False,
        # Query patch params
        num_query_patches: int = 6,
        patch_height: float = 0.6,  # Height of each patch (60% of image)
        patch_width: float = 0.6,   # Width of each patch (60% of image)
        # Model params
        embed_dim: int = 768,
        # Classifier params
        num_classifier_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        # Retrieval params
        k_neighbors: int = 10,
    ):
        super().__init__()

        # Store hyperparameters
        self.num_query_patches = num_query_patches
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim

        # Initialize DiNOv3 backbone
        self.backbone = DiNOv3Backbone(
            model_name=dinov3_model,
            use_auth_token=hf_token_path,
            freeze_backbone=freeze_backbone
        )

        # Update embed_dim from backbone
        self.embed_dim = self.backbone.embed_dim

        # Initialize feature aggregation
        self.feature_aggregation = CEDFeatureAggregation(
            embed_dim=self.embed_dim,
            use_gem_pooling=True,
            use_whitening=True
        )

        # Initialize copy-edit classifier with gradient checkpointing
        self.classifier = CopyEditClassifier(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            num_layers=num_classifier_layers,
            dropout=dropout,
            use_gradient_checkpointing=True
        )

        # Reference corpus storage (for retrieval)
        self.reference_descriptors = None
        self.reference_tokens = None
        self.reference_ids = None

    def extract_fixed_patches(self, image: torch.Tensor) -> list:
        """
        Extract 6 fixed patches from image as described in CED paper.

        Patch layout (overlapping):
        [1] [2]
        [4] [5] [3]
           [6]

        Args:
            image: Input image (C, H, W)

        Returns:
            List of 6 patches, each (C, patch_H, patch_W)
        """
        C, H, W = image.shape

        # Calculate patch dimensions
        patch_h = int(H * self.patch_height)
        patch_w = int(W * self.patch_width)

        # Calculate stride for overlap
        stride_h = int((H - patch_h) / 2.5)  # Overlap patches vertically
        stride_w = int((W - patch_w) / 2.5)  # Overlap patches horizontally

        patches = []

        # Patch 1: Top-left
        patches.append(image[:, 0:patch_h, 0:patch_w])

        # Patch 2: Top-right (shifted horizontally)
        start_w = min(stride_w, W - patch_w)
        patches.append(image[:, 0:patch_h, start_w:start_w+patch_w])

        # Patch 3: Right side (shifted vertically)
        start_h = stride_h
        start_w = W - patch_w
        patches.append(image[:, start_h:start_h+patch_h, start_w:start_w+patch_w])

        # Patch 4: Left side (shifted vertically)
        patches.append(image[:, start_h:start_h+patch_h, 0:patch_w])

        # Patch 5: Center
        start_h = (H - patch_h) // 2
        start_w = (W - patch_w) // 2
        patches.append(image[:, start_h:start_h+patch_h, start_w:start_w+patch_w])

        # Patch 6: Bottom center
        start_h = H - patch_h
        start_w = (W - patch_w) // 2
        patches.append(image[:, start_h:start_h+patch_h, start_w:start_w+patch_w])

        return patches

    def process_image(
        self,
        image: torch.Tensor,
        is_query: bool = False
    ) -> tuple:
        """
        Process an image using fixed patches (baseline CED approach).

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            is_query: If True, extract query patches; else process full image

        Returns:
            Tuple of (deep_descriptor, patch_tokens, cls_token)
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, H, W = image.shape

        if is_query:
            # Extract fixed query patches (6 patches for CED)
            all_descriptors = []
            all_tokens = []
            all_cls = []

            for i in range(B):
                patches = self.extract_fixed_patches(image[i])

                # Process each patch
                for patch in patches:
                    # Resize patch to 224x224 for DiNOv3
                    patch_input = F.interpolate(
                        patch.unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    outputs = self.backbone(patch_input, return_attentions=True)
                    cls_token = outputs['cls_token']
                    patch_tokens = outputs['patch_tokens']
                    cls_attention = outputs.get('cls_attention')

                    # Aggregate features
                    descriptor = self.feature_aggregation(
                        cls_token,
                        patch_tokens,
                        cls_attention
                    )

                    all_descriptors.append(descriptor)
                    all_tokens.append(patch_tokens)
                    all_cls.append(cls_token)

            descriptors = torch.cat(all_descriptors, dim=0)
            tokens = torch.cat(all_tokens, dim=0)
            cls_tokens = torch.cat(all_cls, dim=0)

        else:
            # Process reference images (full image)
            all_descriptors = []
            all_tokens = []
            all_cls = []

            for i in range(B):
                # Resize to 224x224 for DiNOv3
                img_input = F.interpolate(
                    image[i].unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )

                outputs = self.backbone(img_input, return_attentions=True)
                cls_token = outputs['cls_token']
                patch_tokens = outputs['patch_tokens']
                cls_attention = outputs.get('cls_attention')

                # Aggregate features
                descriptor = self.feature_aggregation(
                    cls_token,
                    patch_tokens,
                    cls_attention
                )

                all_descriptors.append(descriptor)
                all_tokens.append(patch_tokens)
                all_cls.append(cls_token)

            descriptors = torch.cat(all_descriptors, dim=0)
            tokens = torch.cat(all_tokens, dim=0)
            cls_tokens = torch.cat(all_cls, dim=0)

        if squeeze_output and not is_query:
            descriptors = descriptors.squeeze(0)

        return descriptors, tokens, cls_tokens

    def forward(
        self,
        query_image: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None,
        return_retrievals: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            query_image: Query image (C, H, W) or (B, C, H, W)
            reference_images: Reference images for comparison (optional)
            return_retrievals: Whether to return retrieval results

        Returns:
            Dictionary containing predictions and components for CEDLoss:
            - scores: Classification scores (logits for training, probabilities for inference)
            - query_cls_tokens: CLS tokens from query images (B, D)
            - ref_cls_tokens: CLS tokens from reference images (B, D)
            - patch_embeddings: Deep descriptors for metric learning (B, D)
            - logits: Raw logits before sigmoid (B,)
        """
        # Process query image with multiple patches
        query_descriptors, query_tokens, query_cls_tokens = self.process_image(
            query_image,
            is_query=True
        )

        results = {
            'query_descriptors': query_descriptors,
            'query_tokens': query_tokens,
            'query_cls_tokens': query_cls_tokens
        }

        # If reference images provided, classify directly
        if reference_images is not None:
            ref_descriptors, ref_tokens, ref_cls_tokens = self.process_image(
                reference_images,
                is_query=False
            )

            # Get batch size
            B = reference_images.shape[0] if reference_images.dim() == 4 else 1

            # query_tokens has shape (B * num_query_patches, N, D) where N is patch tokens per patch
            # We need to compare patches from query_i ONLY with reference_i

            all_logits = []
            for i in range(B):
                # Get the 6 patches for query image i
                patch_start = i * self.num_query_patches
                patch_end = (i + 1) * self.num_query_patches
                query_patches_i = query_tokens[patch_start:patch_end]  # (6, N, D)

                # Get reference tokens for reference image i
                ref_tokens_i = ref_tokens[i].unsqueeze(0)  # (1, N, D)

                # Compare each of the 6 query patches with this reference
                patch_logits = []
                for j in range(self.num_query_patches):
                    q_tokens = query_patches_i[j].unsqueeze(0)  # (1, N, D)
                    logit = self.classifier(q_tokens, ref_tokens_i)
                    patch_logits.append(logit)

                # Take maximum logit across the 6 patches for this pair
                # patch_logits is a list of (1, 1) tensors
                patch_logits_stacked = torch.stack([l.squeeze() for l in patch_logits])  # (6,)
                max_logit = patch_logits_stacked.max()  # scalar
                all_logits.append(max_logit)

            # Stack all B logits
            all_logits = torch.stack(all_logits)  # (B,)

            results['logits'] = all_logits
            results['scores'] = all_logits  # For training, return logits as scores
            results['reference_descriptors'] = ref_descriptors
            results['ref_cls_tokens'] = ref_cls_tokens
            results['patch_embeddings'] = query_descriptors  # Use query descriptors for metric learning

        return results
