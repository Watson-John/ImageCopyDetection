"""
CEDetector with Adaptive Patch Transformers (APT)
Complete model combining CED and APT for efficient copy-edit detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

from .dinov3_backbone import DiNOv3Backbone
from .apt_patch_selector import APTPatchSelector
from .apt_patch_embedding import APTPatchEmbedding
from .feature_aggregation import CEDFeatureAggregation
from .copy_edit_classifier import CopyEditClassifier


class CEDetectorAPT(nn.Module):
    """
    Complete CEDetector model with Adaptive Patch Transformers.
    Combines:
    - APT for adaptive patch sizing
    - DiNOv3 for feature extraction
    - CED feature aggregation
    - Copy-Edit Classifier
    """

    def __init__(
        self,
        # DiNOv3 params
        dinov3_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_token_path: str = None,
        freeze_backbone: bool = False,
        # APT params
        base_patch_size: int = 16,
        num_patch_scales: int = 3,
        entropy_thresholds: List[float] = [5.5, 4.0],
        # Query patch params
        num_query_patches: int = 6,
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
        self.base_patch_size = base_patch_size
        self.num_patch_scales = num_patch_scales
        self.num_query_patches = num_query_patches
        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim

        # Initialize APT patch selector
        self.patch_selector = APTPatchSelector(
            base_patch_size=base_patch_size,
            num_scales=num_patch_scales,
            thresholds=entropy_thresholds
        )

        # Initialize APT patch embedding
        self.patch_embedding = APTPatchEmbedding(
            base_patch_size=base_patch_size,
            num_scales=num_patch_scales,
            in_channels=3,
            embed_dim=embed_dim
        )

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
            use_gradient_checkpointing=True  # Enable for memory savings
        )

        # Reference corpus storage (for retrieval)
        self.reference_descriptors = None
        self.reference_tokens = None
        self.reference_ids = None

    def process_image_with_apt(
        self,
        image: torch.Tensor,
        is_query: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process an image using APT and DiNOv3.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            is_query: If True, extract fixed query patches; else use APT

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
                patches = self.patch_selector.extract_query_patches(
                    image[i],
                    num_patches=self.num_query_patches
                )

                # Process each patch
                for patch in patches:
                    # Get adaptive patches using APT
                    apt_patches, scales, _ = self.patch_selector.get_adaptive_patches(patch)

                    # Embed patches
                    patch_embeds = self.patch_embedding(apt_patches, scales)
                    patch_embeds = patch_embeds.unsqueeze(0)  # (1, N, D)

                    # Pass through backbone
                    # First, we need to prepare the input for DiNOv3
                    # DiNOv3 expects regular images, so we'll use the full patch
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
            # Use APT for reference images
            all_descriptors = []
            all_tokens = []
            all_cls = []

            for i in range(B):
                # Prepare image for DiNOv3 (resize to 224x224)
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

    def build_reference_corpus(
        self,
        reference_images: torch.Tensor,
        reference_ids: Optional[List[str]] = None,
        batch_size: int = 32
    ):
        """
        Build reference corpus for retrieval.

        Args:
            reference_images: Reference images (N, C, H, W)
            reference_ids: Optional IDs for reference images
            batch_size: Batch size for processing
        """
        self.eval()
        all_descriptors = []
        all_tokens = []

        with torch.no_grad():
            for i in range(0, len(reference_images), batch_size):
                batch = reference_images[i:i+batch_size]
                descriptors, tokens, _ = self.process_image_with_apt(batch, is_query=False)
                all_descriptors.append(descriptors)
                all_tokens.append(tokens)

        self.reference_descriptors = torch.cat(all_descriptors, dim=0)
        self.reference_tokens = torch.cat(all_tokens, dim=0)
        self.reference_ids = reference_ids if reference_ids is not None else list(range(len(reference_images)))

    def retrieve_candidates(
        self,
        query_descriptor: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k nearest neighbors from reference corpus.

        Args:
            query_descriptor: Query descriptor (D,) or (B, D)
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (indices, distances)
        """
        if k is None:
            k = self.k_neighbors

        if query_descriptor.dim() == 1:
            query_descriptor = query_descriptor.unsqueeze(0)

        # Compute cosine similarity
        query_norm = F.normalize(query_descriptor, p=2, dim=1)
        ref_norm = F.normalize(self.reference_descriptors, p=2, dim=1)
        similarities = torch.matmul(query_norm, ref_norm.t())

        # Get top-k
        top_k_values, top_k_indices = torch.topk(similarities, k, dim=1)

        return top_k_indices, top_k_values

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
        # Process query image with multiple patches - capture CLS tokens
        query_descriptors, query_tokens, query_cls_tokens = self.process_image_with_apt(
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
            ref_descriptors, ref_tokens, ref_cls_tokens = self.process_image_with_apt(
                reference_images,
                is_query=False
            )

            # Classify each query image against its corresponding reference image
            # query_tokens has shape (B * num_patches_per_image, num_patch_tokens, D)
            # ref_tokens has shape (B, num_patch_tokens, D)
            # We need to compare query[i]'s patches to reference[i] (paired comparison)

            B = reference_images.shape[0] if reference_images.dim() == 4 else 1
            num_patches_per_image = self.num_query_patches  # 6 patches per query image

            all_logits = []
            for batch_idx in range(B):
                # Get all patches for this query image
                patch_start = batch_idx * num_patches_per_image
                patch_end = (batch_idx + 1) * num_patches_per_image

                # Compare all patches of query[batch_idx] to reference[batch_idx]
                patch_logits = []
                for patch_idx in range(patch_start, patch_end):
                    q_tokens = query_tokens[patch_idx].unsqueeze(0)
                    r_tokens = ref_tokens[batch_idx].unsqueeze(0)
                    logit = self.classifier(q_tokens, r_tokens)
                    patch_logits.append(logit)

                # Take maximum logit across all patches for this query-reference pair
                max_logit = torch.stack(patch_logits).max(dim=0)[0]
                all_logits.append(max_logit)

            # Stack to get (B,) or (B, 1)
            max_logits = torch.cat(all_logits, dim=0)
            if max_logits.dim() > 1:
                max_logits = max_logits.squeeze(-1)

            results['logits'] = max_logits
            results['scores'] = max_logits  # For training, return logits as scores
            results['reference_descriptors'] = ref_descriptors
            results['ref_cls_tokens'] = ref_cls_tokens
            results['patch_embeddings'] = query_descriptors  # Use query descriptors for metric learning

        # Retrieval from corpus if available
        elif return_retrievals and self.reference_descriptors is not None:
            # Retrieve for each query patch
            all_indices = []
            all_similarities = []

            for i in range(query_descriptors.shape[0]):
                indices, similarities = self.retrieve_candidates(query_descriptors[i])
                all_indices.append(indices)
                all_similarities.append(similarities)

            results['retrieval_indices'] = torch.stack(all_indices)
            results['retrieval_similarities'] = torch.stack(all_similarities)

        return results
