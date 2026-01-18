"""
CED Feature Aggregation Module
Aggregates patch tokens using attention-weighted pooling
Based on Section 3.1 of the CED paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling
    From "Fine-tuning CNN Image Retrieval with No Human Annotation"
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, D, H, W) or (B, D)

        Returns:
            Pooled tensor of shape (B, D)
        """
        if x.dim() == 4:
            # Spatial dimensions present
            return F.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1))
            ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)
        elif x.dim() == 3:
            # Sequence of tokens (B, N, D)
            x = x.permute(0, 2, 1)  # (B, D, N)
            return F.avg_pool1d(
                x.clamp(min=self.eps).pow(self.p),
                x.size(-1)
            ).pow(1.0 / self.p).squeeze(-1)
        else:
            # Already pooled
            return x


class Whitening(nn.Module):
    """
    Learnable whitening transformation
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        # Initialize as identity (nn.init.eye_ doesn't exist, use torch.eye)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CEDFeatureAggregation(nn.Module):
    """
    Feature Aggregation module from CED paper.
    Combines global CLS token features with salient regional features.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        use_gem_pooling: bool = True,
        use_whitening: bool = True,
        gem_p: float = 3.0,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            use_gem_pooling: Whether to use GeM pooling
            use_whitening: Whether to apply whitening
            gem_p: Power parameter for GeM pooling
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gem_pooling = use_gem_pooling
        self.use_whitening = use_whitening

        # Projection head for CLS token
        self.cls_projection = nn.Linear(embed_dim, embed_dim)

        # GeM pooling for salient features
        if use_gem_pooling:
            self.gem_pooling = GeMPooling(p=gem_p)

        # Whitening transformation
        if use_whitening:
            self.whitening = Whitening(embed_dim)

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        cls_attention: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Aggregate features from CLS token and patch tokens.

        Args:
            cls_token: CLS token embeddings (B, D)
            patch_tokens: Patch token embeddings (B, N, D)
            cls_attention: Attention scores from CLS to patches (B, N)

        Returns:
            Deep image descriptor (B, 2*D) concatenating global and salient features
        """
        B, N, D = patch_tokens.shape

        # 1. Global features from CLS token
        z = self.cls_projection(cls_token)  # (B, D)

        # 2. Salient regional features
        if cls_attention is not None:
            # Element-wise multiplication with attention scores
            # Equation 1 from paper: u = α_CLS ⊗ h_L
            u = cls_attention.unsqueeze(-1) * patch_tokens  # (B, N, D)
        else:
            u = patch_tokens

        # Apply GeM pooling
        if self.use_gem_pooling:
            u = self.gem_pooling(u)  # (B, D)
        else:
            u = u.mean(dim=1)  # (B, D)

        # Apply whitening
        if self.use_whitening:
            u = self.whitening(u)  # (B, D)

        # Concatenate global and salient features
        # Deep image descriptor v = [z; u]
        descriptor = torch.cat([z, u], dim=1)  # (B, 2*D)

        return descriptor

    def get_global_descriptor(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Get only the global descriptor from CLS token."""
        return self.cls_projection(cls_token)

    def get_salient_descriptor(
        self,
        patch_tokens: torch.Tensor,
        cls_attention: torch.Tensor = None
    ) -> torch.Tensor:
        """Get only the salient regional descriptor."""
        if cls_attention is not None:
            u = cls_attention.unsqueeze(-1) * patch_tokens
        else:
            u = patch_tokens

        if self.use_gem_pooling:
            u = self.gem_pooling(u)
        else:
            u = u.mean(dim=1)

        if self.use_whitening:
            u = self.whitening(u)

        return u
