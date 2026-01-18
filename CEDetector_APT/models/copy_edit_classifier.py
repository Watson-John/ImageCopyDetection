"""
Copy-Edit Classifier Module
Uses cross-attention and self-attention to classify whether query is a copy-edit
Based on Section 3.2 of the CED paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention layer.
    Computes attention between query and reference token sequences.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query embeddings (B, N_q, D)
            key_value: Key/Value embeddings (B, N_kv, D)
            attn_mask: Optional attention mask

        Returns:
            Output embeddings (B, N_q, D)
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]

        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # Equation 2 from paper: cross-attention(hr, hq) = (hq·W1)·(hr·W2)^T / sqrt(d) · (hr·W3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, N_q, D)
        output = self.out_proj(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feedforward network.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (reuse normalized x to save memory)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class CopyEditClassifier(nn.Module):
    """
    Copy-Edit Classifier that determines if query is an edited copy of reference.
    Uses cross-attention followed by self-attention blocks.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,  # 2 blocks × 2 layers
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Total number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Cross-attention layer
        self.cross_attention = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Multi-head self-attention blocks
        self.self_attention_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify whether query is a copy-edit of reference.

        Args:
            query_tokens: Query token embeddings (B, N_q, D)
            reference_tokens: Reference token embeddings (B, N_r, D)

        Returns:
            Classification logits (B, 1)
        """
        # Cross-attention between query and reference
        C0 = self.cross_attention(query_tokens, reference_tokens)
        C0 = self.cross_norm(C0)

        # Apply self-attention blocks with optional gradient checkpointing
        x = C0
        for layer in self.self_attention_layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.global_pool(x).squeeze(-1)  # (B, D)

        # Classification
        logits = self.classifier(x)  # (B, 1)

        return logits

    def predict_proba(
        self,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability that query is a copy-edit of reference.

        Args:
            query_tokens: Query token embeddings (B, N_q, D)
            reference_tokens: Reference token embeddings (B, N_r, D)

        Returns:
            Probabilities (B,)
        """
        logits = self.forward(query_tokens, reference_tokens)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs
