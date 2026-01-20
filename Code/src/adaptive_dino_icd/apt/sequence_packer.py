"""
Sequence Packer for efficient variable-length token processing.

Handles packing of variable-length token sequences for efficient
batch processing with attention masks.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SequencePacker(nn.Module):
    """
    Pack variable-length token sequences for efficient processing.

    Phase 1 implementation uses padding + attention masks.
    Future: Flash attention packing optimization for better memory efficiency.

    Args:
        max_tokens: Maximum number of tokens to support
        pad_value: Value to use for padding (default: 0)

    Example:
        packer = SequencePacker(max_tokens=196)
        packed = packer(tokens_list, coords_list)
        # packed['tokens'].shape == (B, max_tokens, D)
    """

    def __init__(
        self,
        max_tokens: int = 196,  # 14x14 for standard ViT-B/16
        pad_value: float = 0.0,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.pad_value = pad_value

    def forward(
        self,
        tokens: List[torch.Tensor],
        coords: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Pack variable-length token sequences with padding.

        Args:
            tokens: List of token tensors, one per image (each: N_i, D)
            coords: Optional list of coordinate tensors (each: N_i, 2)

        Returns:
            Dictionary with:
                - padded_tokens: (B, max_len, D) padded token tensor
                - attention_mask: (B, max_len) boolean mask, True for valid tokens
                - token_counts: (B,) number of tokens per sequence
                - padded_coords: (B, max_len, 2) if coords provided
        """
        if len(tokens) == 0:
            raise ValueError("Cannot pack empty token list")

        B = len(tokens)
        D = tokens[0].shape[-1] if tokens[0].numel() > 0 else 768
        device = tokens[0].device

        # Get token counts
        token_counts = [t.shape[0] for t in tokens]
        max_len = min(max(token_counts), self.max_tokens)

        # Initialize padded tensors
        padded_tokens = torch.full(
            (B, max_len, D),
            fill_value=self.pad_value,
            dtype=tokens[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        # Fill in actual tokens
        for b, (t, count) in enumerate(zip(tokens, token_counts)):
            actual_len = min(count, max_len)
            if actual_len > 0:
                padded_tokens[b, :actual_len] = t[:actual_len]
                attention_mask[b, :actual_len] = True

        result = {
            "padded_tokens": padded_tokens,
            "attention_mask": attention_mask,
            "token_counts": torch.tensor(token_counts, device=device),
        }

        # Handle coords if provided
        if coords is not None:
            padded_coords = torch.zeros(B, max_len, 2, device=device)
            for b, (c, count) in enumerate(zip(coords, token_counts)):
                actual_len = min(count, max_len)
                if actual_len > 0:
                    padded_coords[b, :actual_len] = c[:actual_len]
            result["padded_coords"] = padded_coords

        return result

    def unpack(
        self,
        padded_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        padded_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Unpack padded sequences back to variable-length lists.

        Args:
            padded_tokens: (B, max_len, D) padded token tensor
            attention_mask: (B, max_len) boolean mask
            padded_coords: Optional (B, max_len, 2) coordinate tensor

        Returns:
            tokens: List of token tensors without padding
            coords: List of coord tensors without padding (if provided)
        """
        B = padded_tokens.shape[0]

        tokens = []
        coords = [] if padded_coords is not None else None

        for b in range(B):
            mask = attention_mask[b]
            valid_tokens = padded_tokens[b][mask]
            tokens.append(valid_tokens)

            if padded_coords is not None:
                valid_coords = padded_coords[b][mask]
                coords.append(valid_coords)

        return tokens, coords

    def create_attention_bias(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create attention bias from attention mask for transformer.

        Converts boolean mask to float bias where False -> -inf.

        Args:
            attention_mask: (B, N) boolean mask
            dtype: Output dtype

        Returns:
            Attention bias (B, 1, 1, N) suitable for attention computation
        """
        # Convert mask to attention bias
        # True (valid) -> 0.0, False (padding) -> -inf
        attn_bias = torch.zeros_like(attention_mask, dtype=dtype)
        attn_bias.masked_fill_(~attention_mask, float("-inf"))

        # Reshape for broadcasting in attention
        # (B, N) -> (B, 1, 1, N) for (B, H, Q, K) attention
        return attn_bias.unsqueeze(1).unsqueeze(2)

    def create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create causal attention mask (for autoregressive if needed).

        Args:
            seq_len: Sequence length
            device: Device for tensor
            dtype: Output dtype

        Returns:
            Causal mask (1, 1, seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)
