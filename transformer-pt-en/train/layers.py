"""Transformer building blocks: positional encoding, attention, Pre-LN layers.

All modules follow the paper's shapes:
    q/k/v: (B, seq_len, d_model) -> heads (B, num_heads, seq_len, head_dim)
Masks are boolean with True = "mask out" (PyTorch convention), broadcast
over the head dimension.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the paper (section 3.5).

    Stored as a buffer so it follows ``model.to(device)``; sliced and cast
    to the input's dtype in forward (AMP-friendly).
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd d_model: the cos side has one fewer frequency
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device).type_as(x)


# -------------------------
# Scaled Dot-Product Attention
# -------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention per the paper (section 3.2.1).

    q, k, v: (..., seq_len, head_dim)
    mask: boolean (True -> mask out) or additive float, broadcastable to
          (..., seq_q, seq_k)
    Returns (output, attention weights after softmax).
    """
    dk = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, v)
    return output, attn


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_per_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query/key/value: (B, seq_len, d_model)
        mask: None | (B, seq_k) pad mask | (B, seq_q, seq_k) full mask,
              True = masked position.
        Returns:
            output: (B, seq_q, d_model)
            attn:   (B, num_heads, seq_q, seq_k)
        """
        B, Lq, _ = query.size()
        _, Lk, _ = key.size()

        # project, split heads: (B, num_heads, seq_len, head_dim)
        q = self.w_q(query).view(B, Lq, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.w_k(key).view(B, Lk, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.w_v(value).view(B, Lk, self.num_heads, self.dim_per_head).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 2:    # (B, seq_k)   -> (B, 1, 1, seq_k)
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:  # (B, seq_q, seq_k) -> (B, 1, seq_q, seq_k)
                mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1)

        attn_output, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(attn_output), attn


# -------------------------
# Position-wise Feed-Forward
# -------------------------
class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, xW1 + b1)W2 + b2 (paper section 3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# -------------------------
# Encoder / Decoder Layers (Pre-LN)
# -------------------------
class EncoderLayer(nn.Module):
    """Pre-LN encoder layer: x = x + sublayer(LN(x)).

    Pre-LN (norm before each residual branch) trains far more stably than
    the paper's Post-LN without careful warmup tuning; see the project
    README for the deviation note. Requires a final LN in the Encoder.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        nx = self.norm1(x)
        attn_out, _ = self.self_attn(nx, nx, nx, mask=src_mask)
        x = x + self.dropout1(attn_out)

        nx = self.norm2(x)
        x = x + self.dropout2(self.ffn(nx))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN decoder layer: masked self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # masked self-attention (look-ahead + pad)
        nx = self.norm1(x)
        attn_out, _ = self.self_attn(nx, nx, nx, mask=tgt_mask)
        x = x + self.dropout1(attn_out)

        # cross-attention: keys/values from the encoder memory
        nx = self.norm2(x)
        attn_out, _ = self.enc_dec_attn(nx, memory, memory, mask=memory_mask)
        x = x + self.dropout2(attn_out)

        # feed-forward
        nx = self.norm3(x)
        x = x + self.dropout3(self.ffn(nx))
        return x
