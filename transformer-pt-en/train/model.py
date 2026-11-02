"""Encoder / Decoder / Transformer assembly (Pre-LN with final LayerNorm).

Deviations from the paper (see README):
- Pre-LN blocks + a final LayerNorm on encoder and decoder outputs.
  Pre-LN stacks leave the residual stream unnormalized, and the standard
  practice (GPT-2, fairseq) is a terminal LN before the output projection
  -- especially important here because the output projection is tied to
  the decoder embedding.
- Dropout is applied to embeddings + positional encoding (the paper's
  Pdrop), i.e. AFTER PositionalEncoding, not before.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from layers import DecoderLayer, EncoderLayer, PositionalEncoding


# -------------------------
# Encoder / Decoder stacks
# -------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # final LN (Pre-LN architecture)

    def forward(self, src_tokens: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src_tokens: (B, src_len)
        src_mask: (B, src_len) boolean, True at PAD positions
        """
        x = self.embed(src_tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.dropout(self.pos_enc(x))  # Pdrop over embeddings + PE
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # final LN (Pre-LN architecture)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt_tokens: (B, tgt_len)
        memory: (B, src_len, d_model)
        tgt_mask: (B, tgt_len, tgt_len) boolean (look-ahead + pad)
        memory_mask: (B, src_len) pad mask on encoder outputs
        """
        x = self.embed(tgt_tokens) * math.sqrt(self.embed.embedding_dim)
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)


# -------------------------
# Transformer
# -------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.out = nn.Linear(d_model, tgt_vocab)
        if tie_weights:
            # share decoder embedding and output projection (saves |V|*d params)
            self.out.weight = self.decoder.embed.weight

    def make_src_mask(self, src_tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """(B, seq_src) boolean, True at PAD positions."""
        return src_tokens == pad_idx

    def make_tgt_mask(self, tgt_tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """(B, seq_tgt, seq_tgt) boolean combining look-ahead and pad masks."""
        seq_len = tgt_tokens.size(1)
        # look-ahead: True strictly above the diagonal
        subsequent = torch.triu(
            torch.ones((seq_len, seq_len), device=tgt_tokens.device), diagonal=1
        ).bool()
        # pad columns, broadcast over query positions
        tgt_pad_mask = (tgt_tokens == pad_idx).unsqueeze(1).expand(-1, seq_len, -1)
        return subsequent.unsqueeze(0) | tgt_pad_mask

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor, src_pad_idx: int = 0):
        """
        src_tokens: (B, src_len), tgt_tokens: (B, tgt_len)
        Returns logits: (B, tgt_len, tgt_vocab)
        """
        src_mask = self.make_src_mask(src_tokens, pad_idx=src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt_tokens, pad_idx=src_pad_idx)
        memory = self.encoder(src_tokens, src_mask=src_mask)
        dec = self.decoder(tgt_tokens, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.out(dec)
