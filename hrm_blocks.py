#!/usr/bin/env python3
# hrm_blocks.py
# Core HRM building blocks with stability options:
# - RMSNorm (configurable eps), SwiGLU MLP
# - SelfAttention (attn dropout)
# - TransformerBlock (pre-norm, residual scaling, dropout)
# - HBlock / LBlock encoder-only stacks used by the HRM controller

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm (no bias). Scales each hidden vector by
    the inverse RMS of its components, then applies a learned gain.
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D). Mean over last dim (D)
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x * rms_inv


class SwiGLU(nn.Module):
    """
    SwiGLU MLP: w3( silu(w1(x)) * w2(x) ).
    Uses expansion to grow hidden dim and a final projection back to d.
    """
    def __init__(self, d: int, expansion: float = 4.0, dropout_p: float = 0.0):
        super().__init__()
        h = int(d * expansion)
        self.w1 = nn.Linear(d, h, bias=False)  # gate
        self.w2 = nn.Linear(d, h, bias=False)  # value
        self.w3 = nn.Linear(h, d, bias=False)  # out
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused-style SwiGLU
        y = F.silu(self.w1(x)) * self.w2(x)
        y = self.w3(self.dropout(y))
        return y


class SelfAttention(nn.Module):
    """
    Simple multi-head self-attention.
    Linear weights are registered in __init__ so they train properly.
    """
    def __init__(self, d: int, h: int, attn_dropout_p: float = 0.0):
        super().__init__()
        assert d % h == 0, "Hidden size d must be divisible by number of heads h"
        self.d = d
        self.h = h
        self.hd = d // h

        # One projection for QKV, one for output
        self.w_qkv = nn.Linear(d, 3 * d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout_p) if attn_dropout_p and attn_dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, T, D)
        mask: (B, T) with 1 for valid tokens and 0 for masked keys (optional)
        """
        qkv = self.w_qkv(x)                           # (B, T, 3D)
        q, k, v = qkv.split(self.d, dim=-1)          # each (B, T, D)

        # Reshape to multi-head
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.h)  # (B, h, T, hd)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.h)  # (B, h, T, hd)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.h)  # (B, h, T, hd)

        # Scaled dot-product attention
        att = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5)  # (B, h, T, T)

        # If provided, mask out invalid KEYS (broadcast over heads/queries)
        # NOTE: In our HRM usage, we operate on single-token states (T=1),
        # so we typically don't pass a mask.
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)                    # softmax over KEYS
        att = self.attn_dropout(att)                        # dropout on weights (no-op if p=0)
        y = att @ v                                         # (B, h, T, hd)

        # Merge heads
        y = rearrange(y, 'b h t d -> b t (h d)')            # (B, T, D)
        return self.w_o(y)                                  # (B, T, D)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with:
      x = x + alpha * Dropout(Attn(RMSNorm(x)))
      x = x + alpha * Dropout(MLP(RMSNorm(x)))
    """
    def __init__(
        self,
        d: int,
        h: int,
        mlp_expansion: float = 4.0,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
        resid_alpha: float = 1.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.n1 = RMSNorm(d, eps=norm_eps)
        self.attn = SelfAttention(d, h, attn_dropout_p=attn_dropout_p)
        self.n2 = RMSNorm(d, eps=norm_eps)
        self.mlp = SwiGLU(d, expansion=mlp_expansion, dropout_p=dropout_p)

        self.drop_attn = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()
        self.drop_mlp  = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

        # Residual scaling (fixed scalar, exported as buffer so it moves with .to(device))
        self.register_buffer("resid_alpha", torch.tensor(float(resid_alpha)), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.resid_alpha * self.drop_attn(self.attn(self.n1(x), mask=mask))
        x = x + self.resid_alpha * self.drop_mlp(self.mlp(self.n2(x)))
        return x


class HBlock(nn.Module):
    """
    High-level (slow) HRM module.
    Runs a small stack of encoder-only Transformer blocks over zH + zL.
    """
    def __init__(
        self,
        d: int = 512,
        n_layers: int = 1,
        n_heads: int = 8,
        mlp_expansion: float = 4.0,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
        resid_alpha: float = 1.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d, n_heads,
                    mlp_expansion=mlp_expansion,
                    dropout_p=dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_alpha=resid_alpha,
                    norm_eps=norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d, eps=norm_eps)

    def forward(self, zH: torch.Tensor, zL: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          zH: (B, 1, D) current high-level state
          zL: (B, 1, D) current low-level state (or its summary)
        Returns:
          (B, 1, D) updated high-level state
        """
        x = zH + zL
        for blk in self.layers:
            x = blk(x)   # mask=None (single-token states)
        return self.norm(x)


class LBlock(nn.Module):
    """
    Low-level (fast) HRM module.
    Runs a small stack over zL + zH + x_tilde (pooled prompt features).
    """
    def __init__(
        self,
        d: int = 512,
        n_layers: int = 1,
        n_heads: int = 8,
        mlp_expansion: float = 4.0,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
        resid_alpha: float = 1.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d, n_heads,
                    mlp_expansion=mlp_expansion,
                    dropout_p=dropout_p,
                    attn_dropout_p=attn_dropout_p,
                    resid_alpha=resid_alpha,
                    norm_eps=norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d, eps=norm_eps)

    def forward(self, zL: torch.Tensor, zH: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          zL:      (B, 1, D) current low-level state
          zH:      (B, 1, D) current high-level state
          x_tilde: (B, 1, D) projected pooled prompt embedding
        Returns:
          (B, 1, D) updated low-level state
        """
        x = zL + zH + x_tilde
        for blk in self.layers:
            x = blk(x)   # mask=None (single-token states)
        return self.norm(x)
