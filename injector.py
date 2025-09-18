#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Optional


def _compute_scale(
    gate_param: torch.Tensor,
    scale_vec: Optional[torch.Tensor],
    B: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Combine the learned scalar gate σ(ρ) with an optional external scale.
    scale_vec can be:
      - None: use ones
      - scalar tensor/float: broadcast
      - shape (B,): broadcast over T
      - shape (B,T): per-position
    Returns shape (B,T,1) for easy broadcasting to (B,T,D).
    """
    g = torch.sigmoid(gate_param)  # scalar in (0,1)
    if scale_vec is None:
        s = torch.ones((B, T, 1), device=device, dtype=dtype) * g
        return s
    if not torch.is_tensor(scale_vec):
        scale_vec = torch.as_tensor(scale_vec, device=device, dtype=dtype)
    if scale_vec.dim() == 0:
        s = torch.ones((B, T, 1), device=device, dtype=dtype) * (g * scale_vec)
    elif scale_vec.dim() == 1:
        assert scale_vec.shape[0] == B, "scale_vec (B,) must match batch size"
        s = scale_vec.view(B, 1, 1).expand(B, T, 1) * g
    elif scale_vec.dim() == 2:
        assert scale_vec.shape == (B, T), "scale_vec must be (B,T)"
        s = scale_vec.unsqueeze(-1) * g
    else:
        raise ValueError("scale_vec must be scalar, (B,), or (B,T)")
    return s.to(dtype=dtype)


class InjectorGRB(nn.Module):
    """
    Gated Residual Bias (GRB)
    Adds a projected bias from zH to each token's hidden state, with:
      - learned scalar gate ρ (sigmoid)
      - optional external per-position scale_vec
      - dropout on the injected delta
      - zero-init of projection for identity start

    Inputs:
      last_hidden : (B, T, D)   - LLM hidden states
      zH          : (B, 1, d_h) - HRM high-level state
      scale_vec   : None | scalar | (B,) | (B,T)  (optional extra scale)
    Returns:
      (B, T, D)
    """
    def __init__(
        self,
        d_h: int = 512,
        d_model: int = 2048,
        gate_init: float = -2.0,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.proj = nn.Linear(d_h, d_model, bias=False)
        # Start at identity: zero the delta path so output == input until learned
        nn.init.zeros_(self.proj.weight)

        # Learned scalar gate; sigmoid(gate) ∈ (0,1)
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # Dropout on the delta
        self.inj_dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        last_hidden: torch.Tensor,
        zH: torch.Tensor,
        scale_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = last_hidden.shape
        device, dtype = last_hidden.device, last_hidden.dtype

        # zH: (B,1,d_h) -> bias: (B,1,D) -> broadcast across T
        bias = self.proj(zH).to(dtype)                      # (B,1,D)
        delta = bias.expand(B, T, D)                        # (B,T,D)
        delta = self.inj_dropout(delta)

        s = _compute_scale(self.gate, scale_vec, B, T, device, dtype)  # (B,T,1)
        return last_hidden + s * delta


class CrossAttentionBridge(nn.Module):
    """
    Cross-Attention Bridge (CAB) with multi-token memory.

    Single-hop cross-attention from token states (queries) to a small bank of
    memory tokens derived from zH (keys/values). Output is projected and added
    back (residual), scaled by a learned sigmoid gate and optional external scale.

    Inputs:
      last_hidden : (B, T, D)   - LLM hidden states
      zH          : (B, 1, d_h) - HRM high-level state
      scale_vec   : None | scalar | (B,) | (B,T)
    Returns:
      (B, T, D)
    """
    def __init__(
        self,
        d_h: int = 512,
        d_model: int = 2048,
        n_heads: int = 8,
        mem_tokens: int = 4,
        gate_init: float = -1.5,   # slightly stronger than -2 (~0.18 vs 0.12)
        dropout_p: float = 0.10,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.hd = d_model // n_heads
        self.m = int(mem_tokens)

        # Project zH → m memory tokens in model dim
        # zH: (B,1,d_h) -> mem: (B,m,D)
        self.mem = nn.Linear(d_h, d_model * self.m, bias=False)
        # keep default init for mem so gradients flow even if o is zeroed

        # QKV projections and output projection
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Identity start: zero the output projection so residual=0 at init
        nn.init.zeros_(self.o.weight)

        # Learned scalar gate for the residual
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # Dropout on the attention output delta
        self.inj_dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        last_hidden: torch.Tensor,
        zH: torch.Tensor,
        scale_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = last_hidden.shape
        device, dtype = last_hidden.device, last_hidden.dtype

        # ---- Build memory bank from zH ----
        mem = self.mem(zH)                               # (B,1,D*m)
        mem = mem.view(B, self.m, D)                     # (B,m,D)

        # ---- Project to multi-head Q, K, V ----
        # Q: (B,h,T,hd); K,V: (B,h,m,hd)
        Q = self.q(last_hidden).view(B, T, self.h, self.hd).transpose(1, 2)  # (B,h,T,hd)
        K = self.k(mem).view(B, self.m, self.h, self.hd).transpose(1, 2)     # (B,h,m,hd)
        V = self.v(mem).view(B, self.m, self.h, self.hd).transpose(1, 2)     # (B,h,m,hd)

        # ---- Attention over memory tokens ----
        att_logits = (Q @ K.transpose(-1, -2)) / (self.hd ** 0.5)            # (B,h,T,m)
        att = torch.softmax(att_logits, dim=-1)                               # (B,h,T,m)

        # ---- Aggregate and project back ----
        out = att @ V                                                         # (B,h,T,hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)                  # (B,T,D)
        out = self.o(out)                                                     # (B,T,D)
        out = self.inj_dropout(out)

        s = _compute_scale(self.gate, scale_vec, B, T, device, dtype)         # (B,T,1)
        return last_hidden + s * out
