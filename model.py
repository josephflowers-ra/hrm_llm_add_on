#!/usr/bin/env python3
import os
import contextlib
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F  # KL + BCE utils
from dataclasses import dataclass
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from hrm_blocks import HBlock, LBlock, RMSNorm
from injector import InjectorGRB, CrossAttentionBridge


# -----------------------
#  EOS resolver
# -----------------------
KNOWN_EOS_CANDIDATES = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "</s>"]

def resolve_eos_token(tok, override: str | None = None):
    """
    Pick a compatible EOS token for the current tokenizer.
    Priority:
      1) explicit override (if in vocab)
      2) tokenizer.eos_token (if defined)
      3) a known candidate that exists in vocab
      4) fallback to pad
    Returns (eos_token_str, eos_token_id).
    """
    if override:
        tid = tok.convert_tokens_to_ids(override)
        if tid is not None and tid != getattr(tok, "unk_token_id", None):
            return override, tid
    if getattr(tok, "eos_token", None) is not None and tok.eos_token_id is not None:
        return tok.eos_token, tok.eos_token_id
    for cand in KNOWN_EOS_CANDIDATES:
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != getattr(tok, "unk_token_id", None):
            return cand, tid
    # last resort
    return tok.pad_token, tok.pad_token_id


@dataclass
class HRMConfig:
    """
    Configuration for the HRM controller and its interface to the frozen LLM.
    """
    # Latent width / depth
    d_h: int = 512
    h_layers: int = 4
    l_layers: int = 4
    n_heads: int = 8

    # Unrolled dynamics
    inner_T: int = 3
    segments: int = 3

    # Bridges / options
    use_cab: bool = False     # False → GRB, True → CAB
    use_act: bool = False

    # Small vocab-bias head (z_H → logits bias), gated
    logit_bias_head: bool = True
    logit_bias_init: float = -2.0  # sigmoid(-2) ≈ 0.12 initial strength

    # Label masking for CE
    vocab_ignore_index: int = -100

    # --- New knobs (default safe/off) ---
    eos_override: str | None = None   # allow forcing EOS (e.g., "<|im_end|>")
    inj_dropout_p: float = 0.0        # dropout on injector path (train-time)
    z_noise_std: float = 0.0          # Gaussian noise on x_tilde during training
    gate_l2_coef: float = 1e-3        # penalty used later in training_step (Phase 2)
    kl_lambda: float = 0.0            # used later in training_step (Phase 2)
    cab_mem_tokens: int = 4
    cab_gate_init: float = -1.5
    grb_gate_init: float = -2.0

    # Δlogits head squash scale (after tanh); keeps it bounded & stable
    delta_scale: float = 5.0


class HRMController(nn.Module):
    """
    HRM + LLM hybrid controller (LLM is frozen).

    - Frozen HF CausalLM provides the language backbone ("speaker").
    - HRM (H/L blocks) runs fast/slow latent recurrence on pooled prompt features.
    - Injector (GRB or CAB) conditions the LLM hidden states with the final z_H.
    - Optional vocab/Δlogits heads add tiny, gated biases directly to logits.
    - Only HRM + injector + projections (+ bias heads) are trainable.

    Typical dataflow (per step):
      hidden = LLM(prompt+target) [frozen, w/ hidden states]
      x_pool  = mixed(mean+last of prompt hidden)
      z_H     = HRM(x_pool; inner_T, segments)
      hidden' = Inject(hidden, z_H)
      logits  = LMHead(hidden'); optionally add gated biases
      loss    = CE(logits, labels)  [labels mask prompt tokens with -100]
    """
    def __init__(self, model_name: str, hrm_cfg: HRMConfig):
        super().__init__()
        self.hrm_cfg = hrm_cfg

        # Decide the working dtype for trainable modules (match the frozen LLM on CUDA)
        self.param_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not hasattr(self.tokenizer, "eos_token_id") or self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = self.tokenizer.pad_token

        # Resolve EOS once (used by collator; inference paths may re-resolve with overrides)
        self._eos_tok, self._eos_id = resolve_eos_token(self.tokenizer, self.hrm_cfg.eos_override)

        # Cache unk id (used to ban <unk> during training/generation)
        self._unk_id = getattr(self.tokenizer, "unk_token_id", None)

        # --- Frozen LLM ---
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # keep the backbone in bf16 on GPU
            low_cpu_mem_usage=True
        )
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        d_model = self.lm.config.hidden_size
        V = self.lm.get_output_embeddings().weight.size(0)

        # --- HRM blocks (trainable) ---
        self.h_block = HBlock(d=self.hrm_cfg.d_h, n_layers=self.hrm_cfg.h_layers, n_heads=self.hrm_cfg.n_heads)
        self.l_block = LBlock(d=self.hrm_cfg.d_h, n_layers=self.hrm_cfg.l_layers, n_heads=self.hrm_cfg.n_heads)
        self.in_norm = RMSNorm(self.hrm_cfg.d_h)

        # Prompt pooling: (mean + last) -> mix -> project to HRM width
        self.pool_mix = nn.Linear(d_model * 2, d_model, bias=False)
        self.x_proj = nn.Linear(d_model, self.hrm_cfg.d_h, bias=False)

        # Learnable initial states (expanded per batch)
        self.zH0 = nn.Parameter(torch.zeros(1, 1, self.hrm_cfg.d_h, dtype=self.param_dtype))
        self.zL0 = nn.Parameter(torch.zeros(1, 1, self.hrm_cfg.d_h, dtype=self.param_dtype))

        # --- Injector: GRB or CAB (trainable) ---
        if self.hrm_cfg.use_cab:
            self.injector = CrossAttentionBridge(
                d_h=self.hrm_cfg.d_h,
                d_model=d_model,
                n_heads=self.hrm_cfg.n_heads,
                mem_tokens=self.hrm_cfg.cab_mem_tokens,
                gate_init=self.hrm_cfg.cab_gate_init,
                dropout_p=self.hrm_cfg.inj_dropout_p,
            )
        else:
            self.injector = InjectorGRB(
                d_h=self.hrm_cfg.d_h,
                d_model=d_model,
                gate_init=self.hrm_cfg.grb_gate_init,
                dropout_p=self.hrm_cfg.inj_dropout_p,
            )

        # Optional dropout on injector path
        self.inj_dropout = nn.Dropout(self.hrm_cfg.inj_dropout_p)

        # Optional ACT (halt) head on zH
        self.q_head = nn.Linear(self.hrm_cfg.d_h, 1) if self.hrm_cfg.use_act else None

        # Optional tiny vocab-bias head (trainable)
        if self.hrm_cfg.logit_bias_head:
            self.vocab_bias = nn.Linear(self.hrm_cfg.d_h, V, bias=False)
            self.vocab_gate = nn.Parameter(torch.tensor(self.hrm_cfg.logit_bias_init, dtype=self.param_dtype))
        else:
            self.vocab_bias = None
            self.vocab_gate = None

        # Δlogits head (robust, arch-agnostic) and a learned global gate
        self.delta_logits = nn.Linear(self.hrm_cfg.d_h, V, bias=False)
        nn.init.zeros_(self.delta_logits.weight)

        # A single learned gate; small initial effect (sigmoid ~ 0.12)
        self.inj_gate = nn.Parameter(torch.tensor(-2.0, dtype=self.param_dtype))

        # Stabilize gates to safe, weak values (avoid over-bias at step 0)
        with torch.no_grad():
            self.inj_gate.copy_(torch.tensor(-2.0, dtype=self.param_dtype))   # ~0.12 after sigmoid
            if self.vocab_gate is not None:
                self.vocab_gate.copy_(torch.tensor(-4.0, dtype=self.param_dtype))  # ~0.018 after sigmoid)

        # Move trainable modules to the chosen dtype (match bf16 backbone on GPU)
        def _to_dtype(m: nn.Module):
            for p in m.parameters(recurse=True):
                p.data = p.data.to(self.param_dtype)
                if p.grad is not None:
                    p.grad = p.grad.to(self.param_dtype)
        for mod in [self.h_block, self.l_block, self.in_norm, self.pool_mix, self.x_proj,
                    self.injector, self.delta_logits, *( [self.vocab_bias] if self.vocab_bias is not None else [] ),
                    *( [self.q_head] if self.q_head is not None else [] )]:
            _to_dtype(mod)

        # Debug print toggle (set HRM_DEBUG_SHAPES=1 to enable)
        self.debug_shapes = os.environ.get("HRM_DEBUG_SHAPES", "0") not in ("0", "", "false", "False")

    # -----------------------
    #  Convenience
    # -----------------------
    def set_segments(self, segments: int):
        """Set default number of HRM segments (e.g., deeper reasoning at inference)."""
        self.hrm_cfg.segments = int(segments)

    # -----------------------
    #  LLM helper methods
    # -----------------------
    def forward_llm_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward the frozen LLM and return last hidden states (B, T, D) in the LLM's dtype (bf16 on CUDA).
        """
        with torch.no_grad():
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            return out.hidden_states[-1]

    def _apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Some architectures (LLaMA/Mistral-like, NeoX) expect a final LayerNorm
        before the LM head. Apply it when present; otherwise, identity.
        """
        # LLaMA/Mistral family: lm.model.norm
        try:
            norm = getattr(getattr(self.lm, "model", None), "norm", None)
            if norm is not None:
                return norm(hidden)
        except Exception:
            pass
        # GPT-NeoX family: lm.gpt_neox.final_layer_norm
        try:
            neox = getattr(self.lm, "gpt_neox", None)
            if neox is not None and hasattr(neox, "final_layer_norm"):
                return neox.final_layer_norm(hidden)
        except Exception:
            pass
        # Default: no-op
        return hidden

    # --- Helpers for logits composition ---
    def base_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply final norm (if present) and LM head, enforcing (B,T,V).
        Hidden is kept in the LLM dtype; logits come out in the LM head's dtype (bf16 on CUDA).
        """
        assert hidden.dim() == 3, f"hidden should be (B,T,D), got {tuple(hidden.shape)}"
        B, T, D = hidden.shape
        hidden = self._apply_final_norm(hidden)
        head = self.lm.get_output_embeddings()
        logits = head(hidden.reshape(B * T, D)).reshape(B, T, -1)
        return logits

    # ---- Safe alignment for potentially buggy injector outputs ----
    def _align_injected_to_hidden(self, hidden: torch.Tensor, inj_hidden: torch.Tensor) -> torch.Tensor:
        """
        Injector bugs can reduce/permutate the time dimension (T).
        Make a best-effort correction so (B, T, D) matches `hidden`.
        This is a defensive shim; the real fix belongs in the injector.
        """
        if inj_hidden is None:
            return hidden
        if inj_hidden.dim() != 3:
            return hidden

        B, T, D = hidden.shape
        b, t, d = inj_hidden.shape

        if (b, t, d) == (B, T, D):
            return inj_hidden
        if (b, t, d) == (T, B, D):
            return inj_hidden.transpose(0, 1)
        if (b, t, d) == (B, 1, D):
            return inj_hidden.expand(B, T, D)
        if (b, t, d) == (1, T, D):
            return inj_hidden.expand(B, T, D)
        if (b, t, d) == (B, B, D) and B == T:
            return inj_hidden

        try:
            base = inj_hidden[:, :1, :] if b == B else inj_hidden[:1, :1, :].expand(B, 1, D)
            return base.expand(B, T, D)
        except Exception:
            return hidden

    def _force_btv_logits(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Coerce logits to shape (B, T, V) from a variety of 'almost right' shapes.
        """
        if x.dim() == 3 and x.size(0) == B and x.size(1) == T:
            return x
        if x.dim() == 4 and x.size(0) == B and x.size(1) == B and x.size(2) == T:
            idx = torch.arange(B, device=x.device)
            return x[idx, idx]
        if x.dim() == 4 and x.size(0) == B and x.size(1) == 1:
            return x.squeeze(1)
        if x.dim() == 3 and x.size(0) == T and x.size(1) == B:
            return x.transpose(0, 1)
        if x.dim() == 3 and x.size(0) == 1 and x.size(1) == T:
            return x.expand(B, T, x.size(-1))
        if x.dim() == 2 and x.size(0) == B:
            return x.unsqueeze(1).expand(B, T, x.size(-1))
        V = x.shape[-1]
        return x.reshape(B, -1, V)[:, :T, :]

    def logits_with_hrm(
        self,
        hidden: torch.Tensor,
        zH: torch.Tensor | None,
        scale_vec: torch.Tensor | None = None,
    ):
        """
        Compose base logits with (a) hidden injection (respecting optional scale_vec)
        and (b) Δlogits bias (tanh-squashed), both gated by a learned scalar.
        Returns:
            logits, logits_base  both (B,T,V)
        """
        assert hidden.dim() == 3, f"hidden must be (B,T,D), got {tuple(hidden.shape)}"
        B, T, D = hidden.shape

        # Base logits on the original lattice
        logits_base = self.base_logits_from_hidden(hidden)  # (B,T,V)

        # No HRM path → mirror base (still shape-safe)
        if zH is None:
            logits = self._force_btv_logits(logits_base, B, T)
            if self.debug_shapes:
                print("[HRM logits] no-HRM",
                      "hidden", tuple(hidden.shape),
                      "logits", tuple(logits.shape),
                      "logits_base", tuple(logits_base.shape),
                      flush=True)
            return logits, logits_base

        # Align dtypes (LLM hidden is bf16 on CUDA)
        zH = zH.to(hidden.dtype)

        # Hidden-state injection
        inj_hidden = self.injector(hidden, zH, scale_vec=scale_vec)  # expected (B,T,D)
        inj_hidden = self._align_injected_to_hidden(hidden, inj_hidden)

        # Optional dropout on injector path
        if self.training and self.hrm_cfg.inj_dropout_p > 0.0:
            inj_hidden = self.inj_dropout(inj_hidden)

        # Gated residual mix in hidden space
        g = torch.sigmoid(self.inj_gate.to(hidden.dtype))
        hidden_mix = hidden + g * (inj_hidden - hidden)
        assert hidden_mix.shape == (B, T, D)

        # LM head on the exact lattice → (B,T,V)
        logits = self.base_logits_from_hidden(hidden_mix)

        # Add Δlogits (tanh-squashed and scaled) and optional tiny vocab bias
        dlog = torch.tanh(self.delta_logits(zH)) * float(self.hrm_cfg.delta_scale)  # (B,V)
        logits = logits + g * dlog.to(logits.dtype).unsqueeze(1)                    # (B,1,V)

        if self.vocab_bias is not None:
            vb_gate = torch.sigmoid(self.vocab_gate.to(logits.dtype))
            vb = self.vocab_bias(zH).squeeze(1)                                     # (B,V)
            logits = logits + vb_gate * vb.to(logits.dtype).unsqueeze(1)            # (B,1,V)

        # Ban <unk> to prevent collapse
        if getattr(self, "_unk_id", None) is not None:
            uid = int(self._unk_id)
            if 0 <= uid < logits.size(-1):
                logits[..., uid] = float("-inf")

        # Coerce shapes back to (B,T,V)
        logits = self._force_btv_logits(logits, B, T)
        logits_base = self._force_btv_logits(logits_base, B, T)

        if self.debug_shapes:
            print("[HRM logits]",
                  "hidden", tuple(hidden.shape),
                  "logits", tuple(logits.shape),
                  "logits_base", tuple(logits_base.shape),
                  flush=True)

        assert logits.shape[:2] == (B, T)
        assert logits_base.shape[:2] == (B, T)
        return logits, logits_base



    def logits_from_injected(self, injected_hidden: torch.Tensor, zH: torch.Tensor | None = None) -> torch.Tensor:
        """
        (Kept for backward-compat with existing training code.)
        Apply final norm (if applicable) and LM head → logits (B,T,V), shape-safe.
        If zH is provided and vocab-bias head is enabled, add a gated bias.
        """
        assert injected_hidden.dim() == 3, f"injected_hidden should be (B,T,D), got {tuple(injected_hidden.shape)}"
        B, T, D = injected_hidden.shape
        hidden = self._apply_final_norm(injected_hidden)
        head = self.lm.get_output_embeddings()
        logits = head(hidden.reshape(B * T, D)).reshape(B, T, -1)

        if (zH is not None) and (self.vocab_bias is not None):
            bias = self.vocab_bias(zH).squeeze(1)  # (B,V)
            g = torch.sigmoid(self.vocab_gate.to(logits.dtype))     # scalar in (0,1)
            logits = logits + g * bias.to(logits.dtype).unsqueeze(1)
        if self._unk_id is not None and 0 <= int(self._unk_id) < logits.size(-1):
            logits[..., int(self._unk_id)] = float("-inf")
        return logits

    # -----------------------
    #  HRM dataflow
    # -----------------------
    def pool_tokens(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths) -> torch.Tensor:
        """
        Mixed pooling over ONLY the prompt region per sample → (B, 1, D_model).
          pooled = mix( mean(prompt), last(prompt) )
        """
        pooled = []
        T = hidden_states.size(1)
        for i, Lp in enumerate(prompt_lengths):
            Lp_i = max(1, min(int(Lp), T))
            mean_p = hidden_states[i, :Lp_i, :].mean(0)
            last_p = hidden_states[i, Lp_i - 1, :]
            # Cast to receiver dtype to avoid bf16/float mismatches
            mixed_in = torch.cat([mean_p, last_p], dim=-1).to(self.pool_mix.weight.dtype)
            mixed = self.pool_mix(mixed_in)
            pooled.append(mixed)
        return torch.stack(pooled, 0).unsqueeze(1)  # (B,1,D_model)

    def hrm_segments(self, x_tilde: torch.Tensor, segments: int, inner_T: int):
        """
        Run HRM for `segments` cycles.
        - First inner_T-1 L-steps under no_grad (let L converge)
        - Final L-step + one H-step with grad
        - Deep supervision: detach(zH,zL) between segments  → O(1) memory
        Returns list of zH per segment (len == segments).
        """
        # Ensure x_tilde matches trainable block dtypes
        x_tilde = x_tilde.to(self.param_dtype)

        if self.training and self.hrm_cfg.z_noise_std > 0.0:
            x_tilde = x_tilde + torch.randn_like(x_tilde) * float(self.hrm_cfg.z_noise_std)

        B = x_tilde.size(0)
        zH = self.zH0.expand(B, -1, -1)
        zL = self.zL0.expand(B, -1, -1)
        zH_list = []

        for _ in range(segments):
            with torch.no_grad():
                zH_t, zL_t = zH, zL
                for _ in range(max(0, inner_T - 1)):
                    zL_t = self.l_block(zL_t, zH_t, x_tilde)

            zL = self.l_block(zL_t, zH_t, x_tilde)
            zH = self.h_block(zH_t, zL)
            zH = self.in_norm(zH)

            zH_list.append(zH)
            zH = zH.detach()
            zL = zL.detach()

        return zH_list

    # -----------------------
    #  Training step
    # -----------------------
    def training_step(
        self,
        batch: dict,
        segments: int = None,
        inner_T: int = None,
        use_act: bool = False,
        temperature: float = 1.0,
        act_penalty: float = 0.0,
        injector_scale: float = 1.0,
        inject_over: str = "all",
    ):
        """
        One step of training with deep supervision across segments.
        Returns (total_loss, metrics).
        """
        segments = segments or self.hrm_cfg.segments
        inner_T = inner_T or self.hrm_cfg.inner_T

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prompt_lengths = batch["prompt_lengths"]

        # 1) Frozen LLM forward
        last_hidden = self.forward_llm_hidden(input_ids, attention_mask)  # (B,T,D)
        B, T, _ = last_hidden.shape
        assert last_hidden.shape[:2] == labels.shape[:2]

        # 2) Pool prompt → mix → project to HRM width
        x_pool = self.pool_tokens(last_hidden, attention_mask, prompt_lengths)
        x_tilde = self.x_proj(x_pool.to(self.x_proj.weight.dtype))

        # 3) HRM loop
        zH_list = self.hrm_segments(x_tilde, segments=segments, inner_T=inner_T)

        # 3.5) Per-position scale_vec
        if inject_over not in ("all", "labels"):
            raise ValueError("inject_over must be 'all' or 'labels'")
        scale_vec = torch.full(
            (B, T), float(injector_scale),
            device=last_hidden.device,
            dtype=last_hidden.dtype
        )
        if inject_over == "labels":
            for i, Lp in enumerate(prompt_lengths):
                Lp_i = max(0, min(int(Lp), T))
                scale_vec[i, :Lp_i] = 0.0

        ce = nn.CrossEntropyLoss(ignore_index=self.hrm_cfg.vocab_ignore_index)
        metrics = {"segment_losses": [], "q_losses": []}
        total_loss = torch.zeros((), device=last_hidden.device, dtype=torch.float32)

        zH_mean = float(torch.norm(zH_list[-1], dim=-1).mean().item()) if len(zH_list) > 0 else 0.0

        def _segment_correctness(logits, prompt_lengths, raw_items):
            pred_ids = logits.argmax(-1)
            outs = []
            for i, item in enumerate(raw_items):
                Lp = prompt_lengths[i]
                text = self.tokenizer.decode(pred_ids[i, Lp:], skip_special_tokens=True)
                ok = item.get("verify", None)
                outs.append(1.0 if callable(ok) and ok(text) else 0.0)
            return torch.tensor(outs, device=logits.device)

        use_amp = torch.cuda.is_available()
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

        # 4) Deep supervision
        for zH in zH_list:
            with amp_ctx:
                logits, logits_base = self.logits_with_hrm(last_hidden, zH, scale_vec=scale_vec)
                if temperature != 1.0:
                    logits = logits / temperature

            # CE in fp32
            loss = ce(logits.float().view(-1, logits.size(-1)), labels.view(-1))

            # Regularizers
            if self.hrm_cfg.kl_lambda and self.hrm_cfg.kl_lambda > 0.0:
                p = F.log_softmax(logits.float(), dim=-1)
                q = F.softmax(logits_base.detach().float(), dim=-1)
                loss = loss + self.hrm_cfg.kl_lambda * F.kl_div(p, q, reduction="batchmean")
            if self.hrm_cfg.gate_l2_coef and self.hrm_cfg.gate_l2_coef > 0.0:
                loss = loss + self.hrm_cfg.gate_l2_coef * (torch.sigmoid(self.inj_gate.float()) ** 2)

            # Optional ACT supervision
            if use_act and self.q_head is not None and isinstance(batch.get("raw", None), list):
                q_logit = self.q_head(zH.to(self.q_head.weight.dtype)).squeeze(-1)
                with torch.no_grad():
                    corr = _segment_correctness(logits, prompt_lengths, batch["raw"]).to(q_logit.dtype)
                q_loss = F.binary_cross_entropy_with_logits(q_logit, corr)
                loss = loss + q_loss + act_penalty
                metrics["q_losses"].append(float(q_loss.detach().cpu()))

            metrics["segment_losses"].append(float(loss.detach().cpu()))
            total_loss = total_loss + loss.float()

        metrics["loss"] = float(total_loss.detach().cpu())
        metrics["zH_mean"] = zH_mean
        return total_loss, metrics


    # -----------------------
    #  Collator
    # -----------------------
    def collate(self, batch, max_length: int = 1024):
        """
        Build (input_ids, attention_mask, labels) for CLM with teacher forcing.
        """
        B = len(batch)
        prompts_text: list[str] = []
        targets_text: list[str] = []

        eos_tok, eos_id = resolve_eos_token(self.tokenizer, self.hrm_cfg.eos_override)

        for b in batch:
            system = b.get(
                "system",
                "Answer with the final result ONLY. Do not include any extra text."
            )
            user = b["prompt"]

            if "target" in b and isinstance(b["target"], str):
                tgt = b["target"]
            elif "answer" in b and isinstance(b["answer"], str):
                tgt = f"#### {b['answer'].strip()}"
            else:
                raise KeyError("Each batch item must include 'target' or 'answer'.")

            if hasattr(self.tokenizer, "apply_chat_template") and callable(self.tokenizer.apply_chat_template):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
                prompt_txt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                sys_prefix = (system.strip() + "\n\n") if system else ""
                prompt_txt = f"{sys_prefix}]\n{user}\n"

            tgt = (tgt or "").rstrip()
            if eos_tok and not tgt.endswith(eos_tok):
                tgt = tgt + eos_tok

            prompts_text.append(prompt_txt)
            targets_text.append(tgt)

        enc_p = self.tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc_t = self.tokenizer(
            targets_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids, attention_mask, labels, prompt_lengths = [], [], [], []

        for i in range(B):
            p_ids = enc_p["input_ids"][i]
            t_ids = enc_t["input_ids"][i]

            if t_ids.numel() > 0:
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None and t_ids[0].item() == bos_id:
                    t_ids = t_ids[1:]

            if eos_id is not None:
                if t_ids.numel() == 0 or t_ids[-1].item() != eos_id:
                    t_ids = torch.cat(
                        [t_ids, torch.tensor([eos_id], dtype=t_ids.dtype, device=t_ids.device)],
                        dim=0
                    )

            ids = torch.cat([p_ids, t_ids], dim=0)

            if ids.size(0) > max_length:
                over = ids.size(0) - max_length
                keep_min_prompt = 1
                trim_prompt = min(over, max(keep_min_prompt, p_ids.size(0)) - keep_min_prompt)
                p_ids = p_ids[trim_prompt:]
                ids = torch.cat([p_ids, t_ids], dim=0)

            am = torch.ones_like(ids)

            lab = ids.clone()
            Lp = p_ids.size(0)
            lab[:Lp] = self.hrm_cfg.vocab_ignore_index  # mask prompt region

            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lab)
            prompt_lengths.append(int(Lp))

        maxT = max(x.size(0) for x in input_ids)

        def pad_list(x_list, pad_id):
            out = torch.full((B, maxT), pad_id, dtype=torch.long)
            for i, xi in enumerate(x_list):
                out[i, : xi.size(0)] = xi
            return out

        input_ids = pad_list(input_ids, self.tokenizer.pad_token_id or 0)
        attention_mask = pad_list(attention_mask, 0)
        labels = pad_list(labels, self.hrm_cfg.vocab_ignore_index)

        # Map any <unk> labels to ignore_index to avoid inf loss
        if self._unk_id is not None:
            labels[labels == int(self._unk_id)] = self.hrm_cfg.vocab_ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
            "raw": batch,
            "eos_id": int(eos_id) if eos_id is not None else None,
        }

    # -----------------------
    #  Checkpoint helpers
    # -----------------------
    def trainable_state_dict(self):
        """
        Return only small trainable parts (HRM + injector + projections + bias heads).
        Tokenizer/LLM configs are not saved here—LLM is frozen and reloaded by name/path.
        """
        out = {
            "hrm_cfg": self.hrm_cfg.__dict__,
            "x_proj": self.x_proj.state_dict(),
            "pool_mix": self.pool_mix.state_dict(),
            "h_block": self.h_block.state_dict(),
            "l_block": self.l_block.state_dict(),
            "in_norm": self.in_norm.state_dict(),
            "zH0": self.zH0.detach().cpu(),
            "zL0": self.zL0.detach().cpu(),
            "injector": self.injector.state_dict(),
            "q_head": None if self.q_head is None else self.q_head.state_dict(),
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
            "lm_hidden_size": self.lm.config.hidden_size,
            "delta_logits": self.delta_logits.state_dict(),
            "inj_gate": self.inj_gate.detach().cpu(),
            "vocab_bias": None,
            "vocab_gate": None,
        }
        if self.vocab_bias is not None:
            out["vocab_bias"] = self.vocab_bias.state_dict()
            out["vocab_gate"] = self.vocab_gate.detach().cpu()
        return out


    def _unwrap_legacy_payload(self, payload: dict) -> dict:
        """
        Accepts payloads saved by older scripts:
        - Some saves nest under 'state_dict' or 'trainable' etc.
        - Some used slightly different keys ('proj' -> 'x_proj', etc.).
        Returns a flat dict with best-effort key mapping.
        """
        # Unwrap common wrappers
        cand = payload
        for k in ("trainable", "state_dict", "controller", "hrm"):
            if isinstance(cand, dict) and k in cand and isinstance(cand[k], dict):
                cand = cand[k]

        # Shallow copy we can mutate
        sd = dict(cand)

        # Legacy → current key mapping (best-effort)
        key_map = {
            "proj": "x_proj",
            "proj_in": "x_proj",
            "proj_mix": "pool_mix",
            "mix_proj": "pool_mix",
            "h": "h_block",
            "l": "l_block",
            "norm": "in_norm",
            "gate": "inj_gate",
            "logit_delta": "delta_logits",
            "vocab_head": "vocab_bias",
        }
        for old, new in key_map.items():
            if old in sd and new not in sd:
                sd[new] = sd.pop(old)

        return sd

    def load_trainable_state_dict(self, payload: dict, strict: bool = False):
        """
        Robust loader for small trainable parts.
        Loads what exists, skips missing keys, and prints a short warning once.
        Compatible with older checkpoints that lack newer modules.
        """
        sd = self._unwrap_legacy_payload(payload)

        missing = []

        def _copy_param(param_name: str, dest_tensor: torch.Tensor):
            t = sd.get(param_name, None)
            if t is None:
                missing.append(param_name); return
            with torch.no_grad():
                dest_tensor.copy_(t.to(dest_tensor.device, dtype=dest_tensor.dtype))

        def _load_module(module: nn.Module, key: str):
            state = sd.get(key, None)
            if state is None:
                missing.append(key); return
            try:
                module.load_state_dict(state, strict=strict)
            except Exception as e:
                missing.append(f"{key} (shape/dtype mismatch: {e})")

        # Modules
        _load_module(self.x_proj,     "x_proj")
        _load_module(self.pool_mix,   "pool_mix")
        _load_module(self.h_block,    "h_block")
        _load_module(self.l_block,    "l_block")
        _load_module(self.in_norm,    "in_norm")
        _load_module(self.injector,   "injector")
        _load_module(self.delta_logits, "delta_logits")

        # Optional modules
        if (self.q_head is not None) and ("q_head" in sd):
            _load_module(self.q_head, "q_head")
        if (self.vocab_bias is not None) and ("vocab_bias" in sd):
            _load_module(self.vocab_bias, "vocab_bias")

        # Scalars / tensors
        _copy_param("zH0", self.zH0)
        _copy_param("zL0", self.zL0)

        inj_gate_src = sd.get("inj_gate", None)
        if inj_gate_src is not None:
            with torch.no_grad():
                self.inj_gate.copy_(inj_gate_src.to(self.inj_gate.device, dtype=self.inj_gate.dtype))
        else:
            missing.append("inj_gate")

        if (self.vocab_gate is not None) and ("vocab_gate" in sd):
            with torch.no_grad():
                self.vocab_gate.copy_(sd["vocab_gate"].to(self.vocab_gate.device, dtype=self.vocab_gate.dtype))
        elif self.vocab_gate is not None:
            missing.append("vocab_gate")

        # Quiet, single-line heads-up (only if something was missing)
        if missing:
            warnings.warn(
                "Partial checkpoint load (expected newer keys not found): "
                + ", ".join(missing),
                RuntimeWarning,
            )


    # -----------------------
    #  Inference (AR generate with HRM injection)
    # -----------------------
    @torch.no_grad()
    def generate(self,
                 prompts: List[str],
                 max_new_tokens: int = 128,
                 segments: Optional[int] = None,
                 inner_T: Optional[int] = None,
                 injector_scale: float = 1.0,
                 inject_over: str = "labels",
                 temperature: float = 1.0,
                 top_p: float = 0.0,
                 top_k: int = 0,
                 replan_every: Optional[int] = None,
                 eos_override: Optional[str] = None,
                 system_header: Optional[str] = None,
                 device: Optional[torch.device] = None) -> List[str]:
        """
        True autoregressive decoding with HRM guidance.
        """
        self.eval()
        device = device or next(self.parameters()).device

        eos_tok, eos_id = resolve_eos_token(self.tokenizer, eos_override or self.hrm_cfg.eos_override)
        sys_hdr = system_header or "Answer with the final result ONLY. Do not include any extra text."

        # Render prompts with chat template (same convention as collate)
        rendered = []
        if hasattr(self.tokenizer, "apply_chat_template") and callable(self.tokenizer.apply_chat_template):
            for p in prompts:
                msgs = [{"role": "system", "content": sys_hdr},
                        {"role": "user", "content": p}]
                rendered.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        else:
            for p in prompts:
                rendered.append(f"{sys_hdr}\n\nQuestion:\n{p}\n")

        enc = self.tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        B, _ = input_ids.shape

        # Initial zH from prompt region
        last_hidden = self.forward_llm_hidden(input_ids, attn_mask)
        prompt_lengths = [int(m.nonzero().numel()) for m in attn_mask]
        x_pool = self.pool_tokens(last_hidden, attn_mask, prompt_lengths)
        x_tilde = self.x_proj(x_pool.to(self.x_proj.weight.dtype))
        zH = self.hrm_segments(
            x_tilde,
            segments=segments or self.hrm_cfg.segments,
            inner_T=inner_T or self.hrm_cfg.inner_T
        )[-1]

        # Token lists we will extend
        outs = [input_ids[i, :prompt_lengths[i]].tolist() for i in range(B)]
        finished = [False] * B

        def _filter_row(row: torch.Tensor, tk: int, tp: float) -> torch.Tensor:
            # row: (V,) logits
            if tk and tk > 0:
                k = min(tk, row.numel())
                topk_vals, _ = torch.topk(row, k)
                cutoff = topk_vals[-1]
                row = torch.where(row >= cutoff, row, torch.tensor(float("-inf"), device=row.device, dtype=row.dtype))
            if tp and 0.0 < tp < 1.0:
                probs = torch.softmax(row, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > tp
                mask[..., 0] = False
                undo = torch.zeros_like(mask, dtype=torch.bool)
                undo.scatter_(dim=-1, index=sorted_idx, src=mask)
                row = torch.where(undo, torch.tensor(float("-inf"), device=row.device, dtype=row.dtype), row)
            return row

        for step in range(max_new_tokens):
            # Optional replan of zH
            if replan_every and step > 0 and (step % int(replan_every) == 0):
                maxL = max(len(o) for o in outs)
                cur = torch.full((B, maxL), self.tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
                for i, o in enumerate(outs):
                    cur[i, :len(o)] = torch.tensor(o, device=device)
                mask = (cur != (self.tokenizer.pad_token_id or 0)).long()
                last_hidden = self.forward_llm_hidden(cur, mask)
                new_prompt_lengths = [len(o) for o in outs]
                x_pool = self.pool_tokens(last_hidden, mask, new_prompt_lengths)
                x_tilde = self.x_proj(x_pool.to(self.x_proj.weight.dtype))
                zH = self.hrm_segments(
                    x_tilde,
                    segments=segments or self.hrm_cfg.segments,
                    inner_T=inner_T or self.hrm_cfg.inner_T
                )[-1]

            # Build current batch
            maxL = max(len(o) for o in outs)
            cur = torch.full((B, maxL), self.tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
            for i, o in enumerate(outs):
                cur[i, :len(o)] = torch.tensor(o, device=device)
            mask = (cur != (self.tokenizer.pad_token_id or 0)).long()

            last_hidden = self.forward_llm_hidden(cur, mask)

            # Per-position scaling: inject only from the last token onward
            scale_vec = torch.zeros((B, last_hidden.size(1)), device=device, dtype=last_hidden.dtype)
            for i in range(B):
                scale_vec[i, len(outs[i]) - 1:] = float(injector_scale)

            # Compose logits via the unified path
            logits, _ = self.logits_with_hrm(last_hidden, zH, scale_vec=scale_vec)  # (B,T,V)

            # Ban <unk> at decode time
            if self._unk_id is not None and 0 <= int(self._unk_id) < logits.size(-1):
                logits[..., int(self._unk_id)] = float("-inf")

            # Next-token logits
            next_logits = logits[:, -1, :]  # (B,V)

            # Temperature + filtering
            if temperature != 1.0:
                next_logits = next_logits / max(1e-8, float(temperature))

            if (top_k and top_k > 0) or (top_p and top_p > 0.0):
                rows = []
                for i in range(B):
                    rows.append(_filter_row(next_logits[i], top_k, top_p).unsqueeze(0))
                next_logits = torch.cat(rows, dim=0)

            # Sample/greedy → (B,)
            if (top_k and top_k > 0) or (top_p and top_p > 0.0) or (temperature != 1.0):
                probs = torch.softmax(next_logits, dim=-1)  # (B,V)
                # Guard against NaNs/Infs
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
            else:
                next_ids = next_logits.argmax(dim=-1)  # (B,)

            assert next_ids.dim() == 1 and next_ids.numel() == B, f"next_ids shape bug: {tuple(next_ids.shape)}"

            # Append & check EOS
            all_done = True
            for i in range(B):
                if not finished[i]:
                    nid = int(next_ids[i].item())
                    outs[i].append(nid)
                    if (eos_id is not None) and (nid == int(eos_id)):
                        finished[i] = True
                all_done = all_done and finished[i]
            if all_done:
                break

        # Detokenize generation only (after original prompt)
        texts = []
        for i in range(B):
            gen = outs[i][prompt_lengths[i]:]
            if eos_id is not None and (eos_id in gen):
                gen = gen[:gen.index(eos_id)]
            texts.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        return texts


