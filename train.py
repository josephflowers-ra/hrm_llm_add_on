#!/usr/bin/env python3
"""
Minimal trainer for the HRM + LLM hybrid, with spot-check logging.

Adds (updated):
- --injector {grb,cab} to choose GRB vs CAB
- --save_dir / --save_every / --resume (saves optimizer state too)
- --eval_every + --eval_n for larger, less noisy proxy evals
- --eval_seed_jitter to randomize eval sampling each time
- --eval_segments to run deeper HRM at eval-time (inference scaling test)
- --log_samples_every to print example prompt/target/pred triples
- --max_new_tokens to cap decoded length in spot-checks
- --log_csv to persist eval curves (CSV)
- --grad_clip to tame loss spikes
- NEW: --decode_temperature / --decode_top_p / --decode_top_k for eval/spot-check sampling
- NEW: per-position injection controls for train/eval:
    * --train_injector_scale / --train_inject_over
    * --eval_injector_scale  / --eval_inject_over
- NEW: prompt plumbing:
    * --sentinel (appended after the gold target during training)
    * --system_header (override the default header in RG wrapper)
"""

import argparse, os, random, time, math, csv
import torch
from torch.utils.data import DataLoader

from model import HRMController, HRMConfig
from reasoning_gym_wrapper import build_reasoning_dataset
from collections import Counter


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")



# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # For stricter determinism, you can uncomment:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------
# CSV logging (evaluations)
# ---------------------------
def init_csv(path: str):
    if not path:
        return
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step","elapsed_min","loss","seg1","seg2","acc_proxy","n","zH_mean"]
            )

def log_eval_csv(path: str, step: int, elapsed_min: float,
                 loss: float, seg1: float, seg2: float,
                 acc: float, n: int, zh_mean: float):
    if not path:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [step, f"{elapsed_min:.2f}", f"{loss:.4f}",
             f"{seg1:.3f}", f"{seg2:.3f}", f"{acc:.3f}", n, f"{zh_mean:.3f}"]
        )


# ---------------------------
# Helpers for randomized eval loaders
# ---------------------------
def make_eval_loader(ds_eval, batch_size: int, seed_jitter: bool):
    # Shuffle eval to avoid always seeing the same first batch.
    if seed_jitter:
        # time-based jittered seed (CPU generator)
        seed = int(time.time_ns() % (2**31 - 1))
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None
    return DataLoader(
        ds_eval,
        batch_size=batch_size,
        shuffle=True,          # shuffle eval to diversify samples
        drop_last=False,
        collate_fn=lambda b: b,
        generator=gen
    )


# ---------------------------
# Decoding helpers (sampling or greedy per-token on provided logits)
# ---------------------------
def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """
    Filter a logits vector (1D) using top-k and/or nucleus (top-p).
    Returns masked logits (in-place safe copy) with filtered positions = -inf.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals, _ = torch.topk(logits, top_k)
        cutoff = kth_vals[..., -1]
        logits = torch.where(logits >= cutoff, logits, torch.tensor(float("-inf"), device=logits.device))
    if top_p and top_p > 0.0 and top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False  # keep at least one
        original_mask = torch.zeros_like(mask, dtype=torch.bool)
        original_mask.scatter_(dim=-1, index=sorted_idx, src=mask)
        logits = torch.where(original_mask, torch.tensor(float("-inf"), device=logits.device), logits)
    return logits


def _decode_from_logits(logits: torch.Tensor,
                        prompt_lengths,
                        temperature: float = 1.0,
                        top_p: float = 0.0,
                        top_k: int = 0,
                        greedy_fallback: bool = True) -> torch.Tensor:
    """
    logits: (B, T, V) over full concatenated sequence.
    Returns pred_ids (B, T) for the whole sequence, where only indices >= Lp are meaningful.
    If sampling is disabled (temp==1.0, top_p==0, top_k==0): greedy argmax.
    """
    # --- Defensive shape handling: squeeze accidental singleton dims (e.g., (B,1,T,V)) ---
    if logits.dim() == 4 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    # Clear assertion with helpful message if shape is still off
    assert logits.dim() == 3, f"_decode_from_logits expected (B,T,V); got shape={tuple(logits.shape)}"

    B, T, V = logits.shape
    do_sample = (temperature != 1.0) or (top_p and top_p > 0.0) or (top_k and top_k > 0)
    if not do_sample:
        return logits.argmax(-1)

    # Sampling per position (non-autoregressive over the label window)
    pred_ids = torch.zeros((B, T), dtype=torch.long, device=logits.device)
    for i in range(B):
        Lp = int(prompt_lengths[i])
        if Lp > 0:
            pred_ids[i, :Lp] = logits[i, :Lp].argmax(-1)
        for t in range(Lp, T):
            row = logits[i, t]
            if temperature != 1.0:
                row = row / max(1e-8, temperature)
            row = _top_k_top_p_filtering(row, top_k=top_k, top_p=top_p)
            if torch.isneginf(row).all():
                pred_ids[i, t] = logits[i, t].argmax(-1) if greedy_fallback else torch.randint(0, V, (1,), device=logits.device).item()
            else:
                probs = torch.softmax(row, dim=-1)
                pred_ids[i, t] = torch.multinomial(probs, num_samples=1).squeeze(0)
    return pred_ids


# ---------------------------
# Lightweight proxy evaluation (with scale_vec support)
# ---------------------------
@torch.no_grad()
def evaluate(model: HRMController,
             ds_eval,
             device: str,
             batch_size: int,
             eval_n: int,
             segments_eval: int = None,
             seed_jitter: bool = False,
             decode_temperature: float = 1.0,
             decode_top_p: float = 0.0,
             decode_top_k: int = 0,
             eval_injector_scale: float = 1.0,
             eval_inject_over: str = "labels",
             max_length: int = 2048):
    """
    eval_inject_over: "labels" (default) or "all"

    NOTE: This evaluation path mirrors training by calling `model.logits_with_hrm(...)`
    (gated hidden injection + Δlogits/vocab-bias). This avoids train/eval skew.
    """
    model.eval()

    seg_eval = segments_eval if segments_eval is not None else model.hrm_cfg.segments
    dl = make_eval_loader(ds_eval, batch_size=batch_size, seed_jitter=seed_jitter)

    correct, total = 0, 0
    z_norm_accum = 0.0
    n_batches = math.ceil(max(1, eval_n) / max(1, batch_size))

    batches_seen = 0
    for batch in dl:
        col = model.collate(batch, max_length=max_length)
        for k in ("input_ids", "attention_mask", "labels"):
            col[k] = col[k].to(device)

        last_hidden = model.forward_llm_hidden(col["input_ids"], col["attention_mask"])  # (B,T,D)
        B, T, _ = last_hidden.shape

        x_pool = model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"])
        x_tilde = model.x_proj(x_pool)
        zH_all = model.hrm_segments(x_tilde, segments=seg_eval, inner_T=model.hrm_cfg.inner_T)
        zH = zH_all[-1]

        if eval_inject_over not in ("all", "labels"):
            raise ValueError("eval_inject_over must be 'all' or 'labels'")
        scale_vec = torch.full((B, T), float(eval_injector_scale),
                               device=last_hidden.device, dtype=last_hidden.dtype)
        if eval_inject_over == "labels":
            for i, Lp in enumerate(col["prompt_lengths"]):
                Lp_i = max(0, min(int(Lp), T))
                scale_vec[i, :Lp_i] = 0.0

        # ---------- Unified path with training ----------
        logits, _ = model.logits_with_hrm(last_hidden, zH, scale_vec=scale_vec)

        # Extra safety: ensure (B,T,V) before decoding
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        pred_ids = _decode_from_logits(
            logits,
            prompt_lengths=col["prompt_lengths"],
            temperature=decode_temperature,
            top_p=decode_top_p,
            top_k=decode_top_k
        )

        z_norm_accum += torch.norm(zH, dim=-1).mean().item()

        for i, item in enumerate(batch):
            Lp = col["prompt_lengths"][i]
            text = model.tokenizer.decode(pred_ids[i, Lp:], skip_special_tokens=True)
            ok = item.get("verify", None)
            correct += int(callable(ok) and ok(text))
            total += 1
            if total >= eval_n:
                break

        batches_seen += 1
        if total >= eval_n or batches_seen >= n_batches:
            break

    zh_mean = (z_norm_accum / max(1, batches_seen))
    return {"eval_acc_proxy": correct / max(1, total),
            "n": total,
            "zH_mean": zh_mean}


# ---------------------------
# Spot-check samples (with scale_vec support)
# ---------------------------
@torch.no_grad()
def spot_check_samples(model: HRMController,
                       ds_eval,
                       device: str,
                       max_new_tokens: int,
                       n_print: int = 3,
                       seed_jitter: bool = True,
                       segments_eval: int = None,
                       decode_temperature: float = 1.0,
                       decode_top_p: float = 0.0,
                       decode_top_k: int = 0,
                       eval_injector_scale: float = 1.0,
                       eval_inject_over: str = "labels",
                       max_length: int = 2048):
    """
    eval_inject_over: "labels" (default) or "all"

    NOTE: Spot checks also use `logits_with_hrm(...)` to match training.
    """
    model.eval()

    seg_eval = segments_eval if segments_eval is not None else model.hrm_cfg.segments

    dl_eval = make_eval_loader(ds_eval, batch_size=max(1, n_print), seed_jitter=seed_jitter)
    try:
        batch = next(iter(dl_eval))
    except StopIteration:
        model.train()
        return

    col = model.collate(batch, max_length=max_length)
    for k in ("input_ids", "attention_mask", "labels"):
        col[k] = col[k].to(device)

    last_hidden = model.forward_llm_hidden(col["input_ids"], col["attention_mask"])  # (B,T,D)
    B, T, _ = last_hidden.shape

    x_pool = model.pool_tokens(last_hidden, col["attention_mask"], col["prompt_lengths"])
    x_tilde = model.x_proj(x_pool)
    zH = model.hrm_segments(x_tilde, segments=seg_eval, inner_T=model.hrm_cfg.inner_T)[-1]

    if eval_inject_over not in ("all", "labels"):
        raise ValueError("eval_inject_over must be 'all' or 'labels'")
    scale_vec = torch.full((B, T), float(eval_injector_scale),
                           device=last_hidden.device, dtype=last_hidden.dtype)
    if eval_inject_over == "labels":
        for i, Lp in enumerate(col["prompt_lengths"]):
            Lp_i = max(0, min(int(Lp), T))
            scale_vec[i, :Lp_i] = 0.0

    # ---------- Unified path with training ----------
    logits, _ = model.logits_with_hrm(last_hidden, zH, scale_vec=scale_vec)

    # Extra safety: ensure (B,T,V) before decoding
    if logits.dim() == 4 and logits.size(1) == 1:
        logits = logits.squeeze(1)

    pred_ids = _decode_from_logits(
        logits,
        prompt_lengths=col["prompt_lengths"],
        temperature=decode_temperature,
        top_p=decode_top_p,
        top_k=decode_top_k
    )

    z_norm = torch.norm(zH, dim=-1).mean().item()
    print(f"\n[SPOT CHECK] mean ||z_H|| = {z_norm:.3f}")
    mode = "sample" if ((decode_temperature != 1.0) or (decode_top_p and decode_top_p > 0.0) or (decode_top_k and decode_top_k > 0)) else "greedy"
    print(f"[SPOT CHECK] decode_mode={mode} (T={decode_temperature}, top_p={decode_top_p}, top_k={decode_top_k})")
    print(f"[SPOT CHECK] eval_inject_over={eval_inject_over}, eval_injector_scale={eval_injector_scale}")
    print("[SPOT CHECK] Showing up to", n_print, "examples:")

    # --- Show which tasks are present in this batch (helpful to sanity-check the mix) ---
    tasks = [it.get("metadata", {}).get("task", "?") for it in batch[:n_print]]
    print("[SPOT CHECK] tasks:", Counter(tasks))

    # Resolve eos id once for clean slicing
    eos_id = getattr(model, "_eos_id", getattr(model.tokenizer, "eos_token_id", None))

    for i, item in enumerate(batch[:n_print]):
        task_name = (item.get("metadata", {}) or {}).get("task", "?")
        Lp = col["prompt_lengths"][i]

        prompt_ids = col["input_ids"][i, :Lp]
        target_ids = col["labels"][i, Lp:]
        target_ids = target_ids.masked_fill(
            target_ids == model.hrm_cfg.vocab_ignore_index,
            model.tokenizer.pad_token_id or 0
        )

        # Decode prompt/target (truncated for display)
        prompt_txt = model.tokenizer.decode(prompt_ids[:max_new_tokens], skip_special_tokens=True).strip()
        target_txt = model.tokenizer.decode(target_ids[:max_new_tokens], skip_special_tokens=True).strip()

        # EOS-aware pred decoding
        span = pred_ids[i, Lp: Lp + max_new_tokens].tolist()
        if eos_id is not None and eos_id in span:
            span = span[:span.index(eos_id)]
        pred_txt = model.tokenizer.decode(span, skip_special_tokens=False).strip()

        if not pred_txt:
            # clarify if the very first label token is EOS
            if eos_id is not None and (Lp < pred_ids.size(1)) and int(pred_ids[i, Lp]) == int(eos_id):
                pred_txt = "<EOS @ first label token>"
            else:
                pred_txt = "<empty>"

        print(f"\n-- Example {i+1} [{task_name}] --")
        print("Prompt:", prompt_txt)
        print("Target:", target_txt)
        print("Pred:  ", pred_txt)

    model.train()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Base model + task
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF name or local path to a CausalLM (frozen).")
    ap.add_argument("--task", type=str, default="toy_add", help="reasoning_gym task or 'toy_add'.")
    ap.add_argument("--tasks", type=str, default="",
                    help="Comma-separated RG tasks for a mixed dataset. If set, overrides --task.")

    # Sequence / dataset sizing
    ap.add_argument("--max_seq_len", type=int, default=2048,
                    help="Max total sequence length (prompt+target) used in collation/truncation.")
    ap.add_argument("--train_size", type=int, default=10000,
                    help="Number of training examples to sample/build.")
    ap.add_argument("--eval_size", type=int, default=200,
                    help="Number of eval examples to sample/build.")

    # HRM depth controls
    ap.add_argument("--segments", type=int, default=3, help="Number of HRM segments (H updates).")
    ap.add_argument("--inner_T", type=int, default=3, help="Number of L micro-steps per segment.")

    # Injector choice
    ap.add_argument("--injector", type=str, choices=("grb", "cab"), default="grb",
                    help="GRB = gated residual bias; CAB = cross-attention bridge (gated).")

    # Optional ACT + temperature (training CE)
    ap.add_argument("--use_act", action="store_true", default=False, help="Enable ACT (halt head) training.")
    ap.add_argument("--act_penalty", type=float, default=0.0, help="Compute penalty to bias fewer segments.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Training-time logit temperature for CE.")

    # NEW: decoding controls for eval/spot-checks
    ap.add_argument("--decode_temperature", type=float, default=1.0,
                    help="Eval/spot-check decoding temperature. If !=1.0 or with top_p/top_k, sampling is used.")
    ap.add_argument("--decode_top_p", type=float, default=0.0,
                    help="Eval/spot-check nucleus sampling p (0 disables).")
    ap.add_argument("--decode_top_k", type=int, default=0,
                    help="Eval/spot-check top-k sampling (0 disables).")

    # Optimization + runtime
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=200)

    # Eval controls (upgraded)
    ap.add_argument("--eval_every", type=int, default=50, help="Steps between evals.")
    ap.add_argument("--eval_batches", type=int, default=None,
                    help="(Deprecated) If set, limits eval to this many batches. Prefer --eval_n.")
    ap.add_argument("--eval_n", type=int, default=200, help="Total eval samples per eval call.")
    ap.add_argument("--eval_seed_jitter", action="store_true",
                    help="If set, use a time-jittered seed to shuffle eval each time.")
    ap.add_argument("--eval_segments", type=int, default=4,
                    help="Override HRM segments at eval-time (e.g., train 2, eval 4).")

    # I/O
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_every", type=int, default=1000, help="Save every N steps (e.g., 500 or 1000).")
    ap.add_argument("--resume", type=str, default="", help="Path to .pt checkpoint to resume.")
    ap.add_argument("--log_csv", type=str, default="hrm_training_log.csv",
                    help="Path to CSV file for eval logs (set empty to disable).")

    # Spot-check logs
    ap.add_argument("--log_samples_every", type=int, default=500,
                    help="Steps between printing example prompt/target/pred triples.")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max tokens to decode for targets/preds in spot-checks.")

    # System
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training stability
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Global grad-norm clip (0 to disable).")

    # NEW: per-position injection controls
    ap.add_argument("--train_injector_scale", type=float, default=1.0,
                    help="Global multiplier for injection strength during training.")
    ap.add_argument("--train_inject_over", type=str, choices=("all", "labels"), default="all",
                    help="Where to apply injection during training: 'all' tokens or only 'labels'.")
    ap.add_argument("--eval_injector_scale", type=float, default=1.0,
                    help="Global multiplier for injection strength during eval/spot-check.")
    ap.add_argument("--eval_inject_over", type=str, choices=("all", "labels"), default="labels",
                    help="Where to apply injection during eval/spot-check: 'all' or 'labels' (default).")

    # NEW: prompt plumbing to RG wrapper
    ap.add_argument("--sentinel", type=str, default="",
                    help="If set, appended to targets during training (and used by the verifier).")
    ap.add_argument("--system_header", type=str, default="",
                    help="If set, overrides the default system header string in prompts.")
    ap.add_argument("--cab_mem", type=int, default=4, help="CAB memory tokens (m).")
    ap.add_argument("--cab_gate_init", type=float, default=-1.5, help="CAB gate init.")
    ap.add_argument("--grb_gate_init", type=float, default=-2.0, help="GRB gate init.")
    ap.add_argument("--vocab_gate_init", type=float, default=-2.0, help="Logit-bias gate init.")
    
    args = ap.parse_args()

    # Build HRM config (flip use_cab per injector flag)
    use_cab = (args.injector.lower() == "cab")
    hrm_cfg = HRMConfig(
        inner_T=args.inner_T,
        segments=args.segments,
        use_act=args.use_act,
        use_cab=use_cab,
        logit_bias_init=args.vocab_gate_init,
        cab_mem_tokens=args.cab_mem,
        cab_gate_init=args.cab_gate_init,
        grb_gate_init=args.grb_gate_init
    )

    # Boot
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Injector: {args.injector.upper()} | segments={args.segments} | inner_T={args.inner_T}")
    print(f"[INFO] Train inject: over={args.train_inject_over}, scale={args.train_injector_scale}")
    print(f"[INFO] Eval   inject: over={args.eval_inject_over}, scale={args.eval_injector_scale}")
    if args.sentinel:
        print(f"[INFO] Using sentinel: {repr(args.sentinel)}")
    if args.system_header:
        print(f"[INFO] Overriding system header.")
    print(f"[INFO] Decode mode: "
          f"{'sample' if (args.decode_temperature != 1.0 or args.decode_top_p > 0.0 or args.decode_top_k > 0) else 'greedy'} "
          f"(T={args.decode_temperature}, top_p={args.decode_top_p}, top_k={args.decode_top_k})")
    print(f"[INFO] max_seq_len={args.max_seq_len} | train_size={args.train_size} | eval_size={args.eval_size}")

    set_seed(args.seed)

    # Model (LLM is frozen; HRM + injector trainable)
    model = HRMController(args.model_name, hrm_cfg).to(args.device)

    # Resume (optional)
    os.makedirs(args.save_dir, exist_ok=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_trainable_state_dict(ckpt["state"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        print(f"[INFO] Resumed from {args.resume} (step={start_step})")

    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data
    chosen = args.tasks.strip() if args.tasks.strip() else args.task

    ds_train = build_reasoning_dataset(
        chosen, "train", n=args.train_size, seed=args.seed,
        sentinel=args.sentinel, system_header=args.system_header
    )
    ds_eval  = build_reasoning_dataset(
        chosen, "eval",  n=args.eval_size,  seed=args.seed + 42,
        sentinel=args.sentinel, system_header=args.system_header
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: b
    )

    # CSV init
    init_csv(args.log_csv)

    # Train
    step = start_step
    model.train()
    t0 = time.time()

    while step < args.max_steps:
        for batch in dl_train:
            col = model.collate(batch, max_length=args.max_seq_len)
            for k in ("input_ids", "attention_mask", "labels"):
                col[k] = col[k].to(args.device)

            # ✅ shape sanity checks (right after collate, before training_step)
            B, T = col["input_ids"].shape
            assert col["labels"].shape == (B, T), f"labels shape {col['labels'].shape} != {(B, T)}"
            assert col["attention_mask"].shape == (B, T), f"attn_mask shape {col['attention_mask'].shape} != {(B, T)}"

            loss, metrics = model.training_step(
                col,
                segments=args.segments,
                inner_T=args.inner_T,
                use_act=args.use_act,
                temperature=args.temperature,
                act_penalty=args.act_penalty,
                injector_scale=args.train_injector_scale,   # per-position control (train)
                inject_over=args.train_inject_over
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip
                )

            optim.step()
            step += 1

            if step % 10 == 0:
                seg_losses = [round(x, 3) for x in metrics['segment_losses']]
                q_losses   = [round(x, 3) for x in metrics['q_losses']]
                print(f"[STEP {step}] loss={metrics['loss']:.4f} | seg_losses={seg_losses} | q_losses={q_losses}")

            if step % args.eval_every == 0:
                # Backward-compat for --eval_batches if someone still passes it
                if args.eval_batches is not None:
                    eval_n = args.eval_batches * args.batch_size
                else:
                    eval_n = args.eval_n

                ev = evaluate(
                    model=model,
                    ds_eval=ds_eval,
                    device=args.device,
                    batch_size=args.batch_size,
                    eval_n=eval_n,
                    segments_eval=args.eval_segments,
                    seed_jitter=args.eval_seed_jitter,
                    decode_temperature=args.decode_temperature,
                    decode_top_p=args.decode_top_p,
                    decode_top_k=args.decode_top_k,
                    eval_injector_scale=args.eval_injector_scale,   
                    eval_inject_over=args.eval_inject_over,
                    max_length=args.max_seq_len,
                )
                dt_min = (time.time() - t0) / 60.0

                seg_losses = metrics.get('segment_losses', [float('nan'), float('nan')])
                seg1 = float(seg_losses[0]) if len(seg_losses) > 0 else float('nan')
                seg2 = float(seg_losses[1]) if len(seg_losses) > 1 else float('nan')

                print(f"[EVAL {step}] acc_proxy={ev['eval_acc_proxy']:.3f} on n={ev['n']} | "
                      f"zH_mean={ev['zH_mean']:.3f} | elapsed={dt_min:.1f}m")

                log_eval_csv(
                    path=args.log_csv,
                    step=step,
                    elapsed_min=dt_min,
                    loss=float(metrics.get('loss', 0.0)),
                    seg1=seg1, seg2=seg2,
                    acc=float(ev['eval_acc_proxy']),
                    n=int(ev['n']),
                    zh_mean=float(ev['zH_mean'])
                )

            if step % args.log_samples_every == 0:
                spot_check_samples(
                    model=model,
                    ds_eval=ds_eval,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    n_print=3,
                    seed_jitter=True,
                    segments_eval=args.eval_segments,
                    decode_temperature=args.decode_temperature,
                    decode_top_p=args.decode_top_p,
                    decode_top_k=args.decode_top_k,
                    eval_injector_scale=args.eval_injector_scale,
                    eval_inject_over=args.eval_inject_over,
                    max_length=args.max_seq_len,
                )

            if (step % args.save_every == 0) or (step >= args.max_steps):
                path = os.path.join(args.save_dir, f"hrm_step{step}.pt")
                torch.save(
                    {"step": step,
                     "state": model.trainable_state_dict(),
                     "optim": optim.state_dict()},
                    path
                )
                print(f"[CKPT] Saved {path}")

            if step >= args.max_steps:
                break

    print("[DONE]")


if __name__ == "__main__":
    main()
