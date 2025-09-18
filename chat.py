#!/usr/bin/env python3
import os
import sys
import json
import argparse
import traceback
import torch

# local imports
from model import HRMController, HRMConfig


def load_controller(
    model_name: str,
    injector: str,
    ckpt_path: str | None,
    device: str,
    cfg_overrides: dict,
) -> HRMController:
    # Minimal, explicit config; allow CLI JSON to override
    cfg = HRMConfig(
        use_cab=(injector.lower() == "cab"),
        d_h=cfg_overrides.get("d_h", HRMConfig.d_h),
        h_layers=cfg_overrides.get("h_layers", HRMConfig.h_layers),
        l_layers=cfg_overrides.get("l_layers", HRMConfig.l_layers),
        n_heads=cfg_overrides.get("n_heads", HRMConfig.n_heads),
        inner_T=cfg_overrides.get("inner_T", HRMConfig.inner_T),
        segments=cfg_overrides.get("segments", HRMConfig.segments),
        inj_dropout_p=cfg_overrides.get("inj_dropout_p", 0.0),
        z_noise_std=cfg_overrides.get("z_noise_std", 0.0),
        gate_l2_coef=cfg_overrides.get("gate_l2_coef", 1e-3),
        kl_lambda=cfg_overrides.get("kl_lambda", 0.0),
        cab_mem_tokens=cfg_overrides.get("cab_mem_tokens", 4),
        cab_gate_init=cfg_overrides.get("cab_gate_init", -1.5),
        grb_gate_init=cfg_overrides.get("grb_gate_init", -2.0),
        eos_override=cfg_overrides.get("eos_override", None),
        logit_bias_head=cfg_overrides.get("logit_bias_head", True),
        vocab_ignore_index=-100,
        # Accept optional delta_scale if your model.py defines it
        **({"delta_scale": cfg_overrides["delta_scale"]} if "delta_scale" in cfg_overrides else {}),
    )

    ctrl = HRMController(model_name=model_name, hrm_cfg=cfg)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ctrl.to(dev)

    if ckpt_path:
        if os.path.exists(ckpt_path):
            payload = torch.load(ckpt_path, map_location=dev)
            if isinstance(payload, dict) and "model" in payload:
                payload = payload["model"]
            try:
                ctrl.load_trainable_state_dict(payload, strict=False)
                print(f"[INFO] Loaded checkpoint: {ckpt_path}", flush=True)
            except Exception as e:
                # Keep REPL usable even if ckpt is partial/older
                print(f"[WARN] Failed to fully load checkpoint ({e}). Continuing with current weights.", flush=True)
        else:
            print(f"[WARN] Checkpoint not found: {ckpt_path}", flush=True)

    return ctrl


def run_once(ctrl: HRMController, prompt: str, args) -> str:
    # One-shot wrapper. All knobs pass straight to controller.generate.
    texts = ctrl.generate(
        prompts=[prompt],
        max_new_tokens=args.max_new_tokens,
        segments=args.segments or ctrl.hrm_cfg.segments,
        inner_T=args.inner_T or ctrl.hrm_cfg.inner_T,
        injector_scale=args.injector_scale,
        inject_over=args.inject_over,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        replan_every=(args.replan_every if args.replan_every and args.replan_every > 0 else None),
        eos_override=args.eos_override,
        system_header=args.system_header,  # system message support
        device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")),
    )
    return texts[0] if texts else ""


def main():
    p = argparse.ArgumentParser(description="HRM chat / generation script")
    # model + ckpt
    p.add_argument("--model_name", type=str, required=True, help="HF path or local dir (e.g. ./tiny-rl-sft)")
    p.add_argument("--injector", type=str, default="grb", choices=["grb", "cab"], help="Injector type")
    p.add_argument("--ckpt", type=str, default="", help="Path to checkpoint (hrm_stepXXXX.pt)")
    p.add_argument("--device", type=str, default="", help="cuda | cpu (default: auto)")

    # HRM runtime knobs
    p.add_argument("--segments", type=int, default=None, help="override HRM segments")
    p.add_argument("--inner_T", type=int, default=None, help="override HRM inner_T")
    p.add_argument("--inject_over", type=str, default="labels", choices=["labels", "all"])
    p.add_argument("--injector_scale", type=float, default=1.0)

    # decoding
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--replan_every", type=int, default=0, help="0=off; recompute zH every N tokens if >0")

    # formatting
    p.add_argument("--eos_override", type=str, default=None)
    p.add_argument("--system_header", type=str,
                   default="Answer with the final result ONLY. Do not include any extra text.")

    # optional config overrides as JSON (for HRMConfig)
    p.add_argument("--cfg_json", type=str, default="", help='JSON dict for HRMConfig overrides (e.g. \'{"d_h":512}\')')

    # non-interactive mode
    p.add_argument("--prompt", type=str, default="", help="Run one prompt and exit")

    args = p.parse_args()

    # Parse cfg overrides
    cfg_overrides = {}
    if args.cfg_json:
        try:
            cfg_overrides = json.loads(args.cfg_json)
        except Exception as e:
            print(f"[WARN] Could not parse --cfg_json: {e}", file=sys.stderr)

    # Build controller
    controller = load_controller(
        model_name=args.model_name,
        injector=args.injector,
        ckpt_path=args.ckpt,
        device=args.device,
        cfg_overrides=cfg_overrides,
    )

    # One-off path
    if args.prompt:
        try:
            out = run_once(controller, args.prompt, args)
            print(out)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return

    # Interactive REPL
    print("---- HRM Chat REPL ----")
    print("Type your prompt and press Enter.")
    print("Commands: :q to quit, :cfg to show config, :sys <text> to set system message.\n")
    print(f"[System] {args.system_header}")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue

        if user == ":q":
            break

        if user == ":cfg":
            print("HRMConfig:", controller.hrm_cfg)
            print("Injector:", "CAB" if controller.hrm_cfg.use_cab else "GRB")
            print("Device:", next(controller.parameters()).device)
            print(f"System header: {args.system_header}")
            continue

        if user.startswith(":sys"):
            # Allow changing the system message on the fly
            new_sys = user[len(":sys"):].strip()
            if new_sys:
                args.system_header = new_sys
                print(f"[System updated]")
                print(f"[System] {args.system_header}")
            else:
                print("[System] (unchanged) " + args.system_header)
            continue

        # Normal prompt
        try:
            out = run_once(controller, user, args)
            print(out)
        except Exception:
            traceback.print_exc()
            print("[ERROR] Inference failed; continuing REPL.")

    print("[DONE]")


if __name__ == "__main__":
    main()
