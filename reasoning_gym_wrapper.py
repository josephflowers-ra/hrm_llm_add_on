# reasoning_gym_wrapper.py
"""
Reasoning Gym adapter for HRM training (answer-only targets), with optional sentinel.

- Supports single-task (e.g., "toy_add") or multi-task mixes via comma-separated list.
- Emits items shaped like:
    {
      "prompt": <string>,           # full prompt text given to the tokenizer
      "target": <string>,           # gold final answer ONLY (+ optional sentinel)
      "answer": <string>,           # plain gold answer (for convenience/logging)
      "verify": <callable(str)->bool>,
      "metadata": {...},            # may include source task name, etc.
    }

Defaults:
- Training target is the *answer only*. If you want to teach hard stopping, pass a
  non-empty `sentinel` string to `build_reasoning_dataset(...)` and it will be
  appended to the gold target (e.g., "<|eot_id|>", "<|im_end|>", "</s>", etc).
- At inference, generations may include extra text—verifier robustly extracts the
  candidate answer (handles '####', first line, whitespace, unicode minus).

Backwards compatible with existing train.py calls.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Sequence, Union
import math
import re
import unicodedata
import random

try:
    from reasoning_gym import create_dataset, get_score_answer_fn
except Exception:
    # Graceful fallback if RG isn't installed at import-time
    create_dataset = None
    get_score_answer_fn = None


# ------------------------------
# Prompt / target formatting
# ------------------------------
DEFAULT_SYSTEM_HEADER = (
    "Answer with the final result ONLY. Do not include any extra text."
)

USER_TEMPLATE = """Question:
{question}
"""

def format_prompt(question: str, system_header: str = DEFAULT_SYSTEM_HEADER) -> str:
    return f"{system_header}\n\n{USER_TEMPLATE.format(question=question)}"


# ------------------------------
# Normalization / extraction
# ------------------------------
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"^[+-]?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?$|^[+-]?\d+(?:\.\d+)?$")

def _normalize_answer(s: str) -> str:
    """Unicode/whitespace normalization for stable comparisons."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    # Normalize common minus variants and common quotes
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("´", "'")
    return s

def _maybe_as_number(s: str) -> Optional[str]:
    """
    If the string looks like a number, normalize to a canonical textual form:
    - remove thousands separators (comma or space)
    - keep sign and decimal point
    Returns None if it doesn't look numeric.
    """
    t = s.replace(",", "").replace(" ", "")
    if _NUM_RE.match(s) or _NUM_RE.match(t):
        return t
    return None

def _extract_pred(pred: Optional[str]) -> str:
    """
    Extract a candidate answer from a generated string.
    - Prefer content after the FIRST '####' if present (UI convention).
    - Otherwise use the whole string.
    - Keep only the first line and normalize.
    """
    if not isinstance(pred, str):
        return ""
    s = pred.strip()
    if not s:
        return ""
    if "####" in s:
        s = s.split("####", 1)[1]
    first_line = s.splitlines()[0] if s else ""
    return _normalize_answer(first_line)


# ------------------------------
# Gold answer extraction
# ------------------------------
def extract_gold_answer(item: Dict[str, Any]) -> str:
    """
    Best-effort: RG datasets commonly put the canonical answer under item['answer'].
    Some tasks store answers in metadata. Adjust here if you encounter a variant.
    """
    # Primary
    if "answer" in item and isinstance(item["answer"], str):
        return item["answer"].strip()

    # Fallbacks: Some RG tasks keep answers in metadata
    meta = item.get("metadata", {})
    if isinstance(meta, dict):
        for key in ("answer", "target", "gold"):
            if key in meta and isinstance(meta[key], str):
                return meta[key].strip()

    # As a last resort, just stringify
    return str(item.get("answer", "")).strip()


# ------------------------------
# Verifier factory
# ------------------------------
def _multi_gold_list(gold: Union[str, List[str]]) -> List[str]:
    """
    Allow multiple golds:
    - If gold is a list, use as-is.
    - If gold string contains '||', split into alternatives.
    - Else single gold.
    """
    if isinstance(gold, list):
        return [g for g in gold if isinstance(g, str)]
    if isinstance(gold, str) and "||" in gold:
        return [g.strip() for g in gold.split("||") if g.strip()]
    return [str(gold)]

def make_verifier(source_dataset: Optional[str], gold: Union[str, List[str]]) -> Callable[[str], bool]:
    """
    Returns a function pred -> bool indicating correctness.
    Prefers RG's official scorer when available; otherwise exact match (robust).
    - Supports multiple acceptable gold answers (list or 'a || b' string).
    - Numeric answers compared after numeric canonicalization.
    """
    scorer = None
    if get_score_answer_fn and source_dataset:
        try:
            scorer = get_score_answer_fn(source_dataset)
        except Exception:
            scorer = None

    golds_raw = _multi_gold_list(gold)
    golds_norm = [_normalize_answer(g) for g in golds_raw]
    golds_num  = [_maybe_as_number(g) for g in golds_norm]

    if scorer is not None:
        # Use RG’s scorer with a robust fallback
        def verify_fn(pred: str) -> bool:
            try:
                pred_norm = _extract_pred(pred)
                if not pred_norm:
                    return False
                # If any gold passes, accept
                for g in golds_raw:
                    if scorer(pred_norm, {"answer": g}) or scorer(pred, {"answer": g}):
                        return True
                # Fallback to normalized exact / numeric compare
                return _verify_simple(pred_norm, golds_norm, golds_num)
            except Exception:
                return _verify_simple(_extract_pred(pred), golds_norm, golds_num)
        return verify_fn

    # Simple fallback
    def verify_fallback(pred: str) -> bool:
        pred_norm = _extract_pred(pred)
        return _verify_simple(pred_norm, golds_norm, golds_num)

    return verify_fallback

def _verify_simple(pred_norm: str, golds_norm: List[str], golds_num: List[Optional[str]]) -> bool:
    if not pred_norm:
        return False
    # Direct normalized match
    if pred_norm in golds_norm:
        return True
    # Numeric tolerant match
    p_num = _maybe_as_number(pred_norm)
    if p_num is not None:
        for gnum in golds_num:
            if gnum is not None and p_num == gnum:
                return True
    return False


# ------------------------------
# Dataset containers / mixing
# ------------------------------
@dataclass
class SimpleDataset:
    data: List[Dict[str, Any]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _make_single_task_dataset(task: str,
                              split: str,
                              n: int,
                              seed: int,
                              sentinel: str = "",
                              system_header: str = DEFAULT_SYSTEM_HEADER) -> SimpleDataset:
    """
    Builds a single-task RG dataset with our prompt/verify packaging.

    Args:
      sentinel: if non-empty, append to the gold target (teaches hard stop).
      system_header: override instruction string used in the prompt.
    """
    if create_dataset is None:
        raise RuntimeError("reasoning_gym is not available. Please `pip install reasoning-gym`.")

    # Map split name to a seed range so eval isn't the same as train by accident
    base_seed = seed + (1337 if split == "eval" else 0)

    raw = create_dataset(task, seed=base_seed, size=n)
    packed = []
    for ex in raw:
        q = (ex.get("question", "") or "").strip()
        gold = extract_gold_answer(ex)

        # Source dataset name (when provided by RG)
        src = None
        md = ex.get("metadata", {})
        if isinstance(md, dict):
            src = md.get("source_dataset", task)

        prompt = format_prompt(q, system_header=system_header)
        verify = make_verifier(src, gold)

        # Optionally append sentinel to target to teach a hard stop token.
        target = gold + (sentinel if sentinel else "")

        packed.append({
            "prompt": prompt,
            "target": target,         # <-- answer-only (plus optional sentinel)
            "answer": gold,           # for convenience/logging
            "verify": verify,
            "metadata": {"task": task, "source_dataset": src or task},
        })

    return SimpleDataset(packed)


def _round_robin_mix(datasets: Sequence[SimpleDataset], total_n: int) -> SimpleDataset:
    """
    Interleave examples from multiple datasets to produce a balanced mixed dataset.
    """
    if not datasets:
        return SimpleDataset([])
    idxs = [0] * len(datasets)
    data = []
    while len(data) < total_n:
        for k, ds in enumerate(datasets):
            if len(data) >= total_n:
                break
            if idxs[k] < len(ds):
                data.append(ds[idxs[k]])
                idxs[k] += 1
    return SimpleDataset(data)


def build_reasoning_dataset(task_or_tasks: str,
                            split: str,
                            n: int,
                            seed: int = 1234,
                            sentinel: str = "",
                            system_header: str = DEFAULT_SYSTEM_HEADER):
    """
    Public entry used by train.py (backwards compatible).

    Args
    ----
    task_or_tasks: str
        - Single task name (e.g., "toy_add")
        - OR comma-separated list of tasks for a mixed dataset
          e.g., "basic_arithmetic,gsm_symbolic,chain_sum,simple_equations,propositional_logic"
    split: "train" | "eval"
    n: dataset size
    seed: RNG seed
    sentinel: if non-empty, appended to the gold target (teach hard stop)
    system_header: override the instruction string in the prompt

    Returns
    -------
    SimpleDataset
    """
    task_or_tasks = (task_or_tasks or "").strip()
    if "," not in task_or_tasks:
        return _make_single_task_dataset(task_or_tasks, split, n, seed,
                                         sentinel=sentinel, system_header=system_header)

    # Multi-task mix
    tasks = [t.strip() for t in task_or_tasks.split(",") if t.strip()]
    if not tasks:
        # Fallback if string was e.g. ",,,"
        return _make_single_task_dataset("toy_add", split, n, seed,
                                         sentinel=sentinel, system_header=system_header)

    # Build each sub-dataset with a proportional share (rounded)
    per = max(1, math.floor(n / len(tasks)))
    subs = []
    for i, t in enumerate(tasks):
        # vary seed per sub-dataset so they differ
        subs.append(_make_single_task_dataset(t, split, per, seed + 10 * (i + 1),
                                              sentinel=sentinel, system_header=system_header))

    # If per*len(tasks) < n, top-up with the first dataset
    total = per * len(tasks)
    if total < n and subs:
        extra = _make_single_task_dataset(tasks[0], split, n - total, seed + 777,
                                          sentinel=sentinel, system_header=system_header)
        subs[0] = SimpleDataset(subs[0].data + extra.data)

    return _round_robin_mix(subs, n)
