from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        for key in ("instruction", "input", "output", "meta"):
            if key not in row:
                raise ValueError(f"missing {key!r} at {path}:{line_no}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def record_id(row: dict) -> str:
    payload = json.dumps(
        {k: row[k] for k in ("instruction", "input", "output")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_chat_template_compat(tokenizer, messages: list[dict],
                               add_generation_prompt: bool = True) -> str:
    """Use native chat formatting, folding system into user for Gemma 2.

    Gemma 2's shipped template rejects a separate system role. Folding the
    fixed instruction before the task input preserves the shared protocol and
    changes no curriculum, optimizer, or decoding parameter.
    """
    # Some released Gemma 2 base snapshots intentionally ship no chat
    # template. Use one deterministic plain-text serialization for both SFT
    # and evaluation; never silently mix it with an instruction-tuned format.
    if not getattr(tokenizer, "chat_template", None):
        parts = [str(m.get("content", "")) for m in messages]
        text = "\n\n".join(parts).rstrip() + "\n\n"
        # The trailing newline is a SentencePiece token boundary. A trailing
        # space would be re-segmented when completion text is appended, which
        # breaks TRL's completion-only prefix masking on Gemma base models.
        return text + ("assistant:\n" if add_generation_prompt else "")
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as exc:
        if not any(m.get("role") == "system" for m in messages):
            raise
        system = "\n\n".join(str(m.get("content", "")) for m in messages
                                 if m.get("role") == "system").strip()
        folded = []
        for m in messages:
            if m.get("role") == "system":
                continue
            if m.get("role") == "user" and system:
                folded.append({"role": "user", "content": system + "\n\n" + str(m.get("content", ""))})
            else:
                folded.append(m)
        try:
            return tokenizer.apply_chat_template(
                folded, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            raise exc


def prompt_text(row: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
    ]
    return apply_chat_template_compat(tokenizer, messages, add_generation_prompt=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
