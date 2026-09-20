"""Per-sample base-model loss: an independent ground-truth difficulty signal.

Computes the mean token-level cross-entropy of the BASE Llama-3.2-1B (no
adapter) on each training sample's completion (the XML output) given its
prompt. Used to validate that DUE's difficulty term actually correlates with
how hard each sample is for the model (curriculum/validate_stages.py
--base-loss).

Usage:
  python -m training.base_loss --train outputs/curriculum_due/train_due_staged.jsonl \
      --out outputs/curriculum_due/base_loss.jsonl --batch-size 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "models/llama32-1b"


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    records = load_jsonl(args.train)
    print(f"computing base loss for {len(records)} records")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(0, len(records), args.batch_size):
            chunk = records[i:i + args.batch_size]
            prompts, fulls = [], []
            for r in chunk:
                msgs = [{"role": "system", "content": r["instruction"]},
                        {"role": "user", "content": r["input"]}]
                p = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
                prompts.append(p)
                fulls.append(p + r["output"] + tok.eos_token)
            enc_full = tok(fulls, return_tensors="pt", padding=True,
                           truncation=True, max_length=2560).to(model.device)
            plens = [len(tok(p, truncation=True, max_length=2560)["input_ids"])
                     for p in prompts]
            with torch.no_grad():
                logits = model(**enc_full).logits
            for j in range(len(chunk)):
                ids = enc_full["input_ids"][j]
                mask = enc_full["attention_mask"][j].bool()
                plen = plens[j]
                # predict tokens plen..end-1 from logits plen-1..end-2
                tgt = ids[plen:][mask[plen:]]
                lg = logits[j][plen - 1:-1][mask[plen:]]
                if tgt.numel() == 0:
                    loss = float("nan")
                else:
                    loss = F.cross_entropy(lg.float(), tgt).item()
                f.write(json.dumps({"idx": i + j, "loss": loss}) + "\n")
            f.flush()
            print(f"  {min(i + args.batch_size, len(records))}/{len(records)}",
                  flush=True)
    print(f"wrote losses -> {out_path}")


if __name__ == "__main__":
    main()
