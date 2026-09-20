"""Generate the task-specific SOURCE distribution for DUE scoring.

Runs the BASE Llama-3.2-1B (no LoRA adapter) zero-shot on every training
prompt and stores its raw generations. This is the model's true initial
output distribution on our task -- the principled 'source' for DUCL's
difficulty term, replacing the generic web-text source_dataset.jsonl.

Usage:
  python -m training.gen_source_zero_shot \
      --train outputs/dataset/train_aug10.jsonl \
      --out outputs/source_base_zero_shot.jsonl \
      --batch-size 32 --max-new 1200 --temperature 0.7
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process the first N records")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    records = load_jsonl(args.train)
    if args.limit:
        records = records[:args.limit]
    print(f"generating zero-shot source for {len(records)} records")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
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
            prompts = []
            for r in chunk:
                msgs = [{"role": "system", "content": r["instruction"]},
                        {"role": "user", "content": r["input"]}]
                prompts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2560).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    pad_token_id=tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for j, r in enumerate(chunk):
                text = tok.decode(gen[j][plen:], skip_special_tokens=True)
                f.write(json.dumps({
                    "instruction": r["instruction"], "input": r["input"],
                    "output": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + args.batch_size, len(records))}/{len(records)}",
                  flush=True)
    print(f"wrote source -> {out_path}")


if __name__ == "__main__":
    main()
