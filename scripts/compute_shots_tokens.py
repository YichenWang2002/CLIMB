#!/usr/bin/env python3
"""1-shot/5-shot external stats + token accounting for Table 2.

Per external model and shot count, on the 480-task coordination suite:
  * execution success (symbolic executor, same scorer as eval);
  * mean prompt+completion tokens (provider usage fields).
Per fine-tuned stack (Llama-1B, DeepSeek-1.5B; 4 conditions each):
  * mean prompt+completion tokens via the backbone tokenizer
    (prompt = chat template over {instruction, input}; completion =
    tokenizer count of the saved generation).
"""
import json
import sys
from pathlib import Path

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datagen.executor import execute
from eval.evaluate import reconstruct_task

ROOT = Path("..")
TEST = ROOT / "pipeline/data/test.jsonl"
LB = ROOT / "LLMbase/outputs"
OUT = ROOT / "pipeline/results/outputs/external_shots_stats.json"

API = {
    ("GPT-5.6-Luna", 1): LB / "gpt-5.6-luna/model_1shot.jsonl",
    ("GPT-5.6-Luna", 5): LB / "gpt-5.6-luna/model_5shot.jsonl",
    ("Kimi-K2.5", 1): LB / "kimi-k2.5/kimi-k2.5_1shot_dashscope.jsonl",
    ("Kimi-K2.5", 5): LB / "kimi-k2.5/kimi-k2.5_5shot_dashscope.jsonl",
    ("DeepSeek-V4-Flash", 1): LB / "deepseek/deepseek-v4-flash_1shot_deepseek.jsonl",
    ("DeepSeek-V4-Flash", 5): LB / "deepseek/model_5shot_merged.jsonl",
    ("Qwen3.5-Plus", 1): LB / "qwen3.5-plus/qwen3.5-plus_1shot.jsonl",
    ("Qwen3.5-Plus", 5): LB / "qwen3.5-plus/qwen3.5-plus_5shot.jsonl",
}
FIN = {
    ("Llama-1B", "llama32"): [
        ROOT / "pipeline/results/outputs/spcl_cmp_flat_test.json",
        ROOT / "pipeline/results/outputs/spcl_cmp_method_v2_test.json",
        ROOT / "pipeline/results/outputs/spcl_v2_constrained_test.json",
        ROOT / "pipeline/results/outputs/flat_scd_topology_test.json",
    ],
    ("DS-1.5B", "deepseek15"): [
        ROOT / "pipeline/outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42/eval/flat_test.json",
        ROOT / "pipeline/outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42/eval/spcl_test.json",
        ROOT / "pipeline/outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42/eval/flat_scd_test.json",
        ROOT / "pipeline/outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42/eval/spcl_scd_test.json",
    ],
}
BASES = {"llama32": ROOT / "model/llama32-1b",
         "deepseek15": ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"}
CONDS = ["Flat SFT", "SPCL", "Flat SFT + SCD", "SPCL + SCD"]

import re
def rid_of(row):
    import hashlib
    identity = json.dumps({"instruction": row["instruction"], "input": row["input"]},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()

def extract_xml(text):
    m = re.search(r"<root[^>]*>.*?</root>", text or "", re.DOTALL)
    return m.group(0) if m else ""

rows = []
for line in TEST.open():
    r = json.loads(line)
    rows.append((rid_of(r), r["meta"]["tier"], r))
coord = {rid for rid, t, _ in rows if t in ("T2", "T3")}
meta_by_rid = {rid: r for rid, t, r in rows}
print(f"coordination suite: {len(coord)}")

results = {}

# ---- external: success + tokens --------------------------------------
for (name, shots), path in API.items():
    ok = tok = n = 0
    for line in path.open():
        rec = json.loads(line)
        rid = rec["record_id"]
        if rid not in coord:
            continue
        n += 1
        u = rec.get("usage") or {}
        tok += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        task = reconstruct_task(meta_by_rid[rid]["meta"])
        try:
            ok += execute(task, extract_xml(rec["response"]))["success"]
        except Exception:
            pass
    key = f"{name} ({shots}-shot)"
    results[key] = {"success": 100 * ok / n, "tokens": tok / n, "n": n}
    print(f"{key:34s} n={n} success={100*ok/n:.2f} tokens={tok/n:.0f}", flush=True)

# ---- fine-tuned: tokens via tokenizer --------------------------------
from transformers import AutoTokenizer
from experiments.bt_ducl.common import apply_chat_template_compat

for (name, tokkey), dumps in FIN.items():
    tok = AutoTokenizer.from_pretrained(str(BASES[tokkey]))
    for cond, dump in zip(CONDS, dumps):
        d = json.load(open(dump))
        ptok = ctok = n = 0
        for (rid, tier, row), gen in zip(rows, d["generations"]):
            if tier not in ("T2", "T3"):
                continue
            n += 1
            msgs = [{"role": "system", "content": row["instruction"]},
                    {"role": "user", "content": row["input"]}]
            prompt = apply_chat_template_compat(tok, msgs, add_generation_prompt=True)
            ptok += len(tok(prompt, add_special_tokens=False).input_ids)
            ctok += len(tok(gen, add_special_tokens=False).input_ids)
        key = f"{name} {cond}"
        results[key] = {"success": None, "tokens": (ptok + ctok) / n, "n": n}
        print(f"{key:34s} n={n} tokens={(ptok+ctok)/n:.0f} "
              f"(p={ptok/n:.0f} c={ctok/n:.0f})", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)
