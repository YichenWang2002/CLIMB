#!/usr/bin/env python3
"""Text-overlap scoring (ROUGE-L / corpus BLEU) of generated BT XML vs gold,
on the 480-task coordination suite (T2∪T3 of the 600-task test split) --
the metric family used by BTGenBot/BTGenBot-2, reported for Table 2 systems.

Protocol:
  * gold   = pipeline/data/test.jsonl `output` field, keyed by the
             build_dataset task_key (sha256), recomputed from meta;
  * systems: LLMbase 5-shot dumps (record_id-keyed) and the pipeline eval dump
             for Llama-3.2-1B + SPCL + SCD (generations aligned to test.jsonl
             order; tier sequence asserted before scoring);
  * normalization: parse XML and re-serialize whitespace-free (ET.canonicalize
             fallback: strip gaps from the extracted <root> block), so scores
             measure structure, not indentation;
  * BLEU   = sacrebleu corpus BLEU over pre-tokenized canonical XML
             (tokenize='none');
  * ROUGE-L= rouge_score ROUGE-1/L F1, mean over tasks.

Usage:  python scripts/score_text_overlap.py
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path("..")
TEST = ROOT / "pipeline/data/test.jsonl"
SPCL_SCD = ROOT / "pipeline/results/outputs/spcl_v2_constrained_test.json"
LLMBASE = ROOT / "LLMbase/outputs"
RES = ROOT / "pipeline/results/outputs"
OUT = ROOT / "pipeline/results/outputs/text_overlap_scores.json"

DS = ("./outputs/revision/"
      "deepseek_r1_qwen15b_matched_v2_b16/seed42/eval/")
SYSTEMS_EXTRA = {
    "DS15B Flat SFT": ("pipe_rid", DS + "flat_test.json"),
    "DS15B SPCL": ("pipe_rid", DS + "spcl_test.json"),
    "DS15B Flat+SCD": ("pipe_rid", DS + "flat_scd_test.json"),
    "DS15B SPCL+SCD": ("pipe_rid", DS + "spcl_scd_test.json"),
}
SYSTEMS = {
    "GPT-5.6-Luna (1-shot)": ("api", LLMBASE / "gpt-5.6-luna/model_1shot.jsonl"),
    "Kimi-K2.5 (1-shot)": ("api", LLMBASE / "kimi-k2.5/kimi-k2.5_1shot_dashscope.jsonl"),
    "DeepSeek-V4-Flash (1-shot)": ("api", LLMBASE / "deepseek/deepseek-v4-flash_1shot_deepseek.jsonl"),
    "Qwen3.5-Plus (1-shot)": ("api", LLMBASE / "qwen3.5-plus/qwen3.5-plus_1shot.jsonl"),
    "GPT-5.6-Luna (5-shot)": ("api", LLMBASE / "gpt-5.6-luna/model_5shot.jsonl"),
    "Kimi-K2.5 (5-shot)": ("api", LLMBASE / "kimi-k2.5/kimi-k2.5_5shot_dashscope.jsonl"),
    "DeepSeek-V4-Flash (5-shot)": ("api", LLMBASE / "deepseek/model_5shot_merged.jsonl"),
    "Qwen3.5-Plus (5-shot)": ("api", LLMBASE / "qwen3.5-plus/qwen3.5-plus_5shot.jsonl"),
    "Llama-1B Flat": ("pipe", RES / "spcl_cmp_flat_test.json"),
    "Llama-1B SPCL": ("pipe", RES / "spcl_cmp_method_v2_test.json"),
    "Llama-1B SPCL+SCD": ("pipe", SPCL_SCD),
    "Llama-1B Flat+SCD": ("pipe", RES / "flat_scd_topology_test.json"),
}

TOKEN_RE = re.compile(r"[^<>\s=]+|=")


def task_key_pipe(row: dict) -> str:
    """pipeline record_id: sha256 over {instruction, input, output}."""
    import hashlib
    payload = json.dumps(
        {k: row[k] for k in ("instruction", "input", "output")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def task_key(row: dict) -> str:
    """LLMbase stable_record_id: sha256 over {instruction, input}."""
    import hashlib
    identity = json.dumps(
        {"instruction": row["instruction"], "input": row["input"]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def extract_xml(text: str) -> str:
    m = re.search(r"<root[^>]*>.*?</root>", text or "", re.DOTALL)
    return m.group(0) if m else (text or "").strip()


def canon(xml: str) -> str:
    """Parsed re-serialization: whitespace-free canonical form."""
    try:
        return ET.canonicalize(xml, strip_text=True)
    except Exception:
        return re.sub(r">\s*<", "><", re.sub(r"\s+", " ", xml)).strip()


def canon_lines(xml: str) -> str:
    """Deterministic re-serialization with one XML element per line, so that
    rouge_lsum's sentence split (newline) unions per-element LCS matches --
    the structured analogue of BTGenBot-2's pretty-printed text protocol."""
    try:
        root = ET.fromstring(extract_xml(xml))
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
    except Exception:
        return canon(xml)


def toks(xml: str) -> str:
    return " ".join(TOKEN_RE.findall(xml))


def main():
    gold_by_id, tier_by_id, gold_order = {}, {}, []
    for line in TEST.open():
        r = json.loads(line)
        rid = task_key(r)
        gold_by_id[rid] = r["output"]
        tier_by_id[rid] = r["meta"]["tier"]
        gold_order.append((rid, r["meta"]["tier"], r["output"]))

    coord_ids = {rid for rid, t, _ in gold_order if t in ("T2", "T3")}
    for line in TEST.open():            # pipe-style keys (incl. gold output)
        r = json.loads(line)
        pass  # gold_order rows already carry (rid, tier, output)
    coord_ids_pipe = {task_key_pipe(json.loads(l)) for l in TEST.open()
                      if json.loads(l)["meta"]["tier"] in ("T2", "T3")}
    gold_by_id_pipe = {}
    tier_by_id_pipe = {}
    for l in TEST.open():
        r = json.loads(l)
        gold_by_id_pipe[task_key_pipe(r)] = r["output"]
        tier_by_id_pipe[task_key_pipe(r)] = r["meta"]["tier"]
    print(f"coordination suite: {len(coord_ids)} tasks")

    rs = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
    import os as _os
    only = _os.environ.get("ONLY")
    ALL = {**SYSTEMS, **SYSTEMS_EXTRA}
    results = json.loads(OUT.read_text()) if (only and OUT.exists()) else {}
    _todo = ({k: v for k, v in ALL.items() if k in only.split(",")}
             if only else ALL)
    ALL = {**SYSTEMS, **SYSTEMS_EXTRA}
    for name, (kind, path) in (ALL if not only else
                               {k: v for k, v in ALL.items() if k in only.split(",")}).items():
        # ---- collect (rid, generation) pairs -----------------------------
        pairs = []
        if kind == "pipe_rid":
            d = json.load(open(path))
            gens, det = d["generations"], d["details"]
            assert len(gens) == len(det)
            for g, x in zip(gens, det):
                rid = x["record_id"]
                if rid not in coord_ids_pipe:
                    continue
                pairs.append((rid, extract_xml(g)))
            gold_by_id, tier_backup = gold_by_id_pipe, None
        elif kind == "api":
            for line in path.open():
                r = json.loads(line)
                rid = r["record_id"]
                if rid not in coord_ids:
                    continue
                pairs.append((rid, extract_xml(r["response"])))
        else:
            d = json.load(open(path))
            gens = d["generations"]
            assert len(gens) == len(gold_order) == 600
            for i, ((rid, tier, _), g) in enumerate(zip(gold_order, gens)):
                assert tier == d["details"][i]["tier"], \
                    "eval detail order does not match test.jsonl"
                if tier in ("T2", "T3"):
                    pairs.append((rid, extract_xml(g)))

        # ---- score --------------------------------------------------------
        sys_tok, ref_tok = [], []
        acc = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
        gmap = gold_by_id_pipe if kind == "pipe_rid" else gold_by_id
        for rid, gen in pairs:
            ref_l, sys_l = canon_lines(gmap[rid]), canon_lines(gen)
            sys_tok.append(toks(canon(gen)))
            ref_tok.append(toks(canon(gmap[rid])))
            s = rs.score(ref_l, sys_l)
            for k in acc:
                acc[k].append(s[k].fmeasure)

        bleu = sacrebleu.corpus_bleu(sys_tok, [ref_tok], tokenize="none")
        n_missing = len(coord_ids) - len(pairs)
        results.setdefault(name, {})
        results[name] = {
            "n_scored": len(pairs), "n_missing_from_dump": n_missing,
            "rouge1_f": sum(acc["rouge1"]) / len(pairs) * 100,
            "rouge2_f": sum(acc["rouge2"]) / len(pairs) * 100,
            "rougeL_f": sum(acc["rougeL"]) / len(pairs) * 100,
            "rougeLsum_f": sum(acc["rougeLsum"]) / len(pairs) * 100,
            "bleu": bleu.score,
        }
        r = results[name]
        print(f"{name:32s} n={len(pairs):3d} (missing {n_missing})  "
              f"R1 {r['rouge1_f']:.2f}  R2 {r['rouge2_f']:.2f}  "
              f"RL {r['rougeL_f']:.2f}  RLsum {r['rougeLsum_f']:.2f}  "
              f"BLEU {r['bleu']:.2f}")

    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
