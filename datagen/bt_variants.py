"""Executor-verified BT-variant augmentation.

For each dataset row, enumerate semantics-preserving transformations of the
gold behavior tree, keep only variants that pass the SAME strict check used
at evaluation (parse + schema + symbolic execution), and write augmented
rows sharing the original instruction/input with a different verified
output tree.

Transformations (all justified by the executor's tick semantics, then
verified per-instance anyway):

1. Permute SubTree children of a Parallel node (parallel branches are
   independent; only safe when every child is a SubTree).
2. Permute <BehaviorTree> definition order (lookup is by ID).
3. Permute runs of consecutive pure-condition leaves (IsAtLocation /
   IsItemAt / IsCarrying have no side effects).
4. Re-nest Sequences (flatten a nested Sequence into its parent, or group
   adjacent children; Sequence memory is per-node and leaf order is kept).
5. Fallback nodes and Parallel thresholds are never touched (order matters).

The serializer reproduces the exact gold surface format so the only
difference between gold and variants is tree structure.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

COND_TAGS = ("IsAtLocation", "IsItemAt", "IsCarrying")


# ------------------------------------------------------------ serializer --
def serialize_element(el: ET.Element, level: int = 0) -> str:
    """Gold-style XML: 4-space indent, self-closing leaves without space."""
    indent = "    " * level
    attrs = "".join(f' {k}="{v}"' for k, v in el.attrib.items())
    if len(el) == 0:
        return f"{indent}<{el.tag}{attrs}/>"
    inner = "\n".join(serialize_element(child, level + 1) for child in el)
    return f"{indent}<{el.tag}{attrs}>\n{inner}\n{indent}</{el.tag}>"


def serialize_root(root: ET.Element) -> str:
    return '<?xml version="1.0" ?>\n' + serialize_element(root) + "\n"


# --------------------------------------------------------- transformations
def _permute_parallel_subtrees(root: ET.Element, rng: random.Random) -> bool:
    parallels = [el for el in root.iter("Parallel")
                 if len(el) > 1 and all(c.tag == "SubTree" for c in el)]
    if not parallels:
        return False
    node = rng.choice(parallels)
    order = list(range(len(node)))
    rng.shuffle(order)
    if order == sorted(order):
        order.reverse()
    children = list(node)
    for child in children:
        node.remove(child)
    for index in order:
        node.append(children[index])
    return True


def _permute_tree_definitions(root: ET.Element, rng: random.Random) -> bool:
    trees = [el for el in root.findall("BehaviorTree")]
    if len(trees) < 2:
        return False
    order = list(range(len(trees)))
    rng.shuffle(order)
    if order == sorted(order):
        order.reverse()
    for tree in trees:
        root.remove(tree)
    for index in order:
        root.append(trees[index])
    return True


def _permute_condition_runs(root: ET.Element, rng: random.Random) -> bool:
    sequences = list(root.iter("Sequence"))
    rng.shuffle(sequences)
    for seq in sequences:
        children = list(seq)
        runs, start = [], None
        for i, child in enumerate(children + [None]):
            is_cond = child is not None and child.tag in COND_TAGS
            if is_cond and start is None:
                start = i
            elif not is_cond and start is not None:
                if i - start > 1:
                    runs.append((start, i))
                start = None
        if runs:
            s, e = rng.choice(runs)
            group = children[s:e]
            rng.shuffle(group)
            for child in children:
                seq.remove(child)
            for i, child in enumerate(children):
                seq.append(group.pop(0) if s <= i < e else child)
            return True
    return False


def _renest_sequence(root: ET.Element, rng: random.Random) -> bool:
    sequences = [el for el in root.iter("Sequence")]
    flattenable = [s for s in sequences
                   if any(c.tag == "Sequence" for c in s)]
    nestable = [s for s in sequences if len(s) >= 3]
    choices = (["flatten"] if flattenable else []) + (["nest"] if nestable else [])
    if not choices:
        return False
    if rng.choice(choices) == "flatten":
        seq = rng.choice(flattenable)
        index = [i for i, c in enumerate(seq) if c.tag == "Sequence"][0]
        inner = seq[index]
        seq.remove(inner)
        for offset, child in enumerate(list(inner)):
            seq.insert(index + offset, child)
        return True
    seq = rng.choice(nestable)
    children = list(seq)
    start = rng.randrange(0, len(children) - 1)
    wrapper = ET.Element("Sequence")
    for child in children[start:start + 2]:
        seq.remove(child)
        wrapper.append(child)
    seq.insert(start, wrapper)
    return True


TRANSFORMS = (_permute_parallel_subtrees, _permute_tree_definitions,
              _permute_condition_runs, _renest_sequence)


def make_candidates(xml: str, n_candidates: int, seed: int) -> list[str]:
    """Distinct serialized variants of the gold tree (format preserved)."""
    rng = random.Random(seed)
    try:
        base = ET.fromstring(xml)
    except ET.ParseError:
        return []
    seen, out = {serialize_root(base)}, []
    tries = 0
    while len(out) < n_candidates and tries < n_candidates * 6:
        tries += 1
        root = copy.deepcopy(base)
        changed = False
        for transform in rng.sample(TRANSFORMS, rng.randint(1, 3)):
            changed = transform(root, rng) or changed
        if not changed:
            continue
        text = serialize_root(root)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


# ------------------------------------------------------------- validation --
def _verify(row: dict, xml: str) -> bool:
    from experiments.bt_ducl.strict_eval import reconstruct, validate_xml
    clean, error = validate_xml(xml, row["meta"])
    if error:
        return False
    from datagen.executor import execute
    return bool(execute(reconstruct(row["meta"]), clean).get("success"))


def _augment_row(payload: tuple) -> list[dict]:
    row, candidates, max_variants, seed = payload
    kept = []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for rank, xml in enumerate(candidates):
        if len(kept) >= max_variants:
            break
        if _verify(row, xml):
            kept.append({"instruction": row["instruction"],
                         "input": row["input"], "output": xml,
                         "meta": row["meta"], "source": "variant",
                         "parent_record_id": row["parent_record_id"],
                         "variant_id": rank})
    return kept


def augment(rows: list[dict], max_variants: int, n_candidates: int,
            workers: int, seed: int) -> list[dict]:
    from experiments.bt_ducl.common import record_id
    payloads = [({**row, "parent_record_id": record_id(row)},
                 make_candidates(row["output"], n_candidates, seed + i),
                 max_variants, seed + i)
                for i, row in enumerate(rows)]
    out = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for kept in pool.map(_augment_row, payloads):
                out.extend(kept)
    else:
        for payload in payloads:
            out.extend(_augment_row(payload))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-variants", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from experiments.bt_ducl.common import load_jsonl
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    variants = augment(rows, args.max_variants, args.candidates,
                       args.workers, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in variants:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"tasks": len(rows), "variants_written": len(variants),
               "max_variants": args.max_variants,
               "candidates_per_task": args.candidates}
    (out.parent / (out.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
