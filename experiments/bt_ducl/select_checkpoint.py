"""Select a checkpoint using strict held-out validation execution only."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"step[_-](\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = []
    for path in sorted(Path(args.eval_dir).glob("val_step_*.json")):
        match = STEP_RE.search(path.stem)
        if not match:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        records.append({"step": int(match.group(1)),
                        "strict_success_rate": float(value["strict_success_rate"]),
                        "successes": int(round(value["strict_success_rate"] * value["n"])),
                        "n": int(value["n"]), "adapter": value["adapter"],
                        "eval_file": str(path)})
    if not records:
        raise ValueError("no val_step_*.json files found")
    # Prefer the earlier checkpoint on an exact validation tie. This preserves
    # the convergence-speed claim and avoids needless optimizer exposure.
    selected = max(records, key=lambda row: (row["strict_success_rate"], -row["step"]))
    result = {"selection_metric": "held_out_validation_strict_success_rate",
              "tie_breaker": "fewer_optimizer_steps", "selected": selected,
              "curve": sorted(records, key=lambda row: row["step"])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
