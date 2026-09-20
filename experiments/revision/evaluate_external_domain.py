"""Evaluate an external split through the strict symbolic executor.

This wrapper validates the domain split before delegating to the existing
strict evaluator. It keeps the external-domain command self-documenting and
supports ``--dry-run`` without loading a language model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .protocol import validate_external_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--train", default="outputs/dataset/train_aug10.jsonl")
    parser.add_argument("--val", default="outputs/dataset/val.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validation = validate_external_split(args.data, args.train, args.val, "kitchen")
    command = [sys.executable, "-u", "-m", "experiments.bt_ducl.strict_eval",
               "--data", args.data, "--base", args.base, "--out", args.out,
               "--batch-size", str(args.batch_size), "--max-new", str(args.max_new)]
    if args.adapter:
        command.extend(("--adapter", args.adapter))
    if args.limit:
        command.extend(("--limit", str(args.limit)))
    print("validated external split:", validation["external"]["n"], "rows")
    print("command:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

