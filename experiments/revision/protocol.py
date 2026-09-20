"""Small, dependency-light helpers for revision experiment provenance.

The command-line interface deliberately does not import torch or transformers,
so ``--dry-run`` and split validation work on a CPU-only machine. Model hashes
are content hashes for local snapshots when possible; for a remote identifier
we record an explicit identifier hash and a warning rather than pretending it
is a commit hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from . import PROTOCOL_VERSION


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    source = Path(path)
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {source}:{line_no}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL split: {source}")
    return rows


def row_id(row: dict) -> str:
    """Stable identity used for paired comparisons and split-overlap checks."""
    payload = json.dumps(
        {key: row.get(key) for key in ("instruction", "input", "output")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return _hash_bytes(payload)


def _split_summary(path: str | Path, rows: list[dict]) -> dict:
    domains = Counter(row.get("meta", {}).get("domain", "?") for row in rows)
    tiers = Counter(row.get("meta", {}).get("tier", "?") for row in rows)
    ids = [row_id(row) for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n": len(rows),
        "domains": dict(sorted(domains.items())),
        "tiers": dict(sorted(tiers.items())),
        "duplicate_record_ids": duplicate_ids,
        "record_id_sha256": _hash_bytes("\n".join(ids).encode("ascii")),
    }


def dataset_fingerprint(path: str | Path) -> dict:
    return _split_summary(path, load_jsonl(path))


def validate_aligned_split(path: str | Path, rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else load_jsonl(path)
    summary = _split_summary(path, rows)
    if summary["duplicate_record_ids"]:
        raise ValueError(f"duplicate record IDs in split {path}")
    return summary


def validate_external_split(external_path: str | Path,
                            train_path: str | Path,
                            val_path: str | Path,
                            expected_domain: str = "kitchen") -> dict:
    """Require a genuinely held-out domain and no task identity overlap."""
    external = load_jsonl(external_path)
    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    ext_domains = {row.get("meta", {}).get("domain") for row in external}
    if ext_domains != {expected_domain}:
        raise ValueError(
            f"external split domains={sorted(ext_domains)!r}; expected only {expected_domain!r}")
    train_domains = {row.get("meta", {}).get("domain") for row in train}
    val_domains = {row.get("meta", {}).get("domain") for row in val}
    if expected_domain in train_domains or expected_domain in val_domains:
        raise ValueError("external domain appears in train or validation")
    train_ids = {row_id(row) for row in train}
    val_ids = {row_id(row) for row in val}
    ext_ids = {row_id(row) for row in external}
    if ext_ids & (train_ids | val_ids):
        raise ValueError("external split shares task identities with train/validation")
    return {
        "external": _split_summary(external_path, external),
        "train": _split_summary(train_path, train),
        "val": _split_summary(val_path, val),
        "domain_disjoint": True,
        "record_disjoint": True,
    }


def model_fingerprint(base: str | Path, hash_weights: bool = False) -> dict:
    """Fingerprint a local model snapshot without requiring transformers.

    Weight contents are optional because hashing a multi-gigabyte snapshot is
    expensive. The default still hashes all small configuration/tokenizer
    files and records every weight file's size, mtime, and name. Set
    ``hash_weights=True`` for a content-level audit before a final submission.
    """
    base_text = str(base)
    root = Path(base_text)
    if not root.exists() or not root.is_dir():
        return {
            "identifier": base_text,
            "resolved": False,
            "base_model_hash": _hash_bytes(base_text.encode("utf-8")),
            "base_model_hash_kind": "identifier_sha256",
            "tokenizer_hash": None,
            "warning": "identifier is not a local directory; resolve a pinned snapshot before submission",
        }
    files = sorted(path for path in root.rglob("*") if path.is_file())
    metadata = []
    content_parts = []
    tokenizer_parts = []
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "tokenizer.model", "spiece.model", "vocab.json", "merges.txt"}
    for path in files:
        rel = str(path.relative_to(root))
        stat = path.stat()
        item = {"path": rel, "size": stat.st_size}
        if hash_weights or path.suffix.lower() not in weight_suffixes:
            item["sha256"] = sha256_file(path)
        metadata.append(item)
        if path.name in tokenizer_names:
            tokenizer_parts.append(rel + ":" + item.get("sha256", sha256_file(path)))
        if path.suffix.lower() not in weight_suffixes:
            content_parts.append(rel + ":" + item.get("sha256", sha256_file(path)))
    model_hash = _hash_bytes("\n".join(content_parts + [json.dumps(metadata, sort_keys=True)]).encode())
    tokenizer_hash = _hash_bytes("\n".join(sorted(tokenizer_parts)).encode()) if tokenizer_parts else None
    return {
        "identifier": base_text,
        "resolved": True,
        "base_model_hash": model_hash,
        "base_model_hash_kind": "snapshot_metadata_sha256_with_weights" if hash_weights
        else "snapshot_metadata_sha256_without_weight_contents",
        "tokenizer_hash": tokenizer_hash,
        "files": metadata,
    }


def _parse_json_arg(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON argument: {value!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must be an object")
    return parsed


def build_manifest(*, phase: str, status: str, base: str, seed: int | None,
                   train: str | None, val: str | None, test: str | None,
                   external: str | None = None, optimizer_steps: dict | None = None,
                   decode_config: dict | None = None, extra: dict | None = None,
                   hash_weights: bool = False) -> dict:
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "status": status,
        "base_model": model_fingerprint(base, hash_weights=hash_weights),
        "seed": seed,
        "optimizer_steps": optimizer_steps or {},
        "decode_config": decode_config or {},
    }
    for key, path in (("train", train), ("val", val), ("test", test),
                      ("external", external)):
        if path:
            manifest[f"{key}_fingerprint"] = dataset_fingerprint(path)
    if external and train and val:
        manifest["external_split_validation"] = validate_external_split(
            external, train, val, expected_domain="kitchen")
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing file {target}; pass --overwrite explicitly")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


def _cmd_manifest(args: argparse.Namespace) -> None:
    payload = build_manifest(
        phase=args.phase, status=args.status, base=args.base, seed=args.seed,
        train=args.train, val=args.val, test=args.test, external=args.external,
        optimizer_steps=_parse_json_arg(args.optimizer_steps),
        decode_config=_parse_json_arg(args.decode_config),
        extra=_parse_json_arg(args.extra), hash_weights=args.hash_weights,
    )
    write_json(args.out, payload, overwrite=args.overwrite)
    print(json.dumps({
        "out": str(args.out), "protocol_version": PROTOCOL_VERSION,
        "status": args.status, "base_model_hash": payload["base_model"]["base_model_hash"],
    }, indent=2))


def _cmd_validate_external(args: argparse.Namespace) -> None:
    result = validate_external_split(args.external, args.train, args.val, args.domain)
    if args.out:
        write_json(args.out, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--phase", required=True)
    manifest.add_argument("--status", choices=("planned", "running", "complete", "failed"),
                          default="planned")
    manifest.add_argument("--base", required=True)
    manifest.add_argument("--seed", type=int, default=None)
    manifest.add_argument("--train")
    manifest.add_argument("--val")
    manifest.add_argument("--test")
    manifest.add_argument("--external")
    manifest.add_argument("--optimizer-steps", default="{}")
    manifest.add_argument("--decode-config", default="{}")
    manifest.add_argument("--extra", default="{}")
    manifest.add_argument("--hash-weights", action="store_true")
    manifest.add_argument("--overwrite", action="store_true")
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=_cmd_manifest)

    external = sub.add_parser("validate-external")
    external.add_argument("--external", required=True)
    external.add_argument("--train", required=True)
    external.add_argument("--val", required=True)
    external.add_argument("--domain", default="kitchen")
    external.add_argument("--out")
    external.add_argument("--overwrite", action="store_true")
    external.set_defaults(func=_cmd_validate_external)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

