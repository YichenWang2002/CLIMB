from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.revision.aggregate_revision import _paired
from experiments.revision.build_attribution import build_mixs
from experiments.revision.enrich_eval_metadata import enrich
from experiments.revision.prepare_candidates import build as build_candidates
from experiments.revision.report_revision import build as build_report
from experiments.revision.protocol import (
    model_fingerprint,
    row_id,
    validate_external_split,
    write_json,
)
from experiments.revision.select_adapter import select


def _row(domain: str, tier: str, index: int) -> dict:
    return {
        "instruction": "instruction",
        "input": f"mission {domain} {index}",
        "output": f"<root>{domain}-{index}</root>",
        "meta": {"domain": domain, "tier": tier, "scenario": "relay"},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_model_fingerprint_local_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    first = model_fingerprint(model)
    (model / "config.json").write_text("{\"v\":1}", encoding="utf-8")
    second = model_fingerprint(model)
    assert first["resolved"] is True
    assert first["base_model_hash"] != second["base_model_hash"]
    assert first["tokenizer_hash"] == second["tokenizer_hash"]


def test_external_split_is_domain_and_record_disjoint(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    external = tmp_path / "external.jsonl"
    _write_rows(train, [_row("warehouse", "T1", 0)])
    _write_rows(val, [_row("office", "T2", 1)])
    _write_rows(external, [_row("kitchen", "T3", 2)])
    result = validate_external_split(external, train, val)
    assert result["domain_disjoint"] is True
    assert result["external"]["domains"] == {"kitchen": 1}
    _write_rows(external, [_row("warehouse", "T3", 2)])
    with pytest.raises(ValueError, match="domains"):
        validate_external_split(external, train, val)


def test_mixs_preserves_realized_multiset(tmp_path: Path) -> None:
    train = [_row("warehouse", "T1", i) for i in range(4)]
    source = tmp_path / "source"
    source.mkdir()
    rounds = [[train[0], train[1]], [train[1], train[2]], [train[3], train[0]]]
    for index, rows in enumerate(rounds, 1):
        _write_rows(source / f"round{index}.jsonl", rows)
    out = tmp_path / "mixs"
    report = build_mixs(train, source, out, seed=42)
    assert report["exact_multiset_preserved"] is True
    original = sorted(row_id(row) for rows in rounds for row in rows)
    realized = sorted(row_id(json.loads(line)) for path in out.glob("round*.jsonl")
                      for line in path.read_text().splitlines() if line)
    assert original == realized


def test_write_json_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    write_json(target, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(target, {"a": 2})
    write_json(target, {"a": 2}, overwrite=True)
    assert json.loads(target.read_text())["a"] == 2


def test_adapter_selection_uses_validation_and_rejects_test(tmp_path: Path) -> None:
    val_a = tmp_path / "a_val.json"
    val_b = tmp_path / "b_val.json"
    val_a.write_text(json.dumps({"n": 2, "strict_success_rate": 0.5}))
    val_b.write_text(json.dumps({"n": 2, "strict_success_rate": 0.75}))
    result = select([("a", str(val_a)), ("b", str(val_b))])
    assert result["selected"]["adapter"] == "b"
    test_result = tmp_path / "a_test.json"
    test_result.write_text(json.dumps({"n": 2, "strict_success_rate": 1.0}))
    with pytest.raises(ValueError, match="test result"):
        select([("a", str(test_result))])


def test_paired_aggregation_requires_matching_ids() -> None:
    left = {"details": [{"record_id": "a", "tier": "T2", "success": True}]}
    right = {"details": [{"record_id": "b", "tier": "T2", "success": False}]}
    with pytest.raises(ValueError, match="IDs"):
        _paired(left, right, bootstrap_samples=0, seed=1)


def test_prepare_candidates_aligns_pipeline_ids_and_fixed_k(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    out = tmp_path / "candidates.jsonl"
    rows = [_row("kitchen", "T1", i) for i in range(2)]
    _write_rows(data, rows)
    for path, marker in ((source_a, "a"), (source_b, "b")):
        path.write_text("".join(json.dumps({"index": i, "response": f"<{marker}{i}/>"}) + "\n"
                            for i in range(2)), encoding="utf-8")
    report = build_candidates(data, [source_a, source_b], out)
    assert report["candidate_count"] == 2
    result = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["record_id"] for row in result] == [row_id(row) for row in rows]
    assert all(len(row["candidates"]) == 2 for row in result)


def test_report_revision_emits_paired_subgroups(tmp_path: Path) -> None:
    data = [_row("kitchen", "T1", 0), _row("kitchen", "T3", 1)]
    flat = tmp_path / "flat.json"
    spcl = tmp_path / "spcl.json"
    base_details = [
        {"record_id": row_id(row), "tier": row["meta"]["tier"],
         "scenario": row["meta"]["scenario"], "success": index == 0}
        for index, row in enumerate(data)
    ]
    improved_details = [dict(item, success=True) for item in base_details]
    flat.write_text(json.dumps({"details": base_details}), encoding="utf-8")
    spcl.write_text(json.dumps({"details": improved_details}), encoding="utf-8")
    report = build_report([f"flat={flat}", f"spcl={spcl}"], [], 0, 7)
    assert report["runs"]["spcl"]["rates"]["overall"]["rate"] == 1.0
    paired = report["paired"]["flat,spcl"]["overall"]
    assert paired["delta_b_minus_a"] == 0.5
    assert paired["b_only"] == 1


def test_enrich_eval_metadata_joins_only_by_stable_ids(tmp_path: Path) -> None:
    rows = [_row("kitchen", "T1", 0), _row("kitchen", "T3", 1)]
    data = tmp_path / "data.jsonl"
    _write_rows(data, rows)
    result = tmp_path / "legacy.json"
    details = [{"record_id": row_id(row), "success": index == 0,
                "reason": "success" if index == 0 else "failed"}
               for index, row in enumerate(rows)]
    result.write_text(json.dumps({"details": details, "generations": ["a", "b"]}),
                      encoding="utf-8")
    out = tmp_path / "enriched.json"
    report = enrich(result, data, out)
    assert report["record_ids_verified"] is True
    payload = json.loads(out.read_text())
    assert payload["details"][1]["scenario"] == "relay"
    assert payload["details"][0]["success"] is True
