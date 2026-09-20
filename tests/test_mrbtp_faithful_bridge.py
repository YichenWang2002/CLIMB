from __future__ import annotations

import copy
import json
from pathlib import Path

from baselines_mrbtp.run_mrbtp import run_one
from datagen.executor import execute
from datagen.strips.domains import build_domain


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/test.jsonl"


def _record(source_index: int) -> dict:
    record = json.loads(DATA.read_text().splitlines()[source_index])
    record["_source_index"] = source_index
    return record


def test_executor_supports_faithful_reactive_nodes_and_generic_facts():
    domain = build_domain(
        "library",
        robots=["alpha"],
        items=["novel_stack"],
        extra_static=[
            ("can_reach", "alpha", "front_desk"),
            ("can_reach", "alpha", "reading_hall"),
        ],
        edges=[
            ("connected", "front_desk", "reading_hall"),
            ("connected", "reading_hall", "front_desk"),
        ],
    )
    task = {
        "domain": "library",
        "domain_obj": domain,
        "init_dynamic": [
            ("at", "alpha", "front_desk"),
            ("free", "alpha"),
            ("mobile", "alpha"),
            ("light", "novel_stack"),
        ],
        "goal": [("at", "alpha", "reading_hall")],
        "faults": [],
    }
    xml = """
    <root BTCPP_format="4" main_tree_to_execute="MainTree" mrbtp_bridge="faithful_v1">
      <BehaviorTree ID="MainTree">
        <RetryUntilSuccessful num_attempts="-1">
          <ReactiveSequence>
            <CheckFact predicate="mobile" args="alpha"/>
            <ReactiveFallback>
              <CheckFact predicate="at" args="alpha,reading_hall"/>
              <MoveTo robot="alpha" from="front_desk" to="reading_hall"/>
            </ReactiveFallback>
            <CheckFact predicate="at" args="alpha,reading_hall"/>
          </ReactiveSequence>
        </RetryUntilSuccessful>
      </BehaviorTree>
    </root>
    """
    result = execute(task, xml)
    assert result["success"] is True
    assert result["reason"] == "goal_reached"


def test_faithful_bridge_does_not_read_reference_plan():
    record = _record(1)
    without_plan = copy.deepcopy(record)
    without_plan["meta"].pop("plan")

    original = run_one(record, timeout=20, bridge="faithful")
    rerun = run_one(without_plan, timeout=20, bridge="faithful")

    assert original["success"] is True
    assert rerun["success"] is True
    assert original["xml"] == rerun["xml"]


def test_faults_change_execution_but_not_faithful_generated_tree():
    # Source 5 has a blocked edge on the route selected by the nominal tree.
    faulted = _record(5)
    nominal = copy.deepcopy(faulted)
    nominal["meta"]["faults"] = []

    faulted_result = run_one(faulted, timeout=20, bridge="faithful")
    nominal_result = run_one(nominal, timeout=20, bridge="faithful")

    assert faulted_result["xml"] == nominal_result["xml"]
    assert faulted_result["success"] is False
    assert faulted_result["reason"] == "stalled_no_progress"
    assert nominal_result["success"] is True
    assert "ClearPath" not in faulted_result["xml"]
    assert "Recharge" not in faulted_result["xml"]

