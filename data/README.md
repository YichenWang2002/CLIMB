# CLIMB dataset

Executor-verified pairs of natural-language missions and executable
BehaviorTree.CPP v4 XML for 2–3 robot teams. Every gold tree was produced by
a STRIPS planner, compiled to XML, and **validated by the symbolic executor**
(`datagen/executor.py`) against the task's world model with the injected
faults active — a tree is included only if it ticks to `SUCCESS` and reaches
every mission goal.

Missions come in two coordination families (`meta.scenario`):

- **relay-transport** (`relay`) — an item is relayed across robots via
  explicit handover / synchronization nodes;
- **joint-heavy-transport** (`heavy`) — synchronous joint actions
  (co-carry), optionally with extra goals or services.

Training/validation use four seen domains (warehouse, hospital,
search_rescue, office); the test split uses **only two unseen domains**
(library, greenhouse).

## Splits

| File | Tasks | Domains | Purpose |
|---|---:|---|---|
| `train.jsonl` | 6,000 | 4 seen domains | training set used in all experiments |
| `val.jsonl` | 600 | 4 seen domains | validation; reference distribution for the SPCL utility term |
| `test.jsonl` | **480** | **library, greenhouse (unseen)** | held-out multi-agent benchmark |

Test-suite composition: **320 relay-transport + 160 joint-heavy-transport**
missions; **357/480** contain injected recoverable faults (blocked passages,
battery budgets, fumbled handovers); 240 tasks per domain.

## Record format

One JSON object per line:

```json
{
  "instruction": "fixed system prompt (XML output contract)",
  "input": "natural-language mission",
  "output": "gold BehaviorTree.CPP v4 XML",
  "meta": { "domain": "library", "scenario": "relay",
            "robots": ["alpha", "beta"], "items": ["box_a"],
            "init_dynamic": [["at", "alpha", "dock_a"]],
            "goal": [["item_at", "box_a", "station_b"]],
            "faults": [{"type": "blocked_edge", "edge": ["hall_a", "hall_b"]}],
            "connected": [...], "charge_stations": [...], "can_reach": [...],
            "zones": {...}, "plan": [...], "plan_len": 12, "n_agents": 2 }
}
```

`meta` is exactly what `eval.evaluate.reconstruct_task` needs to rebuild the
STRIPS world model and judge a generated tree. It is evaluation
infrastructure: never send it (or `output`) to a model.

About 10% of the training records carry a renamed primitive
(e.g. `PickUp` → `FetchItem`, recorded in `meta.skill_aliases`): the NL input
is regenerated so the new signature is described in prose, which trains the
meta-skill of grounding unseen skill signatures. No test-domain primitive is
ever used, so nothing leaks into the held-out domains.

## Integrity (sha256)

```
8334db9368c3da6589d6711c8e940eae8ac0e5446652407efdc1ab36308cbf96  train.jsonl
805b64848cf5fb300e685c7eb35ea10407b4874de99ed2b8d823b8c0f071d4ac  val.jsonl
379fba0ce837bef4625e1c0c024f888d6d52a76d476b20246a888ede2e275ed0  test.jsonl
```
