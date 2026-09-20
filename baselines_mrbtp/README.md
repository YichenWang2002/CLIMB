# MRBTP-Adapted symbolic-planner baseline

Symbolic reference from the paper (Table 2, SR 42.1). MRBTP
([cai2025mrbtp]) keeps a symbolic planner as the core generator and uses an
LLM only for action pre-planning. As MRBTP does **not** consume natural
language, this adapter gives it the **oracle formalization** of each test
instance — initial predicates, goals, platform topology, reachability, and
action schemas — and executes the returned per-robot behavior trees in the
**same frozen symbolic executor** used by every other method, without any
reference plan or target tree.

## Setup

MRBTP is third-party software and is not bundled here. Clone it and point the
adapter at it:

```bash
git clone <MRBTP repository url> MRBTP-main
export MRBTP_ROOT=/absolute/path/to/MRBTP-main
pip install -r <MRBTP-main>/requirements.txt   # mabtpg and its deps
cd /path/to/CLIMB-code
```

## Run

```bash
python baselines_mrbtp/run_mrbtp.py                 # 480 tasks, serial, deterministic
python baselines_mrbtp/run_mrbtp.py --workers 8     # parallel
```

Results are written to `outputs/results/mrbtp_test.json` with per-task
executor verdicts and per-scenario breakdowns, directly comparable to the
result JSONs produced by `eval/evaluate.py` for the trained systems.
