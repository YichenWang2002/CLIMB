# CLIMB

**CLIMB: Curriculum Learning for Multi-Agent Behavior Tree Generation with Small Language Models** (ICASSP 2027)

[Paper] · [GitHub](https://github.com/YichenWang2002/CLIMB) · [Project page](https://yichenwang2002.github.io/CLIMB/) · [Dataset](data/)

Given a natural-language mission (including a spoken platform map) for a
heterogeneous multi-robot team, CLIMB generates an executable
BehaviorTree.CPP v4 XML that a shared symbolic executor accepts. This
repository releases the benchmark, the training/evaluation code, the MRBTP
symbolic-planner baseline, and the ROS 2 demo pack.

## Repository layout

```
datagen/              benchmark generator: STRIPS domains (strips/), sampler,
                      NL mission renderer, gold BT compiler, and the shared
                      symbolic executor (executor.py)  <-- scoring oracle
curriculum/           SPCL: stage construction + difficulty/utility scoring
training/             LoRA SFT / DPO entry points
eval/                 evaluate.py (greedy pass@1), eval_constrained.py (SCD),
                      egvd.py (pass@k), paired_compare.py (McNemar/bootstrap)
baselines_mrbtp/      MRBTP (AAAI'25) adapter, faithful bridge  (see its README)
baselines_prompted/   hosted-LLM 1-/5-shot evaluation            (see its README)
ros2_demo/            ROS 2 / Gazebo execution demo pack         (see its README)
scripts/              reproduction shell scripts for reported tables
data/                 train/val/test splits + prompted-baseline prompt files
results/              per-task JSON outputs behind the paper tables
docs/                 project page source
```

## Setup

```bash
conda create -n climb python=3.10 && conda activate climb
pip install -r requirements.txt
# base models under models/: Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B
```

## Dataset

`data/test.jsonl` — 600 rows; the paper's suite is the 480 rows with
`meta.tier ∈ {T2, T3}` (two/three agents; library + greenhouse domains,
unseen in training). Training: `data/train_aug10.jsonl`; supervision
ablations: `train_llmteacher.jsonl`, `train_planner_matched.jsonl`.
Each row:

```json
{"instruction": "...system prompt...",
 "input":       "natural-language mission incl. platform map",
 "output":      "gold BTCPP v4 XML",
 "meta":        {"domain","tier","scenario","robots","items","init_dynamic",
                 "goal","faults","connected","can_reach","charge_stations",
                 "plan","plan_len","n_agents"}}
```

`meta` is the formal task representation consumed by the MRBTP baseline;
neither the reference plan nor the target tree is ever exposed to a baseline
(verified by `tests/test_mrbtp_faithful_bridge.py`). Regenerate with
`python -m datagen.build_dataset`. `data/` is ~130 MB — use git-lfs or
GitHub Releases when publishing.

## Reproducing the paper

```bash
# CLIMB (Table: numerical simulation) — train then evaluate
bash scripts/run_spcl_v2_multiseed.sh
PYTHONHASHSEED=0 python -m eval.eval_constrained \
    --data data/test.jsonl --adapter outputs/checkpoints/<run> \
    --out results/spcl_v2_constrained_test.json

# MRBTP baseline (faithful protocol; deterministic = single process + seed 0)
git clone https://github.com/DIDS-EI/MRBTP && pip install -e MRBTP
export MRBTP_ROOT=/absolute/path/to/MRBTP
PYTHONHASHSEED=0 python baselines_mrbtp/run_mrbtp.py \
    --workers 1 --timeout 20.0 --out results/mrbtp_faithful_480.json

# paired significance tests between any two result files
python -m eval.paired_compare results/A.json results/B.json
```

MRBTP protocol: official MABTP search (optional LLM subtree plugin disabled),
oracle nominal formalization from `meta` (no faults, no reference plan), and
the generated reactive per-robot trees are executed natively by the shared
executor. Environment and file hashes: `results/mrbtp_official_manifest.json`.

## License & citation

MIT (see `LICENSE`).

```bibtex
@inproceedings{climb2027,
  title     = {CLIMB: Curriculum Learning for Multi-Agent Behavior Tree
               Generation with Small Language Models},
  author    = {Wang, Yichen and Cai, Zhongxuan and Liu, Yinuo and Wang, Yuhao
               and Jiang, Tianjian and Peng, Yuanxi},
  booktitle = {Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```
