# Prompted-LLM baselines (1-shot / 5-shot)

Runs any OpenAI-compatible chat model (cloud API, vLLM, Ollama, ...) on the
**same** held-out test suite as the fine-tuned systems, and scores the
responses locally with the **same frozen symbolic executor** used for corpus
construction and all paper evaluations. Only execution success counts; the
gold XML never leaves this repository (it is not in this folder at all).

## Fair-comparison protocol

- Test set fixed to `data/test_prompts.jsonl` (aligned by index with the
  repository's `data/test.jsonl`).
- 0-shot: no examples; 1-shot: the first of the fixed demo list; 5-shot: the
  same fixed 5 demos. Demos come from the training domains only — no
  retrieval from the test split.
- `temperature=0`; identical model, decoding parameters, and task order
  across shot settings.
- The model sees only `instruction` + `input`. The executor metadata in
  `data/test_eval.jsonl` is never sent to the model.
- Demo selection rule and file hashes are recorded in `data/manifest.json`.

## Usage

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...                 # or EMPTY for a local server
export OPENAI_BASE_URL=https://api.openai.com/v1   # or http://127.0.0.1:8000/v1

python run_api.py --model "$MODEL" --shots 1 --out outputs/model_1shot.jsonl
python run_api.py --model "$MODEL" --shots 5 --out outputs/model_5shot.jsonl

python evaluate_outputs.py --responses outputs/model_5shot.jsonl \
    --out outputs/model_5shot_scored.jsonl                 # executor verdicts
python summarize.py outputs/model_5shot_scored.jsonl       # success-rate table
```
