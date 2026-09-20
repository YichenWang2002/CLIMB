"""LoRA SFT for mBT, following BTGenBot-2's recipe (4-bit nf4 + LoRA r16/a32),
with two changes: prompt/completion masking (loss on XML only) and staged
curriculum training.

Modes:
  # staged curriculum (ours): train stage1 -> stage2 -> stage3 sequentially
  python -m training.sft_lora --mode staged \
      --stages outputs/curriculum/ours_stage1.jsonl outputs/curriculum/ours_stage2.jsonl outputs/curriculum/ours_stage3.jsonl \
      --val outputs/dataset/val.jsonl --run-name ours

  # flat baseline: single shuffled file
  python -m training.sft_lora --mode flat \
      --stages outputs/curriculum/baseline_random.jsonl \
      --val outputs/dataset/val.jsonl --run-name baseline_random

Each stage saves its adapter + eval metrics; forgetting is measured in the
eval phase by re-scoring earlier-stage held-out sets after every stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from experiments.bt_ducl.common import apply_chat_template_compat

BASE = "models/llama32-1b"
CKPT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints"


class SaveEachEpoch(TrainerCallback):
    """Save the (LoRA) adapter at the end of every epoch into
    out_root/epoch{N}, for learning-curve evaluation. The trainer reference is
    attached after SFTTrainer construction (callback needs the trainer to save)."""
    def __init__(self, out_root):
        self.out_root = Path(out_root)
        self.trainer = None
        self.done = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(round(state.epoch))
        if ep in self.done or ep <= 0:
            return
        self.done.add(ep)
        d = self.out_root / f"epoch{ep}"
        d.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(d))
        print(f"[SaveEachEpoch] saved adapter -> {d}", flush=True)


class OrderedSFTTrainer(SFTTrainer):
    """SFTTrainer variant that preserves the dataset's curriculum order."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return SequentialSampler(dataset)


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _canon(n: str) -> str:
    """Canonical parameter key: strip PEFT wrapper prefixes so that names are
    stable across construction paths (fresh LoRA via peft_config vs
    PeftModel.from_pretrained). Without this, fisher/anchor lookups silently
    miss and the EWC penalty silently degrades to zero."""
    i = n.find("model.layers.")
    return n[i:] if i >= 0 else n


class EWCTrainer(SFTTrainer):
    """SFTTrainer + Elastic Weight Consolidation penalty.

    ewc_state = {"fisher": {canon(name): tensor(cpu), mean-normalised},
                 "anchor":  {canon(name): tensor(cpu), params after previous stage}}
    Penalty: (lambda/2) * sum_i F_i (theta_i - anchor_i)^2 over trainable
    (LoRA) parameters only."""

    def __init__(self, *args, ewc_state=None, ewc_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_state = ewc_state or {}
        self.ewc_lambda = ewc_lambda
        self._ewc_checked = False

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True,
            num_items_in_batch=num_items_in_batch)
        if self.ewc_lambda > 0 and self.ewc_state.get("fisher"):
            device = loss.device
            penalty = torch.zeros((), device=device)
            anchor = self.ewc_state["anchor"]
            matched = 0
            for n, p in model.named_parameters():
                f = self.ewc_state["fisher"].get(_canon(n))
                if f is not None:
                    penalty = penalty + (f.to(device) * (p - anchor[_canon(n)].to(device)) ** 2).sum()
                    matched += 1
            if not self._ewc_checked:
                self._ewc_checked = True
                total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
                print(f"[EWC] penalty active on {matched}/{total} trainable params",
                      flush=True)
                if matched == 0:
                    raise RuntimeError("EWC penalty matched 0 params -- name "
                                       "mismatch; refusing to train a silently "
                                       "naive model")
            loss = loss + 0.5 * self.ewc_lambda * penalty
        return (loss, outputs) if return_outputs else loss


def compute_fisher(model, tokenizer, records, n_batches, batch_size, max_len,
                   seed=0):
    """Diagonal Fisher information of the task loss over trainable (LoRA)
    params, estimated on n_batches of the stage's data. Mean-normalised so
    ewc_lambda is comparable across stages and runs."""
    import random as _rnd
    device = model.device
    fisher = {n: torch.zeros(p.shape) for n, p in model.named_parameters()
              if p.requires_grad}
    if not fisher:
        return fisher
    rng = _rnd.Random(seed)
    take = min(n_batches * batch_size, len(records))
    idxs = rng.sample(range(len(records)), take)
    done = 0
    for i in range(0, len(idxs), batch_size):
        chunk = [records[j] for j in idxs[i:i + batch_size]]
        prompts, fulls = [], []
        for r in chunk:
            msgs = [{"role": "system", "content": r["instruction"]},
                    {"role": "user", "content": r["input"]}]
            p = apply_chat_template_compat(tokenizer, msgs, add_generation_prompt=True)
            prompts.append(p)
            fulls.append(p + r["output"] + tokenizer.eos_token)
        enc = tokenizer(fulls, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        for j, p in enumerate(prompts):
            plen = len(tokenizer(p, truncation=True, max_length=max_len)["input_ids"])
            labels[j, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        model.zero_grad(set_to_none=True)
        model(**enc, labels=labels).loss.backward()
        for n, p_ in model.named_parameters():
            if p_.requires_grad and p_.grad is not None:
                fisher[n] += (p_.grad.detach() ** 2).cpu()
        done += 1
        if done >= n_batches:
            break
    for n in fisher:
        fisher[n] /= max(done, 1)
    flat = torch.cat([v.flatten() for v in fisher.values()])
    m = flat.mean()
    if m > 0:
        for n in fisher:
            fisher[n] /= m
    model.zero_grad(set_to_none=True)
    return fisher


def to_prompt_completion(records: list, tokenizer) -> Dataset:
    rows = []
    for r in records:
        messages = [
            {"role": "system", "content": r["instruction"]},
            {"role": "user", "content": r["input"]},
        ]
        prompt = apply_chat_template_compat(tokenizer, messages, add_generation_prompt=True)
        rows.append({"prompt": prompt, "completion": r["output"] + tokenizer.eos_token})
    return Dataset.from_list(rows)


def train_one_stage(model, tokenizer, train_ds, eval_ds, out_dir,
                    epochs, lr, batch, accum, max_len, seed, peft_config=None,
                    save_epochs=False, ewc_state=None, ewc_lambda=0.0,
                    preserve_order=False, loss_type="chunked_nll"):
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        max_length=max_len,
        completion_only_loss=True,
        # TRL's chunked implementation preserves the NLL while avoiding a
        # full-vocabulary entropy materialization in every micro-batch.
        loss_type=loss_type,
        seed=seed,
        report_to=[],
    )
    callbacks = []
    if save_epochs:
        cb = SaveEachEpoch(out_dir)
        callbacks.append(cb)
    trainer_cls = OrderedSFTTrainer if preserve_order else EWCTrainer
    trainer_kwargs = dict(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if not preserve_order:
        trainer_kwargs.update(ewc_state=ewc_state, ewc_lambda=ewc_lambda)
    trainer = trainer_cls(**trainer_kwargs)
    if save_epochs:
        cb.trainer = trainer
    trainer.train()
    metrics = trainer.evaluate()
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["staged", "flat"], required=True)
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base", default=BASE,
                    help="base causal LM identifier or local snapshot")
    ap.add_argument("--checkpoint-root", default=str(CKPT_ROOT),
                    help="root directory for adapters; isolates revision runs")
    ap.add_argument("--epochs", type=str, default="3",
                    help="single value or comma list per stage, e.g. '3,3,3,1'")
    ap.add_argument("--lr", type=str, default="1e-4",
                    help="single value or comma list per stage, e.g. '1e-4,1e-4,1e-4,5e-5'")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--loss-type", choices=("nll", "chunked_nll"),
                    default="chunked_nll",
                    help="TRL loss implementation; chunked_nll is numerically "
                         "equivalent and has a lower memory peak")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-adapter", default=None,
                    help="continue training from this LoRA adapter instead of a fresh LoRA")
    ap.add_argument("--save-epochs", action="store_true",
                    help="save the adapter at the end of EVERY epoch into "
                         "<run>/stage{i}/epoch{N} (for learning curves)")
    ap.add_argument("--preserve-order", action="store_true",
                    help="use SequentialSampler so the input JSONL order is trained")
    ap.add_argument("--ewc-lambda", type=float, default=0.0,
                    help="EWC penalty strength (0 = disabled). Fisher is "
                         "mean-normalised, so lambda is comparable across runs")
    ap.add_argument("--ewc-fisher-batches", type=int, default=200,
                    help="batches used to estimate the diagonal Fisher after "
                         "each stage")
    args = ap.parse_args()

    n_stages = len(args.stages)

    def per_stage(value_str, cast, name):
        parts = [cast(x) for x in str(value_str).split(",")]
        if len(parts) == 1:
            return parts * n_stages
        if len(parts) != n_stages:
            raise ValueError(f"--{name}: expected 1 or {n_stages} values, got {len(parts)}")
        return parts

    epochs_per_stage = per_stage(args.epochs, int, "epochs")
    lr_per_stage = per_stage(args.lr, float, "lr")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    run_dir = Path(args.checkpoint_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    val_records = load_jsonl(args.val)
    eval_ds = to_prompt_completion(val_records, tokenizer)
    all_metrics = {}

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    prev_adapter = args.init_adapter
    ewc_state = {"fisher": None, "anchor": None}
    for si, stage_file in enumerate(args.stages, 1):
        print(f"=== stage {si}: {stage_file} (continue_from={prev_adapter}) ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
        if prev_adapter is None:
            model = base
            stage_peft = peft_config
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, prev_adapter, is_trainable=True)
            stage_peft = None
        train_records = load_jsonl(stage_file)
        train_ds = to_prompt_completion(train_records, tokenizer)
        stage_dir = run_dir / f"stage{si}"
        metrics = train_one_stage(model, tokenizer, train_ds, eval_ds, stage_dir,
                                  epochs_per_stage[si - 1], lr_per_stage[si - 1],
                                  args.batch, args.accum,
                                  args.max_len, args.seed + si, peft_config=stage_peft,
                                  save_epochs=args.save_epochs,
                                  ewc_state=ewc_state, ewc_lambda=args.ewc_lambda,
                                  preserve_order=args.preserve_order,
                                  loss_type=args.loss_type)
        all_metrics[f"stage{si}"] = {"n_train": len(train_records), **metrics}
        print(f"stage {si} metrics: {metrics}", flush=True)
        if args.ewc_lambda > 0:
            print(f"[EWC] computing Fisher after stage {si} ...", flush=True)
            new_fisher = compute_fisher(model, tokenizer, train_records,
                                        args.ewc_fisher_batches, args.batch,
                                        args.max_len, seed=args.seed + si)
            if ewc_state["fisher"] is None:
                ewc_state["fisher"] = new_fisher
            else:
                for n in ewc_state["fisher"]:
                    ewc_state["fisher"][n] += new_fisher[n]
            ewc_state["anchor"] = {n: p.detach().clone().cpu()
                                   for n, p in model.named_parameters()
                                   if p.requires_grad}
        prev_adapter = str(stage_dir)
        del model, base
        torch.cuda.empty_cache()

    all_metrics["_metadata"] = {
        "base_model": args.base,
        "checkpoint_root": str(args.checkpoint_root),
        "run_name": args.run_name,
    }
    (run_dir / "train_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
