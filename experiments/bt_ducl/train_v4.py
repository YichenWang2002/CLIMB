"""Fair v4 semantic-loss trainer for flat SFT and ability-stage DUCL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import Sampler
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, TrainerCallback)
from trl import SFTConfig, SFTTrainer

from .common import load_jsonl, record_id, set_seed
from .curriculum_v4 import AbilityCurriculumSampler, order_hash
from .semantic import semantic_dataset_row

BASE_DEFAULT = "models/llama32-1b"


class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_steps: set[int]):
        self.checkpoint_steps = set(checkpoint_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.checkpoint_steps:
            control.should_save = True
            control.should_evaluate = True
        return control


class MatchedV4Trainer(SFTTrainer):
    def __init__(self, *args, sampler_rows: list[dict], sampler_mode: str,
                 sampler_seed: int, effective_batch: int, epochs: int, **kwargs):
        self.sampler_rows = sampler_rows
        self.sampler_mode = sampler_mode
        self.sampler_seed = sampler_seed
        self.effective_batch = effective_batch
        self.sampler_epochs = epochs
        self._epoch_sampler: AbilityCurriculumSampler | None = None
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        if self._epoch_sampler is None:
            self._epoch_sampler = AbilityCurriculumSampler(
                self.sampler_rows, self.sampler_mode, self.sampler_seed,
                self.effective_batch, self.sampler_epochs,
            )
        if len(dataset) != len(self._epoch_sampler):
            raise ValueError("sampler rows and processed dataset length differ")
        return self._epoch_sampler


def _load_model(base: str, lora_r: int, lora_alpha: int):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    peft = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    return model, tokenizer, peft


def _processed_dataset(rows: list[dict], tokenizer, max_len: int,
                       loss_scope: str) -> Dataset:
    return Dataset.from_list(
        [semantic_dataset_row(row, tokenizer, max_len, scope=loss_scope) for row in rows])


def _checkpoint_steps(updates_per_epoch: int, epochs: int) -> list[int]:
    values = set()
    for epoch in range(1, epochs + 1):
        values.add(epoch * updates_per_epoch)
        # Also save mid-epoch in every epoch, including the last one, so the
        # constant-LR final-epoch trajectory is covered (run_v4_single_seed.sh
        # evaluates all of these steps).
        values.add(epoch * updates_per_epoch - updates_per_epoch // 2)
    return sorted(x for x in values if x > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("flat", "ducl"), required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--scores", default=None,
                    help="original-order v4 scores; required for ducl")
    ap.add_argument("--val", required=True)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss-scope", choices=("completion", "semantic"),
                    default="completion",
                    help="labels scope; 'completion' supervises every completion "
                         "token (default, known-good). 'semantic' reproduces the "
                         "failed v4 mask and must not be used for headline runs.")
    args = ap.parse_args()
    if args.mode == "ducl" and not args.scores:
        raise ValueError("DUCL requires --scores in original row order")
    if args.loss_scope == "semantic":
        print("WARNING: loss-scope=semantic masks all XML structural tokens. "
              "The v4 run with this loss scored 0/600 strict on both arms. "
              "Use completion for any real run.", flush=True)
    if args.batch * args.accum <= 0:
        raise ValueError("invalid effective batch")
    set_seed(args.seed)
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    sampler_rows = train_rows
    if args.mode == "ducl":
        sampler_rows = load_jsonl(args.scores)
        if len(sampler_rows) != len(train_rows):
            raise ValueError("scores and train lengths differ")
        train_ids = [record_id(row) for row in train_rows]
        #
        score_ids = [row.get("v4_record_id") for row in sampler_rows]
        if any(a != b for a, b in zip(train_ids, score_ids)):
            raise ValueError("scores are not in original train row order")
    if args.limit:
        if args.limit % (args.batch * args.accum):
            raise ValueError("--limit must be divisible by effective batch")
        train_rows = train_rows[:args.limit]
        sampler_rows = sampler_rows[:args.limit]
        val_rows = val_rows[:max(1, args.limit // 10)]

    model, tokenizer, peft = _load_model(args.base, args.lora_r, args.lora_alpha)
    train_ds = _processed_dataset(train_rows, tokenizer, args.max_len, args.loss_scope)
    val_ds = _processed_dataset(val_rows, tokenizer, args.max_len, args.loss_scope)
    updates_per_epoch = math.ceil(len(train_rows) / (args.batch * args.accum))
    total_updates = updates_per_epoch * args.epochs
    checkpoints = _checkpoint_steps(updates_per_epoch, args.epochs)
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1, optim="paged_adamw_32bit", bf16=True,
        logging_steps=25, eval_strategy="no", save_strategy="no",
        max_length=args.max_len, completion_only_loss=False,
        loss_type="nll", seed=args.seed, data_seed=args.seed,
        report_to=[], remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                      label_pad_token_id=-100, return_tensors="pt")
    trainer = MatchedV4Trainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=peft, processing_class=tokenizer, data_collator=collator,
        sampler_rows=sampler_rows, sampler_mode=args.mode, sampler_seed=args.seed,
        effective_batch=args.batch * args.accum, epochs=args.epochs,
        callbacks=[CheckpointCallback(set(checkpoints))],
    )
    train_result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    if trainer._epoch_sampler is None or trainer._epoch_sampler.epoch != args.epochs:
        raise RuntimeError("sampler did not produce exactly one order per epoch")
    # Hard protocol gate: every epoch must be a strict permutation of the
    # training set. Reusing or omitting rows would change per-sample gradient
    # exposure and confound the curriculum comparison against flat SFT.
    for report in trainer._epoch_sampler.epoch_reports:
        if report.get("duplicate_draws") or report["unique_rows_exposed"] != len(train_rows) \
                or report.get("exposure_count_min") != 1 or report.get("exposure_count_max") != 1:
            raise RuntimeError("sampler violated the equal-exposure permutation protocol: "
                               + json.dumps(report))
    metadata = {
        "protocol": "bt_ducl_v4", "mode": args.mode, "seed": args.seed,
        "n_train": len(train_rows), "n_val": len(val_rows), "epochs": args.epochs,
        "per_device_batch": args.batch, "gradient_accumulation": args.accum,
        "effective_batch": args.batch * args.accum,
        "updates_per_epoch": updates_per_epoch, "total_optimizer_updates": total_updates,
        "learning_rate": args.lr, "lr_scheduler": "constant_with_warmup",
        "max_len": args.max_len, "loss_scope": args.loss_scope,
        "loss_mask": ("all_completion_tokens_including_structure"
                      if args.loss_scope == "completion"
                      else "semantic_only_FAILED_v4_do_not_use"),
        "checkpoint_steps": checkpoints, "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "sampler_order_hashes": [order_hash(x) for x in trainer._epoch_sampler.orders],
        "sampler_epoch_reports": trainer._epoch_sampler.epoch_reports,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
