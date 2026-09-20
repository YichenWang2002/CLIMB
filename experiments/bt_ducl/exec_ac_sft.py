"""Executor-guided Actor-Curator SFT with stratified tabular OSMD.

The actor is trained with ordinary full-completion SFT.  A tabular curator
chooses which prompts receive each update using strict executor rewards from
the actor's sampled XML.  The candidate proposal is uniform, the curator has
an explicit uniform exploration floor, and a statistical block detector
increases sharing when the observed utility becomes non-stationary.

The tabular arms are prompt-only strata induced before online training.  This
keeps the curator identifiable when the number of online draws is smaller than
the number of rows, without using tier, domain, fault, output XML, validation,
or test information.  The importance-weighted update follows the
OSMD/Actor-Curator construction; practical reward estimates are logged so
their variance and coverage can be audited.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .common import load_jsonl, prompt_text, record_id, set_seed
from .semantic import semantic_dataset_row
from .strict_eval import reconstruct, validate_xml
from datagen.executor import execute


BASE_DEFAULT = "models/llama32-1b"


class TabularOSMD:
    """Exponentiated-gradient curator with fixed-share exploration.

    ``weights`` are the unconstrained Hedge weights.  ``distribution`` is
    always a mixture with the uniform distribution, so every arm remains
    sampleable.  Sampling is with replacement inside a candidate set; this
    makes the per-draw conditional probability explicit and keeps the
    importance-weighted estimator unbiased for the expected selected batch.
    """

    def __init__(self, n_arms: int, exploration_floor: float = 0.05,
                 eta: float = 0.5, fixed_share: float = 0.0,
                 prior: torch.Tensor | list[float] | None = None):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if not 0.0 <= exploration_floor < 1.0:
            raise ValueError("exploration_floor must be in [0, 1)")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        if not 0.0 <= fixed_share < 1.0:
            raise ValueError("fixed_share must be in [0, 1)")
        self.n_arms = n_arms
        self.exploration_floor = exploration_floor
        self.eta = eta
        self.fixed_share = fixed_share
        if prior is None:
            prior_tensor = torch.full((n_arms,), 1.0 / n_arms,
                                      dtype=torch.float64)
        else:
            prior_tensor = torch.as_tensor(prior, dtype=torch.float64).clone()
            if prior_tensor.shape != (n_arms,) or not torch.isfinite(prior_tensor).all():
                raise ValueError("prior must contain one finite mass per arm")
            if (prior_tensor <= 0).any():
                raise ValueError("prior masses must be positive")
            prior_tensor /= prior_tensor.sum()
        self.prior = prior_tensor
        self.weights = prior_tensor.clone()
        self.updates = 0
        self.restarts = 0

    def distribution(self) -> torch.Tensor:
        base = self.weights / self.weights.sum()
        return ((1.0 - self.exploration_floor) * base
                + self.exploration_floor * self.prior)

    def sample_candidate(self, candidate_size: int, rng: random.Random) -> list[int]:
        if not 1 <= candidate_size <= self.n_arms:
            raise ValueError("candidate_size must be in [1, n_arms]")
        return rng.sample(range(self.n_arms), candidate_size)

    def sample_from_candidate(self, candidate: list[int], count: int,
                              rng: random.Random) -> tuple[list[int], list[float]]:
        if not 1 <= count:
            raise ValueError("count must be positive")
        if not candidate or len(set(candidate)) != len(candidate):
            raise ValueError("candidate must be a non-empty set of unique arms")
        if count > 0 and not candidate:
            raise ValueError("empty candidate")
        probs = self.distribution()
        local = probs[torch.tensor(candidate, dtype=torch.long)].tolist()
        total = sum(local)
        if total <= 0.0 or not math.isfinite(total):
            raise RuntimeError("invalid curator probability mass")
        local = [x / total for x in local]
        chosen, selected_probs = [], []
        for _ in range(count):
            u = rng.random()
            cumulative = 0.0
            pos = len(candidate) - 1
            for i, p in enumerate(local):
                cumulative += p
                if u <= cumulative:
                    pos = i
                    break
            chosen.append(candidate[pos])
            selected_probs.append(local[pos])
        return chosen, selected_probs

    def update(self, utilities: dict[int, float]) -> None:
        if not utilities:
            return
        values = torch.zeros_like(self.weights)
        for arm, utility in utilities.items():
            if not 0 <= arm < self.n_arms:
                raise IndexError(f"arm out of range: {arm}")
            if not math.isfinite(float(utility)):
                raise ValueError("utility must be finite")
            values[arm] = float(utility)
        # Do not center only the observed coordinates: doing so would change
        # their update relative to unobserved arms and would no longer be the
        # OSMD exponentiated-gradient step. Clipping is a numerical guard.
        exponent = torch.clamp(self.eta * values, -20.0, 20.0)
        self.weights = self.weights * torch.exp(exponent)
        self.weights = self.weights / self.weights.sum()
        if self.fixed_share:
            self.weights = ((1.0 - self.fixed_share) * self.weights
                            + self.fixed_share * self.prior)
        self.updates += 1

    def restart(self, share: float | None = None) -> None:
        amount = self.fixed_share if share is None else share
        if not 0.0 <= amount <= 1.0:
            raise ValueError("restart share must be in [0, 1]")
        self.weights = (1.0 - amount) * self.weights + amount * self.prior
        self.restarts += 1


def automatic_bandit_hyperparameters(updates: int, draws_per_update: int,
                                     n_arms: int) -> tuple[float, float]:
    """Return the multiple-play Hedge rate and explicit exploration mass."""
    if updates <= 0 or draws_per_update <= 0 or n_arms <= 1:
        raise ValueError("invalid bandit budget")
    eta = math.sqrt(2.0 * draws_per_update * math.log(n_arms)
                    / (updates * n_arms))
    exploration = min(0.5, math.sqrt(
        n_arms * math.log(n_arms) / (updates * draws_per_update)))
    return eta, exploration


def induce_prompt_strata(rows: list[dict], n_strata: int,
                         seed: int) -> tuple[list[int], dict]:
    """Deterministically cluster instruction+input without label/meta access."""
    if not 1 < n_strata <= len(rows):
        raise ValueError("n_strata must be in [2, n_rows]")
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [f"{row.get('instruction', '')}\n{row.get('input', '')}"
             for row in rows]
    vectorizer = TfidfVectorizer(
        max_features=8192, min_df=2, ngram_range=(1, 2),
        sublinear_tf=True, strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts)
    components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 2:
        raise RuntimeError("prompt corpus is too small for automatic strata")
    reduced = TruncatedSVD(n_components=components, random_state=seed).fit_transform(matrix)
    reduced = normalize(reduced)
    clusterer = MiniBatchKMeans(
        n_clusters=n_strata, random_state=seed, n_init=10,
        batch_size=min(1024, len(rows)), reassignment_ratio=0.0,
    )
    labels = clusterer.fit_predict(reduced).tolist()
    sizes = [labels.count(arm) for arm in range(n_strata)]
    repaired_empty = 0
    for empty_arm in [arm for arm, size in enumerate(sizes) if size == 0]:
        source_arm = max(range(n_strata), key=lambda arm: sizes[arm])
        source_rows = [index for index, arm in enumerate(labels)
                       if arm == source_arm]
        if len(source_rows) <= 1:
            raise RuntimeError("cannot repair an empty prompt stratum")
        center = clusterer.cluster_centers_[source_arm]
        move = max(source_rows, key=lambda index: float(
            ((reduced[index] - center) ** 2).sum()))
        labels[move] = empty_arm
        sizes[source_arm] -= 1
        sizes[empty_arm] += 1
        repaired_empty += 1
    assignment_hash = __import__("hashlib").sha256(
        json.dumps(labels, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return labels, {
        "method": "prompt_only_tfidf_svd_minibatch_kmeans",
        "n_strata": n_strata,
        "sizes": sizes,
        "assignment_sha256": assignment_hash,
        "svd_components": components,
        "tfidf_features": int(matrix.shape[1]),
        "empty_strata_repaired": repaired_empty,
        "uses_output": False,
        "uses_meta": False,
        "uses_validation_or_test": False,
    }


def sample_stratified_rows(curator: TabularOSMD, members: list[list[int]],
                           count: int, rng: random.Random
                           ) -> tuple[list[int], list[int], list[float]]:
    """Draw strata from the curator and one row uniformly inside each."""
    if len(members) != curator.n_arms or any(not group for group in members):
        raise ValueError("members must contain one non-empty list per arm")
    probabilities = curator.distribution().tolist()
    selected_rows, selected_arms, arm_probabilities = [], [], []
    for _ in range(count):
        u = rng.random()
        cumulative = 0.0
        arm = curator.n_arms - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if u <= cumulative:
                arm = index
                break
        group = members[arm]
        selected_rows.append(group[rng.randrange(len(group))])
        selected_arms.append(arm)
        arm_probabilities.append(probabilities[arm])
    return selected_rows, selected_arms, arm_probabilities


class BlockVariationDetector:
    """Two-block confidence test for non-stationary utility.

    No data-category threshold is used.  A restart occurs only after two
    completed blocks have enough observations and their means differ by a
    pooled three-sigma bound.  This is a statistical change detector rather
    than a hand-written curriculum phase boundary.
    """

    def __init__(self, block_size: int = 8, z_threshold: float = 3.0):
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self.block_size = block_size
        self.z_threshold = z_threshold
        self.previous: list[float] = []
        self.current: list[float] = []
        self.tests = 0

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _variance(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

    def update(self, value: float) -> dict:
        if not math.isfinite(value):
            raise ValueError("detector value must be finite")
        self.current.append(float(value))
        result = {"tested": False, "restart": False, "z": 0.0}
        if len(self.current) < self.block_size:
            return result
        if self.previous:
            mean_a, mean_b = self._mean(self.previous), self._mean(self.current)
            se = math.sqrt(self._variance(self.previous) / len(self.previous)
                           + self._variance(self.current) / len(self.current) + 1e-12)
            z = abs(mean_b - mean_a) / se if se > 0 else (math.inf if mean_a != mean_b else 0.0)
            result = {"tested": True, "restart": z >= self.z_threshold, "z": z,
                      "previous_mean": mean_a, "current_mean": mean_b}
            self.tests += 1
        self.previous = self.current
        self.current = []
        return result


@dataclass
class RolloutBatch:
    output_ids: torch.Tensor
    attention_mask: torch.Tensor
    continuation_mask: torch.Tensor
    prompt_width: int
    old_logprob: torch.Tensor
    rewards: list[float]
    reasons: list[str]
    texts: list[str]
    timings: dict[str, float]


def _training_text(raw_text: str, clean: str) -> str:
    """Format-preserving verified completion for RFT replacement.

    ``re_extract`` strips the XML declaration and any preamble, while every
    gold completion starts with ``<?xml ...?>``.  Training on stripped bodies
    created a first-token format conflict that destabilized the root element
    (observed drift: declaration dropout, then ``main_tree_to_execute``
    attr dropout, then executor collapse).  Keep the model's own preamble
    and append the validated clean body plus the gold-style trailing newline.
    """
    index = raw_text.find("<root")
    prefix = raw_text[:index] if index >= 0 else ""
    return prefix + clean + "\n"


def _verified_completion(row: dict, text: str) -> tuple[str | None, str]:
    """Return (clean XML, "success") iff the sample passes strict execution.

    The training-time replacement uses the extracted clean XML rather than
    the raw decoded sample, so the actor never learns preamble or trailing
    junk that the validator would have stripped anyway.
    """
    clean, error = validate_xml(text, row["meta"])
    if error:
        return None, error
    result = execute(reconstruct(row["meta"]), clean)
    if result.get("success"):
        return clean, "success"
    return None, str(result.get("reason", "execution_failure"))


def _strict_reward(row: dict, text: str) -> tuple[float, str]:
    clean, reason = _verified_completion(row, text)
    if clean is not None:
        return 1.0, "success"
    return 0.0, reason


def build_train_rows(selected_rows: list[dict], rollout: RolloutBatch,
                     rollouts_per_arm: int, train_mix: str,
                     max_replaced: int, rng: random.Random
                     ) -> tuple[list[dict], dict]:
    """Optimal-difficulty gating and completion-provenance composition.

    Each draw's pass rate p comes from its own K rollouts.  With 0/1
    verifiable rewards the policy-improvement signal of a prompt is
    proportional to p(1-p), so:

    - p == 1 (mastered): skipped entirely in BOTH arms — zero learning
      signal, and skipping saves the compute;
    - 0 < p < 1 (learnable): train_mix="rft" trains on one deduplicated
      executor-verified self-generated completion, train_mix="gold"
      (matched control) trains on gold with rollouts discarded;
    - p == 0 (unsolved): both arms keep gold, the only available signal.

    If every draw is mastered, the batch falls back to all-gold so the
    optimizer step still happens (logged via ``mastered_fallback``).  Both
    arms apply the identical gating rule to their own rollouts, so the only
    remaining difference is completion provenance on learnable draws.
    """
    if train_mix not in ("gold", "rft"):
        raise ValueError(f"unsupported train mix: {train_mix}")
    train_rows = []
    n_replaced = verified_draws = n_mastered = 0
    for draw, row in enumerate(selected_rows):
        start = draw * rollouts_per_arm
        rewards = rollout.rewards[start:start + rollouts_per_arm]
        texts = rollout.texts[start:start + rollouts_per_arm]
        n_success = sum(1 for reward in rewards if reward == 1.0)
        if n_success > 0:
            verified_draws += 1
        if n_success == rollouts_per_arm:
            n_mastered += 1
            continue
        verified = sorted({text for text, reward in zip(texts, rewards)
                           if reward == 1.0})
        if verified and train_mix == "rft" and n_replaced < max_replaced:
            replacement = verified[rng.randrange(len(verified))]
            train_rows.append({**row, "output": replacement})
            n_replaced += 1
        else:
            train_rows.append(row)
    mastered_fallback = not train_rows
    if mastered_fallback:
        train_rows = list(selected_rows)
    stats = {"n_replaced": n_replaced, "verified_draws": verified_draws,
             "n_mastered_skipped": n_mastered, "batch_slots": len(train_rows),
             "mastered_fallback": mastered_fallback}
    return train_rows, stats


def _load_actor(base: str, adapter: str, torch_dtype: torch.dtype,
                load_mode: str = "bf16"):
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if load_mode == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map="auto",
            torch_dtype=torch_dtype, attn_implementation="sdpa",
        )
    elif load_mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto", torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        )
    else:
        raise ValueError(f"unsupported load mode: {load_mode}")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    return model, tokenizer


def _continuation_logprob(model, output_ids: torch.Tensor,
                          attention_mask: torch.Tensor, prompt_width: int,
                          eos_id: int, batch_size: int | None = None,
                          continuation_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Log probability of generated suffix, including its first EOS only."""
    if batch_size is not None and batch_size <= 0:
        raise ValueError("log-prob batch size must be positive")
    batch_size = batch_size or output_ids.shape[0]
    values = []
    with torch.no_grad():
        for start in range(0, output_ids.shape[0], batch_size):
            ids = output_ids[start:start + batch_size]
            mask_input = attention_mask[start:start + batch_size]
            logits = model(input_ids=ids, attention_mask=mask_input,
                           use_cache=False).logits[:, :-1]
            targets = ids[:, 1:]
            # Fused CE returns only the selected next-token log-probability;
            # materializing FP32 log_softmax over the full 128k vocabulary
            # costs several GB for long XML batches.
            token_log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1), reduction="none",
            ).reshape(targets.shape)
            generated = ids[:, prompt_width:]
            selected = token_log_probs[:, prompt_width - 1:]
            if continuation_mask is None:
                eos_seen = generated.eq(eos_id).cumsum(dim=1)
                score_mask = ((eos_seen == 0)
                              | (generated.eq(eos_id) & (eos_seen == 1)))
            else:
                score_mask = continuation_mask[start:start + batch_size].to(
                    device=generated.device, dtype=torch.bool)
            values.append((selected * score_mask).sum(dim=1).detach())
            del logits, targets, token_log_probs, selected
    return torch.cat(values, dim=0)


def _sample_rollouts(model, tokenizer, rows: list[dict], max_new: int,
                     temperature: float, top_p: float,
                     logprob_batch: int | None = None,
                     generation_batch: int | None = None,
                     compute_logprob: bool = True) -> RolloutBatch:
    prompts = [prompt_text(row, tokenizer) for row in rows]
    if generation_batch is not None and generation_batch <= 0:
        raise ValueError("generation batch size must be positive")
    generation_batch = generation_batch or len(rows)
    model.eval()
    generation_started = time.perf_counter()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), generation_batch):
            prompt_chunk = prompts[start:start + generation_batch]
            enc = tokenizer(prompt_chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=2560).to(model.device)
            output_ids = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                # A behavior tree is complete at its closing root tag. This
                # stops malformed/no-EOS samples at the first complete tree,
                # while strict_reward still performs the same parse/schema/
                # executor checks on the returned text.
                stop_strings=["</root>"], tokenizer=tokenizer,
                # Training disables the cache for backward compatibility, but
                # rollout generation is no-grad. Re-enable it here so each XML
                # token does not recompute the complete prefix.
                use_cache=True,
            )
            width = enc["input_ids"].shape[1]
            generated = output_ids[:, width:].detach()
            real_mask = torch.zeros_like(generated, dtype=torch.bool)
            for row_index, row_ids in enumerate(generated):
                eos_positions = (row_ids == tokenizer.eos_token_id).nonzero(
                    as_tuple=False)
                end = int(eos_positions[0].item()) + 1 if len(eos_positions) else generated.shape[1]
                real_mask[row_index, :end] = True
            chunks.append((enc["input_ids"].detach(),
                           enc["attention_mask"].detach(), generated,
                           real_mask, width))
    generation_seconds = time.perf_counter() - generation_started
    prompt_width = max(width for _, _, _, _, width in chunks)
    max_generated = max(generated.shape[1] for _, _, generated, _, _ in chunks)
    padded_outputs, padded_attention, generated_rows, generated_masks = [], [], [], []
    pad_id = (tokenizer.pad_token_id
              if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    for input_ids, input_attention, generated, real_mask, width in chunks:
        left = prompt_width - width
        if left:
            input_ids = torch.cat([torch.full(
                (input_ids.shape[0], left), pad_id, dtype=input_ids.dtype,
                device=input_ids.device), input_ids], dim=1)
            input_attention = torch.cat([torch.zeros(
                (input_attention.shape[0], left), dtype=input_attention.dtype,
                device=input_attention.device), input_attention], dim=1)
        if generated.shape[1] < max_generated:
            padding = max_generated - generated.shape[1]
            generated = torch.cat([generated, torch.full(
                (generated.shape[0], padding),
                tokenizer.eos_token_id, dtype=generated.dtype,
                device=generated.device)], dim=1)
            real_mask = torch.cat([real_mask, torch.zeros(
                (real_mask.shape[0], padding), dtype=torch.bool,
                device=real_mask.device)], dim=1)
        padded_outputs.append(torch.cat([input_ids, generated], dim=1))
        padded_attention.append(torch.cat([
            input_attention, torch.ones_like(generated)], dim=1))
        generated_rows.append(generated)
        generated_masks.append(real_mask)
    output_ids = torch.cat(padded_outputs, dim=0)
    extended_attention = torch.cat(padded_attention, dim=0)
    generated = torch.cat(generated_rows, dim=0)
    continuation_mask = torch.cat(generated_masks, dim=0)
    old_logprob_started = time.perf_counter()
    if compute_logprob:
        old_logprob = _continuation_logprob(
            model, output_ids, extended_attention, prompt_width,
            tokenizer.eos_token_id, batch_size=logprob_batch,
            continuation_mask=continuation_mask,
        ).detach()
    else:
        # Uniform-curator arms never consume policy ratios; skip the forward.
        old_logprob = torch.zeros(len(rows), dtype=torch.float64)
    old_logprob_seconds = time.perf_counter() - old_logprob_started
    executor_started = time.perf_counter()
    rewards, reasons, texts = [], [], []
    for row, ids in zip(rows, generated):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        clean, reason = _verified_completion(row, text)
        rewards.append(1.0 if clean is not None else 0.0)
        reasons.append(reason)
        texts.append(_training_text(text, clean) if clean is not None else text)
    executor_seconds = time.perf_counter() - executor_started
    return RolloutBatch(output_ids=output_ids.detach(),
                        attention_mask=extended_attention.detach(),
                        continuation_mask=continuation_mask.detach(),
                        prompt_width=prompt_width,
                        old_logprob=old_logprob,
                        rewards=rewards, reasons=reasons, texts=texts,
                        timings={"generation_s": generation_seconds,
                                 "old_logprob_s": old_logprob_seconds,
                                 "executor_s": executor_seconds})


def _supervised_token_logprobs(logits: torch.Tensor, labels: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities at supervised (label != -100) positions."""
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    token_logp = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)
    return token_logp, mask


def _reference_adapter_kl(model, batch: dict, logits: torch.Tensor,
                          kl_adapter: str) -> torch.Tensor:
    """k3 KL estimate between the trainable adapter and the frozen reference.

    Computed on the supervised tokens of the training batch only.  k3 is
    non-negative: k3 = exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1.
    """
    token_logp, mask = _supervised_token_logprobs(logits, batch["labels"])
    model.set_adapter(kl_adapter)
    with torch.no_grad():
        ref_logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
    model.set_adapter("default")
    ref_logp, _ = _supervised_token_logprobs(ref_logits, batch["labels"])
    log_ratio = ref_logp - token_logp
    k3 = log_ratio.exp() - log_ratio - 1.0
    return (k3 * mask).sum() / mask.sum().clamp_min(1)


def _sft_update(model, tokenizer, rows: list[dict], max_len: int,
                micro_batch: int, optimizer, scheduler,
                kl_beta: float = 0.0, kl_adapter: str | None = None):
    encoded = [semantic_dataset_row(row, tokenizer, max_len, scope="completion")
               for row in rows]
    completion_tokens = sum(
        1 for example in encoded for label in example["labels"]
        if label != -100)
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                      label_pad_token_id=-100,
                                      return_tensors="pt")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, kls = [], []
    n_chunks = math.ceil(len(encoded) / micro_batch)
    for start in range(0, len(encoded), micro_batch):
        batch = collator(encoded[start:start + micro_batch])
        batch = {key: value.to(model.device) for key, value in batch.items()}
        if kl_beta > 0.0 and kl_adapter is not None:
            out = model(**batch)
            kl = _reference_adapter_kl(model, batch, out.logits, kl_adapter)
            loss = (out.loss + kl_beta * kl) / n_chunks
            kls.append(float(kl.detach().cpu()))
        else:
            loss = model(**batch).loss / n_chunks
        loss.backward()
        losses.append(float(loss.detach().cpu()) * n_chunks)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    model.eval()
    mean_kl = sum(kls) / len(kls) if kls else 0.0
    return sum(losses) / max(1, len(losses)), completion_tokens, mean_kl


def _new_logprob(model, rollout: RolloutBatch, eos_id: int,
                 logprob_batch: int | None = None) -> torch.Tensor:
    return _continuation_logprob(model, rollout.output_ids,
                                 rollout.attention_mask, rollout.prompt_width,
                                 eos_id, batch_size=logprob_batch,
                                 continuation_mask=rollout.continuation_mask).detach()


def _resume_signature(args: argparse.Namespace, n_rows: int, n_strata: int,
                      stratum_info: dict, eta: float,
                      exploration_floor: float) -> dict:
    """Configuration that must remain identical across a resumed arm."""
    keys = (
        "curator", "base", "warm_start", "load_mode", "updates",
        "select_size", "micro_batch", "rollouts_per_arm", "lr",
        "warmup_ratio", "max_len", "max_new", "logprob_batch",
        "generation_batch", "temperature", "top_p", "fixed_share",
        "utility_clip", "restart_share", "restart_block", "restart_z",
        "baseline_momentum", "actor_log_ratio_clip", "seed", "limit",
        "train_mix", "max_replaced", "kl_beta",
    )
    return {
        **{key: getattr(args, key) for key in keys},
        "n_rows": n_rows,
        "n_strata": n_strata,
        "assignment_sha256": stratum_info["assignment_sha256"],
        "effective_eta": eta,
        "effective_exploration_floor": exploration_floor,
    }


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_resume_checkpoint(out_dir: Path) -> Path | None:
    candidates = []
    for pattern in ("checkpoint-*", "recovery-*"):
        for path in out_dir.glob(pattern):
            if (path.is_dir() and (path / "trainer_state.pt").is_file()
                    and (path / "adapter_config.json").is_file()):
                candidates.append(path)
    return max(candidates, key=_checkpoint_step) if candidates else None


def _training_state(args: argparse.Namespace, signature: dict, step: int,
                    loss: float, optimizer, scheduler, curator: TabularOSMD,
                    detector: BlockVariationDetector, baseline: torch.Tensor,
                    row_exposure: Counter, arm_exposure: Counter,
                    round_reports: list[dict], rng: random.Random) -> dict:
    return {
        "version": 1,
        "signature": signature,
        "step": step,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "curator_weights": curator.weights.clone(),
        "curator_updates": curator.updates,
        "curator_restarts": curator.restarts,
        "detector_previous": detector.previous,
        "detector_current": detector.current,
        "detector_tests": detector.tests,
        "baseline": baseline.clone(),
        "row_exposure": dict(row_exposure),
        "arm_exposure": dict(arm_exposure),
        "round_reports": round_reports,
        "local_rng_state": rng.getstate(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else []),
        "recovery_every": args.recovery_every,
    }


def _save_training_checkpoint(
        adapter_dir: Path, model, tokenizer, metadata: dict, state: dict) -> None:
    """Atomically publish a complete adapter and resumable trainer state."""
    temporary = adapter_dir.with_name(adapter_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    # Only the trainable adapter is persisted; the frozen "klref" reference is
    # rebuilt from --warm-start on every launch, so checkpoint layout and the
    # resume/eval loaders stay single-adapter.
    model.save_pretrained(temporary, selected_adapters=["default"])
    tokenizer.save_pretrained(temporary)
    (temporary / "exec_ac_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(state, temporary / "trainer_state.pt")
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    temporary.replace(adapter_dir)


def _restore_training_state(state: dict, signature: dict, optimizer,
                            scheduler, curator: TabularOSMD,
                            detector: BlockVariationDetector,
                            rng: random.Random):
    if state.get("version") != 1:
        raise RuntimeError("unsupported recovery state version")
    if state.get("signature") != signature:
        raise RuntimeError("recovery checkpoint protocol signature mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    curator.weights = state["curator_weights"].to(dtype=torch.float64).clone()
    curator.updates = int(state["curator_updates"])
    curator.restarts = int(state["curator_restarts"])
    detector.previous = list(state["detector_previous"])
    detector.current = list(state["detector_current"])
    detector.tests = int(state["detector_tests"])
    rng.setstate(state["local_rng_state"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return (
        state["baseline"].to(dtype=torch.float64).clone(),
        Counter(state["row_exposure"]),
        Counter(state["arm_exposure"]),
        list(state["round_reports"]),
        int(state["step"]) + 1,
        float(state["loss"]),
    )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    rng = random.Random(args.seed + 9137)
    rows = load_jsonl(args.train)
    if args.limit:
        rows = rows[:args.limit]
    if args.select_size % args.micro_batch:
        raise ValueError("select_size must be divisible by micro_batch")
    n_strata = args.n_strata or max(2, math.ceil(math.sqrt(args.updates)))
    strata, stratum_info = induce_prompt_strata(rows, n_strata, args.seed)
    members = [[] for _ in range(n_strata)]
    for row_index, arm in enumerate(strata):
        members[arm].append(row_index)
    prior = torch.tensor([len(group) / len(rows) for group in members],
                         dtype=torch.float64)
    auto_eta, auto_exploration = automatic_bandit_hyperparameters(
        args.updates, args.select_size, n_strata)
    eta = args.osmd_eta if args.osmd_eta > 0.0 else auto_eta
    exploration_floor = (args.exploration_floor
                         if args.exploration_floor >= 0.0
                         else auto_exploration)
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = _resume_signature(
        args, len(rows), n_strata, stratum_info, eta, exploration_floor)
    resume_dir = _latest_resume_checkpoint(out_dir) if args.resume else None
    actor_adapter = str(resume_dir) if resume_dir else args.warm_start
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")
    model, tokenizer = _load_actor(args.base, actor_adapter, dtype,
                                   args.load_mode)
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("warm-start adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    kl_adapter = None
    if args.kl_beta > 0.0:
        # Frozen reference adapter holding the warm-start weights on the same
        # quantized base.  The KL anchor keeps reward-tilted self-training
        # from drifting into entropy collapse; the reference never moves,
        # even when the trainable adapter resumes from a recovery checkpoint.
        kl_adapter = "klref"
        model.load_adapter(args.warm_start, adapter_name=kl_adapter,
                           is_trainable=False)
        for param_name, param in model.named_parameters():
            if f".{kl_adapter}." in param_name:
                param.requires_grad = False
        model.set_adapter("default")
    warmup_updates = max(1, int(args.updates * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_updates))
    curator = TabularOSMD(n_strata, exploration_floor, eta,
                          args.fixed_share, prior=prior)
    detector = BlockVariationDetector(args.restart_block, args.restart_z)
    baseline = torch.full((len(rows),), 0.5, dtype=torch.float64)
    row_exposure = Counter()
    arm_exposure = Counter()
    round_reports = []
    start_step = 1
    last_loss = math.nan
    if resume_dir is not None:
        state = torch.load(resume_dir / "trainer_state.pt", map_location="cpu",
                           weights_only=False)
        (baseline, row_exposure, arm_exposure, round_reports,
         start_step, last_loss) = _restore_training_state(
            state, signature, optimizer, scheduler, curator, detector, rng)
        print(json.dumps({
            "event": "resumed",
            "adapter": str(resume_dir),
            "completed_step": start_step - 1,
            "next_step": start_step,
        }), flush=True)
    for step in range(start_step, args.updates + 1):
        selected, selected_arms, arm_probs = sample_stratified_rows(
            curator, members, args.select_size, rng)
        selected_rows = [rows[i] for i in selected]
        rollout_rows = [row for row in selected_rows
                        for _ in range(args.rollouts_per_arm)]
        need_ratios = args.curator == "osmd"
        rollout = _sample_rollouts(model, tokenizer, rollout_rows, args.max_new,
                                   args.temperature, args.top_p,
                                   logprob_batch=args.logprob_batch,
                                   generation_batch=args.generation_batch,
                                   compute_logprob=need_ratios)
        # A per-arm EMA is used only as a control-variate baseline.  The raw
        # reward and utility are retained in metadata for variance auditing.
        utilities: dict[int, float] = {}
        rollout_advantages = []
        draw_mean_rewards = []
        importance_denominators = []
        for draw, (row_index, arm, arm_probability) in enumerate(zip(
                selected, selected_arms, arm_probs)):
            start = draw * args.rollouts_per_arm
            end = start + args.rollouts_per_arm
            rewards = rollout.rewards[start:end]
            reward_sum = sum(rewards)
            draw_mean_reward = reward_sum / args.rollouts_per_arm
            draw_mean_rewards.append(draw_mean_reward)
            old_base = float(baseline[row_index])
            if args.rollouts_per_arm > 1:
                advantages = [reward - (reward_sum - reward) / (args.rollouts_per_arm - 1)
                              for reward in rewards]
            else:
                # A single rollout cannot provide an independent Monte Carlo
                # baseline. This fallback is supported for cheap diagnostics;
                # headline runs use leave-one-out groups (K >= 2).
                advantages = [rewards[0] - old_base]
            rollout_advantages.extend(advantages)
            baseline[row_index] = (args.baseline_momentum * baseline[row_index]
                                   + (1.0 - args.baseline_momentum) * draw_mean_reward)
            denom = max(args.select_size * arm_probability, 1e-12)
            importance_denominators.append(denom)
            row_exposure[row_index] += 1
            arm_exposure[arm] += 1
        sft_started = time.perf_counter()
        # Replacement sampling uses a per-step RNG derived from (seed, step)
        # rather than the shared stream, so both headline arms draw the exact
        # same prompt sequence regardless of how many slots are replaced.
        train_rows, mix_stats = build_train_rows(
            selected_rows, rollout, args.rollouts_per_arm, args.train_mix,
            args.max_replaced, random.Random(f"replace:{args.seed}:{step}"))
        loss, train_completion_tokens, mean_kl = _sft_update(
            model, tokenizer, train_rows, args.max_len,
            args.micro_batch, optimizer, scheduler,
            kl_beta=args.kl_beta, kl_adapter=kl_adapter)
        sft_seconds = time.perf_counter() - sft_started
        # The new-policy full-sequence log-probability forward only feeds the
        # OSMD policy-ratio correction.  Matched uniform-curator arms never
        # use it, so headline runs skip the forward and report zero ratios.
        if need_ratios:
            new_logprob_started = time.perf_counter()
            new_logp = _new_logprob(model, rollout, tokenizer.eos_token_id,
                                    logprob_batch=args.logprob_batch)
            new_logprob_seconds = time.perf_counter() - new_logprob_started
            log_ratios = new_logp - rollout.old_logprob
        else:
            new_logprob_seconds = 0.0
            log_ratios = torch.zeros(len(rollout.rewards),
                                     dtype=torch.float64)
        clipped_log_ratios = torch.clamp(
            log_ratios, -args.actor_log_ratio_clip, args.actor_log_ratio_clip)
        ratios = torch.exp(clipped_log_ratios)
        # The policy ratio is the actor-curator performance-improvement
        # correction.  It is multiplied into the already importance-weighted
        # strict-reward advantage before the OSMD update.
        ratio_mean = float(ratios.mean().cpu())
        draw_utilities = []
        draw_policy_improvements = []
        ratio_values = ratios.tolist()
        utility_clip_count = 0
        bounded_policy_improvements = []
        for draw, (arm, denom) in enumerate(zip(selected_arms,
                                                 importance_denominators)):
            start = draw * args.rollouts_per_arm
            end = start + args.rollouts_per_arm
            improvement = sum(ratio * advantage for ratio, advantage in zip(
                ratio_values[start:end], rollout_advantages[start:end])) / args.rollouts_per_arm
            bounded_improvement = max(-args.utility_clip,
                                      min(args.utility_clip, improvement))
            utility_clip_count += int(bounded_improvement != improvement)
            bounded_policy_improvements.append(bounded_improvement)
            contribution = bounded_improvement / denom
            utilities[arm] = utilities.get(arm, 0.0) + contribution
            draw_utilities.append(contribution)
            draw_policy_improvements.append(improvement)
        if args.curator == "osmd":
            curator.update(utilities)
            detector_result = detector.update(
                sum(bounded_policy_improvements)
                / len(bounded_policy_improvements))
        else:
            # Uniform-online is the matched control: it performs the same
            # stochastic rollouts and actor updates, but never adapts arm
            # probabilities from the observed rewards.
            detector_result = {"tested": False, "restart": False, "z": 0.0,
                               "disabled": True}
        restarted = False
        if args.curator == "osmd" and detector_result.get("restart"):
            curator.restart(max(args.fixed_share, args.restart_share))
            restarted = True
        probs = curator.distribution()
        row_probabilities = torch.tensor([
            float(probs[arm]) / len(members[arm]) for arm in strata
        ], dtype=torch.float64)
        report = {
            "step": step, "loss": loss, "mean_reward": sum(rollout.rewards) / len(rollout.rewards),
            "successes": int(sum(rollout.rewards)),
            "mean_advantage": sum(rollout_advantages) / len(rollout_advantages),
            "mean_policy_improvement": (sum(draw_policy_improvements)
                                        / len(draw_policy_improvements)),
            "mean_policy_ratio": ratio_mean,
            "mean_log_policy_ratio": float(log_ratios.mean().cpu()),
            "policy_ratio_clip_fraction": float(
                (log_ratios.abs() > args.actor_log_ratio_clip).float().mean().cpu()),
            "selected_record_ids": [record_id(rows[i]) for i in selected],
            "selected_indices": selected,
            "selected_strata": selected_arms,
            "train_mix": args.train_mix,
            "n_replaced_slots": mix_stats["n_replaced"],
            "replace_rate": mix_stats["n_replaced"] / args.select_size,
            "verified_draws": mix_stats["verified_draws"],
            "n_mastered_skipped": mix_stats["n_mastered_skipped"],
            "batch_slots": mix_stats["batch_slots"],
            "mastered_fallback": mix_stats["mastered_fallback"],
            "kl_beta": args.kl_beta,
            "mean_kl": mean_kl,
            "train_completion_tokens": train_completion_tokens,
            "stratum_probabilities": arm_probs,
            "row_selection_probabilities": [
                probability / len(members[arm])
                for probability, arm in zip(arm_probs, selected_arms)
            ],
            "strict_rewards": rollout.rewards,
            "draw_mean_rewards": draw_mean_rewards,
            "rollout_advantages": rollout_advantages,
            "importance_denominators": importance_denominators,
            "policy_ratios": ratios.tolist(),
            "draw_policy_improvements": draw_policy_improvements,
            "utility_clip_fraction": utility_clip_count / args.select_size,
            "draw_utilities": draw_utilities,
            "aggregated_arm_utilities": {str(k): v for k, v in utilities.items()},
            "reasons": dict(Counter(rollout.reasons)),
            "stratum_effective_sample_size": float(1.0 / (probs.pow(2).sum())),
            "row_effective_sample_size": float(1.0 / row_probabilities.pow(2).sum()),
            "max_stratum_probability": float(probs.max()),
            "min_stratum_probability": float(probs.min()),
            "max_row_probability": float(row_probabilities.max()),
            "min_row_probability": float(row_probabilities.min()),
            "detector": detector_result,
            "restart": restarted,
            "curator_restarts": curator.restarts,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "timings": {**rollout.timings, "sft_s": sft_seconds,
                        "new_logprob_s": new_logprob_seconds,
                        "total_measured_s": (sum(rollout.timings.values())
                                             + sft_seconds
                                             + new_logprob_seconds)},
        }
        round_reports.append(report)
        last_loss = loss
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(report, ensure_ascii=False), flush=True)
        if step % args.save_every == 0 or step == args.updates:
            adapter_dir = out_dir / f"checkpoint-{step}"
            metadata = _metadata(
                args, rows, curator, detector, baseline, row_exposure,
                arm_exposure, strata, stratum_info, members, eta,
                exploration_floor, round_reports, step, loss)
            state = _training_state(
                args, signature, step, loss, optimizer, scheduler, curator,
                detector, baseline, row_exposure, arm_exposure, round_reports,
                rng)
            _save_training_checkpoint(
                adapter_dir, model, tokenizer, metadata, state)
        if (step % args.recovery_every == 0 and step != args.updates
                and step % args.save_every != 0):
            recovery_dir = out_dir / f"recovery-{step}"
            metadata = _metadata(
                args, rows, curator, detector, baseline, row_exposure,
                arm_exposure, strata, stratum_info, members, eta,
                exploration_floor, round_reports, step, loss)
            state = _training_state(
                args, signature, step, loss, optimizer, scheduler, curator,
                detector, baseline, row_exposure, arm_exposure, round_reports,
                rng)
            _save_training_checkpoint(
                recovery_dir, model, tokenizer, metadata, state)
            for old_recovery in out_dir.glob("recovery-*"):
                if old_recovery != recovery_dir and old_recovery.is_dir():
                    shutil.rmtree(old_recovery)
    model.save_pretrained(out_dir / "final", selected_adapters=["default"])
    tokenizer.save_pretrained(out_dir / "final")
    metadata = _metadata(
        args, rows, curator, detector, baseline, row_exposure, arm_exposure,
        strata, stratum_info, members, eta, exploration_floor, round_reports,
        args.updates, last_loss)
    (out_dir / "train_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _metadata(args, rows, curator, detector, baseline, row_exposure,
              arm_exposure, strata, stratum_info, members, eta,
              exploration_floor, reports, step, loss):
    probs = curator.distribution()
    row_probabilities = torch.tensor([
        float(probs[arm]) / len(members[arm]) for arm in strata
    ], dtype=torch.float64)
    return {
        "protocol": "exec_ac_sft_v2_prompt_strata",
        "curator": args.curator,
        "train_mix": args.train_mix,
        "max_replaced": args.max_replaced,
        "kl_beta": args.kl_beta,
        "actor_objective": "full_completion_loss_including_xml_structure",
        "curator_objective": "strict_executor_policy_improvement_reward",
        "seed": args.seed, "warm_start": args.warm_start,
        "n_train": len(rows), "step": step,
        "updates": args.updates,
        "select_size": args.select_size, "micro_batch": args.micro_batch,
        "rollouts_per_arm": args.rollouts_per_arm,
        "max_len": args.max_len, "max_new": args.max_new,
        "generation_batch": args.generation_batch,
        "load_mode": args.load_mode,
        "resume_enabled": args.resume,
        "recovery_every": args.recovery_every,
        "logprob_batch": args.logprob_batch,
        "temperature": args.temperature, "top_p": args.top_p,
        "learning_rate": args.lr, "loss": loss,
        "exploration_floor": exploration_floor,
        "osmd_eta": eta, "fixed_share": args.fixed_share,
        "hyperparameters_auto_calibrated": {
            "osmd_eta": args.osmd_eta <= 0.0,
            "exploration_floor": args.exploration_floor < 0.0,
            "n_strata": args.n_strata == 0,
        },
        "utility_clip": args.utility_clip,
        "strata": stratum_info,
        "restart_share": args.restart_share,
        "restart_block": args.restart_block, "restart_z": args.restart_z,
        "actor_log_ratio_clip": args.actor_log_ratio_clip,
        "curator_updates": curator.updates, "curator_restarts": curator.restarts,
        "stratum_effective_sample_size": float(1.0 / probs.pow(2).sum()),
        "row_effective_sample_size": float(1.0 / row_probabilities.pow(2).sum()),
        "min_stratum_probability": float(probs.min()),
        "max_stratum_probability": float(probs.max()),
        "min_row_probability": float(row_probabilities.min()),
        "max_row_probability": float(row_probabilities.max()),
        "row_exposure_min": min(row_exposure.values()) if row_exposure else 0,
        "row_exposure_max": max(row_exposure.values()) if row_exposure else 0,
        "unique_rows_exposed": len(row_exposure),
        "row_exposure_counts": dict(row_exposure),
        "stratum_exposure_min": min(arm_exposure.values()) if arm_exposure else 0,
        "stratum_exposure_max": max(arm_exposure.values()) if arm_exposure else 0,
        "unique_strata_exposed": len(arm_exposure),
        "stratum_exposure_counts": dict(arm_exposure),
        "arm_baseline_mean": float(baseline.mean()),
        "round_reports": reports,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--warm-start", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--curator", choices=("osmd", "uniform"), default="osmd",
                    help="adaptive OSMD curator or matched uniform-online control")
    ap.add_argument("--train-mix", choices=("gold", "rft"), default="gold",
                    help="gold: every slot keeps its gold completion (matched "
                         "SFT control); rft: slots with an executor-verified "
                         "rollout train on one deduplicated verified "
                         "self-generated completion instead of gold")
    ap.add_argument("--max-replaced", type=int, default=16,
                    help="cap on verified-replacement slots per update; "
                         "remaining slots always keep gold")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                    help="weight of the k3 KL anchor to the frozen warm-start "
                         "reference adapter; >0 stabilizes reward-tilted "
                         "self-training against entropy collapse")
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--load-mode", choices=("bf16", "4bit"), default="4bit",
                    help="4bit must be used with the QLoRA warm start; BF16 is "
                         "available only for adapters trained on a BF16 base")
    ap.add_argument("--updates", type=int, default=225)
    ap.add_argument("--select-size", type=int, default=16)
    ap.add_argument("--n-strata", type=int, default=0,
                    help="prompt-only tabular arms; 0 uses ceil(sqrt(updates))")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--rollouts-per-arm", type=int, default=4,
                    help="strict-executor rollouts per selected draw; K>=2 uses "
                         "an independent leave-one-out reward baseline")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="small online actor step; keeps the policy-ratio "
                         "first-order approximation stable after warm start")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--logprob-batch", type=int, default=8,
                    help="micro-batch for old/new rollout log-probability "
                         "forwards; lower this if long XML batches OOM")
    ap.add_argument("--generation-batch", type=int, default=16,
                    help="stochastic generation micro-batch for rollout XMLs")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--exploration-floor", type=float, default=-1.0,
                    help="prior-mixture mass; negative uses the budget formula")
    ap.add_argument("--osmd-eta", type=float, default=0.0,
                    help="OSMD learning rate; 0 uses the budget formula")
    ap.add_argument("--fixed-share", type=float, default=0.0,
                    help="per-step prior mixing; headline runs use 0 because "
                         "detected changes trigger explicit restarts")
    ap.add_argument("--utility-clip", type=float, default=1.0,
                    help="bound on per-draw policy improvement for OSMD")
    ap.add_argument("--restart-share", type=float, default=1.0,
                    help="mass mixed toward uniform after a detected change; "
                         "1.0 is a full non-stationary restart")
    ap.add_argument("--restart-block", type=int, default=8)
    ap.add_argument("--restart-z", type=float, default=3.0)
    ap.add_argument("--baseline-momentum", type=float, default=0.9)
    ap.add_argument("--actor-log-ratio-clip", type=float, default=10.0,
                    help="symmetric numerical bound before exponentiating the "
                         "full-sequence actor log-probability ratio")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--recovery-every", type=int, default=5,
                    help="rolling resumable checkpoint interval; recovery "
                         "directories are excluded from validation selection")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest complete checkpoint/recovery "
                         "state under run-dir after protocol verification")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.updates <= 0 or args.save_every <= 0 or args.recovery_every <= 0:
        raise ValueError("updates, save-every, and recovery-every must be positive")
    if (args.select_size <= 0 or args.micro_batch <= 0
            or args.rollouts_per_arm <= 0):
        raise ValueError("select-size, micro-batch, and rollouts-per-arm "
                         "must be positive")
    if args.max_replaced < 0 or args.max_replaced > args.select_size:
        raise ValueError("max-replaced must be in [0, select-size]")
    if args.kl_beta < 0.0:
        raise ValueError("kl-beta must be non-negative")
    if args.lr <= 0.0 or args.max_len < 2 or args.max_new <= 0:
        raise ValueError("invalid optimizer or sequence-length configuration")
    if args.logprob_batch <= 0:
        raise ValueError("logprob-batch must be positive")
    if args.generation_batch <= 0:
        raise ValueError("generation-batch must be positive")
    if not 0.0 <= args.restart_share <= 1.0:
        raise ValueError("restart-share must be in [0, 1]")
    if args.n_strata < 0 or args.utility_clip <= 0.0:
        raise ValueError("n-strata and utility-clip are invalid")
    if args.exploration_floor >= 1.0:
        raise ValueError("exploration-floor must be below 1")
    if args.fixed_share < 0.0 or args.fixed_share >= 1.0:
        raise ValueError("fixed-share must be in [0, 1)")
    if not 0.0 < args.baseline_momentum < 1.0:
        raise ValueError("baseline-momentum must be in (0, 1)")
    if args.actor_log_ratio_clip <= 0.0:
        raise ValueError("actor-log-ratio-clip must be positive")
    metadata = train(args)
    print(json.dumps({k: metadata[k] for k in (
        "protocol", "step", "loss", "stratum_effective_sample_size",
        "row_effective_sample_size", "curator_restarts",
        "stratum_exposure_min", "stratum_exposure_max",
        "unique_strata_exposed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
