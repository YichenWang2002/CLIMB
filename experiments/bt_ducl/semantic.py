"""Shared semantic-token supervision for BT-XML scoring and training."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .common import prompt_text


TAG_NAME_RE = re.compile(r"<\s*(?:\?|/)?\s*([A-Za-z_][\w:.-]*)")
ATTR_VALUE_RE = re.compile(r"=\s*([\"'])(.*?)\1", re.DOTALL)


@dataclass(frozen=True)
class SemanticEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    semantic_tokens: int
    completion_tokens: int
    truncated: bool


def semantic_spans(xml: str) -> list[tuple[int, int]]:
    """Return tag-name and quoted attribute-value character spans."""
    spans = [match.span(1) for match in TAG_NAME_RE.finditer(xml)]
    spans.extend(match.span(2) for match in ATTR_VALUE_RE.finditer(xml))
    return sorted((start, end) for start, end in spans if end > start)


def _overlaps(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = offset
    return end > start and any(start < span_end and end > span_start
                               for span_start, span_end in spans)


def encode_semantic_example(row: dict, tokenizer, max_len: int,
                            scope: str = "semantic") -> SemanticEncoding:
    """Tokenize one prompt/completion and build labels for the chosen scope.

    scope="completion" supervises every completion token (the standard SFT
    loss; the v4 "semantic" scope that masked structural tokens produced
    0/600 strict success and must not be used for training).
    scope="semantic" supervises only tag names, attribute values and EOS; it
    remains valid for frozen-base Difficulty scoring, never for training.

    Offsets are computed over the concatenated text, so the first completion
    token is identical to the token used during normal causal generation.
    """
    if scope not in ("semantic", "completion"):
        raise ValueError("scope must be 'semantic' or 'completion'")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("semantic token masking requires a fast tokenizer")
    if max_len < 2:
        raise ValueError("max_len must be at least 2")

    prompt = prompt_text(row, tokenizer)
    xml = row["output"]
    eos = tokenizer.eos_token or ""
    completion = xml + eos
    full_text = prompt + completion
    encoded = tokenizer(full_text, add_special_tokens=False,
                        return_offsets_mapping=True, truncation=True,
                        max_length=max_len)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    attention = [1] * len(input_ids)
    if scope == "completion":
        shifted_spans = [(len(prompt), len(full_text))]
    else:
        shifted_spans = [(len(prompt) + start, len(prompt) + end)
                         for start, end in semantic_spans(xml)]
        if eos:
            shifted_spans.append((len(prompt) + len(xml), len(full_text)))

    labels = [token_id if _overlaps(offset, shifted_spans) else -100
              for token_id, offset in zip(input_ids, offsets)]
    # Position zero has no causal predecessor and can never contribute to NLL.
    if labels:
        labels[0] = -100

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    semantic_tokens = sum(label != -100 for label in labels)
    if semantic_tokens == 0:
        raise ValueError("completion has no semantic XML or EOS tokens")
    return SemanticEncoding(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        semantic_tokens=semantic_tokens,
        completion_tokens=min(len(completion_ids), max(0, max_len - len(prompt_ids))),
        truncated=len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) > max_len,
    )


def semantic_dataset_row(row: dict, tokenizer, max_len: int,
                         scope: str = "completion") -> dict:
    """Build a training row. Default scope is full-completion supervision."""
    encoded = encode_semantic_example(row, tokenizer, max_len, scope=scope)
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        "labels": encoded.labels,
    }
