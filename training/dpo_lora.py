"""Gold-anchored DPO on execution-verified preference pairs.

chosen  = gold BT (dataset answer)      -- never model dialect, anchors style
rejected= model sample that failed executor verification
ref     = same model with LoRA adapters disabled (no extra memory)

loss = DPO(beta) + sft_alpha * NLL(chosen)   (SFT anchor against drift)
"""
import argparse, json, math, random
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = "models/llama32-1b"


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def seq_logp(model, input_ids, attn_mask, comp_start, grad=True):
    """sum logp of tokens from comp_start..end (per row)."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(tok_lp)
    for i, s in enumerate(comp_start):
        mask[i, s - 1:] = 1.0            # token at position s predicted by logits s-1
    mask = mask * attn_mask[:, 1:]
    return (tok_lp * mask).sum(-1), mask.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--val-pairs", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-alpha", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=2, help="pairs per step")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = load_jsonl(args.pairs)
    if args.limit: pairs = pairs[: args.limit]
    random.shuffle(pairs)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable/1e6:.1f}M", flush=True)

    def encode(pair):
        msgs = [{"role": "system", "content": pair["instruction"]},
                {"role": "user", "content": pair["input"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows = []
        for text in (pair["chosen"], pair["rejected"]):
            full = prompt + text + tok.eos_token
            ids = tok(full, add_special_tokens=False)["input_ids"][: args.max_len]
            plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
            rows.append((ids, min(plen, len(ids))))
        return rows

    steps = math.ceil(len(pairs) / args.bs / args.accum)
    print(f"pairs={len(pairs)} opt_steps={steps} (bs={args.bs} accum={args.accum})", flush=True)
    hist = []
    gstep = 0; micro = 0; opt.zero_grad()
    acc = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    done = 0
    epochs_left = args.epochs
    while epochs_left > 0:
        epochs_left -= 1
        for i in range(0, len(pairs), args.bs):
            batch = pairs[i:i + args.bs]
            enc = []
            for p in batch: enc += encode(p)          # chosen, rejected alternating
            maxlen = max(len(r[0]) for r in enc)
            ids = torch.full((len(enc), maxlen), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(enc), maxlen), dtype=torch.long)
            starts = []
            for j, (seq, st) in enumerate(enc):
                ids[j, :len(seq)] = torch.tensor(seq); am[j, :len(seq)] = 1
                starts.append(st)
            ids, am = ids.to(model.device), am.to(model.device)
            pol_lp, _ = seq_logp(model, ids, am, starts, grad=True)
            with model.disable_adapter(), torch.no_grad():
                ref_lp, _ = seq_logp(model, ids, am, starts, grad=False)
            pc, pr = pol_lp[0::2], pol_lp[1::2]
            rc, rr = ref_lp[0::2], ref_lp[1::2]
            logits = args.beta * ((pc - rc) - (pr - rr))
            dpo = -F.logsigmoid(logits).mean()
            sft = -(pc / torch.tensor([maxlen - s for s in starts[0::2]],
                     device=model.device).clamp(min=1)).mean()
            loss = dpo + args.sft_alpha * sft
            (loss / args.accum).backward()
            acc["loss"] += loss.item(); acc["n"] += 1
            acc["acc"] += (logits > 0).float().mean().item()
            acc["margin"] += logits.mean().item()
            micro += 1; done += len(batch)
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); gstep += 1
                if gstep % 10 == 0:
                    m = {k: v / max(acc["n"], 1) for k, v in acc.items() if k != "n"}
                    print(f"step {gstep}/{steps} loss={m['loss']:.4f} pair_acc={m['acc']:.3f} margin={m['margin']:.3f}", flush=True)
                    hist.append({"step": gstep, **m}); acc.update({"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0})

    out = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / args.run_name / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out)); tok.save_pretrained(str(out))
    json.dump({"args": vars(args), "pairs": len(pairs), "opt_steps": gstep, "history": hist},
              open(out / "train_metrics.json", "w"), indent=1)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
