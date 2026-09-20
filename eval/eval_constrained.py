"""Map-grounded constrained decoding (SPCL-ER lever-1 pilot).

Parses the NL mission prompt into (robots, items, locations, adjacency),
then constrains XML attribute VALUES during greedy decoding:
  - robot/giver/receiver/robot1/robot2 -> parsed robot ids
  - item                              -> parsed item ids
  - location/station                  -> parsed location ids
  - from/edge_from                    -> parsed location ids
  - to/edge_to                        -> neighbours of the same tag's from/edge_from
Anything unexpected -> fall back to unconstrained (never crash, never block).
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import torch

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datagen.executor import execute
from eval.evaluate import reconstruct_task, load_jsonl
from experiments.bt_ducl.common import record_id, apply_chat_template_compat

BASE = "models/llama32-1b"

NAME_ID = re.compile(r"([A-Za-z][A-Za-z'\- ]*?)\s*\(([a-z_0-9]+)\)")
ATTR_RE = re.compile(r'(from|to|location|item|robot|station|giver|receiver|robot1|robot2|edge_from|edge_to)="([^"]*)$')
TAG_RE = re.compile(r"<([A-Za-z]+)((?:\s+\w+=\"[^\"]*\")*)\s*$")
CONNECT_VERBS = re.compile(r"\b(connects to|connected to|connecting|links to|links|linked to|linked|joins|joined to|joins|leads to)\b")


STOP = {"at","the","a","an","to","where","you","your","start","starts","take","it",
        "directly","and","connects","connect","is","are","in","on","of","wakes",
        "already","left","into","one","two","has","have","exits","exit","straight",
        "through","coordinate","routes","each","other","watch","for","be","ready",
        "retry","may","fumble","between","only","available","report","when","both",
        "items","places","place","avoid","unnecessary","delays","shared","corridors",
        "charging","station","stations","recharging","recharge","set",
        "leads","lead","links","link","linked","joins","joined","begin","begins",
        "from","mission","objective","objectives","relocate","relocates","deposit",
        "plan","approach","transport","execute","use","layout","damaging","property",
        "needed","need","must","will","then","next","finally","first","second",
        "team","robots","robot","wakes","wake","starts","started","beginning"}

def _clean_name(name):
    ws = [w for w in name.strip().lower().split() if w not in STOP]
    return " ".join(ws)

def parse_mission(text):
    name2id = {}
    for name, iid in NAME_ID.findall(text):
        cn = _clean_name(name)
        if cn:
            name2id[cn] = iid
    ids = set(name2id.values())
    occ = []
    variants = []
    for name, iid in name2id.items():
        ws = name.split()
        variants.append((name, iid))
        for k in range(1, len(ws) - 1):  # suffixes with >=2 words
            variants.append((" ".join(ws[k:]), iid))
    for name, iid in sorted(set(variants), key=lambda x: -len(x[0])):
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            occ.append((m.start(), m.end(), iid))
    occ.sort()
    clean, last_end = [], -1
    for s, e, i in occ:
        if s >= last_end:
            clean.append((s, e, i)); last_end = e
    edges = defaultdict(set)
    for cm in CONNECT_VERBS.finditer(text):
        left = max(text.rfind(";", 0, cm.start()), text.rfind(".", 0, cm.start()),
                   text.rfind(":", 0, cm.start())) + 1
        right_c = [p for p in (text.find(";", cm.end()), text.find(".", cm.end())) if p > 0]
        right = min(right_c) if right_c else len(text)
        before = [i for s, e, i in clean if e <= cm.start() and s >= left]
        after = [i for s, e, i in clean if s >= cm.end() and e <= right]
        if before and after:
            src = before[-1]
            for tgt in after:
                if tgt != src:
                    edges[src].add(tgt); edges[tgt].add(src)
    for m in re.finditer(r"\bexits?:([^;.]+)", text, re.I):
        clause_ids = [i for s, e, i in clean if s >= m.start() and e <= m.end()]
        before = [i for s, e, i in clean if e <= m.start()]
        if before and clause_ids:
            src = before[-1]
            for tgt in clause_ids:
                if tgt != src:
                    edges[src].add(tgt); edges[tgt].add(src)
    locs = set(edges) | {b for bs in edges.values() for b in bs}
    robots = {i for i in ids if re.fullmatch(r"alpha|beta|gamma|delta|epsilon", i)}
    items = ids - locs - robots
    return robots, items, locs, dict(edges)


class PlatformMap:
    """Minimal public deployment input for SCD.

    Built from the task declaration via from_meta() -- the ONLY function
    allowed to touch raw meta. Any field access outside the whitelist
    raises AssertionError, so privilege escalation crashes loudly.
    """
    ALLOWED = ("robots", "items", "edges", "charge_stations")
    FORBIDDEN = ("goal", "goals", "faults", "fault", "initial", "init",
                 "gold", "output", "plan", "trace", "success", "xml")

    def __init__(self, robots, items, edges, charge_stations):
        self._d = {"robots": sorted(robots), "items": sorted(items),
                   "edges": [tuple(e) for e in edges],
                   "charge_stations": sorted(charge_stations)}

    def __getitem__(self, k):
        assert k in self.ALLOWED, f"SCD accessed forbidden field: {k}"
        return self._d[k]

    @classmethod
    def from_meta(cls, meta):
        """whitelist extraction from task declaration (platform static facts)"""
        edges = [tuple(e) for e in meta["connected"]]
        return cls(robots=meta["robots"], items=meta["items"],
                   edges=edges, charge_stations=meta.get("charge_stations", []))


class ConstrainedState:
    """Per-row stateful attribute-value constraint.

    level:
      topology - attr values constrained to typed vocab; to/edge_to restricted
                 to map neighbours of the same tag's from/edge_from (full SCD)
      type     - attr values constrained to typed vocab only (robot/item/loc),
                 no topology restriction
      grammar  - only XML tag names and attribute names constrained to the
                 schema vocabulary; attribute VALUES are unconstrained
    """
    # Token decoding is independent of the task instance.  Sharing this cache
    # across per-row states avoids decoding the full vocabulary 600 times.
    _GLOBAL_TOKEN_CACHE = {}

    def __init__(self, tok, robots, items, locs, edges, level="topology", schema=None):
        self.tok = tok
        self.level = level
        self.schema = schema or {}   # tag -> sorted attr names
        self.vocabs = {"robot": sorted(robots), "giver": sorted(robots),
                       "receiver": sorted(robots), "robot1": sorted(robots),
                       "robot2": sorted(robots), "item": sorted(items),
                       "location": sorted(locs), "station": sorted(locs)}
        self.locs, self.edges = sorted(locs), edges
        self.text = ""
        self.seen = 0
        self._tok_cache = {"__tok_id__": id(tok)}

    def _allowed_tokens(self, targets):
        key = tuple(targets)
        if key in self._tok_cache:
            return self._tok_cache[key]
        allowed = []
        for tid in range(self.tok.vocab_size):
            s = ConstrainedState._GLOBAL_TOKEN_CACHE.get((id(self.tok), tid))
            if s is None:
                try: s = self.tok.decode([tid])
                except Exception: s = ""
                ConstrainedState._GLOBAL_TOKEN_CACHE[(id(self.tok), tid)] = s
            if not s:
                continue
            s2 = s.lstrip() if self.level == "grammar" else ""
            for w in targets:
                if (w.startswith(s) or s.startswith(w)
                        or (s2 and (w.startswith(s2) or s2.startswith(w)))):
                    allowed.append(tid); break
        self._tok_cache[key] = allowed
        return allowed

    def mask(self, gen_ids):
        # update decoded text with new tokens
        if len(gen_ids) > self.seen:
            self.text += self.tok.decode(gen_ids[self.seen:])
            self.seen = len(gen_ids)
        if self.level == "grammar":
            return self._mask_grammar()
        m = ATTR_RE.search(self.text)
        if not m:
            return None
        attr, typed = m.group(1), m.group(2)
        tm = TAG_RE.search(self.text[:m.start()])
        tag = tm.group(1) if tm else ""
        attrs_done = tm.group(2) if tm else ""
        # tag-aware routing: to/from are LOCATIONS only for moves, ROBOTS for signals
        if attr in ("to", "edge_to"):
            if tag in ("SignalReady",):
                values = self.vocabs["robot"]
            else:
                fm = re.search(r'(?:from|edge_from)="([^"]+)"', attrs_done)
                frm = fm.group(1) if fm else None
                if self.level == "topology" and frm and frm in self.edges:
                    values = sorted(self.edges[frm])
                else:
                    values = self.locs
        elif attr in ("from", "edge_from"):
            if tag in ("WaitReady",):
                values = self.vocabs["robot"]
            else:
                values = self.locs
        else:
            values = self.vocabs.get(attr)
            if not values:
                return None
        targets = [v + '"' for v in values if v.startswith(typed)]
        if typed + '"' in targets:
            targets = [typed + '"']
        return self._filter_tokens(targets, typed)

    def _filter_tokens(self, targets, typed, nospace=False):
        """tokens s.t. typed+s stays on some target (or completes it)."""
        targets = [w for w in targets if w.startswith(typed)]
        if not targets:
            return None  # already off-vocab: let it finish (executor will fail it)
        if self.level == "grammar":
            # typed-aware filter over a class-level cached candidate pool
            out = []
            for tid, s in self._grammar_pool():
                if not s or (nospace and s != s.strip()):
                    continue
                for ps in {typed + s, typed + s.lstrip()}:
                    if any(w.startswith(ps) or ps.startswith(w) for w in targets):
                        out.append(tid); break
            return out or None
        allowed = self._allowed_tokens(targets)
        out = []
        for tid in allowed:
            s = ConstrainedState._GLOBAL_TOKEN_CACHE.get((id(self.tok), tid), "")
            if nospace and s != s.strip():
                continue  # inside a tag name: no whitespace-bearing tokens
            for ps in {typed + s, typed + s.lstrip()}:
                if any(w.startswith(ps) or ps.startswith(w) for w in targets):
                    out.append(tid); break
        return out or None

    _GPOOL = {}  # class-level: schema-fingerprint -> [tid,...]

    def _grammar_pool(self):
        """candidate tokens that can ever continue ANY schema target."""
        key = tuple(sorted((t, tuple(v)) for t, v in sorted(self.schema.items())))
        pool = self._GPOOL.get(key)
        if pool is None:
            allw = set(self.schema) | {">", "/>"}
            for v in self.schema.values():
                allw.update(a + '=\"' for a in v)
            pool = []
            for tid in range(self.tok.vocab_size):
                s = self._tok_cache.get(("t", tid))
                if s is None:
                    try: s = self.tok.decode([tid])
                    except Exception: s = ""
                    self._tok_cache[("t", tid)] = s
                if not s:
                    continue
                if s.strip() == "":
                    pool.append((tid, s))  # separators between attributes
                    continue
                s2 = s.lstrip()
                if any(w.startswith(s2) or s2.startswith(w) or s2 in w for w in allw):
                    pool.append((tid, s))
            self._GPOOL[key] = pool
        return pool

    def _mask_grammar(self):
        """constrain tag names / attribute names to schema vocab; values free."""
        # attribute VALUE position -> unconstrained
        if ATTR_RE.search(self.text):
            return None
        # tag name position: '<Na' or '</Na' partially typed
        m = re.search(r"</?([A-Za-z]*)$", self.text)
        if m:
            typed = m.group(1)
            if typed in self.schema:
                return None  # tag name complete; next comes space or '>'
            return self._filter_tokens(sorted(self.schema), typed, nospace=True)
        # inside an open tag: attribute-name position (or tag close)
        if self.text.rfind("<") > self.text.rfind(">"):
            tm = TAG_RE.search(self.text)
            if not tm:
                return None
            tag, attrs_done = tm.group(1), tm.group(2)
            if tag not in self.schema:
                return None
            used = set(re.findall(r"(\w+)=", attrs_done))
            tail = re.search(r"([A-Za-z]*)$", self.text).group(1)
            remaining = [a + '="' for a in self.schema[tag] if a not in used]
            targets = remaining + [">", "/>"]
            return self._filter_tokens(targets, tail)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--level", choices=["topology", "type", "grammar"],
                    default="topology")
    ap.add_argument("--schema-from", default="data/train.jsonl",
                    help="jsonl whose gold outputs define the tag/attr schema (grammar level)")
    ap.add_argument("--base", default=BASE,
                    help="base causal LM identifier or local snapshot")
    a = ap.parse_args()

    recs = load_jsonl(a.data)
    if a.limit: recs = recs[:a.limit]

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb,
                                                 device_map="auto", dtype=torch.bfloat16)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    # constraint vocab from the platform-provided map (meta static facts),
    # mirroring BTGenBot-2's closed-vocabulary validator
    parsed = []
    for r in recs:
        pm = PlatformMap.from_meta(r["meta"])   # whitelist-only extraction
        edges = defaultdict(set)
        for a_, b_ in pm["edges"]:
            edges[a_].add(b_)  # keep original directionality semantics
        locs = {x for e in pm["edges"] for x in e} | set(pm["charge_stations"])
        parsed.append((set(pm["robots"]), set(pm["items"]), locs, dict(edges)))
    print("constraint vocab: PlatformMap (whitelisted deployment input)")
    schema = None
    if a.level == "grammar":
        from collections import defaultdict as _dd2
        schema = _dd2(set)
        for r in load_jsonl(a.schema_from):
            gold = r.get("output", "")
            for tm_ in re.finditer(r"<([A-Za-z]+)((?:\s+\w+=\"[^\"]*\")*)", gold):
                schema[tm_.group(1)].update(re.findall(r"(\w+)=", tm_.group(2)))
        schema = {t: sorted(v) for t, v in schema.items()}
        print(f"grammar schema: {len(schema)} tags, "
              f"{sum(len(v) for v in schema.values())} (tag,attr) pairs")

    details, generations = [], []
    for s in range(0, len(recs), a.batch_size):
        chunk = recs[s:s+a.batch_size]
        prompts = []
        for r in chunk:
            msgs = [{"role": "system", "content": r["instruction"]},
                    {"role": "user", "content": r["input"]}]
            prompts.append(apply_chat_template_compat(tok, msgs, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        states = [ConstrainedState(tok, *parsed[s+j], level=a.level, schema=schema)
                  for j in range(len(chunk))]
        plen = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc, max_new_tokens=a.max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
            logits_processor=[_Proc(states, plen)])
        for j, r in enumerate(chunk):
            text = tok.decode(gen[j][plen:], skip_special_tokens=True)
            generations.append(text)
            try:
                res = execute(reconstruct_task(r["meta"]), text)
            except Exception as e:
                res = {"success": False, "reason": f"error: {e}"}
            details.append({"record_id": record_id(r),
                            "tier": r["meta"]["tier"], "domain": r["meta"]["domain"],
                            "scenario": r["meta"].get("scenario", ""),
                            "success": res["success"], "reason": res["reason"]})
        done = sum(d["success"] for d in details)
        print(f"[{min(s+a.batch_size,len(recs))}/{len(recs)}] acc {done}/{len(details)}", flush=True)

    from collections import Counter
    per = {}
    for t in ("T1","T2","T3"):
        sub = [d for d in details if d["tier"]==t]
        per[t] = {"n": len(sub), "ok": sum(d["success"] for d in sub)}
    n = len(details); ok = sum(d["success"] for d in details)
    out = {"n": n, "base": a.base, "exec_success_rate": ok/n, "per_tier": per,
           "fail_reasons": dict(Counter(d["reason"] for d in details if not d["success"])),
           "details": details, "generations": generations}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({"overall": ok/n, "per_tier": per}, indent=1))


class _Proc:
    def __init__(self, states, plen):
        self.states, self.plen = states, plen
    def __call__(self, input_ids, scores):
        import torch as T
        for j, st in enumerate(self.states):
            allowed = st.mask(input_ids[j][self.plen:].tolist())
            if allowed is not None:
                keep = T.full_like(scores[j], float("-inf"))
                keep[allowed] = scores[j][allowed]
                scores[j] = keep
        return scores


if __name__ == "__main__":
    main()
