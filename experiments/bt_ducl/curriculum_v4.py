"""Ability-stage, stratified, DUE-diverse sampler used by v4 DUCL."""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import Sampler


TRAIN_TIER = {"T1": 0.30, "T2": 0.50, "T3": 0.20}
TARGET_TIER = {"T1": 0.20, "T2": 0.40, "T3": 0.40}
EARLY_TIER = {"T1": 0.50, "T2": 0.35, "T3": 0.15}
TIER_ORDER = ("T1", "T2", "T3")


def _interpolate(a: dict[str, float], b: dict[str, float], amount: float) -> dict[str, float]:
    amount = min(1.0, max(0.0, amount))
    return {key: (1.0 - amount) * a[key] + amount * b[key] for key in TIER_ORDER}


def tier_profile(epoch: int, progress: float, epochs: int = 3) -> dict[str, float]:
    """Return the target tier mix at a fractional epoch position."""
    if epoch == 0:
        if progress <= 0.5:
            return _interpolate(EARLY_TIER, TRAIN_TIER, progress / 0.5)
        return _interpolate(TRAIN_TIER, TARGET_TIER, (progress - 0.5) / 0.5)
    if epoch == 1:
        return _interpolate(TRAIN_TIER, TARGET_TIER, progress)
    return dict(TARGET_TIER)


def _largest_remainder(total: int, weights: dict[str, float], keys: tuple[str, ...]) -> dict[str, int]:
    raw = {key: total * max(0.0, float(weights.get(key, 0.0))) for key in keys}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(keys, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


class AbilityCurriculumSampler(Sampler[int]):
    """Sampler with fixed compute and a strict per-epoch permutation guarantee.

    Flat mode is the ordinary seeded permutation. DUCL mode also emits exactly
    one permutation of the dataset per epoch: the ability profile guides only
    *ordering* (which strata are drawn early vs late), while per-sample
    exposure stays exactly one. Rows are never reused or omitted. When a quota
    stratum is exhausted (e.g. the epoch-3 T3 target exceeds the T3 pool), the
    slot spills to the nearest unused rows and the deviation is recorded in
    the epoch report.
    """

    def __init__(self, rows: list[dict], mode: str, seed: int, batch_size: int = 16,
                 epochs: int = 3, due_key: str = "v4_due", n_bands: int = 4):
        if mode not in ("flat", "ducl"):
            raise ValueError("mode must be flat or ducl")
        if batch_size <= 0 or len(rows) % batch_size:
            raise ValueError("dataset length must be divisible by effective batch size")
        self.rows, self.mode, self.seed = rows, mode, seed
        self.batch_size, self.epochs, self.due_key = batch_size, epochs, due_key
        self.n_bands = max(1, n_bands)
        self.epoch = 0
        self.orders: list[list[int]] = []
        self.epoch_reports: list[dict] = []
        self._pools: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        self._tier_fault_pools: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._tier_pools: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            meta = row["meta"]
            tier = meta.get("tier")
            fault = str(min(2, len(meta.get("faults", []))))
            domain = meta.get("domain", "unknown")
            self._pools[(tier, fault, domain)].append(i)
            self._tier_fault_pools[(tier, fault)].append(i)
            self._tier_pools[tier].append(i)
        for pool in list(self._pools.values()) + list(self._tier_fault_pools.values()) + list(self._tier_pools.values()):
            pool.sort(key=lambda i: (float(rows[i].get(due_key, 0.0)), i))

    def __len__(self):
        return len(self.rows)

    def _fault_profile(self, tier: str, epoch: int, progress: float) -> dict[str, float]:
        if tier == "T1":
            return {"0": 1.0}
        if tier == "T2":
            # Observed train mix is approximately 47/53 (no fault/one fault).
            p = 0.53 + (0.60 - 0.53) * min(1.0, max(0.0, (epoch + progress) / 2.0))
            return {"0": 1.0 - p, "1": p}
        # T3 contains one- and two-fault cases; progressively favor two faults.
        p = 0.48 + (0.60 - 0.48) * min(1.0, max(0.0, (epoch + progress) / 2.0))
        return {"1": 1.0 - p, "2": p}

    def _choose_from_pool(self, pool: list[int], fraction: float, band: int,
                          used: set[int], rng: random.Random) -> int | None:
        """Pick one unused row inside the DUE window, or None if exhausted.

        Preference order: the requested diversity band inside the window, then
        the full window, then the full pool. The band index is chosen
        independently of the domain round-robin so every DUE quarter of each
        stratum stays reachable.
        """
        if not pool:
            raise RuntimeError("empty curriculum stratum")
        opened = max(1, min(len(pool), int(math.ceil(fraction * len(pool)))))
        window = pool[:opened]
        lo = int(math.floor(band * opened / self.n_bands))
        hi = int(math.ceil((band + 1) * opened / self.n_bands))
        band_candidates = [i for i in window[lo:hi] if i not in used]
        candidates = band_candidates or [i for i in window if i not in used]
        if not candidates:
            candidates = [i for i in pool if i not in used]
        if not candidates:
            return None
        chosen = candidates[rng.randrange(len(candidates))]
        used.add(chosen)
        return chosen

    def _ducl_order(self, epoch: int) -> tuple[list[int], dict]:
        n_batches = len(self.rows) // self.batch_size
        used: set[int] = set()
        order: list[int] = []
        fallbacks = 0
        spillage = 0
        batch_profiles = []
        tier_expected, tier_actual = Counter(), Counter()
        fault_expected, fault_actual = defaultdict(Counter), defaultdict(Counter)
        rng = random.Random(self.seed + 100003 * epoch)
        for batch_index in range(n_batches):
            progress = batch_index / max(1, n_batches - 1)
            profile = tier_profile(epoch, progress, self.epochs)
            for key in TIER_ORDER:
                tier_expected[key] += self.batch_size * profile[key]
            tier_quota = {key: 0 for key in TIER_ORDER}
            for _ in range(self.batch_size):
                key = max(TIER_ORDER, key=lambda x: (tier_expected[x] - tier_actual[x], -TIER_ORDER.index(x)))
                tier_quota[key] += 1
                tier_actual[key] += 1
            batch_profiles.append(profile)
            batch_indices = []
            for tier in TIER_ORDER:
                fault_profile = self._fault_profile(tier, epoch, progress)
                fault_keys = tuple(fault_profile)
                for key in fault_keys:
                    fault_expected[tier][key] += tier_quota[tier] * fault_profile[key]
                fault_quota = {key: 0 for key in fault_keys}
                for _ in range(tier_quota[tier]):
                    key = max(fault_keys, key=lambda x: (fault_expected[tier][x] - fault_actual[tier][x], x))
                    fault_quota[key] += 1
                    fault_actual[tier][key] += 1
                for fault, count in fault_quota.items():
                    domains = sorted({self.rows[i]["meta"].get("domain", "unknown")
                                      for i in self._tier_fault_pools.get((tier, fault), [])})
                    if not domains:
                        # A target stratum may not exist for a tiny dataset; use
                        # the nearest ability pool and record the deviation.
                        domains = sorted({self.rows[i]["meta"].get("domain", "unknown")
                                          for i in self._tier_pools.get(tier, [])})
                        fault_pool = self._tier_pools.get(tier, [])
                        fallback_fault = True
                    else:
                        fault_pool = self._tier_fault_pools[(tier, fault)]
                        fallback_fault = False
                    for slot in range(count):
                        if not fault_pool:
                            raise RuntimeError(f"no examples for tier={tier} fault={fault}")
                        band = rng.randrange(self.n_bands)
                        if fallback_fault:
                            chosen = self._choose_from_pool(
                                fault_pool, 0.10 + 0.90 * progress, band, used, rng)
                            fallbacks += 1
                        else:
                            domain_offset = TIER_ORDER.index(tier) + int(fault)
                            domain = domains[(batch_index + slot + domain_offset) % len(domains)]
                            pool = self._pools[(tier, fault, domain)]
                            # Open a window over each semantic ability stratum;
                            # band diversity prevents adjacent-rank batches.
                            fraction = 0.10 + 0.90 * min(1.0, (epoch + progress) / 1.8)
                            chosen = self._choose_from_pool(
                                pool, fraction, band, used, rng)
                        if chosen is None:
                            # Permutation constraint: the quota stratum is
                            # exhausted, so spill to the nearest unused rows
                            # (same tier first) and record the deviation from
                            # the target profile in the epoch report.
                            spare = [i for i in self._tier_pools.get(tier, [])
                                     if i not in used]
                            if not spare:
                                spare = [i for i in range(len(self.rows))
                                         if i not in used]
                            if not spare:
                                raise RuntimeError("permutation pool exhausted")
                            chosen = spare[rng.randrange(len(spare))]
                            used.add(chosen)
                            spillage += 1
                        batch_indices.append(chosen)
            if len(batch_indices) != self.batch_size:
                raise AssertionError("stratified quota did not fill effective batch")
            rng.shuffle(batch_indices)
            order.extend(batch_indices)
        if len(order) != len(self.rows) or len(set(order)) != len(self.rows):
            raise AssertionError("curriculum order is not a strict permutation")
        exposure = Counter(order)
        tier_counts = Counter(self.rows[i]["meta"].get("tier") for i in order)
        fault_counts = Counter(str(min(2, len(self.rows[i]["meta"].get("faults", []))))
                              for i in order)
        domain_counts = Counter(self.rows[i]["meta"].get("domain", "unknown") for i in order)
        report = {
            "epoch": epoch + 1,
            "tier_profile_start": batch_profiles[0],
            "tier_profile_end": batch_profiles[-1],
            "exposure_count_min": min(exposure.values()),
            "exposure_count_max": max(exposure.values()),
            "unique_rows_exposed": len(exposure),
            "duplicate_draws": len(order) - len(exposure),
            "permutation": len(exposure) == len(self.rows),
            "spillage_slots": spillage,
            "tier_counts": dict(tier_counts),
            "fault_counts": dict(fault_counts),
            "domain_counts": dict(domain_counts),
            "stratum_fallbacks": fallbacks,
        }
        return order, report

    def __iter__(self):
        epoch = self.epoch
        if self.mode == "flat":
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.rows), generator=generator).tolist()
            report = {
                "epoch": epoch + 1,
                "tier_counts": dict(Counter(self.rows[i]["meta"].get("tier") for i in order)),
                "fault_counts": dict(Counter(str(min(2, len(self.rows[i]["meta"].get("faults", []))))
                                               for i in order)),
                "unique_rows_exposed": len(self.rows), "duplicate_draws": 0,
                "exposure_count_min": 1, "exposure_count_max": 1,
                "permutation": True, "spillage_slots": 0,
            }
        else:
            order, report = self._ducl_order(epoch)
        self.orders.append(order)
        self.epoch_reports.append(report)
        self.epoch += 1
        return iter(order)


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, order)).encode("ascii")).hexdigest()
