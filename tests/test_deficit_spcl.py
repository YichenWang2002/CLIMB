import unittest

import numpy as np

from curriculum.run_deficit_spcl import (
    adaptive_bucket_quotas,
    assign_quantile_buckets,
    filter_reference_rows,
    jeffreys_deficits,
    largest_remainder,
    sample_adaptive_round,
    split_probe_indices,
    task_key,
)
from eval.paired_compare import compare_results
from eval.evaluate import stable_record_id


def _row(index, bucket=0):
    return {
        "instruction": "generate",
        "input": f"task {index}",
        "output": f"<root id='{index}'/>",
        "meta": {
            "domain": "office",
            "tier": ("T1", "T2", "T3")[bucket % 3],
            "robots": ["alpha"],
            "init_dynamic": [["at", "alpha", f"p{index}"]],
            "goal": [["at", "alpha", f"q{index}"]],
            "faults": [],
        },
    }


class DeficitSPCLTest(unittest.TestCase):
    def test_evaluation_record_id_ignores_gold_output(self):
        left = _row(1)
        right = dict(left, output="different held-out answer")
        self.assertEqual(stable_record_id(left), stable_record_id(right))
        self.assertNotEqual(stable_record_id(left), stable_record_id(_row(2)))

    def test_quantile_buckets_and_probe_are_exact_and_deterministic(self):
        difficulty = np.linspace(0.0, 1.0, 40)
        buckets = assign_quantile_buckets(difficulty, 4)
        self.assertEqual(np.bincount(buckets).tolist(), [10, 10, 10, 10])
        core_a, probe_a = split_probe_indices(buckets, 2, 42)
        core_b, probe_b = split_probe_indices(buckets, 2, 42)
        self.assertTrue(np.array_equal(core_a, core_b))
        self.assertTrue(np.array_equal(probe_a, probe_b))
        self.assertFalse(set(core_a) & set(probe_a))
        self.assertEqual(np.bincount(buckets[probe_a]).tolist(), [2, 2, 2, 2])

    def test_jeffreys_deficit_is_data_derived_and_nonzero(self):
        errors = np.asarray([0.0, 0.2, 0.5, 0.7])
        buckets = np.asarray([0, 0, 1, 1])
        deficit = jeffreys_deficits(errors, buckets, 2)
        self.assertAlmostEqual(deficit[0], (0.2 + 0.5) / 3.0)
        self.assertAlmostEqual(deficit[1], (1.2 + 0.5) / 3.0)
        self.assertGreater(deficit[1], deficit[0])

    def test_largest_remainder_preserves_budget(self):
        quotas = largest_remainder(11, np.asarray([0.1, 0.2, 0.7]))
        self.assertEqual(int(quotas.sum()), 11)
        self.assertEqual(quotas.tolist(), [1, 2, 8])

    def test_bucket_allocation_tracks_size_times_deficit(self):
        buckets = np.asarray([0] * 10 + [1] * 20 + [2] * 30)
        deficits = np.asarray([0.1, 0.2, 0.9])
        masses, quotas = adaptive_bucket_quotas(buckets, deficits, 2, 60)
        self.assertAlmostEqual(masses[0], 0.2)
        self.assertAlmostEqual(masses[1], 0.8)
        self.assertEqual(masses[2], 0.0)
        self.assertEqual(quotas.tolist(), [12, 48, 0])

    def test_sampling_is_deterministic_and_never_draws_inactive_bucket(self):
        rows = [_row(i, i // 5) for i in range(15)]
        buckets = np.asarray([0] * 5 + [1] * 5 + [2] * 5)
        utility = np.linspace(0.2, 1.0, 15)
        deficits = np.asarray([0.1, 0.3, 0.8])
        first, report_a = sample_adaptive_round(
            rows, buckets, utility, deficits, 2, 30, 7)
        second, report_b = sample_adaptive_round(
            rows, buckets, utility, deficits, 2, 30, 7)
        self.assertEqual([row["input"] for row in first],
                         [row["input"] for row in second])
        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a["n_draws"], 30)
        self.assertEqual(report_a["realized_bucket_draws"][2], 0)

    def test_reference_filter_removes_structured_overlap(self):
        source = [_row(1), _row(2)]
        duplicate = dict(_row(1), input="different prose", output="different xml")
        reference = [duplicate, _row(3)]
        kept, removed = filter_reference_rows(reference, source)
        self.assertEqual(removed, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(task_key(duplicate), task_key(source[0]))

    def test_paired_comparison_uses_aligned_outcomes(self):
        def result(values):
            return {"details": [
                {"record_id": str(index), "domain": "d", "tier": "T1",
                 "scenario": "s", "success": value}
                for index, value in enumerate(values)]}
        comparison = compare_results(
            result([1, 1, 0, 0]), result([1, 0, 1, 1]),
            bootstrap_samples=1_000, seed=3)
        self.assertEqual(comparison["flat_only"], 1)
        self.assertEqual(comparison["method_only"], 2)
        self.assertAlmostEqual(comparison["absolute_difference"], 0.25)

    def test_paired_comparison_rejects_record_reordering(self):
        left = {"details": [
            {"record_id": "a", "domain": "d", "tier": "T1",
             "scenario": "s", "success": True}]}
        right = {"details": [
            {"record_id": "b", "domain": "d", "tier": "T1",
             "scenario": "s", "success": True}]}
        with self.assertRaises(ValueError):
            compare_results(left, right, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
