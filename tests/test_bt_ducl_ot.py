import unittest

import numpy as np

from experiments.bt_ducl.scoring_v2 import window_order_from_due
from experiments.bt_ducl.train import EpochOrderSampler


class WindowOrderingTest(unittest.TestCase):
    def test_order_is_deterministic_complete_permutation(self):
        due = np.linspace(0.01, 10.0, 101)
        first = window_order_from_due(due, batch_size=8, alpha=0.8, seed=42)
        second = window_order_from_due(due, batch_size=8, alpha=0.8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(101)))

    def test_curriculum_anneals_to_matched_flat_order(self):
        due = list(np.linspace(0.01, 10.0, 100))
        ducl = EpochOrderSampler(100, "ducl", 7, 16, due, 0.8, 2)
        flat = EpochOrderSampler(100, "flat", 7, 16, None, 0.8, 0)
        ducl_orders = [list(iter(ducl)) for _ in range(3)]
        flat_orders = [list(iter(flat)) for _ in range(3)]
        self.assertNotEqual(ducl_orders[0], flat_orders[0])
        self.assertNotEqual(ducl_orders[1], flat_orders[1])
        self.assertEqual(ducl_orders[2], flat_orders[2])
        for order in ducl_orders:
            self.assertEqual(sorted(order), list(range(100)))


if __name__ == "__main__":
    unittest.main()
