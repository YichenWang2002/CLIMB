import unittest

from experiments.bt_ducl.curriculum_v4 import AbilityCurriculumSampler, tier_profile


class CurriculumSamplerTest(unittest.TestCase):
    @staticmethod
    def _rows(n=96):
        rows = []
        domains = ("office", "hospital", "warehouse", "search_rescue")
        for i in range(n):
            tier = ("T1", "T2", "T3")[i % 3]
            faults = 0 if tier == "T1" else (1 if i % 2 else 0)
            if tier == "T3" and i % 4 == 0:
                faults = 2
            rows.append({"meta": {"tier": tier, "domain": domains[i % 4],
                                   "faults": [{} for _ in range(faults)]},
                         "v4_due": float(i)})
        return rows

    def test_profile_progresses_and_sampler_is_deterministic(self):
        self.assertGreater(tier_profile(0, 0.0)["T1"], tier_profile(0, 1.0)["T1"])
        rows = self._rows()
        a = AbilityCurriculumSampler(rows, "ducl", 42, batch_size=16, epochs=3)
        b = AbilityCurriculumSampler(rows, "ducl", 42, batch_size=16, epochs=3)
        ao = [list(iter(a)) for _ in range(3)]
        bo = [list(iter(b)) for _ in range(3)]
        self.assertEqual(ao, bo)
        self.assertTrue(all(len(x) == len(rows) for x in ao))
        self.assertTrue(all(min(x) >= 0 and max(x) < len(rows) for x in ao))
        self.assertTrue(all(r["stratum_fallbacks"] == 0 for r in a.epoch_reports))
        self.assertEqual(a.epoch_reports[0]["duplicate_draws"], 0)

    def test_flat_is_a_permutation_each_epoch(self):
        rows = self._rows()
        sampler = AbilityCurriculumSampler(rows, "flat", 42, batch_size=16, epochs=3)
        for _ in range(3):
            order = list(iter(sampler))
            self.assertEqual(sorted(order), list(range(len(rows))))
            self.assertEqual(sampler.epoch_reports[-1]["duplicate_draws"], 0)


if __name__ == "__main__":
    unittest.main()
