import math
import random
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path

import torch

from experiments.bt_ducl.common import load_jsonl
from experiments.bt_ducl.exec_ac_sft import (
    BlockVariationDetector,
    RolloutBatch,
    TabularOSMD,
    automatic_bandit_hyperparameters,
    _continuation_logprob,
    _latest_resume_checkpoint,
    _restore_training_state,
    _strict_reward,
    _supervised_token_logprobs,
    _training_state,
    _training_text,
    build_train_rows,
    induce_prompt_strata,
    sample_stratified_rows,
)


class TestTabularOSMD(unittest.TestCase):
    def test_distribution_and_floor(self):
        curator = TabularOSMD(11, exploration_floor=0.2, eta=0.7,
                              fixed_share=0.1)
        probabilities = curator.distribution()
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertGreaterEqual(float(probabilities.min()), 0.2 / 11.0)

        rng = __import__("random").Random(7)
        candidate = curator.sample_candidate(5, rng)
        self.assertEqual(len(candidate), len(set(candidate)))
        selected, selected_probabilities = curator.sample_from_candidate(
            candidate, 13, rng)
        self.assertEqual(len(selected), len(selected_probabilities))
        self.assertTrue(all(p > 0 and math.isfinite(p)
                            for p in selected_probabilities))

        curator.update({selected[0]: 100.0, selected[-1]: -20.0})
        self.assertAlmostEqual(float(curator.distribution().sum()), 1.0,
                               places=12)
        self.assertGreaterEqual(float(curator.distribution().min()), 0.2 / 11.0)
        curator.restart(0.5)
        self.assertAlmostEqual(float(curator.distribution().sum()), 1.0,
                               places=12)
        self.assertEqual(curator.restarts, 1)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            TabularOSMD(0)
        with self.assertRaises(ValueError):
            TabularOSMD(2, exploration_floor=1.0)
        curator = TabularOSMD(2)
        with self.assertRaises(ValueError):
            curator.sample_candidate(3, __import__("random").Random(1))
        with self.assertRaises(ValueError):
            curator.sample_from_candidate([0, 0], 1, __import__("random").Random(1))

    def test_update_is_exact_exponentiated_gradient(self):
        curator = TabularOSMD(3, exploration_floor=0.0, eta=0.5,
                              fixed_share=0.0)
        curator.update({0: 1.0})
        expected = math.exp(0.5) / (math.exp(0.5) + 2.0)
        self.assertAlmostEqual(float(curator.distribution()[0]), expected,
                               places=12)

    def test_nonuniform_prior_preserves_uniform_row_distribution(self):
        curator = TabularOSMD(2, exploration_floor=0.2, eta=0.5,
                              prior=[0.25, 0.75])
        self.assertTrue(torch.allclose(
            curator.distribution(), torch.tensor([0.25, 0.75],
                                                  dtype=torch.float64)))
        members = [[0], [1, 2, 3]]
        rows, arms, probabilities = sample_stratified_rows(
            curator, members, 20000, random.Random(4))
        counts = [rows.count(index) / len(rows) for index in range(4)]
        self.assertTrue(all(abs(probability - 0.25) < 0.02
                            for probability in counts), counts)
        self.assertEqual(len(arms), len(probabilities))

    def test_automatic_hyperparameters_are_budget_scaled(self):
        eta, exploration = automatic_bandit_hyperparameters(225, 16, 15)
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)
        self.assertGreater(exploration, 0.0)
        self.assertLess(exploration, 0.5)

    def test_prompt_strata_do_not_read_output_or_meta(self):
        rows = [
            {"instruction": "make a tree", "input": f"task family {i % 3} item {i}",
             "output": f"gold-{i}", "meta": {"secret": i}}
            for i in range(30)
        ]
        labels_a, info_a = induce_prompt_strata(rows, 3, seed=9)
        changed = [dict(row, output="changed", meta={"other": 999}) for row in rows]
        labels_b, info_b = induce_prompt_strata(changed, 3, seed=9)
        self.assertEqual(labels_a, labels_b)
        self.assertEqual(info_a["assignment_sha256"], info_b["assignment_sha256"])
        self.assertFalse(info_a["uses_output"])


class TestBlockVariationDetector(unittest.TestCase):
    def test_stable_blocks_do_not_restart(self):
        detector = BlockVariationDetector(block_size=4, z_threshold=3.0)
        results = [detector.update(0.25) for _ in range(8)]
        self.assertFalse(results[-1]["restart"])
        self.assertTrue(results[-1]["tested"])

    def test_large_change_restarts(self):
        detector = BlockVariationDetector(block_size=4, z_threshold=3.0)
        for _ in range(4):
            detector.update(0.0)
        result = None
        for _ in range(4):
            result = detector.update(1.0)
        self.assertTrue(result["tested"])
        self.assertTrue(result["restart"])
        self.assertGreaterEqual(result["z"], 3.0)

    def test_partial_block_is_not_tested(self):
        detector = BlockVariationDetector(block_size=4)
        result = detector.update(1.0)
        self.assertFalse(result["tested"])
        self.assertFalse(result["restart"])


class TestBuildTrainRows(unittest.TestCase):
    """Optimal-difficulty gating and provenance contract of the RFT batch."""

    @staticmethod
    def _rollout(texts, rewards):
        n = len(texts)
        return RolloutBatch(
            output_ids=torch.zeros((n, 4), dtype=torch.long),
            attention_mask=torch.ones((n, 4), dtype=torch.long),
            continuation_mask=torch.ones((n, 2), dtype=torch.bool),
            prompt_width=2,
            old_logprob=torch.zeros(n),
            rewards=list(rewards),
            reasons=["success" if r else "parse:bad" for r in rewards],
            texts=list(texts),
            timings={},
        )

    def _rows(self, n):
        return [{"instruction": "i", "input": f"in{k}",
                 "output": f"<gold{k}/>", "meta": {}} for k in range(n)]

    def test_gold_mix_skips_mastered_keeps_rest_gold(self):
        rows = self._rows(3)
        # draw0 mastered (2/2), draw1 learnable (1/2), draw2 unsolved (0/2)
        rollout = self._rollout(
            ["<a/>", "<b/>", "<c/>", "bad", "bad", "bad"],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        train_rows, stats = build_train_rows(
            rows, rollout, 2, "gold", 16, random.Random(0))
        self.assertEqual(train_rows, rows[1:])
        self.assertEqual(stats["n_replaced"], 0)
        self.assertEqual(stats["verified_draws"], 2)
        self.assertEqual(stats["n_mastered_skipped"], 1)
        self.assertEqual(stats["batch_slots"], 2)
        self.assertFalse(stats["mastered_fallback"])

    def test_rft_replaces_learnable_skips_mastered_keeps_unsolved_gold(self):
        rows = self._rows(3)
        rollout = self._rollout(
            ["<a/>", "<b/>", "<win1/>", "bad", "bad", "bad"],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        train_rows, stats = build_train_rows(
            rows, rollout, 2, "rft", 16, random.Random(0))
        self.assertEqual([r["input"] for r in train_rows], ["in1", "in2"])
        self.assertEqual(train_rows[0]["output"], "<win1/>")
        self.assertEqual(train_rows[1]["output"], "<gold2/>")
        self.assertEqual(stats["n_replaced"], 1)
        self.assertEqual(stats["n_mastered_skipped"], 1)
        # The original rows are not mutated.
        self.assertEqual(rows[1]["output"], "<gold1/>")

    def test_rft_uses_only_deduplicated_verified_texts(self):
        rows = self._rows(1)
        rollout = self._rollout(["<w/>", "<w/>", "bad", "<w/>"],
                                [1.0, 1.0, 0.0, 1.0])
        train_rows, stats = build_train_rows(
            rows, rollout, 4, "rft", 16, random.Random(0))
        self.assertEqual(stats["n_replaced"], 1)
        self.assertEqual(train_rows[0]["output"], "<w/>")

    def test_rft_respects_max_replaced_cap(self):
        rows = self._rows(3)
        rollout = self._rollout(["<w0/>", "bad", "<w1/>", "bad", "<w2/>", "bad"],
                                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        train_rows, stats = build_train_rows(
            rows, rollout, 2, "rft", 1, random.Random(0))
        self.assertEqual(stats["n_replaced"], 1)
        self.assertEqual(stats["batch_slots"], 3)
        replaced = [k for k, r in enumerate(train_rows)
                    if r["output"] != f"<gold{k}/>"]
        self.assertEqual(len(replaced), 1)

    def test_all_mastered_falls_back_to_all_gold(self):
        rows = self._rows(2)
        rollout = self._rollout(["<a/>", "<b/>"], [1.0, 1.0])
        train_rows, stats = build_train_rows(
            rows, rollout, 1, "rft", 16, random.Random(0))
        self.assertEqual(train_rows, rows)
        self.assertTrue(stats["mastered_fallback"])
        self.assertEqual(stats["n_mastered_skipped"], 2)

    def test_invalid_mix_is_rejected(self):
        rows = self._rows(1)
        rollout = self._rollout(["<w/>"], [1.0])
        with self.assertRaises(ValueError):
            build_train_rows(rows, rollout, 1, "dpo", 16, random.Random(0))


class TestSupervisedTokenLogprobs(unittest.TestCase):
    def test_mask_selects_only_supervised_shifted_positions(self):
        logits = torch.zeros((1, 5, 7))
        labels = torch.tensor([[10, -100, 3, 4, -100]])
        token_logp, mask = _supervised_token_logprobs(logits, labels)
        self.assertEqual(token_logp.shape, (1, 4))
        self.assertEqual(mask.tolist(), [[False, True, True, False]])

    def test_k3_is_zero_for_identical_distributions(self):
        labels = torch.tensor([[10, 3, 4]])
        logits = torch.zeros((1, 3, 8))
        token_logp, mask = _supervised_token_logprobs(logits, labels)
        log_ratio = token_logp - token_logp
        k3 = log_ratio.exp() - log_ratio - 1.0
        self.assertAlmostEqual(float((k3 * mask).sum() / mask.sum()), 0.0)


class TestTrainingText(unittest.TestCase):
    def test_keeps_model_preamble_and_adds_newline(self):
        raw = '<?xml version="1.0"?>\n<root BTCPP_format="4"><BehaviorTree ID="M"/></root>\njunk'
        clean = '<root BTCPP_format="4"><BehaviorTree ID="M"/></root>'
        self.assertEqual(
            _training_text(raw, clean),
            '<?xml version="1.0"?>\n' + clean + "\n")

    def test_without_root_prefix_is_empty(self):
        clean = "<root><BehaviorTree ID=\"M\"/></root>"
        self.assertEqual(_training_text("garbage", clean), clean + "\n")


class TestStrictReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = load_jsonl("outputs/dataset/train.jsonl")[0]

    def test_gold_tree_succeeds(self):
        reward, reason = _strict_reward(self.row, self.row["output"])
        self.assertEqual(reward, 1.0)
        self.assertEqual(reason, "success")

    def test_malformed_tree_is_zero(self):
        reward, reason = _strict_reward(self.row, "not xml")
        self.assertEqual(reward, 0.0)
        self.assertTrue(reason.startswith("parse:"))

    def test_executable_but_wrong_tree_is_zero(self):
        # A syntactically valid tree with a missing required action cannot
        # satisfy the task and must never receive a positive curator reward.
        xml = '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence/></BehaviorTree></root>'
        reward, reason = _strict_reward(self.row, xml)
        self.assertEqual(reward, 0.0)
        self.assertNotEqual(reason, "success")


class _NextTokenModel:
    """Returns a high logit for the actual next token at each position."""

    def __call__(self, input_ids, attention_mask, use_cache=False):
        class Output:
            pass
        logits = torch.full((*input_ids.shape, 32), -20.0)
        for row in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1] - 1):
                logits[row, pos, int(input_ids[row, pos + 1])] = 20.0
        Output.logits = logits
        return Output()


class TestContinuationLogprob(unittest.TestCase):
    def test_first_token_and_first_eos_only(self):
        # Two left-padded examples. The suffixes contain an EOS followed by a
        # token; only the first EOS and tokens before it belong to the score.
        ids = torch.tensor([[0, 0, 4, 5, 6, 1, 7],
                            [0, 8, 9, 1, 10, 11, 12]])
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 1, 1, 1]])
        values = _continuation_logprob(_NextTokenModel(), ids, mask,
                                       prompt_width=3, eos_id=1)
        chunked = _continuation_logprob(_NextTokenModel(), ids, mask,
                                        prompt_width=3, eos_id=1, batch_size=1)
        # Each selected token is predicted with ~0 log loss. If the first
        # generated token were shifted out, or post-EOS tokens were included,
        # this value would be materially negative.
        self.assertTrue(torch.all(values > -1e-3), values)
        self.assertTrue(torch.allclose(values, chunked), (values, chunked))


class TestRecoveryState(unittest.TestCase):
    def test_latest_checkpoint_requires_complete_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("checkpoint-5", "recovery-8", "recovery-9"):
                path = root / name
                path.mkdir()
                (path / "adapter_config.json").write_text("{}")
            (root / "checkpoint-5" / "trainer_state.pt").touch()
            (root / "recovery-8" / "trainer_state.pt").touch()
            self.assertEqual(_latest_resume_checkpoint(root).name,
                             "recovery-8")

    def test_training_state_round_trip(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, (step + 1) / 4))
        curator = TabularOSMD(3, exploration_floor=0.1, eta=0.2)
        detector = BlockVariationDetector(block_size=2)
        rng = random.Random(17)
        signature = {"protocol": "test"}
        curator.update({1: 0.4})
        detector.update(0.25)
        rng.random()
        state = _training_state(
            Namespace(recovery_every=1), signature, 7, 0.25,
            optimizer, scheduler, curator, detector,
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            Counter({1: 2}), Counter({0: 3}), [{"step": 7}], rng)

        restored_curator = TabularOSMD(
            3, exploration_floor=0.1, eta=0.2)
        restored_detector = BlockVariationDetector(block_size=2)
        restored_rng = random.Random(99)
        baseline, rows, arms, reports, start, loss = _restore_training_state(
            state, signature, optimizer, scheduler, restored_curator,
            restored_detector, restored_rng)
        self.assertEqual(start, 8)
        self.assertEqual(loss, 0.25)
        self.assertTrue(torch.equal(baseline, state["baseline"]))
        self.assertEqual(rows, Counter({1: 2}))
        self.assertEqual(arms, Counter({0: 3}))
        self.assertEqual(reports, [{"step": 7}])
        self.assertTrue(torch.equal(restored_curator.weights,
                                    state["curator_weights"]))
        expected_rng = random.Random()
        expected_rng.setstate(state["local_rng_state"])
        self.assertEqual(restored_rng.random(), expected_rng.random())


if __name__ == "__main__":
    unittest.main()
