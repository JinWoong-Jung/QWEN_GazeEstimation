from __future__ import annotations

import math
import unittest

import torch

from model.utils.rl_utils import (
    build_answer_span_mask,
    compute_grpo_loss,
    compute_object_reward,
    compute_point_reward,
    compute_token_logprobs_sum,
    compute_total_reward,
    group_normalize_advantages,
)


# ---------------------------------------------------------------------------
# compute_point_reward
# ---------------------------------------------------------------------------

class TestComputePointReward(unittest.TestCase):
    def test_zero_l2_approaches_one(self):
        self.assertAlmostEqual(compute_point_reward(0.0, beta=10.0), 1.0, places=9)

    def test_large_l2_approaches_zero(self):
        self.assertLess(compute_point_reward(10.0, beta=10.0), 1e-4)

    def test_monotone_decreasing(self):
        for beta in (5.0, 10.0):
            r0 = compute_point_reward(0.0, beta)
            r1 = compute_point_reward(0.1, beta)
            r2 = compute_point_reward(0.5, beta)
            self.assertGreater(r0, r1)
            self.assertGreater(r1, r2)

    def test_negative_l2_clamped_to_zero(self):
        self.assertAlmostEqual(compute_point_reward(-1.0, beta=10.0), 1.0, places=9)


# ---------------------------------------------------------------------------
# compute_object_reward
# ---------------------------------------------------------------------------

class TestComputeObjectReward(unittest.TestCase):
    def test_exact_match_returns_one(self):
        self.assertEqual(compute_object_reward(3, [1, 3, 5]), 1.0)

    def test_no_match_returns_zero(self):
        self.assertEqual(compute_object_reward(9, [1, 3, 5]), 0.0)

    def test_none_pred_returns_zero(self):
        self.assertEqual(compute_object_reward(None, [1, 2]), 0.0)

    def test_empty_gt_returns_zero(self):
        self.assertEqual(compute_object_reward(1, []), 0.0)

    def test_single_gt_match(self):
        self.assertEqual(compute_object_reward(7, [7]), 1.0)


# ---------------------------------------------------------------------------
# compute_total_reward
# ---------------------------------------------------------------------------

class TestComputeTotalReward(unittest.TestCase):
    def _gt(self) -> torch.Tensor:
        return torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    def _kwargs(self) -> dict:
        return dict(
            reward_point_weight=1.0,
            reward_object_weight=0.75,
            reward_joint_bonus=0.25,
            reward_extra_penalty=0.5,
            reward_point_beta=10.0,
        )

    def test_invalid_format_returns_minus_one(self):
        parsed = {"valid_format": False, "has_extra_text": False,
                  "point_xy": None, "object_id": None}
        result = compute_total_reward(parsed, self._gt(), [1])
        self.assertEqual(result["reward_total"], -1.0)
        self.assertFalse(result["valid_format"])

    def test_valid_correct_both_gives_high_reward(self):
        parsed = {"valid_format": True, "has_extra_text": False,
                  "point_xy": (0.5, 0.5), "object_id": 1}
        result = compute_total_reward(parsed, self._gt(), [1], **self._kwargs())
        self.assertTrue(result["valid_format"])
        self.assertGreater(result["reward_point"], 0.99)
        self.assertEqual(result["reward_object"], 1.0)
        self.assertEqual(result["reward_joint"], 1.0)
        self.assertGreater(result["reward_total"], 1.5)

    def test_extra_text_penalty_applied(self):
        kw = self._kwargs()
        parsed_extra = {"valid_format": True, "has_extra_text": True,
                        "point_xy": (0.5, 0.5), "object_id": 1}
        parsed_clean = {**parsed_extra, "has_extra_text": False}
        r_extra = compute_total_reward(parsed_extra, self._gt(), [1], **kw)
        r_clean = compute_total_reward(parsed_clean, self._gt(), [1], **kw)
        self.assertLess(r_extra["reward_total"], r_clean["reward_total"])
        self.assertAlmostEqual(
            r_clean["reward_total"] - r_extra["reward_total"], 0.5, places=5
        )

    def test_wrong_object_no_joint_bonus(self):
        parsed = {"valid_format": True, "has_extra_text": False,
                  "point_xy": (0.5, 0.5), "object_id": 99}
        result = compute_total_reward(parsed, self._gt(), [1], **self._kwargs())
        self.assertEqual(result["reward_object"], 0.0)
        self.assertEqual(result["reward_joint"], 0.0)

    def test_none_gt_points_gives_zero_point_reward(self):
        parsed = {"valid_format": True, "has_extra_text": False,
                  "point_xy": (0.5, 0.5), "object_id": 1}
        result = compute_total_reward(parsed, None, [1], **self._kwargs())
        self.assertEqual(result["reward_point"], 0.0)


# ---------------------------------------------------------------------------
# group_normalize_advantages
# ---------------------------------------------------------------------------

class TestGroupNormalizeAdvantages(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(group_normalize_advantages([]), [])

    def test_all_same_reward_gives_zero_advantages(self):
        adv = group_normalize_advantages([1.0, 1.0, 1.0, 1.0])
        for a in adv:
            self.assertAlmostEqual(a, 0.0, places=6)

    def test_normalized_mean_near_zero(self):
        adv = group_normalize_advantages([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(adv) / len(adv), 0.0, places=6)

    def test_monotone_order_preserved(self):
        adv = group_normalize_advantages([0.0, 0.5, 1.0, 2.0])
        for i in range(len(adv) - 1):
            self.assertLess(adv[i], adv[i + 1])


# ---------------------------------------------------------------------------
# build_answer_span_mask
# ---------------------------------------------------------------------------

class TestBuildAnswerSpanMask(unittest.TestCase):
    def test_masks_positions_after_prompt(self):
        ids = torch.zeros(2, 10, dtype=torch.long)
        mask = build_answer_span_mask(ids, prompt_len=5, pad_token_id=None)
        self.assertEqual(mask.shape, (2, 10))
        self.assertFalse(mask[:, :5].any().item())
        self.assertTrue(mask[:, 5:].all().item())

    def test_pad_tokens_excluded(self):
        ids = torch.zeros(1, 8, dtype=torch.long)  # all zeros = pad token
        mask = build_answer_span_mask(ids, prompt_len=4, pad_token_id=0)
        self.assertFalse(mask[0, 4:].any().item())

    def test_prompt_len_at_seq_end_gives_empty_mask(self):
        ids = torch.zeros(1, 5, dtype=torch.long)
        mask = build_answer_span_mask(ids, prompt_len=5, pad_token_id=None)
        self.assertFalse(mask.any().item())


# ---------------------------------------------------------------------------
# compute_token_logprobs_sum
# ---------------------------------------------------------------------------

class TestComputeTokenLogprobsSum(unittest.TestCase):
    def test_returns_correct_shapes(self):
        B, L, V = 2, 8, 32
        logits = torch.randn(B, L, V)
        ids = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, 4:] = True
        lp_sum, n_tok = compute_token_logprobs_sum(logits, ids, mask)
        self.assertEqual(lp_sum.shape, (B,))
        self.assertEqual(n_tok.shape, (B,))

    def test_n_tokens_matches_mask_after_causal_shift(self):
        # mask[:, 5:] = True → positions 5-9 (5 entries).
        # After shift (mask[:, 1:]), positions 5-9 remain True → 5 tokens.
        B, L, V = 3, 10, 16
        logits = torch.randn(B, L, V)
        ids = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, 5:] = True
        _, n_tok = compute_token_logprobs_sum(logits, ids, mask)
        self.assertTrue((n_tok == 5.0).all().item())

    def test_zero_mask_gives_zero_logprob_and_zero_ntokens(self):
        B, L, V = 2, 6, 8
        logits = torch.randn(B, L, V)
        ids = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        lp_sum, n_tok = compute_token_logprobs_sum(logits, ids, mask)
        self.assertTrue((lp_sum == 0.0).all().item())
        self.assertTrue((n_tok == 0.0).all().item())

    def test_logprob_sum_is_nonpositive(self):
        B, L, V = 2, 8, 64
        logits = torch.randn(B, L, V)
        ids = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        lp_sum, _ = compute_token_logprobs_sum(logits, ids, mask)
        self.assertTrue((lp_sum <= 0.0).all().item())


# ---------------------------------------------------------------------------
# compute_grpo_loss
# ---------------------------------------------------------------------------

class TestComputeGrpoLoss(unittest.TestCase):
    def _inputs(self, B: int = 4):
        lp = torch.zeros(B)
        n = torch.ones(B) * 5.0
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        return lp, n, adv

    def test_returns_finite_loss(self):
        lp, n, adv = self._inputs()
        loss, _ = compute_grpo_loss(lp, lp.clone(), lp.clone(), n, adv)
        self.assertTrue(math.isfinite(float(loss)))

    def test_ratio_one_pg_loss_equals_neg_adv_mean(self):
        # ratio=1 → pg_loss = -mean(A)
        lp, n, adv = self._inputs()
        _, stats = compute_grpo_loss(lp, lp.clone(), lp.clone(), n, adv,
                                     clip_eps=0.2, kl_beta=0.0)
        self.assertAlmostEqual(stats["pg_loss"], -float(adv.mean()), places=4)

    def test_kl_zero_when_new_equals_ref(self):
        lp, n, adv = self._inputs()
        _, stats = compute_grpo_loss(lp, lp.clone(), lp.clone(), n, adv,
                                     clip_eps=0.2, kl_beta=0.02)
        self.assertAlmostEqual(stats["kl_mean"], 0.0, places=6)

    def test_loss_allows_backward(self):
        B = 4
        new_lp = torch.zeros(B, requires_grad=True)
        old_lp = torch.zeros(B)
        ref_lp = torch.zeros(B)
        n = torch.ones(B) * 5.0
        adv = torch.randn(B)
        loss, _ = compute_grpo_loss(new_lp, old_lp, ref_lp, n, adv)
        loss.backward()
        self.assertIsNotNone(new_lp.grad)

    def test_stats_keys_present(self):
        lp, n, adv = self._inputs()
        _, stats = compute_grpo_loss(lp, lp.clone(), lp.clone(), n, adv)
        for key in ("pg_loss", "kl_mean", "ratio_mean", "ratio_std"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
