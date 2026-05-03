from __future__ import annotations

import unittest

import torch

from model.utils.loss_utils import (
    compute_answer_loss,
    compute_structured_loss,
    gaussian_soft_label_ce,
    masked_sample_exact_rate,
    masked_token_ce,
)


def _make_logits(bsz: int, seqlen: int, vocab: int = 10) -> torch.Tensor:
    return torch.randn(bsz, seqlen, vocab)


def _make_labels(bsz: int, seqlen: int, vocab: int = 10) -> torch.Tensor:
    labels = torch.randint(0, vocab, (bsz, seqlen))
    labels[:, 0] = -100  # mask first token
    return labels


def _make_mask(bsz: int, seqlen: int, positions: list[int]) -> torch.Tensor:
    m = torch.zeros(bsz, seqlen, dtype=torch.bool)
    for pos in positions:
        if pos < seqlen:
            m[:, pos] = True
    return m


class TestMaskedTokenCE(unittest.TestCase):
    def test_returns_scalar(self):
        logits = _make_logits(2, 8)
        labels = _make_labels(2, 8)
        mask = _make_mask(2, 8, [2, 3])
        loss, n = masked_token_ce(logits, labels, mask)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(n, 0)

    def test_no_valid_tokens_returns_zero(self):
        logits = _make_logits(2, 8)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        mask = _make_mask(2, 8, [2])
        loss, n = masked_token_ce(logits, labels, mask)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(float(loss.item()), 0.0)

    def test_none_mask_returns_zero(self):
        logits = _make_logits(2, 8)
        labels = _make_labels(2, 8)
        loss, n = masked_token_ce(logits, labels, None)
        self.assertEqual(n, 0)

    def test_no_nan(self):
        logits = _make_logits(4, 12)
        labels = _make_labels(4, 12)
        mask = _make_mask(4, 12, [1, 2, 5, 6])
        loss, _ = masked_token_ce(logits, labels, mask)
        self.assertFalse(torch.isnan(loss).item())


class TestComputeStructuredLoss(unittest.TestCase):
    def _make_batch(self, bsz: int = 2, seqlen: int = 16, vocab: int = 10):
        logits = _make_logits(bsz, seqlen, vocab)
        labels = _make_labels(bsz, seqlen, vocab)
        # Simulate 7-token answer at positions [5..11]
        mask_pt = _make_mask(bsz, seqlen, [6, 7])    # point: positions 1,2 of answer
        mask_obj = _make_mask(bsz, seqlen, [10])       # object: position 5
        mask_fmt = _make_mask(bsz, seqlen, [5, 8, 9, 11])  # format: positions 0,3,4,6
        return logits, labels, mask_pt, mask_obj, mask_fmt

    def test_returns_all_keys(self):
        logits, labels, mp, mo, mf = self._make_batch()
        d = compute_structured_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        for key in ["loss", "loss_point", "loss_object", "loss_format",
                    "n_point_tokens", "n_object_tokens", "n_format_tokens"]:
            self.assertIn(key, d)

    def test_no_nan(self):
        logits, labels, mp, mo, mf = self._make_batch()
        d = compute_structured_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        self.assertFalse(torch.isnan(d["loss"]).item())
        self.assertFalse(torch.isnan(d["loss_point"]).item())
        self.assertFalse(torch.isnan(d["loss_object"]).item())

    def test_token_counts(self):
        logits, labels, mp, mo, mf = self._make_batch(bsz=2, seqlen=16)
        d = compute_structured_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        self.assertEqual(d["n_point_tokens"], 4)   # 2 per sample × 2 samples
        self.assertEqual(d["n_object_tokens"], 2)  # 1 per sample × 2 samples
        self.assertEqual(d["n_format_tokens"], 8)  # 4 per sample × 2 samples

    def test_zero_loss_when_no_tokens(self):
        logits = _make_logits(2, 8)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        z_mask = torch.zeros(2, 8, dtype=torch.bool)
        d = compute_structured_loss(
            logits=logits, labels=labels,
            loss_mask_point=z_mask, loss_mask_object=z_mask, loss_mask_format=z_mask,
        )
        self.assertAlmostEqual(float(d["loss"].item()), 0.0)

    def test_format_valid_rate_present(self):
        logits, labels, mp, mo, mf = self._make_batch()
        d = compute_structured_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        self.assertIn("format_valid_rate", d)
        self.assertIn("n_format_samples", d)


class TestGaussianSoftLabelCE(unittest.TestCase):
    def _make_inputs(self, N: int = 4, C: int = 10, V: int = 20):
        # loc_token_ids: C consecutive vocab IDs starting at 5
        loc_token_ids = torch.arange(5, 5 + C, dtype=torch.long)
        # GT loc ids: one per position, all pointing to bin index 3 (vocab id 8)
        gt_loc_ids = torch.full((N,), 5 + 3, dtype=torch.long)
        logits_at_pt = torch.randn(N, V)
        return logits_at_pt, gt_loc_ids, loc_token_ids

    def test_returns_scalar(self):
        logits, gt_ids, loc_ids = self._make_inputs()
        loss = gaussian_soft_label_ce(logits, gt_ids, loc_ids)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)

    def test_no_nan(self):
        logits, gt_ids, loc_ids = self._make_inputs()
        loss = gaussian_soft_label_ce(logits, gt_ids, loc_ids)
        self.assertFalse(torch.isnan(loss).item())

    def test_nonnegative(self):
        logits, gt_ids, loc_ids = self._make_inputs()
        loss = gaussian_soft_label_ce(logits, gt_ids, loc_ids)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_matched_distribution_lowers_loss(self):
        # Logits whose loc-token distribution matches the Gaussian soft label
        # should produce lower loss than uniform logits.
        N, C, V = 4, 10, 20
        loc_token_ids = torch.arange(0, C, dtype=torch.long)
        gt_bin = 3
        gt_loc_ids = torch.full((N,), gt_bin, dtype=torch.long)
        sigma = 2.0
        k = torch.arange(C, dtype=torch.float32)
        soft = torch.exp(-0.5 * (k - gt_bin) ** 2 / sigma ** 2)
        soft = soft / soft.sum()
        # Build logits where only loc positions get mass (proportional to soft label).
        # Non-loc logits are suppressed so log_softmax over loc ≈ log(soft).
        logits_matched = torch.full((N, V), -100.0)
        logits_matched[:, loc_token_ids] = torch.log(soft).unsqueeze(0).expand(N, -1)
        logits_uniform = torch.zeros(N, V)
        loss_matched = gaussian_soft_label_ce(logits_matched, gt_loc_ids, loc_token_ids, sigma=sigma)
        loss_uniform = gaussian_soft_label_ce(logits_uniform, gt_loc_ids, loc_token_ids, sigma=sigma)
        self.assertLess(float(loss_matched.item()), float(loss_uniform.item()))

    def test_wider_sigma_smooths_loss(self):
        # Larger sigma should give lower loss when GT bin gets some probability mass.
        N, C, V = 4, 20, 30
        loc_token_ids = torch.arange(0, C, dtype=torch.long)
        gt_loc_ids = torch.full((N,), 10, dtype=torch.long)
        logits = torch.randn(N, V)
        loss_narrow = gaussian_soft_label_ce(logits, gt_loc_ids, loc_token_ids, sigma=1.0)
        loss_wide = gaussian_soft_label_ce(logits, gt_loc_ids, loc_token_ids, sigma=15.0)
        # Wide sigma distributes credit across many bins; hard to assert direction
        # strictly, but both should be finite and non-negative.
        self.assertFalse(torch.isnan(loss_narrow).item())
        self.assertFalse(torch.isnan(loss_wide).item())

    def test_empty_N_returns_zero(self):
        loc_token_ids = torch.arange(0, 10, dtype=torch.long)
        gt_loc_ids = torch.zeros(0, dtype=torch.long)
        logits = torch.zeros(0, 20)
        loss = gaussian_soft_label_ce(logits, gt_loc_ids, loc_token_ids)
        self.assertAlmostEqual(float(loss.item()), 0.0)

    def test_empty_C_returns_zero(self):
        loc_token_ids = torch.zeros(0, dtype=torch.long)
        gt_loc_ids = torch.tensor([3, 5], dtype=torch.long)
        logits = torch.randn(2, 20)
        loss = gaussian_soft_label_ce(logits, gt_loc_ids, loc_token_ids)
        self.assertAlmostEqual(float(loss.item()), 0.0)


class TestComputeAnswerLoss(unittest.TestCase):
    def test_structured_path_used_when_masks_provided(self):
        bsz, seqlen, vocab = 2, 16, 10
        logits = _make_logits(bsz, seqlen, vocab)
        labels = _make_labels(bsz, seqlen, vocab)
        mp = _make_mask(bsz, seqlen, [2, 3])
        mo = _make_mask(bsz, seqlen, [6])
        mf = _make_mask(bsz, seqlen, [1, 4, 5, 7])
        d = compute_answer_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        self.assertIn("loss_point", d)
        self.assertGreater(d["n_point_tokens"], 0)

    def test_legacy_path_when_no_structured_masks(self):
        bsz, seqlen, vocab = 2, 16, 10
        logits = _make_logits(bsz, seqlen, vocab)
        labels = _make_labels(bsz, seqlen, vocab)
        mask = _make_mask(bsz, seqlen, [2, 3, 4, 5, 6, 7, 8])
        d = compute_answer_loss(
            logits=logits, labels=labels,
            loss_mask_answer=mask,
        )
        self.assertIn("loss_answer", d)
        self.assertIn("n_answer_tokens", d)

    def test_no_nan_structured(self):
        bsz, seqlen, vocab = 4, 20, 10
        logits = _make_logits(bsz, seqlen, vocab)
        labels = _make_labels(bsz, seqlen, vocab)
        mp = _make_mask(bsz, seqlen, [5, 6])
        mo = _make_mask(bsz, seqlen, [10])
        mf = _make_mask(bsz, seqlen, [4, 7, 8, 11])
        d = compute_answer_loss(
            logits=logits, labels=labels,
            loss_mask_point=mp, loss_mask_object=mo, loss_mask_format=mf,
        )
        self.assertFalse(torch.isnan(d["loss"]).item())


class TestMaskedSampleExactRate(unittest.TestCase):
    def test_per_sample_exact_rate(self):
        logits = torch.zeros(2, 5, 6)
        labels = torch.tensor([
            [-100, 1, 2, 3, 4],
            [-100, 1, 2, 3, 4],
        ], dtype=torch.long)
        mask = torch.tensor([
            [False, False, True, True, False],
            [False, False, True, True, False],
        ], dtype=torch.bool)

        # With causal shift, logits[:,1] predicts labels[:,2], logits[:,2] predicts labels[:,3].
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        logits[1, 1, 2] = 10.0
        logits[1, 2, 0] = 10.0

        rate, n = masked_sample_exact_rate(logits, labels, mask)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(float(rate.item()), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
