from __future__ import annotations

import unittest

import torch

from model.utils.loss_utils import retrieval_ce_full_bank


class TestRetrievalLoss(unittest.TestCase):
    def test_full_bank_infonce_prefers_correct_label(self) -> None:
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([0, 1], dtype=torch.long)
        valid = torch.ones((2,), dtype=torch.float32)

        loss_good, n_good = retrieval_ce_full_bank(
            pred_object_emb=pred,
            target_label=target,
            target_object_valid=valid,
            label_embedding_bank=bank,
            temperature=0.07,
        )
        loss_bad, n_bad = retrieval_ce_full_bank(
            pred_object_emb=pred,
            target_label=torch.tensor([2, 2], dtype=torch.long),
            target_object_valid=valid,
            label_embedding_bank=bank,
            temperature=0.07,
        )
        self.assertEqual(n_good, 2)
        self.assertEqual(n_bad, 2)
        self.assertLess(float(loss_good.item()), float(loss_bad.item()))

    def test_full_bank_infonce_ignores_invalid_targets(self) -> None:
        bank = torch.eye(4, dtype=torch.float32)
        pred = torch.eye(4, dtype=torch.float32)[:2]
        target = torch.tensor([-1, 99], dtype=torch.long)
        valid = torch.tensor([1.0, 0.0], dtype=torch.float32)
        loss, n = retrieval_ce_full_bank(
            pred_object_emb=pred,
            target_label=target,
            target_object_valid=valid,
            label_embedding_bank=bank,
            temperature=0.07,
        )
        self.assertEqual(n, 0)
        self.assertEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
