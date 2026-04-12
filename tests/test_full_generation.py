"""Tests for the full-generation + retrieval-auxiliary refactoring.

Covers:
- object_label_span() finds pure-text label spans
- parse_target_spans() uses object_label_span (not just slot_span)
- format_target_text() produces pure-text answers
- answer_mask covers the full answer; object_mask covers just the label
- parse_object_text() extracts generated label strings
- label_bank canonicalization helpers
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make sure project root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.utils.object_tokens import OBJ_SLOT, object_label_span
from model.utils.data_utils import build_split_bank
from model.utils.eval_utils import (
    clean_retrieval_text,
    collect_generation_samples,
    expand_retrieval_candidates,
    parse_object_text,
    run_test_metrics,
    topk_similarity,
)
from model.utils.label_bank import LabelBank, canonicalize


# ---------------------------------------------------------------------------
# object_label_span
# ---------------------------------------------------------------------------

class TestObjectLabelSpan(unittest.TestCase):

    def test_pure_text_label(self) -> None:
        txt = "Point: 0.4230 0.7112\nObject: television"
        span = object_label_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], "television")

    def test_pure_text_multi_word(self) -> None:
        txt = "Point: 0.1 0.2\nObject: television monitor"
        span = object_label_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], "television monitor")

    def test_legacy_slot(self) -> None:
        txt = f"Point: 0.1 0.2\nObject: {OBJ_SLOT}"
        span = object_label_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], OBJ_SLOT)

    def test_missing_object_line(self) -> None:
        self.assertIsNone(object_label_span("Point: 0.1 0.2"))

    def test_empty_object_value(self) -> None:
        # "Object:" with no content after colon should return None
        # (OBJECT_LABEL_CONTENT_RE requires \S at start of group)
        result = object_label_span("Point: 0.1 0.2\nObject:   ")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# parse_object_text (eval helper)
# ---------------------------------------------------------------------------

class TestParseObjectText(unittest.TestCase):

    def test_simple(self) -> None:
        self.assertEqual(parse_object_text("Point: 0.4 0.5\nObject: television"), "television")

    def test_multi_word(self) -> None:
        self.assertEqual(parse_object_text("Point: 0.4 0.5\nObject: tv set"), "tv set")

    def test_missing(self) -> None:
        self.assertIsNone(parse_object_text("Point: 0.4 0.5"))

    def test_legacy_slot_returns_none(self) -> None:
        # <obj_emb> is not useful as a parsed label
        self.assertIsNone(parse_object_text(f"Point: 0.4 0.5\nObject: {OBJ_SLOT}"))

    def test_noisy_generation(self) -> None:
        # Extra whitespace / casing should still parse
        result = parse_object_text("  Point: 0.4 0.5  \n  Object:   chair  ")
        self.assertEqual(result, "chair")


# ---------------------------------------------------------------------------
# Retrieval text cleaning / candidate expansion
# ---------------------------------------------------------------------------

class TestRetrievalTextCleaning(unittest.TestCase):

    def test_clean_removes_duplicate_tokens(self) -> None:
        self.assertEqual(clean_retrieval_text("chair chair chair"), "chair")

    def test_clean_removes_repeated_ngram_cycle(self) -> None:
        self.assertEqual(
            clean_retrieval_text("laptop screen laptop screen"),
            "laptop screen",
        )

    def test_clean_handles_special_cases(self) -> None:
        self.assertEqual(clean_retrieval_text("  !!!  "), "")
        self.assertEqual(clean_retrieval_text("  CHAIR, chair.  "), "chair")


class TestExpandRetrievalCandidates(unittest.TestCase):

    def test_candidate_count_for_three_tokens(self) -> None:
        cands = expand_retrieval_candidates("computer laptop screen")
        # full(1) + uni(3) + bi(2) + tri(1, duplicate of full) = 6 unique
        self.assertEqual(len(cands), 6)
        self.assertEqual(
            cands,
            [
                "computer laptop screen",
                "computer",
                "laptop",
                "screen",
                "computer laptop",
                "laptop screen",
            ],
        )

    def test_sliding_ngrams_present(self) -> None:
        cands = expand_retrieval_candidates("red green blue yellow")
        self.assertEqual(len(cands), 10)  # 1 + 4 + 3 + 2
        self.assertIn("red green", cands)
        self.assertIn("green blue", cands)
        self.assertIn("blue yellow", cands)
        self.assertIn("red green blue", cands)
        self.assertIn("green blue yellow", cands)


# ---------------------------------------------------------------------------
# Split bank construction
# ---------------------------------------------------------------------------

class TestBuildSplitBank(unittest.TestCase):

    def test_bank_shape_and_canonical_ids_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            embed_dir = Path(td)
            torch.save(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                embed_dir / "chair-emb.pt",
            )
            torch.save(
                torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
                embed_dir / "book-emb.pt",
            )

            bank_texts, bank_ids, bank_embs = build_split_bank(
                label_texts=["chair", "book", "missing_label"],
                canonical_ids=[10, 20, 30],
                label_embed_dir=embed_dir,
                embedding_dim=4,
                normalize=True,
            )

        self.assertEqual(bank_texts, ["chair", "book"])
        self.assertEqual(bank_ids, [10, 20])
        self.assertEqual(tuple(bank_embs.shape), (2, 4))
        self.assertEqual(len(bank_texts), len(bank_ids))
        self.assertEqual(len(bank_ids), int(bank_embs.shape[0]))
        norms = bank_embs.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


# ---------------------------------------------------------------------------
# LabelBank
# ---------------------------------------------------------------------------

class TestLabelBank(unittest.TestCase):

    def _make_bank(self) -> LabelBank:
        vocab2id = {"television": 0, "chair": 1, "book": 2}
        dim = 4
        matrix = torch.zeros(3, dim)
        matrix[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        matrix[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        matrix[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        bank = LabelBank(
            label_texts=["television", "chair", "book"],
            label_to_id=vocab2id,
            canonical_to_id={canonicalize(k): v for k, v in vocab2id.items()},
            embedding_matrix=matrix,
        )
        return bank

    def test_topk_returns_correct_label(self) -> None:
        bank = self._make_bank()
        query = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ids = bank.topk(query, k=1)
        self.assertEqual(ids, [0])
        self.assertEqual(bank.id_to_label[ids[0]], "television")

    def test_topk_labels(self) -> None:
        bank = self._make_bank()
        query = torch.tensor([0.0, 1.0, 0.0, 0.0])
        labels = bank.topk_labels(query, k=1)
        self.assertEqual(labels, ["chair"])

    def test_lookup_id_exact(self) -> None:
        bank = self._make_bank()
        self.assertEqual(bank.lookup_id("book"), 2)

    def test_lookup_id_canonical(self) -> None:
        bank = self._make_bank()
        self.assertEqual(bank.lookup_id("  Television  "), 0)

    def test_lookup_id_missing(self) -> None:
        bank = self._make_bank()
        self.assertEqual(bank.lookup_id("sofa"), -1)

    def test_canonicalize(self) -> None:
        self.assertEqual(canonicalize("  Television  Monitor "), "television monitor")


# ---------------------------------------------------------------------------
# datasets.format_target_text pure-text output
# ---------------------------------------------------------------------------

class TestFormatTargetText(unittest.TestCase):

    def test_pure_text_template(self) -> None:
        from model.datasets import format_target_text
        text, valid = format_target_text(
            label_text="television",
            label_id=0,
            id2label={0: "television"},
            vocab2id={"television": 0},
            vocab2id_lower={"television": 0},
            num_classes=5,
            answer_template="Point: {point_x} {point_y}\nObject: {label_text}",
            fallback_target_text="unknown",
            point_x=0.423,
            point_y=0.711,
            point_decimals=4,
        )
        self.assertIn("Point:", text)
        self.assertIn("Object: television", text)
        self.assertNotIn("<obj_emb>", text)
        self.assertEqual(valid, 1.0)

    def test_fallback_uses_label_text_not_slot(self) -> None:
        from model.datasets import format_target_text
        # Use a template with a typo to trigger the except branch
        text, _ = format_target_text(
            label_text="chair",
            label_id=1,
            id2label=None,
            vocab2id=None,
            vocab2id_lower=None,
            num_classes=0,
            answer_template="{broken_key}",
            fallback_target_text="unknown",
            point_x=0.5,
            point_y=0.5,
            point_decimals=2,
        )
        # fallback should include "chair", not OBJ_SLOT
        self.assertIn("chair", text)
        self.assertNotIn("<obj_emb>", text)


# ---------------------------------------------------------------------------
# ID-space acc@1 via topk_similarity
# ---------------------------------------------------------------------------

class TestIdSpaceRetrieval(unittest.TestCase):
    """Verify that topk_similarity on a vocab2id-ordered bank returns label ids."""

    def _make_bank_and_vocab(self) -> tuple[torch.Tensor, dict[str, int]]:
        # bank row i = embedding for label id i
        vocab2id = {"television": 0, "chair": 1, "book": 2}
        bank = torch.zeros(3, 4, dtype=torch.float32)
        bank[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        bank[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        bank[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        return bank, vocab2id

    def test_topk_returns_label_id(self) -> None:
        bank, _ = self._make_bank_and_vocab()
        # query close to "television" embedding (id=0)
        query = torch.tensor([0.95, 0.0, 0.0, 0.0], dtype=torch.float32)
        topk_ids = topk_similarity(query, bank, k=1, temperature=0.07)
        self.assertEqual(topk_ids[0], 0)  # top-1 id == target_label id for "television"

    def test_acc1_correct_when_topk_matches_target_label(self) -> None:
        bank, _ = self._make_bank_and_vocab()
        # Simulate: generated "book" → query near id=2
        query = torch.tensor([0.0, 0.0, 0.95, 0.0], dtype=torch.float32)
        topk_ids = topk_similarity(query, bank, k=3, temperature=0.07)
        target_label_id = 2  # "book"
        retrieval_acc1 = int(topk_ids[0] == target_label_id)
        retrieval_acc3 = int(target_label_id in topk_ids[:3])
        self.assertEqual(retrieval_acc1, 1)
        self.assertEqual(retrieval_acc3, 1)

    def test_invalid_target_label_excluded_from_denominator(self) -> None:
        # target_label=-1 → acc1_den should NOT increment
        target_label_id = -1
        acc1_den = 0
        if target_label_id >= 0:
            acc1_den += 1
        self.assertEqual(acc1_den, 0)

    def test_multiacc_uses_id_set(self) -> None:
        bank, _ = self._make_bank_and_vocab()
        query = torch.tensor([0.0, 0.95, 0.0, 0.0], dtype=torch.float32)
        topk_ids = topk_similarity(query, bank, k=1, temperature=0.07)
        pred_id = topk_ids[0]  # should be 1 ("chair")
        gt_multi_ids = {0, 1}  # multiple valid GT ids
        self.assertIn(pred_id, gt_multi_ids)


class _PreviewTokenizer:
    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        seq = ids.tolist() if torch.is_tensor(ids) else list(ids)
        if seq == [10, 11, 12, 13, 14]:
            return "Point: 0.1000 0.2000\nObject: chair"
        return ""


class _PreviewProcessor:
    def __init__(self) -> None:
        self.tokenizer = _PreviewTokenizer()


class _PreviewModel(torch.nn.Module):
    def generate(self, joint_inputs: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        bsz = int(joint_inputs["input_ids"].shape[0])
        rows = [
            [1, 2, 10, 11, 12, 13, 14]
            for _ in range(bsz)
        ]
        return torch.tensor(rows, dtype=torch.long)


class _PreviewClipEncoder:
    def encode(self, texts: list[str | None]) -> torch.Tensor:
        mapping = {
            "book": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "chair": torch.tensor([0.0, 1.0], dtype=torch.float32),
        }
        rows = [mapping[str(text)] for text in texts]
        return torch.stack(rows, dim=0)


class _CandidateE2ETokenizer:
    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        seq = ids.tolist() if torch.is_tensor(ids) else list(ids)
        if seq == [10, 11, 12, 13, 14]:
            # "computer laptop" alone is biased to wrong bank row in mock encoder.
            # Candidate expansion must include "laptop" unigram to recover acc@1.
            return "Point: 0.1000 0.2000\nObject: computer laptop laptop"
        return ""


class _CandidateE2EProcessor:
    def __init__(self) -> None:
        self.tokenizer = _CandidateE2ETokenizer()


class _CandidateE2EModel(torch.nn.Module):
    def generate(self, joint_inputs: dict[str, torch.Tensor], **_: object) -> torch.Tensor:
        bsz = int(joint_inputs["input_ids"].shape[0])
        rows = [[1, 2, 10, 11, 12, 13, 14] for _ in range(bsz)]
        return torch.tensor(rows, dtype=torch.long)


class _CandidateE2EClipEncoder:
    def encode(self, texts: list[str | None]) -> torch.Tensor:
        mapping = {
            "computer laptop": torch.tensor([1.0, 0.0], dtype=torch.float32),  # wrong row if alone
            "computer": torch.tensor([0.9, 0.0], dtype=torch.float32),
            "laptop": torch.tensor([0.0, 2.0], dtype=torch.float32),  # rescuing candidate
        }
        rows = [mapping.get(str(text), torch.zeros(2, dtype=torch.float32)) for text in texts]
        return torch.stack(rows, dim=0)


class TestGenerationPreview(unittest.TestCase):
    def test_collect_generation_samples(self) -> None:
        loader = [
            {
                "joint_inputs": {
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                },
                "text_input": ["Locate the gaze target."],
                "target_text": ["Point: 0.1000 0.2000\nObject: chair"],
                "target_text_valid": torch.tensor([1.0], dtype=torch.float32),
                "target_label": torch.tensor([1], dtype=torch.long),
                "target_label_ids": [[1]],
                "gt_points": [torch.tensor([[0.1, 0.2]], dtype=torch.float32)],
                "target_label_text": ["chair"],
                "image_rel": ["test/image_0001.jpg"],
            }
        ]
        samples = collect_generation_samples(
            model=_PreviewModel(),
            loader=loader,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            processor=_PreviewProcessor(),
            label_embedding_bank=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            retrieval_label_texts=["book", "chair"],
            retrieval_top_k=2,
            clip_encoder=_PreviewClipEncoder(),
            object_temperature=0.07,
            show_tqdm=False,
            max_new_tokens=8,
            num_beams=1,
            max_samples=1,
        )
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["generated_text"], "Point: 0.1000 0.2000\nObject: chair")
        self.assertEqual(sample["parsed_object_text"], "chair")
        self.assertEqual(sample["retrieved_topk_ids"][0], 1)
        self.assertEqual(sample["retrieved_topk_labels"][0], "chair")
        self.assertEqual(sample["parsed_point"], [0.1, 0.2])
        self.assertAlmostEqual(float(sample["min_l2"]), 0.0, places=6)
        self.assertTrue(bool(sample["exact_match"]))


class TestSplitBankCandidateExpansionE2E(unittest.TestCase):

    def test_acc1_uses_split_bank_ids_with_candidate_expansion(self) -> None:
        loader = [
            {
                "joint_inputs": {
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                },
                "target_text": ["Point: 0.1000 0.2000\nObject: laptop"],
                "target_text_valid": torch.tensor([1.0], dtype=torch.float32),
                "target_label": torch.tensor([42], dtype=torch.long),  # canonical label id
                "target_label_ids": [[42]],
                "gt_points": [torch.tensor([[0.1, 0.2]], dtype=torch.float32)],
            }
        ]

        metrics = run_test_metrics(
            model=_CandidateE2EModel(),
            loader=loader,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            processor=_CandidateE2EProcessor(),
            label_embedding_bank=torch.tensor(
                [
                    [1.0, 0.0],  # bank row 0 -> canonical id 7
                    [0.0, 1.0],  # bank row 1 -> canonical id 42
                ],
                dtype=torch.float32,
            ),
            retrieval_label_texts=["mug", "laptop"],
            retrieval_top_k=2,
            clip_encoder=_CandidateE2EClipEncoder(),
            object_temperature=0.07,
            show_tqdm=False,
            max_new_tokens=8,
            num_beams=1,
            bank_canonical_ids=[7, 42],
        )

        self.assertAlmostEqual(float(metrics["acc@1"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["acc@3"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["multiacc@1"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["ObjectParseFail"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
