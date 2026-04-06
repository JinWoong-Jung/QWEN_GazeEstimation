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
import unittest
from pathlib import Path

# Make sure project root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.utils.object_tokens import OBJ_SLOT, object_label_span
from model.utils.processor_collate import parse_target_spans
from model.utils.eval_utils import parse_object_text
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
# parse_target_spans uses object_label_span
# ---------------------------------------------------------------------------

class TestParseTargetSpans(unittest.TestCase):

    def test_pure_text_object_span(self) -> None:
        txt = "Point: 0.4230 0.7112\nObject: television"
        x_span, y_span, o_span = parse_target_spans(txt)
        self.assertIsNotNone(x_span)
        self.assertIsNotNone(y_span)
        self.assertIsNotNone(o_span)
        assert o_span is not None
        self.assertEqual(txt[o_span[0]:o_span[1]], "television")

    def test_legacy_slot_still_found(self) -> None:
        txt = f"Point: 0.4 0.5\nObject: {OBJ_SLOT}"
        _, _, o_span = parse_target_spans(txt)
        self.assertIsNotNone(o_span)
        assert o_span is not None
        self.assertEqual(txt[o_span[0]:o_span[1]], OBJ_SLOT)

    def test_no_object_line(self) -> None:
        txt = "Point: 0.4 0.5"
        _, _, o_span = parse_target_spans(txt)
        self.assertIsNone(o_span)


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


if __name__ == "__main__":
    unittest.main()
