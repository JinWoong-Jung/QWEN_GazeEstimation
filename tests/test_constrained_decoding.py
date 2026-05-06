from __future__ import annotations

import re
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch

from model.utils.eval_utils import (
    _append_token_to_joint,
    _loc_token_ids,
    _obj_token_ids,
    _single_token_id,
    constrained_generate_structured,
)
from model.utils.gaze_tokens import parse_structured_output_text


# ---------------------------------------------------------------------------
# Minimal mock tokenizer
# ---------------------------------------------------------------------------

def _make_tokenizer(coord_bins: int = 8, num_classes: int = 4) -> MagicMock:
    """Mock tokenizer whose encode/decode/batch_decode mirror gaze_tokens format."""
    loc_w = max(3, len(str(coord_bins - 1)))
    obj_w = max(3, len(str(num_classes - 1)))

    # token_str → fixed single int id
    _vocab: dict[str, int] = {}
    _id2tok: dict[int, str] = {}
    next_id = 100

    def _register(tok: str) -> int:
        nonlocal next_id
        if tok not in _vocab:
            _vocab[tok] = next_id
            _id2tok[next_id] = tok
            next_id += 1
        return _vocab[tok]

    _register("<|gaze_point|>")
    _register("<|gaze_object|>")
    for i in range(coord_bins):
        _register(f"<loc_{i:0{loc_w}d}>")
    for i in range(num_classes):
        _register(f"<obj_{i:0{obj_w}d}>")

    tok = MagicMock()
    tok.encode.side_effect = lambda s, add_special_tokens=False: (
        [_vocab[s]] if s in _vocab else [999]
    )
    tok.batch_decode.side_effect = lambda ids_list, skip_special_tokens=False: [
        "".join(_id2tok.get(int(i), "?") for i in row)
        for row in ids_list
    ]
    return tok


def _make_joint(bsz: int = 1, seq_len: int = 5, device: str = "cpu") -> dict[str, Any]:
    return {
        "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long, device=device),
        "attention_mask": torch.ones(bsz, seq_len, dtype=torch.long, device=device),
        "pixel_values": torch.zeros(bsz, 3, 16, 16, device=device),  # image tensor — must be unchanged
    }


def _make_model(vocab_size: int = 500) -> MagicMock:
    """Model whose forward always returns uniform logits, so argmax picks index 0."""
    def _forward(joint_inputs: dict, use_cache: bool = False) -> dict:
        bsz = int(joint_inputs["input_ids"].shape[0])
        logits = torch.zeros(bsz, 1, vocab_size)
        return {"logits": logits}

    return MagicMock(side_effect=_forward)


# ---------------------------------------------------------------------------
# Tests: _single_token_id
# ---------------------------------------------------------------------------

class TestSingleTokenId(unittest.TestCase):

    def test_raises_on_multi_token(self) -> None:
        tok = MagicMock()
        tok.encode.return_value = [1, 2]
        with self.assertRaises(ValueError):
            _single_token_id(tok, "<multi>")

    def test_raises_on_empty(self) -> None:
        tok = MagicMock()
        tok.encode.return_value = []
        with self.assertRaises(ValueError):
            _single_token_id(tok, "<empty>")

    def test_returns_int_for_single_token(self) -> None:
        tok = MagicMock()
        tok.encode.return_value = [42]
        result = _single_token_id(tok, "<ok>")
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)


# ---------------------------------------------------------------------------
# Tests: _loc_token_ids / _obj_token_ids
# ---------------------------------------------------------------------------

class TestTokenIdLists(unittest.TestCase):

    def test_loc_token_ids_length(self) -> None:
        coord_bins = 8
        tok = _make_tokenizer(coord_bins=coord_bins, num_classes=4)
        ids = _loc_token_ids(tok, coord_bins=coord_bins)
        self.assertEqual(len(ids), coord_bins)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_obj_token_ids_length(self) -> None:
        num_classes = 4
        tok = _make_tokenizer(coord_bins=8, num_classes=num_classes)
        ids = _obj_token_ids(tok, num_classes=num_classes)
        self.assertEqual(len(ids), num_classes)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_loc_token_ids_are_unique(self) -> None:
        tok = _make_tokenizer(coord_bins=8, num_classes=4)
        ids = _loc_token_ids(tok, coord_bins=8)
        self.assertEqual(len(ids), len(set(ids)))

    def test_obj_token_ids_are_unique(self) -> None:
        tok = _make_tokenizer(coord_bins=8, num_classes=4)
        ids = _obj_token_ids(tok, num_classes=4)
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# Tests: _append_token_to_joint
# ---------------------------------------------------------------------------

class TestAppendTokenToJoint(unittest.TestCase):

    def test_input_ids_grows_by_one(self) -> None:
        joint = _make_joint(bsz=2, seq_len=5)
        token = torch.tensor([99, 99], dtype=torch.long)
        out = _append_token_to_joint(joint, token)
        self.assertEqual(out["input_ids"].shape, (2, 6))

    def test_attention_mask_grows_by_one(self) -> None:
        joint = _make_joint(bsz=2, seq_len=5)
        token = torch.tensor([99, 99], dtype=torch.long)
        out = _append_token_to_joint(joint, token)
        self.assertEqual(out["attention_mask"].shape, (2, 6))
        # new mask positions should be 1
        self.assertTrue(out["attention_mask"][:, -1].all())

    def test_pixel_values_unchanged(self) -> None:
        joint = _make_joint(bsz=1, seq_len=5)
        orig_pv = joint["pixel_values"].clone()
        token = torch.tensor([99], dtype=torch.long)
        out = _append_token_to_joint(joint, token)
        self.assertTrue(torch.equal(out["pixel_values"], orig_pv))

    def test_original_joint_not_mutated(self) -> None:
        joint = _make_joint(bsz=1, seq_len=5)
        orig_len = int(joint["input_ids"].shape[1])
        token = torch.tensor([99], dtype=torch.long)
        _append_token_to_joint(joint, token)
        self.assertEqual(int(joint["input_ids"].shape[1]), orig_len)


# ---------------------------------------------------------------------------
# Tests: constrained_generate_structured
# ---------------------------------------------------------------------------

_PT_OBJ_RE = re.compile(
    r"^<\|gaze_point\|>(<loc_\d+>)(<loc_\d+>)<\|gaze_object\|>(<obj_\d+>)$"
)
_OBJ_PT_RE = re.compile(
    r"^<\|gaze_object\|>(<obj_\d+>)<\|gaze_point\|>(<loc_\d+>)(<loc_\d+>)$"
)


class TestConstrainedGenerateStructured(unittest.TestCase):

    def setUp(self) -> None:
        self.coord_bins = 8
        self.num_classes = 4
        self.tok = _make_tokenizer(self.coord_bins, self.num_classes)

        self.processor = MagicMock()
        self.processor.tokenizer = self.tok

        # vocab_size large enough to cover all token ids from mock tokenizer
        self.model = _make_model(vocab_size=500)

    def test_point_object_order_valid_format(self) -> None:
        joint = _make_joint(bsz=1, seq_len=3)
        results = constrained_generate_structured(
            model=self.model,
            joint=joint,
            processor=self.processor,
            num_classes=self.num_classes,
            coord_bins=self.coord_bins,
            amp_dtype=torch.float32,
            target_order="point_object",
        )
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(_PT_OBJ_RE.match(results[0]), msg=f"No match: {results[0]!r}")

    def test_object_point_order_valid_format(self) -> None:
        joint = _make_joint(bsz=1, seq_len=3)
        results = constrained_generate_structured(
            model=self.model,
            joint=joint,
            processor=self.processor,
            num_classes=self.num_classes,
            coord_bins=self.coord_bins,
            amp_dtype=torch.float32,
            target_order="object_point",
        )
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(_OBJ_PT_RE.match(results[0]), msg=f"No match: {results[0]!r}")

    def test_unsupported_order_raises_value_error(self) -> None:
        joint = _make_joint(bsz=1, seq_len=3)
        # "reasoning_only" is a training-only target order; constrained eval must reject it.
        with self.assertRaises(ValueError):
            constrained_generate_structured(
                model=self.model,
                joint=joint,
                processor=self.processor,
                num_classes=self.num_classes,
                coord_bins=self.coord_bins,
                amp_dtype=torch.float32,
                target_order="reasoning_only",
            )

    def test_batch_size_2(self) -> None:
        joint = _make_joint(bsz=2, seq_len=3)
        results = constrained_generate_structured(
            model=self.model,
            joint=joint,
            processor=self.processor,
            num_classes=self.num_classes,
            coord_bins=self.coord_bins,
            amp_dtype=torch.float32,
            target_order="point_object",
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsNotNone(_PT_OBJ_RE.match(r), msg=f"No match: {r!r}")


# ---------------------------------------------------------------------------
# Integration: point_object output is parsed as valid by parse_structured_output_text
# ---------------------------------------------------------------------------

class TestPointObjectEvalOrderMatchesTrainer(unittest.TestCase):
    """Verify that constrained output with point_object order parses as valid_format=True."""

    def setUp(self) -> None:
        self.coord_bins = 8
        self.num_classes = 4
        self.tok = _make_tokenizer(self.coord_bins, self.num_classes)
        self.processor = MagicMock()
        self.processor.tokenizer = self.tok
        self.model = _make_model(vocab_size=500)

    def test_parser_accepts_constrained_point_object_output(self) -> None:
        joint = _make_joint(bsz=1, seq_len=3)
        results = constrained_generate_structured(
            model=self.model,
            joint=joint,
            processor=self.processor,
            num_classes=self.num_classes,
            coord_bins=self.coord_bins,
            amp_dtype=torch.float32,
            target_order="point_object",
        )
        parsed = parse_structured_output_text(results[0], self.num_classes, coord_bins=self.coord_bins)
        self.assertTrue(parsed["valid_format"], msg=f"Parser rejected: {results[0]!r} -> {parsed}")
        self.assertIsNotNone(parsed["point_xy"])
        self.assertIsNotNone(parsed["point_bins"])
        self.assertIsNotNone(parsed["object_id"])

    def test_parser_accepts_constrained_object_point_output(self) -> None:
        joint = _make_joint(bsz=1, seq_len=3)
        results = constrained_generate_structured(
            model=self.model,
            joint=joint,
            processor=self.processor,
            num_classes=self.num_classes,
            coord_bins=self.coord_bins,
            amp_dtype=torch.float32,
            target_order="object_point",
        )
        parsed = parse_structured_output_text(results[0], self.num_classes, coord_bins=self.coord_bins)
        self.assertTrue(parsed["valid_format"], msg=f"Parser rejected: {results[0]!r} -> {parsed}")


if __name__ == "__main__":
    unittest.main()
