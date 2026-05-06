from __future__ import annotations

import re
import unittest

import torch

from model.utils.gaze_tokens import (
    build_structured_target_text,
    build_structured_target_text_with_reasoning,
)
from model.utils.processor_collate import build_structured_masks


_ATOM_RE = re.compile(r"(<[^>]+>|\n| )|([^\s<]+)")


class _SimpleTokenizer:
    """Word-level tokenizer that keeps <...> tokens atomic.

    Designed for unit tests only. Splits on spaces and newlines while
    preserving <loc_NNN>, <obj_NNN>, and active span markers as single tokens.
    """

    def __init__(self) -> None:
        self._v: dict[str, int] = {}
        self._r: dict[int, str] = {}
        self._next = 1

    def _id(self, tok: str) -> int:
        if tok not in self._v:
            self._v[tok] = self._next
            self._r[self._next] = tok
            self._next += 1
        return self._v[tok]

    def _encode(self, text: str) -> list[int]:
        return [self._id(m.group()) for m in _ATOM_RE.finditer(text) if m.group()]

    def __call__(self, text_or_texts: str | list[str], add_special_tokens: bool = False, return_attention_mask: bool = False) -> dict:
        # HuggingFace tokenizers return a flat list for single-string input and
        # a list-of-lists for batch input. _tok_ids in build_structured_masks
        # relies on this distinction.
        if isinstance(text_or_texts, str):
            return {"input_ids": self._encode(text_or_texts)}
        return {"input_ids": [self._encode(t) for t in list(text_or_texts)]}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._r.get(int(i), "<unk>") for i in ids]

    def convert_tokens_to_ids(self, token: str | list[str]) -> int | list[int]:
        if isinstance(token, str):
            return self._v.get(token, 0)
        return [self._v.get(t, 0) for t in token]


class _MockProcessor:
    def __init__(self) -> None:
        self.tokenizer = _SimpleTokenizer()


class TestReasoningMasks(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = _MockProcessor()

    def _joint(self, text: str) -> dict:
        ids = self.proc.tokenizer._encode(text)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def test_no_reasoning_mask_all_false(self):
        text = build_structured_target_text(0.5, 0.5, 10, 100)
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertFalse(mrsn.any().item(), "reasoning mask must be all False when no reasoning block")

    def test_with_reasoning_has_content(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "person looks left")
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertTrue(mrsn.any().item(), "reasoning mask must have True tokens for reasoning content")

    def test_empty_reasoning_scaffold_is_format_only(self):
        text = build_structured_target_text_with_reasoning(
            0.5,
            0.5,
            10,
            100,
            "",
            force_reasoning_format=True,
        )
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertFalse(mrsn.any().item(), "empty reasoning scaffold must not create reasoning-content loss")
        self.assertTrue(mfmt.any().item(), "empty reasoning scaffold tokens should be covered by format loss")
        self.assertEqual(int(mpt.sum().item()), 2)
        self.assertEqual(int(mobj.sum().item()), 1)

    def test_masks_are_disjoint(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "some reasoning text")
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertFalse((mpt & mobj).any().item(), "point ∩ object must be empty")
        self.assertFalse((mpt & mfmt).any().item(), "point ∩ format must be empty")
        self.assertFalse((mpt & mrsn).any().item(), "point ∩ reasoning must be empty")
        self.assertFalse((mobj & mfmt).any().item(), "object ∩ format must be empty")
        self.assertFalse((mobj & mrsn).any().item(), "object ∩ reasoning must be empty")
        self.assertFalse((mfmt & mrsn).any().item(), "format ∩ reasoning must be empty")

    def test_masks_cover_full_answer(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "covering test")
        joint = self._joint(text)
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, joint, [text], torch.ones(1)
        )
        union = mpt | mobj | mfmt | mrsn
        n_union = int(union.sum().item())
        n_seq = int(joint["input_ids"].shape[1])
        self.assertGreater(n_union, 0, "union must cover at least one token")
        # union cannot exceed sequence length
        self.assertLessEqual(n_union, n_seq)

    def test_point_mask_hits_loc_tokens(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "test")
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        # There must be exactly 2 <loc_*> tokens (x and y bins)
        self.assertEqual(int(mpt.sum().item()), 2, "point mask must cover exactly 2 loc tokens")

    def test_object_mask_hits_obj_token(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "test")
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertEqual(int(mobj.sum().item()), 1, "object mask must cover exactly 1 obj token")

    def test_invalid_sample_all_masks_zero(self):
        text = build_structured_target_text_with_reasoning(0.5, 0.5, 10, 100, "reason")
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.zeros(1)  # invalid
        )
        union = mpt | mobj | mfmt | mrsn
        self.assertFalse(union.any().item(), "all masks must be zero for invalid sample")

    def test_no_reasoning_still_has_point_and_object(self):
        text = build_structured_target_text(0.5, 0.5, 10, 100)
        mpt, mobj, mfmt, mrsn = build_structured_masks(
            self.proc, self._joint(text), [text], torch.ones(1)
        )
        self.assertEqual(int(mpt.sum().item()), 2)
        self.assertEqual(int(mobj.sum().item()), 1)
        self.assertFalse(mrsn.any().item())


if __name__ == "__main__":
    unittest.main()
