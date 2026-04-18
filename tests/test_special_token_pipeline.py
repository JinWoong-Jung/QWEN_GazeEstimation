from __future__ import annotations

import re
import unittest

import torch

from model.utils.gaze_tokens import (
    COORD_BINS,
    FORMAT_TOKENS,
    build_gaze_special_tokens,
    register_gaze_special_tokens,
)
from model.utils.object_tokens import OBJ_SLOT, add_slot_token


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.additional_special_tokens: list[str] = []
        self.tok2id: dict[str, int] = {"<pad>": 0}
        self.id2tok: dict[int, str] = {0: "<pad>"}

    def ensure(self, tok: str) -> int:
        if tok in self.tok2id:
            return int(self.tok2id[tok])
        idx = len(self.tok2id)
        self.tok2id[tok] = idx
        self.id2tok[idx] = tok
        return idx

    def tokenize_with_offsets(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        for m in re.finditer(r"<obj_emb>|[^\s]+", str(text)):
            tokens.append(m.group(0))
            offsets.append((int(m.start()), int(m.end())))
        return tokens, offsets

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del add_special_tokens
        del return_attention_mask
        toks, offs = self.tokenize_with_offsets(str(text))
        ids = [self.ensure(t) for t in toks]
        out: dict[str, list[int] | list[tuple[int, int]]] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def add_special_tokens(
        self,
        payload: dict[str, list[str]],
        replace_additional_special_tokens: bool = False,
    ) -> int:
        del replace_additional_special_tokens
        toks = [str(x) for x in payload.get("additional_special_tokens", [])]
        added = 0
        for t in toks:
            if t not in self.tok2id:
                self.ensure(t)
                added += 1
        self.additional_special_tokens = toks
        return int(added)

    def add_tokens(self, tokens: list[str], special_tokens: bool = True) -> int:
        del special_tokens
        added = 0
        for t in [str(x) for x in tokens]:
            if t not in self.tok2id:
                self.ensure(t)
                added += 1
        return int(added)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.tok2id)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.tok2id.get(str(token), -1))


class DummyProcessor:
    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer


class TestRegisterGazeSpecialTokens(unittest.TestCase):
    """Integration test: register_gaze_special_tokens preserves pre-existing tokens."""

    def _make_tokenizer_with_existing(self, existing: list[str]) -> DummyTokenizer:
        tok = DummyTokenizer()
        # Simulate pre-existing Qwen special tokens (e.g. vision tokens, chat tokens)
        for t in existing:
            tok.ensure(t)
        tok.additional_special_tokens = list(existing)
        return tok

    def test_existing_tokens_preserved_in_vocab(self) -> None:
        qwen_specials = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
        tok = self._make_tokenizer_with_existing(qwen_specials)
        original_ids = {t: tok.tok2id[t] for t in qwen_specials}

        register_gaze_special_tokens(tok, num_classes=5)

        for t, orig_id in original_ids.items():
            self.assertIn(t, tok.get_vocab(), f"pre-existing token {t!r} lost after registration")
            self.assertEqual(tok.tok2id[t], orig_id, f"token {t!r} id changed after registration")

    def test_all_gaze_tokens_registered(self) -> None:
        tok = DummyTokenizer()
        num_classes = 10
        token_id_map = register_gaze_special_tokens(tok, num_classes=num_classes)

        expected = build_gaze_special_tokens(num_classes)
        for t in expected:
            self.assertIn(t, tok.get_vocab(), f"gaze token {t!r} not added to vocab")
            self.assertIn(t, token_id_map, f"gaze token {t!r} missing from returned id map")

    def test_vocab_size_grows_by_correct_amount(self) -> None:
        tok = DummyTokenizer()
        num_classes = 5
        before = len(tok.get_vocab())
        register_gaze_special_tokens(tok, num_classes=num_classes)
        after = len(tok.get_vocab())
        # 1000 loc tokens + 5 obj tokens + 1 unknown obj token
        expected_new = COORD_BINS + num_classes + 1
        self.assertEqual(after - before, expected_new)

    def test_double_registration_is_idempotent(self) -> None:
        tok = DummyTokenizer()
        num_classes = 5
        register_gaze_special_tokens(tok, num_classes=num_classes)
        size_after_first = len(tok.get_vocab())
        register_gaze_special_tokens(tok, num_classes=num_classes)
        size_after_second = len(tok.get_vocab())
        self.assertEqual(size_after_first, size_after_second)

    def test_format_tokens_have_valid_ids(self) -> None:
        tok = DummyTokenizer()
        token_id_map = register_gaze_special_tokens(tok, num_classes=5)
        for ft in FORMAT_TOKENS:
            self.assertIn(ft, token_id_map)
            self.assertGreaterEqual(token_id_map[ft], 0)


class TestSpecialTokenPipeline(unittest.TestCase):
    def test_add_slot_token(self) -> None:
        tok = DummyTokenizer()
        _ = tok.add_special_tokens({"additional_special_tokens": ["<sys_special>"]})
        added = add_slot_token(tok)
        self.assertEqual(added, 1)
        self.assertIn(OBJ_SLOT, tok.get_vocab())
        added2 = add_slot_token(tok)
        self.assertEqual(added2, 0)


if __name__ == "__main__":
    unittest.main()
