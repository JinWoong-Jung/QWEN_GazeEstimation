from __future__ import annotations

import re
import unittest

import torch

from model.utils.eval_utils import topk_similarity
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


class DummyProcessor:
    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer


class TestSpecialTokenPipeline(unittest.TestCase):
    def test_add_slot_token(self) -> None:
        tok = DummyTokenizer()
        _ = tok.add_special_tokens({"additional_special_tokens": ["<sys_special>"]})
        added = add_slot_token(tok)
        self.assertEqual(added, 1)
        self.assertIn(OBJ_SLOT, tok.get_vocab())
        added2 = add_slot_token(tok)
        self.assertEqual(added2, 0)

    def test_topk_similarity(self) -> None:
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        q = torch.tensor([0.9, 0.1], dtype=torch.float32)
        topk = topk_similarity(q, bank, k=2, temperature=0.07)
        self.assertEqual(topk[0], 0)
        self.assertEqual(len(topk), 2)


if __name__ == "__main__":
    unittest.main()
