from __future__ import annotations

import re
import unittest

import torch

from model.utils.eval_utils import (
    _build_object_token_id_to_label_map,
    _parse_object_id_from_token_ids,
    _parse_object_id_with_fallback,
)
from model.utils.object_tokens import register_object_special_tokens
from model.utils.processor_collate import _build_component_loss_masks


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.additional_special_tokens: list[str] = []
        self._tok2id: dict[str, int] = {"<pad>": 0}
        self._id2tok: dict[int, str] = {0: "<pad>"}

    def _ensure_token(self, tok: str) -> int:
        if tok in self._tok2id:
            return int(self._tok2id[tok])
        idx = len(self._tok2id)
        self._tok2id[tok] = idx
        self._id2tok[idx] = tok
        return idx

    def _tokenize_with_offsets(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        for m in re.finditer(r"<obj_\d+>|[^\s]+", str(text)):
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
        toks, offs = self._tokenize_with_offsets(str(text))
        ids = [self._ensure_token(t) for t in toks]
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
            if t not in self._tok2id:
                self._ensure_token(t)
                added += 1
        self.additional_special_tokens = toks
        return int(added)

    def add_tokens(self, tokens: list[str], special_tokens: bool = True) -> int:
        del special_tokens
        added = 0
        for t in [str(x) for x in tokens]:
            if t not in self._tok2id:
                self._ensure_token(t)
                added += 1
        return int(added)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tok2id)

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if torch.is_tensor(token_ids):
            ids = [int(x) for x in token_ids.tolist()]
        else:
            ids = [int(x) for x in token_ids]
        return " ".join(self._id2tok.get(i, "<unk>") for i in ids)


class DummyProcessor:
    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer


class TestSpecialTokenPipeline(unittest.TestCase):
    def test_tokenizer_extension_contains_expected_object_tokens(self) -> None:
        tok = DummyTokenizer()
        # Keep existing special token to check preservation behavior.
        _ = tok.add_special_tokens({"additional_special_tokens": ["<sys_special>"]})

        added, width, required = register_object_special_tokens(tok, num_classes=4, width=3)
        self.assertEqual(width, 3, "Object token width should be fixed to 3 in this test.")
        self.assertEqual(added, 4, "Expected four object tokens to be added on first registration.")
        vocab = tok.get_vocab()
        for t in required:
            self.assertIn(t, vocab, f"Missing expected object token in tokenizer vocab: {t}")

        # Re-registering should not add duplicates.
        added2, _, _ = register_object_special_tokens(tok, num_classes=4, width=3)
        self.assertEqual(added2, 0, "Expected zero added tokens on second registration.")

    def test_collator_object_span_masking(self) -> None:
        tok = DummyTokenizer()
        proc = DummyProcessor(tok)
        target = "Point: 0.1000 0.2000\nObject: <obj_127>"
        encoded = tok(
            target,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
        attn = torch.ones_like(ids, dtype=torch.long)
        joint_inputs = {"input_ids": ids, "attention_mask": attn}
        target_valid = torch.tensor([1.0], dtype=torch.float32)

        answer_mask, point_mask, object_mask = _build_component_loss_masks(
            processor=proc,
            joint_inputs=joint_inputs,
            target_texts=[target],
            target_valid=target_valid,
        )

        self.assertTrue(bool(answer_mask.any().item()), "Answer mask should contain supervised tokens.")
        self.assertGreaterEqual(int(point_mask.sum().item()), 2, "Point mask should cover x and y coordinate tokens.")
        self.assertEqual(int(object_mask.sum().item()), 1, "Object mask should cover exactly one object token.")

        obj_id = tok.get_vocab()["<obj_127>"]
        masked_obj_ids = ids[0][object_mask[0]].tolist()
        self.assertIn(obj_id, masked_obj_ids, "Object mask should target <obj_127> token position.")

    def test_evaluation_text_parsing_fallback(self) -> None:
        obj, src = _parse_object_id_with_fallback(
            pred_text="Point: 0.1 0.2\nObject: <obj_127>",
            generated_token_ids=[],
            object_token_id_to_label={},
        )
        self.assertEqual(obj, 127, "Text fallback should parse object label id from Object line.")
        self.assertEqual(src, "object_line", "Expected object_line source when token-id path is unavailable.")

        obj_any, src_any = _parse_object_id_with_fallback(
            pred_text="random prefix <obj_009> random suffix",
            generated_token_ids=[],
            object_token_id_to_label={},
        )
        self.assertEqual(obj_any, 9, "Regex fallback should parse object token when Object line is missing.")
        self.assertEqual(src_any, "text_regex", "Expected text_regex source for fallback regex parsing.")

        obj2, src2 = _parse_object_id_with_fallback(
            pred_text="Point: 0.1 0.2\nObject: ???",
            generated_token_ids=[],
            object_token_id_to_label={},
        )
        self.assertIsNone(obj2, "Malformed object output should not map to any class id.")
        self.assertEqual(src2, "failed", "Malformed output should be marked as parse failure.")

    def test_evaluation_parse_priority_token_ids_first(self) -> None:
        # Token-level path should win even if decoded text also contains an object token.
        obj, src = _parse_object_id_with_fallback(
            pred_text="Point: 0.1 0.2\nObject: <obj_007>",
            generated_token_ids=[101, 202, 303],
            object_token_id_to_label={202: 123},
        )
        self.assertEqual(obj, 123, "Token-level object parsing should take priority over text parsing.")
        self.assertEqual(src, "token_ids", "Expected token_ids source when token-level parsing is available.")

    def test_bonus_token_level_extraction_from_generated_ids(self) -> None:
        token_map = {77: 127}
        obj = _parse_object_id_from_token_ids([5, 77, 9], token_map)
        self.assertEqual(obj, 127, "Token-level extraction should detect object token id anywhere in generated ids.")

    def test_build_object_token_id_map_from_tokenizer_vocab(self) -> None:
        tok = DummyTokenizer()
        _ = register_object_special_tokens(tok, num_classes=3, width=3)
        proc = DummyProcessor(tok)
        mapping = _build_object_token_id_to_label_map(proc)
        vocab = tok.get_vocab()
        self.assertEqual(mapping[vocab["<obj_000>"]], 0)
        self.assertEqual(mapping[vocab["<obj_001>"]], 1)
        self.assertEqual(mapping[vocab["<obj_002>"]], 2)


if __name__ == "__main__":
    unittest.main()
