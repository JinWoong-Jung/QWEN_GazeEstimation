"""Tests for data utilities and label bank used in the structured-token pipeline.

Covers:
- object_label_span() finds pure-text label spans
- build_prompt() formats bbox into prompt strings
- build_split_bank() constructs embedding bank from label dir
- load_test_groups() loads and groups test annotations
- LabelBank topk retrieval
- load_label_map() / load_test_label_map() vocab lookup
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make sure project root is on the path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from model.utils.object_tokens import OBJ_SLOT, object_label_span
from model.utils.data_utils import (
    build_prompt,
    build_split_bank,
    load_label_map,
    load_test_groups,
    load_test_label_map,
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
        result = object_label_span("Point: 0.1 0.2\nObject:   ")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):

    def test_prompt_formats_bbox_and_point_decimals(self) -> None:
        prompt = build_prompt(
            (0.1, 0.2, 0.3, 0.4),
            "",
            "bbox=[{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}] decimals={point_decimals}",
            point_decimals=2,
        )
        self.assertEqual(prompt, "bbox=[0.10, 0.20, 0.30, 0.40] decimals=2")

    def test_config_prompts_describe_token_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = "\n".join(
            (root / name).read_text(encoding="utf-8")
            for name in ("sft.yaml", "config_rl.yaml")
        )
        self.assertIn("<|gaze_reasoning|><your reasoning here><|gaze_point|>", cfg)
        self.assertIn("<|gaze_point|><loc_NNN><loc_MMM><|gaze_object|><obj_KKK>", cfg)
        self.assertNotIn("Reasoning: <your reasoning here>", cfg)
        self.assertNotIn("Point: <loc_NNN><loc_MMM>", cfg)
        self.assertNotIn("Object: <obj_KKK>", cfg)

    def test_sft_uses_ratio_based_multiview_sampling(self) -> None:
        cfg = (Path(__file__).resolve().parents[1] / "sft.yaml").read_text(encoding="utf-8")
        self.assertIn("direct_view_ratio: 0.8", cfg)
        self.assertIn("reasoning_view_ratio: 0.2", cfg)
        self.assertNotIn("full_dual_view=True", cfg)


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
        norms = bank_embs.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


# ---------------------------------------------------------------------------
# Test group loading
# ---------------------------------------------------------------------------

class TestLoadTestGroups(unittest.TestCase):

    def test_path_level_label_attached_for_single_bbox_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_root = root / "images"
            img_rel = "test2/00000000/00000001.jpg"
            img_path = image_root / "00000000/00000001.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color="white").save(img_path)

            ann = root / "test_ann.txt"
            ann.write_text(
                (
                    "test2/00000000/00000001.jpg,1,0,0,0,0,0,0,0.25,0.75,10,20,30,40,pascal,img.jpg\n"
                    "test2/00000000/00000001.jpg,2,0,0,0,0,0,0,0.35,0.65,10,20,30,40,pascal,img.jpg\n"
                ),
                encoding="utf-8",
            )

            groups = load_test_groups(
                annotation_file=ann,
                image_root=image_root,
                test_label_map={img_rel: 42},
                test_label_text_map={img_rel: "laptop"},
                test_label_ids_map={img_rel: [42, 99]},
                split_prefix="test2/",
                strip_split_prefix=True,
                bbox_round_decimals=3,
            )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].label_id, 42)
        self.assertEqual(groups[0].label_text, "laptop")
        self.assertEqual(groups[0].label_ids, [42, 99])
        self.assertEqual(len(groups[0].gt_points), 2)

    def test_path_level_label_dropped_for_ambiguous_multi_bbox_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_root = root / "images"
            img_rel = "test2/00000000/00000002.jpg"
            img_path = image_root / "00000000/00000002.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color="white").save(img_path)

            ann = root / "test_ann.txt"
            ann.write_text(
                (
                    "test2/00000000/00000002.jpg,1,0,0,0,0,0,0,0.25,0.75,10,20,30,40,pascal,img.jpg\n"
                    "test2/00000000/00000002.jpg,2,0,0,0,0,0,0,0.35,0.65,11,21,31,41,pascal,img.jpg\n"
                ),
                encoding="utf-8",
            )

            groups = load_test_groups(
                annotation_file=ann,
                image_root=image_root,
                test_label_map={img_rel: 7},
                test_label_text_map={img_rel: "chair"},
                test_label_ids_map={img_rel: [7, 8]},
                split_prefix="test2/",
                strip_split_prefix=True,
                bbox_round_decimals=3,
            )

        self.assertEqual(len(groups), 2)
        self.assertTrue(all(int(g.label_id) < 0 for g in groups))
        self.assertTrue(all(str(g.label_text) == "" for g in groups))
        self.assertTrue(all(list(g.label_ids) == [] for g in groups))


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
# Label map loading
# ---------------------------------------------------------------------------

class TestSemgazeStyleLabelMapping(unittest.TestCase):

    def test_train_label_map_uses_direct_vocab_lookup_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels_csv = root / "gaze-labels-train.csv"
            labels_csv.write_text(
                "path,id,split,gaze_pseudo_label,label_id\n"
                "train/a.jpg,1,train,chair,0\n"
                "train/b.jpg,2,train,Chair,0\n",
                encoding="utf-8",
            )

            label_map, stats = load_label_map(
                labels_csv=labels_csv,
                vocab2id={"chair": 7},
                vocab2id_lower={"chair": 7},
                text_key="gaze_pseudo_label",
                use_embed_fallback=False,
            )

        self.assertEqual(label_map[("train/a.jpg", 1)], 7)
        self.assertEqual(label_map[("train/b.jpg", 2)], -1)
        self.assertEqual(int(stats["mapped"]), 1)
        self.assertEqual(int(stats["unknown_text"]), 1)
        self.assertEqual(int(stats["embed_fallback_mapped"]), 0)
        self.assertEqual(int(stats["fallback_mapped"]), 0)

    def test_test_label_map_keeps_multi_labels_without_forcing_primary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels_csv = root / "gaze-labels-test.csv"
            labels_csv.write_text(
                "path,eyes_closed,outside_frame,uncertain,gaze_gt_labels,gaze_gt_label,label_id,test_label_id\n"
                "test2/a.jpg,False,False,False,book-table,chair,0,0\n",
                encoding="utf-8",
            )

            id_map, text_map, multi_id_map, stats = load_test_label_map(
                labels_csv=labels_csv,
                vocab2id={"chair": 1, "book": 2, "table": 3},
                vocab2id_lower={"chair": 1, "book": 2, "table": 3},
            )

        self.assertEqual(id_map["test2/a.jpg"], 1)
        self.assertEqual(text_map["test2/a.jpg"], "chair")
        self.assertEqual(multi_id_map["test2/a.jpg"], [2, 3])
        self.assertEqual(int(stats["mapped"]), 1)


if __name__ == "__main__":
    unittest.main()
