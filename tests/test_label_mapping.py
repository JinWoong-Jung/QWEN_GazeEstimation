from __future__ import annotations

import unittest

from model.datasets import format_structured_target_text
from model.utils.data_utils import alias_label_text, direct_vocab_id
from model.utils.gaze_tokens import GAZE_OBJ_UNKNOWN


class TestSafeAliasMapping(unittest.TestCase):
    def test_alias_label_text_maps_known_person_variant(self):
        self.assertEqual(alias_label_text("man"), "person")

    def test_direct_vocab_id_uses_alias_map(self):
        vocab = {"person": 3, "phone": 8}
        self.assertEqual(direct_vocab_id("woman", vocab), 3)
        self.assertEqual(direct_vocab_id("smartphone", vocab), 8)


class TestStructuredTargetFallback(unittest.TestCase):
    def test_unmapped_label_keeps_text_but_disables_object_supervision(self):
        target_text, obj_id, text_valid, object_valid = format_structured_target_text(
            label_text="totally unknown object",
            label_id=-1,
            id2label=None,
            vocab2id={"person": 1},
            vocab2id_lower={"person": 1},
            num_classes=10,
            point_x=0.25,
            point_y=0.75,
        )
        self.assertEqual(obj_id, -1)
        self.assertEqual(text_valid, 1.0)
        self.assertEqual(object_valid, 0.0)
        self.assertIn(GAZE_OBJ_UNKNOWN, target_text)


if __name__ == "__main__":
    unittest.main()
