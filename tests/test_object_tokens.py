from __future__ import annotations

import unittest

from model.utils.object_tokens import OBJ_SLOT, slot_span


class TestObjectTokens(unittest.TestCase):
    def test_slot_span_from_object_line(self) -> None:
        txt = f"Point: 0.1000 0.2000\nObject: {OBJ_SLOT}"
        span = slot_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], OBJ_SLOT)

    def test_slot_span_from_fallback_match(self) -> None:
        txt = f"prefix {OBJ_SLOT} suffix"
        span = slot_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], OBJ_SLOT)

    def test_slot_span_missing(self) -> None:
        self.assertIsNone(slot_span("Point: 0.1 0.2\nObject: none"))


if __name__ == "__main__":
    unittest.main()
