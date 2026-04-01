from __future__ import annotations

import unittest

from model.utils.object_tokens import (
    build_all_object_tokens,
    build_object_token,
    format_answer,
    is_object_token,
    parse_object_id_from_object_line,
    parse_object_id_from_text,
    parse_object_id_from_text_regex,
    parse_object_id_from_text_with_source,
    parse_object_token,
    parse_object_token_span,
)


class TestObjectTokens(unittest.TestCase):
    def test_build_and_parse_object_token(self) -> None:
        tok = build_object_token(7, width=3)
        self.assertEqual(tok, "<obj_007>")
        self.assertEqual(parse_object_token(tok), 7)
        self.assertIsNone(parse_object_token("obj_007"))

    def test_build_all_object_tokens(self) -> None:
        tokens = build_all_object_tokens(4, width=3)
        self.assertEqual(tokens, ["<obj_000>", "<obj_001>", "<obj_002>", "<obj_003>"])

    def test_is_object_token(self) -> None:
        self.assertTrue(is_object_token("<obj_127>"))
        self.assertFalse(is_object_token("Object: <obj_127>"))

    def test_parse_object_id_from_text(self) -> None:
        txt = "Point: 0.1000 0.2000\nObject: <obj_127>"
        self.assertEqual(parse_object_id_from_text(txt), 127)
        self.assertEqual(parse_object_id_from_text("foo <obj_009> bar"), 9)
        self.assertIsNone(parse_object_id_from_text("Object: 127"))

    def test_parse_object_id_with_source_order(self) -> None:
        txt_line = "Point: 0.1000 0.2000\nObject: <obj_127>"
        self.assertEqual(parse_object_id_from_object_line(txt_line), 127)
        self.assertEqual(parse_object_id_from_text_regex(txt_line), 127)
        obj, src = parse_object_id_from_text_with_source(txt_line)
        self.assertEqual(obj, 127)
        self.assertEqual(src, "object_line")

        txt_any = "noise <obj_009> trailing"
        self.assertIsNone(parse_object_id_from_object_line(txt_any))
        self.assertEqual(parse_object_id_from_text_regex(txt_any), 9)
        obj2, src2 = parse_object_id_from_text_with_source(txt_any)
        self.assertEqual(obj2, 9)
        self.assertEqual(src2, "text_regex")

    def test_parse_object_token_span(self) -> None:
        txt = "Point: 0.1000 0.2000\nObject: <obj_042>"
        span = parse_object_token_span(txt)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertEqual(txt[span[0]:span[1]], "<obj_042>")

    def test_format_answer(self) -> None:
        out = format_answer(0.123456, 0.987654, 5, point_decimals=4, width=3)
        self.assertEqual(out, "Point: 0.1235 0.9877\nObject: <obj_005>")


if __name__ == "__main__":
    unittest.main()
