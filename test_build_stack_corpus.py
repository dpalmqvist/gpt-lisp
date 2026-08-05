"""Unit tests for the corpus builder's pure logic (no network/HF needed)."""

import unittest

from build_stack_corpus import ascii_filter, shuffle_docs


class TestAsciiFilter(unittest.TestCase):
    def test_keeps_printable_tab_newline(self):
        self.assertEqual(ascii_filter("(def x 1)\n\tok"), b"(def x 1)\n\tok")

    def test_strips_non_ascii_and_control(self):
        self.assertEqual(ascii_filter("λ(def\x00 ü1)\r"), b"(def 1)")

    def test_empty(self):
        self.assertEqual(ascii_filter(""), b"")


class TestShuffle(unittest.TestCase):
    def test_deterministic_for_seed(self):
        docs = [bytes([i]) for i in range(50)]
        self.assertEqual(shuffle_docs(list(docs), 7), shuffle_docs(list(docs), 7))

    def test_actually_shuffles(self):
        docs = [bytes([i]) for i in range(50)]
        self.assertNotEqual(shuffle_docs(list(docs), 7), docs)


if __name__ == "__main__":
    unittest.main()
