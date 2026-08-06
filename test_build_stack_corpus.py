"""Unit tests for the corpus builder's pure logic (no network/HF needed)."""

import unittest

from build_stack_corpus import ascii_filter, shuffle_docs, byte_budget, capped_docs


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


class TestByteBudget(unittest.TestCase):
    def test_clamps_to_mlx_limit(self):
        self.assertEqual(byte_budget(5.0), 2**31 - 1)

    def test_small_budget_unchanged(self):
        self.assertEqual(byte_budget(0.001), 1_000_000)


class TestCappedDocs(unittest.TestCase):
    def test_never_exceeds_budget_when_docs_overshoot(self):
        docs = [bytes([i % 256]) * 30 for i in range(10)]  # 10 docs, 30B each = 300B
        out = capped_docs(docs, 100)
        self.assertLessEqual(sum(len(d) for d in out), 100)

    def test_truncates_boundary_doc_rather_than_dropping_it(self):
        docs = [b"a" * 30, b"b" * 30, b"c" * 30]  # total 90B
        out = capped_docs(docs, 50)  # boundary falls inside doc 2 (b's)
        self.assertEqual(out[0], b"a" * 30)
        self.assertEqual(out[1], b"b" * 20)  # truncated, not dropped
        self.assertEqual(len(out), 2)
        self.assertEqual(sum(len(d) for d in out), 50)

    def test_docs_under_budget_pass_through_unchanged(self):
        docs = [b"a" * 10, b"b" * 10, b"c" * 10]  # total 30B
        out = capped_docs(docs, 1000)
        self.assertEqual(out, docs)


if __name__ == "__main__":
    unittest.main()
