"""Tests for lisp.py, focused on silent-failure bugs in the interpreter."""

import unittest

from lisp import standard_env, evaluate, tokenize, parse, balanced


def run(src, env=None):
    env = env or standard_env()
    tokens = tokenize(src)
    result = None
    while tokens:
        result = evaluate(parse(tokens), env)
    return result


class TestArity(unittest.TestCase):
    def test_too_many_args_raises(self):
        with self.assertRaises(TypeError):
            run("(define (f a b) (+ a b)) (f 1 2 3)")

    def test_too_few_args_raises(self):
        with self.assertRaises(TypeError):
            run("(define (f a b) (+ a b)) (f 1)")

    def test_exact_args_ok(self):
        self.assertEqual(run("(define (f a b) (+ a b)) (f 1 2)"), 3)

    def test_non_list_params_rejected(self):
        with self.assertRaises(SyntaxError):
            run("(lambda xs xs)")


class TestMultiFormBody(unittest.TestCase):
    def test_define_evaluates_all_body_forms(self):
        # first body form runs for effect, last form is the return value
        self.assertEqual(run("(define (f x) (set! seen x) (+ x 1)) "
                             "(define seen 0) (f 3)"), 4)

    def test_define_body_side_effects_happen(self):
        self.assertEqual(run("(define seen 0) "
                             "(define (f x) (set! seen x) (+ x 1)) "
                             "(f 3) seen"), 3)

    def test_lambda_evaluates_all_body_forms(self):
        self.assertEqual(run("(define seen 0) "
                             "((lambda (x) (set! seen x) (* x 2)) 5) seen"), 5)


class TestAddition(unittest.TestCase):
    def test_plus_concatenates_all_strings(self):
        self.assertEqual(run('(+ "a" "b" "c")'), "abc")

    def test_plus_numbers_still_works(self):
        self.assertEqual(run("(+ 1 2 3)"), 6)


class TestBalanced(unittest.TestCase):
    def test_paren_inside_string_is_ignored(self):
        self.assertTrue(balanced('(display "(")'))

    def test_close_paren_inside_string_is_ignored(self):
        self.assertFalse(balanced('(display ")"'))

    def test_paren_inside_comment_is_ignored(self):
        self.assertFalse(balanced("(f ; )"))

    def test_plain_forms(self):
        self.assertTrue(balanced("(+ 1 2)"))
        self.assertFalse(balanced("(+ 1 (2"))

    def test_unterminated_string_means_incomplete(self):
        self.assertFalse(balanced('(display "abc'))


class TestStrings(unittest.TestCase):
    def test_escaped_backslash_n_is_not_newline(self):
        # lisp source "a\\nb" is backslash + n, not a newline
        self.assertEqual(run(r'"a\\nb"'), "a\\nb")

    def test_newline_escape(self):
        self.assertEqual(run(r'"a\nb"'), "a\nb")

    def test_tab_escape(self):
        self.assertEqual(run(r'"a\tb"'), "a\tb")

    def test_quote_escape(self):
        self.assertEqual(run(r'"a\"b"'), 'a"b')

    def test_unterminated_string_raises(self):
        with self.assertRaises(SyntaxError):
            tokenize('"abc')


class TestRegressions(unittest.TestCase):
    def test_recursion(self):
        self.assertEqual(run("(define (fact n) (if (= n 0) 1 (* n (fact (- n 1))))) "
                             "(fact 10)"), 3628800)

    def test_tail_call_depth(self):
        self.assertEqual(run("(define (loop n) (if (= n 0) 0 (loop (- n 1)))) "
                             "(loop 100000)"), 0)

    def test_macro_and_quasiquote(self):
        self.assertEqual(run("(defmacro (when test body) `(if ,test ,body)) "
                             "(when (> 2 1) 42)"), 42)

    def test_variadic_map(self):
        self.assertEqual(run("(map + (list 1 2) (list 10 20))"), [11, 22])


if __name__ == "__main__":
    unittest.main()
