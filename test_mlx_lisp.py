"""Tests for the MLX Lisp layer. Auto-skips when mlx is not installed."""

import os
import tempfile
import unittest

try:
    import mlx.core as mx  # noqa: F401
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

if HAVE_MLX:
    from mlx_lisp import gpu_env
    from lisp import tokenize, parse, evaluate


def run(src, env):
    tokens = tokenize(src)
    result = None
    while tokens:
        result = evaluate(parse(tokens), env)
    return result


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestPrimitives(unittest.TestCase):
    def setUp(self):
        self.env = gpu_env()

    def test_randint_shape(self):
        self.assertEqual(run("(shape (randint 0 10 (list 4 3)))", self.env),
                         [4, 3])

    def test_window_gather_shape(self):
        r = run('(define flat (array (range 10) "int32")) '
                '(define offs (array (list 0 5) "int32")) '
                "(shape (take flat (+ (reshape offs (list 2 1)) "
                '(array (range 3) "int32"))))', self.env)
        self.assertEqual(r, [2, 3])

    def test_transpose_axes(self):
        r = run("(shape (transpose-axes (zeros (list 2 3 4)) (list 1 0 2)))",
                self.env)
        self.assertEqual(r, [3, 2, 4])

    def test_pow(self):
        self.assertAlmostEqual(run("(pow 0.9 2)", self.env), 0.81)

    def test_save_load_tree_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), "t.npz")
        run(f'(save-tree "{path}" '
            "(list (ones (list 2)) (list (zeros (list 3)) (ones (list 1)))))",
            self.env)
        self.assertEqual(
            run(f'(item (sum (car (load-tree "{path}"))))', self.env), 2.0)
        self.assertEqual(
            run(f'(shape (car (nth (load-tree "{path}") 1)))', self.env), [3])

    def test_read_file_and_exists(self):
        self.assertTrue(run('(exists? "mlx_lisp.py")', self.env))
        self.assertFalse(run('(exists? "no-such-file")', self.env))
        self.assertIn("gpu_env", run('(read-file "mlx_lisp.py")', self.env))

    def test_load_include(self):
        with tempfile.NamedTemporaryFile("w", suffix=".lisp",
                                         delete=False) as f:
            f.write("(define loaded-value 42)")
        self.assertEqual(run(f'(load "{f.name}") loaded-value', self.env), 42)

    def test_seed_reproducible(self):
        a = run("(seed 7) (item (sum (randn (list 4))))", self.env)
        b = run("(seed 7) (item (sum (randn (list 4))))", self.env)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
