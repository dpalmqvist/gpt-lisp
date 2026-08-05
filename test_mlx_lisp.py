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


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestModel(unittest.TestCase):
    """Runs model.lisp in smoke config (T=32, D=32, H=2, L=2), in a tempdir
    so the repo root never sees the SMOKE flag file."""

    def setUp(self):
        import shutil
        self.repo = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(self.repo, "model.lisp"), self.tmp)
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.repo)
        open("SMOKE", "w").close()
        self.env = gpu_env()
        run('(load "model.lisp")', self.env)

    def test_smoke_config_active(self):
        self.assertEqual(run("(list T D H L)", self.env), [32, 32, 2, 2])

    def test_gpt_output_shape(self):
        r = run("(define p (init-params)) "
                "(shape (gpt p (randint 0 V (list 2 T))))", self.env)
        self.assertEqual(r, [2, 32, 128])

    def test_initial_loss_near_uniform(self):
        loss = run("(define p (init-params)) "
                   "(item (batch-loss p (randint 0 V (list 4 T)) "
                   "(randint 0 V (list 4 T))))", self.env)
        self.assertGreater(loss, 3.0)
        self.assertLess(loss, 7.0)

    def test_sample_text_length(self):
        r = run("(define p (init-params)) (sample-text p 5 1.0)", self.env)
        self.assertEqual(len(r), 13)


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestSaveAtomic(unittest.TestCase):
    def test_save_replaces_not_truncates(self):
        # an existing checkpoint must survive being overwritten atomically:
        # after save-tree, no .tmp file remains and the target is loadable
        env = gpu_env()
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.npz")
        run(f'(save-tree "{path}" (list (ones (list 2))))', env)
        run(f'(save-tree "{path}" (list (zeros (list 3))))', env)
        self.assertEqual(run(f'(shape (car (load-tree "{path}")))', env), [3])
        self.assertEqual([f for f in os.listdir(d) if "tmp" in f], [])


@unittest.skipUnless(HAVE_MLX and os.path.exists("corpus.txt"),
                     "mlx or corpus.txt not available")
class TestTraining(unittest.TestCase):
    """End-to-end smoke training in a tempdir: loss must decrease and
    checkpoints must appear. ~30s."""

    def test_smoke_training_loss_decreases(self):
        import re
        import shutil
        import subprocess
        import sys
        repo = os.getcwd()
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(repo, "model.lisp"), tmp)
        shutil.copy(os.path.join(repo, "train.lisp"), tmp)
        with open(os.path.join(repo, "corpus.txt")) as f:
            head = f.read(200_000)
        with open(os.path.join(tmp, "corpus.txt"), "w") as f:
            f.write(head)
        open(os.path.join(tmp, "SMOKE"), "w").close()
        out = subprocess.run(
            [sys.executable, os.path.join(repo, "mlx_lisp.py"), "train.lisp"],
            cwd=tmp, capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        losses = [float(m) for m in
                  re.findall(r'"train" ([0-9.]+)', out.stdout)]
        self.assertGreaterEqual(len(losses), 3, out.stdout[-2000:])
        self.assertLess(losses[-1], losses[0] - 0.5)
        self.assertTrue(os.path.exists(os.path.join(tmp, "ckpt.npz")))
        self.assertTrue(os.path.exists(os.path.join(tmp, "ckpt-best.npz")))


if __name__ == "__main__":
    unittest.main()
