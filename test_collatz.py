"""Tests for collatz.lisp. Auto-skips when mlx is not installed."""

import os
import subprocess
import sys
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

REPO = os.path.dirname(os.path.abspath(__file__))
RUN_MARKER = "; --- run ---"


def run(src, env):
    tokens = tokenize(src)
    result = None
    while tokens:
        result = evaluate(parse(tokens), env)
    return result


def collatz_env():
    """gpu_env plus collatz.lisp's definitions (not its driver section)."""
    with open(os.path.join(REPO, "collatz.lisp")) as f:
        defs = f.read().split(RUN_MARKER)[0]
    env = gpu_env()
    run(defs, env)
    return env


def py_glide_peak(n0):
    """Reference implementation on Python bignums."""
    n, peak = n0, n0
    while n >= n0:
        n = n // 2 if n % 2 == 0 else (3 * n + 1) // 2
        peak = max(peak, n)
    return peak


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestPrimitives(unittest.TestCase):
    def setUp(self):
        self.env = gpu_env()

    def test_floordiv_is_exact_on_bignums(self):
        n = 8528817511 * 10 ** 12  # far beyond int64
        self.assertEqual(run(f"(// (+ (* 3 {n}) 1) 2)", self.env),
                         (3 * n + 1) // 2)

    def test_int64_where_argsort(self):
        self.assertEqual(
            run('(item (sum (int64 (> (array (list 1 5 9) "int64") 4))))',
                self.env), 2)
        self.assertEqual(
            run('(item (where #t (array (list 7) "int64") (array (list 8) "int64")))',
                self.env), 7)
        self.assertEqual(
            run('(item (at (argsort (array (list 3 1 2) "int64")) 0))',
                self.env), 1)

    def test_iota_exact_past_float32_precision(self):
        # float32 arange would round 2^24+1; iota must not
        self.assertEqual(run("(item (at (iota 16777220) 16777217))", self.env),
                         16777217)


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestCollatzUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = collatz_env()

    def test_scalar_glides(self):
        self.assertIs(run("(scalar-glides? 27)", self.env), True)
        # glide peak 9.07e18 exceeds int64's safe range: bignum territory
        self.assertIs(run("(scalar-glides? 8528817511)", self.env), True)

    def test_extract_values(self):
        self.assertEqual(
            run('(extract-values (array (list 10 20 30) "int64") '
                '(array (list 1 0 1) "int64"))', self.env), [10, 30])

    def test_sieve_mod16_survivors_exact(self):
        self.assertEqual(run("(sieve-survivors 4)", self.env), [7, 11, 15])

    def test_sieve_mod256_sound_and_consistent(self):
        surv = run("(sieve-survivors 8)", self.env)
        # refinement: every mod-256 survivor reduces to a mod-16 survivor
        self.assertTrue(all(r % 16 in (7, 11, 15) for r in surv))
        self.assertLess(len(surv), 40)
        # soundness: every *eliminated* residue really glides, k = 1..3
        eliminated = sorted(set(range(256)) - set(surv))
        for r in eliminated[::17] + eliminated[-3:]:
            for k in (1, 2, 3):
                self.assertIs(
                    run(f"(scalar-glides? {256 * k + r})", self.env), True,
                    f"sieve wrongly eliminated r={r} (n={256 * k + r})")

    def test_glide_lanes_verifies_and_escalates(self):
        # 8528817511 overflows int64 mid-glide -> must escalate to tier 3
        self.assertGreater(py_glide_peak(8528817511), (2 ** 63 - 2) // 3)
        run("(set! escalated 0) (set! failures 0)", self.env)
        run('(glide-lanes (array (list 4 27 100003 8528817511) "int64"))',
            self.env)
        self.assertEqual(run("escalated", self.env), 1)
        self.assertEqual(run("failures", self.env), 0)


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestSmokeRun(unittest.TestCase):
    def test_smoke_verifies_ten_million(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "SMOKE"), "w").close()
            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "mlx_lisp.py"),
                 os.path.join(REPO, "collatz.lisp")],
                cwd=tmp, capture_output=True, text=True, timeout=600,
                env={**os.environ, "PYTHONPATH": REPO},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("VERIFIED", proc.stdout)
            self.assertIn('"failures" 0', proc.stdout)
            self.assertNotIn("COUNTEREXAMPLE", proc.stdout)


if __name__ == "__main__":
    unittest.main()
