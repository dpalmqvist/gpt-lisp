# Ambitious Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a ~4.8M-parameter multi-head GPT, written entirely in Lisp, on a ~2-4MB downloaded Lisp/Scheme corpus, with minibatched Adam, an mx.compile'd training step, checkpoints, resume, and standalone sampling.

**Architecture:** Three new Lisp files (`model.lisp` = config + pure architecture functions, `train.lisp` = data/Adam/loop/checkpoints, `sample.lisp` = generate from checkpoint) running on the existing interpreter; new capabilities are one-line MLX primitives added to `mlx_lisp.py`. A shell script builds the corpus. The existing `gpt.lisp` and `lisp.py` are NOT modified.

**Tech Stack:** Python 3 stdlib, MLX (`mlx.core`, `mlx.utils.tree_flatten/tree_unflatten`), POSIX sh, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-04-ambitious-training-run-design.md`
- Do NOT modify `gpt.lisp` or `lisp.py`.
- Anything importing MLX must run with `/tmp/mlx-venv/bin/python`. If that venv is missing, recreate it: `python3 -m venv /tmp/mlx-venv && /tmp/mlx-venv/bin/pip install mlx`.
- Plain `python3 -m unittest test_lisp` must stay green; `test_mlx_lisp.py` must auto-skip when MLX is absent.
- Model config (real / smoke): T=256/32, D=256/32, H=8/2, HD=32/16, L=6/2, B=64/8, V=128, steps=50000/50. Smoke mode = a file named `SMOKE` exists in the working directory.
- Adam: lr=3e-4, beta1=0.9, beta2=0.99, eps=1e-8, bias-corrected, step counter starts at 1.
- Checkpoint files: `ckpt.npz` (every 1000 steps, and resume source), `ckpt-best.npz` (on improved val loss). A checkpoint is the whole tree `(list step-array params m v)`.
- Git commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (use a heredoc).
- Known simplification (accepted in design): `best-val` resets on resume; `SMOKE` must not exist when training or sampling the real model (dims are read from the flag at load time).

---

### Task 1: Corpus pipeline

**Files:**
- Create: `fetch_corpus.sh`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `corpus.txt` in the repo root — printable-ASCII-only (bytes 0x09, 0x0A, 0x20–0x7E) concatenated Lisp/Scheme source, ≥1MB. Task 4 reads it via `(read-file "corpus.txt")`.

- [ ] **Step 1: Write `fetch_corpus.sh`**

```sh
#!/bin/sh
# Build corpus.txt: ASCII Lisp/Scheme source from two well-known repos.
set -e
mkdir -p corpus_repos
[ -d corpus_repos/paip-lisp ] || \
    git clone --depth 1 https://github.com/norvig/paip-lisp corpus_repos/paip-lisp
[ -d corpus_repos/chibi-scheme ] || \
    git clone --depth 1 https://github.com/ashinn/chibi-scheme corpus_repos/chibi-scheme
find corpus_repos \( -name '*.lisp' -o -name '*.scm' \) -type f | sort | xargs cat \
    | LC_ALL=C tr -cd '\11\12\40-\176' > corpus.txt
wc -c corpus.txt
```

- [ ] **Step 2: Add ignore entries**

Append to `.gitignore`:

```
corpus_repos/
corpus.txt
*.npz
SMOKE
```

- [ ] **Step 3: Run and verify**

Run: `chmod +x fetch_corpus.sh && ./fetch_corpus.sh`
Expected: `wc -c` reports ≥ 1000000 bytes.

Then verify ASCII-cleanliness (must print `clean`):

```sh
python3 -c "d=open('corpus.txt','rb').read(); assert all(b in (9,10) or 32<=b<127 for b in d); print('clean', len(d))"
```

- [ ] **Step 4: Commit**

```bash
git add fetch_corpus.sh .gitignore
git commit -m "$(cat <<'EOF'
Add corpus fetch script (paip-lisp + chibi-scheme, ASCII-filtered)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: MLX primitives

**Files:**
- Modify: `mlx_lisp.py` (imports + the `env.update` table + after it)
- Create: `test_mlx_lisp.py`

**Interfaces:**
- Consumes: existing `gpu_env`, `tokenize`, `parse`, `evaluate` from `lisp.py`/`mlx_lisp.py`.
- Produces Lisp primitives used by Tasks 3-5: `(randint lo hi shape)`, `(transpose-axes a axes)`, `(pow a b)`, `(save-tree path tree)`, `(load-tree path)`, `(read-file path)`, `(exists? path)`, `(load path)`, `(seed n)`.
- Produces test helper `run(src, env)` in `test_mlx_lisp.py` reused by Task 3's tests.

- [ ] **Step 1: Write the failing tests**

Create `test_mlx_lisp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v`
Expected: FAIL/ERROR with `unbound symbol: randint` (and similar) — NOT import errors.

Also run: `python3 -m unittest test_mlx_lisp -v`
Expected: all tests skipped ("mlx not installed") — only if the system python3 lacks mlx.

- [ ] **Step 3: Implement the primitives**

In `mlx_lisp.py`, extend the imports:

```python
import functools
import operator as op
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
```

Add to the `env.update({...})` table (before the closing `})`):

```python
        # --- scaling-run primitives ---
        "randint": lambda lo, hi, shape: mx.random.randint(lo, hi, shape),
        "transpose-axes": lambda a, axes: mx.transpose(a, axes),
        "pow": lambda a, b: a ** b,
        "save-tree": lambda path, tree: mx.savez(path, **dict(tree_flatten(tree))),
        "load-tree": lambda path: tree_unflatten(list(mx.load(path).items())),
        "read-file": lambda path: String(open(path).read()),
        "exists?": os.path.exists,
        "seed": lambda n: mx.random.seed(n),
```

After the `env.update({...})` call, before `return env`, add `load` (it needs the env itself):

```python
    def lisp_load(path):
        with open(path) as f:
            toks = tokenize(f.read())
        while toks:
            evaluate(parse(toks), env)

    env["load"] = lisp_load
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v`
Expected: all PASS.
Run: `python3 -m unittest test_lisp`
Expected: 23 tests, OK (interpreter untouched).

- [ ] **Step 5: Commit**

```bash
git add mlx_lisp.py test_mlx_lisp.py
git commit -m "$(cat <<'EOF'
Add MLX primitives for the scaled run: randint, transpose-axes, pow,
save-tree/load-tree, read-file, exists?, load, seed

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: model.lisp — config + architecture

**Files:**
- Create: `model.lisp`
- Modify: `test_mlx_lisp.py` (append a test class)

**Interfaces:**
- Consumes: Task 2 primitives; existing primitives (`matmul`, `swap`, `softmax`, `meank`, `square`, `relu`, `take`, `causal-mask`, `log-softmax`, `onehot`, `sumlast`, `sample`, `at`, `text->ids`, `ids->text`).
- Produces (used by Tasks 4-5): config symbols `smoke T V D H HD L B`; `(init-params)` → params tree; `(gpt p ids)` → (batch, t, V) logits; `(batch-loss p xb yb)` → scalar loss; `(sample-text p n temp)` → String of n generated chars from prompt `"(define "`.
- Params tree layout: index 0 token emb (V,D); 1 position emb (T,D); 2..L+1 blocks, each `(ln1 q k v out ln2 up down)` where ln = `(gamma beta)` and linear = `(weight bias)`; L+2 final ln; L+3 head linear (D,V).

- [ ] **Step 1: Write the failing tests**

Append to `test_mlx_lisp.py`:

```python
@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestModel(unittest.TestCase):
    """Runs model.lisp in smoke config (T=32, D=32, H=2, L=2)."""

    def setUp(self):
        open("SMOKE", "w").close()
        self.env = gpu_env()
        run('(load "model.lisp")', self.env)

    def tearDown(self):
        os.remove("SMOKE")

    def test_smoke_config_active(self):
        self.assertEqual(run("(list T D H L)", self.env), [32, 32, 2, 2])

    def test_gpt_output_shape(self):
        r = run("(define p (init-params)) "
                "(shape (gpt p (randint 0 V (list 2 T))))", self.env)
        self.assertEqual(r, [2, 32, 128])

    def test_initial_loss_near_uniform(self):
        # untrained loss should be near -log(1/128) = 4.85
        loss = run("(define p (init-params)) "
                   "(item (batch-loss p (randint 0 V (list 4 T)) "
                   "(randint 0 V (list 4 T))))", self.env)
        self.assertGreater(loss, 3.0)
        self.assertLess(loss, 7.0)

    def test_sample_text_length(self):
        r = run("(define p (init-params)) (sample-text p 5 1.0)", self.env)
        # prompt "(define " is 8 chars + 5 generated
        self.assertEqual(len(r), 13)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp.TestModel -v`
Expected: ERROR — `model.lisp` does not exist (FileNotFoundError from `load`).

- [ ] **Step 3: Write model.lisp**

```lisp
; ============================================================
; model.lisp — config + architecture for the scaled Lisp GPT.
; Pure functions only; no training state. Loaded by train.lisp
; and sample.lisp. A file named SMOKE selects the tiny config.
; ============================================================

(define smoke (exists? "SMOKE"))

(define T (if smoke 32 256))     ; context length
(define V 128)                   ; vocab = ASCII
(define D (if smoke 32 256))     ; model width
(define H (if smoke 2 8))        ; attention heads
(define HD (if smoke 16 32))     ; head dim = D / H
(define L (if smoke 2 6))        ; transformer blocks
(define B (if smoke 8 64))       ; batch size

; --- parameters ---
(define (linear nin nout)
  (list (* (randn (list nin nout)) (/ 1.0 (sqrt (* 1.0 nin))))
        (zeros (list nout))))

(define (ln-params) (list (ones (list D)) (zeros (list D))))

(define (block-params)           ; ln1 q k v out ln2 up down
  (list (ln-params) (linear D D) (linear D D) (linear D D) (linear D D)
        (ln-params) (linear D (* 4 D)) (linear (* 4 D) D)))

(define (init-params)            ; 0: tok emb, 1: pos emb, 2..L+1: blocks,
  (append                        ; L+2: final ln, L+3: head
    (list (* (randn (list V D)) 0.02)
          (* (randn (list T D)) 0.02))
    (map (lambda (i) (block-params)) (range L))
    (list (ln-params) (linear D V))))

; --- architecture ---
(define (dense p x) (+ (matmul x (car p)) (nth p 1)))

(define (ln p x)                 ; layernorm with learned scale + shift
  (begin
    (define xc (- x (meank x)))
    (+ (* (car p) (/ xc (sqrt (+ (meank (square xc)) 0.00001))))
       (nth p 1))))

(define (heads x)                ; (B T D) -> (B H T HD)
  (transpose-axes
    (reshape x (list (nth (shape x) 0) (nth (shape x) 1) H HD))
    (list 0 2 1 3)))

(define (unheads x)              ; (B H T HD) -> (B T D)
  (begin
    (define y (transpose-axes x (list 0 2 1 3)))
    (reshape y (list (nth (shape y) 0) (nth (shape y) 1) D))))

(define (attn p x m)             ; p: (q k v out) — multi-head causal
  (begin
    (define q (heads (dense (nth p 0) x)))
    (define k (heads (dense (nth p 1) x)))
    (define v (heads (dense (nth p 2) x)))
    (define scores (+ (/ (matmul q (swap k)) (sqrt (* 1.0 HD))) m))
    (dense (nth p 3) (unheads (matmul (softmax scores) v)))))

(define (block p x m)            ; p: (ln1 q k v out ln2 up down)
  (begin
    (define h (+ x (attn (slice p 1 5) (ln (car p) x) m)))
    (+ h (dense (nth p 7) (relu (dense (nth p 6) (ln (nth p 5) h)))))))

(define (run-blocks p i x m)
  (if (= i L) x (run-blocks p (+ i 1) (block (nth p (+ 2 i)) x m) m)))

(define (gpt p ids)
  (begin
    (define t (nth (shape ids) 1))
    (define x (+ (take (car p) ids)
                 (take (nth p 1) (array (range t) "int32"))))
    (define xf (run-blocks p 0 x (causal-mask t)))
    (dense (nth p (+ L 3)) (ln (nth p (+ L 2)) xf))))

(define (batch-loss p xb yb)     ; mean cross-entropy over batch
  (- (mean (sumlast (* (log-softmax (gpt p xb)) (onehot yb V))))))

; --- generation ---
(define (last-k xs k)
  (if (<= (length xs) k) xs (slice xs (- (length xs) k) (length xs))))

(define (gen p ctx n temp)
  (if (= n 0)
      ctx
      (begin
        (define logits (gpt p (array (list (last-k ctx T)) "int32")))
        (define nxt (item (sample (at logits 0 -1) temp)))
        (gen p (append ctx (list nxt)) (- n 1) temp))))

(define (sample-text p n temp)
  (ids->text (gen p (text->ids "(define ") n temp)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v`
Expected: all PASS (primitives + model classes).

- [ ] **Step 5: Commit**

```bash
git add model.lisp test_mlx_lisp.py
git commit -m "$(cat <<'EOF'
Add model.lisp: 6-block 8-head GPT architecture in pure Lisp

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: train.lisp — Adam loop, checkpoints, resume

**Files:**
- Create: `train.lisp`

**Interfaces:**
- Consumes: `model.lisp` (config, `init-params`, `batch-loss`, `sample-text`), `corpus.txt` (Task 1), primitives (Task 2).
- Produces: `ckpt.npz` / `ckpt-best.npz` — trees `(list step-array params m v)` where `step-array` is a 1-element int32 array. Task 5 reads index 1 (params).

- [ ] **Step 1: Write train.lisp**

```lisp
; ============================================================
; train.lisp — minibatched Adam training with checkpoints/resume.
; Run: /tmp/mlx-venv/bin/python mlx_lisp.py train.lisp
; Smoke test: touch SMOKE first (tiny model, 50 steps).
; ============================================================

(load "model.lisp")
(defmacro (when test body) `(if ,test ,body))

(define steps (if smoke 50 50000))
(define lr 0.0003)
(define b1 0.9)
(define b2 0.99)
(define eps 0.00000001)
(define log-every (if smoke 10 250))
(define ckpt-every (if smoke 25 1000))
(define nval-batch 4)

; --- data: char ids, last 5% is validation ---
(define ids (text->ids (read-file "corpus.txt")))
(define n (length ids))
(define nval (floor (* 0.05 n)))
(define ntrain (- n nval))
(define train-ids (array (slice ids 0 ntrain) "int32"))
(define val-ids (array (slice ids ntrain n) "int32"))
(define max-off (- ntrain T))

(define tvec (array (range T) "int32"))
(define (get-batch src nmax)     ; -> (xb yb), each (B T) int32
  (begin
    (define pos (+ (reshape (randint 0 nmax (list B)) (list B 1)) tvec))
    (list (take src pos) (take src (+ pos 1)))))

(seed 1234)                      ; fixed val batches for comparable losses
(define val-batches
  (map (lambda (i) (get-batch val-ids (- nval T))) (range nval-batch)))
(seed 42)

(define (val-loss p)
  (/ (apply + (map (lambda (vb) (item (batch-loss p (car vb) (nth vb 1))))
                   val-batches))
     (* 1.0 nval-batch)))

; --- Adam as generic tree maps, like gpt.lisp's sgd ---
(define (tree-map2 f a b)
  (if (list? a) (map (lambda (x y) (tree-map2 f x y)) a b) (f a b)))

(define (tree-map3 f a b c)
  (if (list? a) (map (lambda (x y z) (tree-map3 f x y z)) a b c) (f a b c)))

(define (tree-zeros p) (if (list? p) (map tree-zeros p) (* 0.0 p)))

(define (adam-step p g m v t)    ; -> (p m v) updated
  (begin
    (define m2 (tree-map2 (lambda (mi gi) (+ (* b1 mi) (* (- 1.0 b1) gi))) m g))
    (define v2 (tree-map2 (lambda (vi gi) (+ (* b2 vi) (* (- 1.0 b2) (square gi)))) v g))
    (define c1 (- 1.0 (pow b1 t)))
    (define c2 (- 1.0 (pow b2 t)))
    (define p2 (tree-map3
                 (lambda (pi mi vi)
                   (- pi (* lr (/ (/ mi c1) (+ (sqrt (/ vi c2)) eps)))))
                 p m2 v2))
    (list p2 m2 v2)))

; --- compiled train step: traced through the interpreter ONCE ---
(define step-fn (jit (vgrad batch-loss)))

(define (save-ckpt path step p m v)
  (save-tree path (list (array (list step) "int32") p m v)))

(define best-val 999.0)

(define (train-loop p m v step)
  (if (> step steps)
      p
      (begin
        (define bpair (get-batch train-ids max-off))
        (define lg (step-fn p (car bpair) (nth bpair 1)))
        (define pmv (adam-step p (nth lg 1) m v step))
        (define p2 (car pmv))
        (define m2 (nth pmv 1))
        (define v2 (nth pmv 2))
        (force-tree (list p2 m2 v2))
        (when (= (mod step log-every) 0)
          (begin
            (define vl (val-loss p2))
            (display (list "step" step "train" (item (car lg)) "val" vl))
            (when (< vl best-val)
              (begin
                (set! best-val vl)
                (save-ckpt "ckpt-best.npz" step p2 m2 v2)))))
        (when (= (mod step ckpt-every) 0)
          (begin
            (save-ckpt "ckpt.npz" step p2 m2 v2)
            (display (sample-text p2 150 0.5))))
        (train-loop p2 m2 v2 (+ step 1)))))

; --- init or resume ---
(define state0
  (if (exists? "ckpt.npz")
      (begin
        (display "resuming from ckpt.npz")
        (define st (load-tree "ckpt.npz"))
        (list (item (car st)) (nth st 1) (nth st 2) (nth st 3)))
      (begin
        (define p (init-params))
        (list 0 p (tree-zeros p) (tree-zeros p)))))

(display (list "device" (device) "corpus-chars" n "smoke" smoke
               "start-step" (+ (car state0) 1) "of" steps))

(define final
  (train-loop (nth state0 1) (nth state0 2) (nth state0 3)
              (+ (car state0) 1)))

(display (list "done. final val loss" (val-loss final)))
(display (sample-text final 300 0.5))
```

- [ ] **Step 2: Smoke run end-to-end**

```sh
rm -f ckpt.npz ckpt-best.npz && touch SMOKE
/tmp/mlx-venv/bin/python mlx_lisp.py train.lisp
```

Expected: prints device/config line with `start-step 1 of 50`; loss starts near 4.85 and *decreases* by step 50 (any clear downward trend passes); `ckpt.npz` and `ckpt-best.npz` exist afterwards.

- [ ] **Step 3: Verify resume**

Run again without deleting checkpoints:

```sh
/tmp/mlx-venv/bin/python mlx_lisp.py train.lisp
```

Expected: prints `resuming from ckpt.npz` and `start-step 51 of 50`, then exits immediately printing final val loss + a sample (training already complete).

- [ ] **Step 4: Clean up smoke artifacts**

```sh
rm -f SMOKE ckpt.npz ckpt-best.npz
```

- [ ] **Step 5: Commit**

```bash
git add train.lisp
git commit -m "$(cat <<'EOF'
Add train.lisp: minibatched Adam, compiled train step, checkpoints, resume

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: sample.lisp

**Files:**
- Create: `sample.lisp`

**Interfaces:**
- Consumes: `model.lisp` (`sample-text`), checkpoint trees from Task 4 (`params` at index 1).

- [ ] **Step 1: Write sample.lisp**

```lisp
; sample.lisp — generate from the best checkpoint.
; Run: /tmp/mlx-venv/bin/python mlx_lisp.py sample.lisp
(load "model.lisp")

(define ckpt (if (exists? "ckpt-best.npz") "ckpt-best.npz" "ckpt.npz"))
(define st (load-tree ckpt))
(display (list "checkpoint" ckpt "step" (item (car st))))
(display (sample-text (nth st 1) 500 0.6))
```

- [ ] **Step 2: Verify against a smoke checkpoint**

```sh
touch SMOKE && rm -f ckpt.npz ckpt-best.npz
/tmp/mlx-venv/bin/python mlx_lisp.py train.lisp > /dev/null
/tmp/mlx-venv/bin/python mlx_lisp.py sample.lisp
rm -f SMOKE ckpt.npz ckpt-best.npz
```

Expected: sample.lisp prints the checkpoint name + step, then ~508 chars of (gibberish-ish, 50-step) text starting with `(define `.

- [ ] **Step 3: Commit**

```bash
git add sample.lisp
git commit -m "$(cat <<'EOF'
Add sample.lisp: standalone generation from checkpoints

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Launch the real run

**Files:** none created (produces `train.log`, `ckpt.npz`, `ckpt-best.npz` — all gitignored; add `train.log` to `.gitignore`).

**Interfaces:**
- Consumes: everything above; `corpus.txt` must exist (re-run `./fetch_corpus.sh` if not).

- [ ] **Step 1: Preflight**

```sh
ls corpus.txt || ./fetch_corpus.sh
rm -f SMOKE ckpt.npz ckpt-best.npz
echo train.log >> .gitignore
```

- [ ] **Step 2: Launch in background with logging**

Run as a background task (nohup so it survives the session):

```sh
nohup /tmp/mlx-venv/bin/python mlx_lisp.py train.lisp > train.log 2>&1 &
```

- [ ] **Step 3: Verify liftoff**

After the first log lines appear (allow a few minutes — the jit trace of the full model happens on step 1):

```sh
tail -5 train.log
```

Expected: config line shows `smoke #f`, `corpus-chars` ≥ 1000000, `start-step 1 of 50000`; first logged `train` loss is 3-7 and subsequent logs decrease. If step 250 hasn't logged within ~30 minutes, investigate before leaving it overnight.

- [ ] **Step 4: Commit the gitignore tweak**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
Ignore train.log

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: README + publish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Scaled training run" section to README.md**

Insert before the `## Tests` section:

````markdown
## Scaled training run

The ambitious version: a ~4.8M-parameter, 6-block, 8-head GPT trained on real
Lisp — Norvig's *Paradigms of AI Programming* code plus the chibi-scheme
library tree (~MBs of ASCII source).

```sh
./fetch_corpus.sh                        # build corpus.txt
python3 mlx_lisp.py train.lisp           # train (resumes from ckpt.npz)
python3 mlx_lisp.py sample.lisp          # generate from best checkpoint
```

- `model.lisp` — config + architecture (multi-head attention, learned
  layernorm), pure functions.
- `train.lisp` — minibatched Adam in Lisp tree-maps, an `mx.compile`'d train
  step (the interpreter is traced once, then replays as a fused graph),
  checkpoints every 1000 steps, automatic resume.
- `sample.lisp` — load `ckpt-best.npz`, generate.
- `touch SMOKE` first for a 1-minute end-to-end pipeline test.
````

(The README uses plain `python3` — readers install mlx themselves per the Run section. Local runs in this environment still use `/tmp/mlx-venv/bin/python`.)

- [ ] **Step 2: Verify test suites one final time**

```sh
python3 -m unittest test_lisp
/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp
```

Expected: both OK.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Document the scaled training run

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
git push
```

- [ ] **Step 4: Report training progress**

Check `tail train.log` and report the current step, train/val loss, and the latest generated sample to the user. Next morning, run `sample.lisp` for the final showcase output.
