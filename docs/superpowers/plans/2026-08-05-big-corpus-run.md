# Big-Corpus Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the run-protecting checkpoint fixes, grow the corpus to ~60MB, and launch a fresh 50,000-step run that shouldn't overfit.

**Architecture:** Same three-file Lisp pipeline as run 1. Changes: atomic `save-tree` in `mlx_lisp.py`; checkpoint tree gains a persisted `best-val`; `sample.lisp` gains a guard + format compat; tests move to tempdirs and gain a real training test; `fetch_corpus.sh` gains Emacs/SBCL/Guile.

**Tech Stack:** Python 3 + MLX (`/tmp/mlx-venv/bin/python` locally), POSIX sh, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-big-corpus-run-design.md`
- Run 1 artifacts to preserve: the best checkpoint (archived as `run1-best.npz`). Never delete `corpus_repos/` clones (re-fetch is expensive).
- New checkpoint tree format: `(list step-array best-array params m v)` — step-array 1-elem int32, best-array 1-elem float32, params at index 2. Legacy 4-element trees (run 1) must still load in `sample.lisp` via a length check.
- MLX runs use `/tmp/mlx-venv/bin/python`; `python3 -m unittest test_lisp` must stay green (23 tests).
- Do NOT modify `gpt.lisp`, `lisp.py`, or `model.lisp`.
- Training launches use `PYTHONUNBUFFERED=1` (runbook — plain nohup buffers stdout invisibly).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (heredoc).

---

### Task 1: Checkpoint hardening + test overhaul

**Files:**
- Modify: `mlx_lisp.py` (the `save-tree` line)
- Modify: `train.lisp` (save-ckpt, state0, best-val, train-loop save calls)
- Modify: `sample.lisp` (guard + new format)
- Modify: `test_mlx_lisp.py` (TestModel tempdir; new TestTraining)
- Modify: `README.md` (Tests section)

**Interfaces:**
- Produces: checkpoint format `(list step-array best-array params m v)` (see Global Constraints) — Task 3's run writes it; `sample.lisp` consumes it.
- Consumes: existing `model.lisp` functions unchanged.

- [ ] **Step 1: Write the failing tests**

In `test_mlx_lisp.py`, REPLACE the existing `TestModel` class with the version below (tempdir cwd, exception-safe cleanup), and ADD `TestSaveAtomic` and `TestTraining`:

```python
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
```

- [ ] **Step 2: Establish baseline + RED evidence**

This task is a behavior-preserving refactor plus new coverage, so RED comes
from sabotage, not from missing features:

1. Run `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v` with the new
   tests but UNCHANGED production code → all green (baseline; proves the new
   tests accept current behavior).
2. Temporarily edit `save-tree` to write only `path + ".tmp.npz"` and skip
   the final rename → re-run: `TestSaveAtomic` and `TestTraining` must FAIL
   (proves the tests detect a broken checkpoint pipeline). Record this
   output as RED evidence.
3. Revert the sabotage.

- [ ] **Step 3: Implement the changes**

`mlx_lisp.py` — replace the `save-tree` line:

```python
        "save-tree": lambda path, tree: (
            mx.savez(path + ".tmp.npz", **dict(tree_flatten(tree))),
            os.replace(path + ".tmp.npz", path))[-1],
```

`train.lisp` — replace `save-ckpt`, the best-val definition, both save calls, `state0`, and the final `train-loop` invocation:

```lisp
(define (save-ckpt path step best p m v)
  (save-tree path (list (array (list step) "int32")
                        (array (list best) "float32") p m v)))
```

In `train-loop`, the log branch becomes:

```lisp
        (when (= (mod step log-every) 0)
          (begin
            (define vl (val-loss p2))
            (display (list "step" step "train" (item (car lg)) "val" vl))
            (when (< vl best-val)
              (begin
                (set! best-val vl)
                (save-ckpt "ckpt-best.npz" step vl p2 m2 v2)))))
        (when (= (mod step ckpt-every) 0)
          (begin
            (save-ckpt "ckpt.npz" step best-val p2 m2 v2)
            (display (sample-text p2 150 0.5))))
```

`state0` / `best-val` / final call become:

```lisp
(define state0
  (if (exists? "ckpt.npz")
      (begin
        (display "resuming from ckpt.npz")
        (define st (load-tree "ckpt.npz"))
        (list (item (car st)) (item (nth st 1))
              (nth st 2) (nth st 3) (nth st 4)))
      (begin
        (define p (init-params))
        (list 0 999.0 p (tree-zeros p) (tree-zeros p)))))

(define best-val (nth state0 1))

(display (list "device" (device) "corpus-chars" n "smoke" smoke
               "start-step" (+ (car state0) 1) "of" steps
               "best-val" best-val))

(define final
  (train-loop (nth state0 2) (nth state0 3) (nth state0 4)
              (+ (car state0) 1)))
```

(Delete the old standalone `(define best-val 999.0)` near the top; keep everything else in the file unchanged.)

`sample.lisp` — full new content:

```lisp
; sample.lisp — generate from the best checkpoint.
; Run: python3 mlx_lisp.py sample.lisp
(load "model.lisp")

(define ckpt (if (exists? "ckpt-best.npz") "ckpt-best.npz"
             (if (exists? "ckpt.npz") "ckpt.npz" #f)))
(if (equal? ckpt #f)
    (display "no checkpoint found - run train.lisp first")
    (begin
      (define st (load-tree ckpt))
      (define params (if (= (length st) 5) (nth st 2) (nth st 1)))
      (display (list "checkpoint" ckpt "step" (item (car st))))
      (display (sample-text params 500 0.6))))
```

`train.lisp` header comment: change `/tmp/mlx-venv/bin/python` to `python3`.

`README.md` Tests section — replace the single command block with:

```sh
python3 -m unittest test_lisp        # interpreter (no mlx needed)
python3 -m unittest test_mlx_lisp    # MLX layer + model + training smoke
```

- [ ] **Step 4: Run tests to verify green**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v` (expect all pass incl. TestTraining, ~40s)
Run: `python3 -m unittest test_lisp` (expect 23 OK)
Also verify legacy-format loading: `/tmp/mlx-venv/bin/python mlx_lisp.py sample.lisp` with run-1's 4-element `run1-best.npz` copied to `ckpt-best.npz` in a tempdir alongside `model.lisp` — prints step 4000 and generates. (If run1 archive is not yet created, use `ckpt-best.backup.npz` as the source.)

- [ ] **Step 5: Commit**

```bash
git add mlx_lisp.py train.lisp sample.lisp test_mlx_lisp.py README.md
git commit -m "$(cat <<'EOF'
Harden checkpoints: atomic saves, persist best-val, sample guard, training test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Corpus expansion to ~60MB

**Files:**
- Modify: `fetch_corpus.sh`
- Modify: `README.md` (corpus description sentence)

**Interfaces:**
- Produces: `corpus.txt` ≥ 40MB (expected 50-70MB), ASCII-only, from 5 sources. Task 3 trains on it.

- [ ] **Step 1: Rewrite fetch_corpus.sh**

```sh
#!/bin/sh
# Build corpus.txt: ASCII Lisp/Scheme/Elisp source from well-known repos.
set -e
mkdir -p corpus_repos
clone() { [ -d "corpus_repos/$2" ] || git clone --depth 1 "$1" "corpus_repos/$2"; }
clone https://github.com/norvig/paip-lisp paip-lisp
clone https://github.com/ashinn/chibi-scheme chibi-scheme
clone https://github.com/sbcl/sbcl sbcl
clone https://git.savannah.gnu.org/git/guile.git guile || echo "warning: guile clone failed, continuing without it"
if [ ! -d corpus_repos/emacs ]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/emacs-mirror/emacs corpus_repos/emacs
    git -C corpus_repos/emacs sparse-checkout set lisp
fi
find corpus_repos \( -name '*.lisp' -o -name '*.scm' -o -name '*.el' \) -type f \
    | sort | xargs cat | LC_ALL=C tr -cd '\11\12\40-\176' > corpus.txt
wc -c corpus.txt
```

Note: `clone ... || echo` keeps `set -e` from aborting if Savannah is down. If the emacs sparse clone fails (old git), fall back to a plain `--depth 1` clone.

- [ ] **Step 2: README corpus sentence**

In the "Scaled training run" section, replace the corpus sentence with:

```markdown
Lisp — Norvig's *Paradigms of AI Programming* code, chibi-scheme, SBCL,
Guile, and GNU Emacs's Elisp tree (~60MB of ASCII source).
```

- [ ] **Step 3: Run and verify**

Run: `./fetch_corpus.sh` (emacs sparse clone downloads tens of MB; allow ~10 min)
Expected: `wc -c` ≥ 40000000.
Verify ASCII: `python3 -c "d=open('corpus.txt','rb').read(); assert all(b in (9,10) or 32<=b<127 for b in d); print('clean', len(d))"`

- [ ] **Step 4: Commit**

```bash
git add fetch_corpus.sh README.md
git commit -m "$(cat <<'EOF'
Grow corpus to ~60MB: add SBCL, Guile, and Emacs lisp/ (sparse)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fresh launch

**Files:** none committed (produces run artifacts, all gitignored).

- [ ] **Step 1: Archive run 1 and clear state**

```sh
[ -f run1-best.npz ] || mv ckpt-best.npz run1-best.npz
rm -f ckpt.npz ckpt-best.npz SMOKE
```

(`ckpt-best.backup.npz` may be deleted after `run1-best.npz` exists.)

- [ ] **Step 2: Launch unbuffered**

```sh
nohup env PYTHONUNBUFFERED=1 /tmp/mlx-venv/bin/python mlx_lisp.py train.lisp > train.log 2>&1 &
```

- [ ] **Step 3: Verify liftoff**

Within ~5 min `train.log` must show: `smoke #f`, `corpus-chars` ≥ 40000000, `start-step 1 of 50000`, `best-val 999.0`; first logged train loss 3-7 and decreasing on subsequent logs. The first minutes include corpus load (~55M-char `text->ids`) and the jit trace — silence for 2-4 min is normal.

---

### Task 4: Push + wrap-up

- [ ] **Step 1: Final suites** — `python3 -m unittest test_lisp` and `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp` both OK.
- [ ] **Step 2: Push the branch**; integration decision goes through superpowers:finishing-a-development-branch (controller).
- [ ] **Step 3 (controller):** monitor the run; report val trajectory; next morning run `sample.lisp` for the showcase.
