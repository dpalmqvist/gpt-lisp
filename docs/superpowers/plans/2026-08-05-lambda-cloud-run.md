# Lambda Cloud Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a ~38M-param Lisp GPT on ~5GB of The Stack's Lisp-family code on a Lambda A100, same MLX codebase via CUDA backend.

**Architecture:** Additive changes only: `CLOUD` config tier, `read-corpus`/`int32` primitives + `corpus.bin` data path, jit'd val loss, split sample cadence, corpus-builder script, Lambda API helper scripts. Tasks 1-3 are local code (TDD); Tasks 4-6 are cloud operations run by the controller.

**Tech Stack:** MLX (+CUDA on box), numpy, HF `datasets` (box only), POSIX sh + curl (Lambda API), ssh/scp/tmux.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-lambda-cloud-run-design.md`
- Config tiers (SMOKE / LOCAL / CLOUD): T=32/256/512, D=32/256/512, H=2/8/8, HD=16/32/64, L=2/6/12, B=8/64/64, steps=50/50000/100000, sample-every=25/1000/5000. Flag files `SMOKE` and `CLOUD` in cwd; SMOKE wins if both exist.
- Laptop behavior with neither flag must be byte-identical to today's (LOCAL) behavior; `corpus.txt` path keeps working. Run 2 may still be training locally — never touch the running trainer, `train.log`, repo-root `*.npz`, `corpus.txt`, `corpus_repos/`, and never create `SMOKE`/`CLOUD` in the repo root.
- Local MLX runs: `/tmp/mlx-venv/bin/python` (pip install numpy into it if missing). `python3 -m unittest test_lisp` stays green (23).
- Secrets: Lambda key at `~/.lambda-api-key`, HF token at `~/.hf-token` — read from files, NEVER echoed into logs, transcripts, or committed files.
- Do NOT modify `gpt.lisp`, `lisp.py`.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (heredoc).

---

### Task 1: CLOUD tier + GB-scale data path

**Files:**
- Modify: `mlx_lisp.py` (imports + two primitives)
- Modify: `model.lisp` (config block only)
- Modify: `train.lisp` (steps/sample-every, data load, get-batch casts, jit'd val, sample cadence)
- Modify: `test_mlx_lisp.py` (new tests)

**Interfaces:**
- Produces: `(read-corpus path)` → uint8 mx.array of the file's bytes; `(int32 a)` → a cast to int32; config symbol `cloud`; `train.lisp` prefers `corpus.bin` over `corpus.txt`.
- Consumes: everything else unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test_mlx_lisp.py`:

```python
@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestCorpusBin(unittest.TestCase):
    def test_read_corpus_uint8_roundtrip(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.bin")
        with open(path, "wb") as f:
            f.write(bytes([40, 100, 101, 102, 41, 10]))  # "(def)\n"
        env = gpu_env()
        self.assertEqual(run(f'(shape (read-corpus "{path}"))', env), [6])
        self.assertEqual(run(f'(item (at (read-corpus "{path}") 0))', env), 40)

    def test_int32_cast(self):
        env = gpu_env()
        # sum of uint8 values would overflow at 255+255 if not widened
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.bin")
        with open(path, "wb") as f:
            f.write(bytes([255, 255]))
        r = run(f'(item (sum (int32 (read-corpus "{path}"))))', env)
        self.assertEqual(r, 510)


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class TestCloudConfig(unittest.TestCase):
    def test_cloud_flag_selects_cloud_tier(self):
        import shutil
        repo = os.getcwd()
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(repo, "model.lisp"), tmp)
        os.chdir(tmp)
        self.addCleanup(os.chdir, repo)
        open("CLOUD", "w").close()
        env = gpu_env()
        run('(load "model.lisp")', env)
        self.assertEqual(run("(list T D H HD L B)", env),
                         [512, 512, 8, 64, 12, 64])

    def test_smoke_beats_cloud(self):
        import shutil
        repo = os.getcwd()
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(repo, "model.lisp"), tmp)
        os.chdir(tmp)
        self.addCleanup(os.chdir, repo)
        open("CLOUD", "w").close()
        open("SMOKE", "w").close()
        env = gpu_env()
        run('(load "model.lisp")', env)
        self.assertEqual(run("(list T D L)", env), [32, 32, 2])
```

And inside `TestTraining`, add a second test that exercises the `corpus.bin`
branch (mirror of the existing test, but writing the 200KB slice as
`corpus.bin` bytes and NOT writing `corpus.txt`):

```python
    def test_smoke_training_from_corpus_bin(self):
        import re
        import shutil
        import subprocess
        import sys
        repo = os.getcwd()
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(repo, "model.lisp"), tmp)
        shutil.copy(os.path.join(repo, "train.lisp"), tmp)
        with open(os.path.join(repo, "corpus.txt"), "rb") as f:
            head = f.read(200_000)
        with open(os.path.join(tmp, "corpus.bin"), "wb") as f:
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
```

- [ ] **Step 2: Verify RED**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp.TestCorpusBin test_mlx_lisp.TestCloudConfig -v`
Expected: failures/errors — `unbound symbol: read-corpus`, `int32`, and CLOUD config assertions get `[256, 256, ...]` (no cloud tier yet). (`pip install numpy` into the venv first if missing.)
`test_smoke_training_from_corpus_bin` fails: train.lisp requires corpus.txt (FileNotFoundError in read-file).

- [ ] **Step 3: Implement**

`mlx_lisp.py` — add `import numpy as np` to the imports, and to the primitive table:

```python
        "read-corpus": lambda path: mx.array(np.fromfile(path, dtype=np.uint8)),
        "int32": lambda a: a.astype(mx.int32),
```

`model.lisp` — replace the config block:

```lisp
(define smoke (exists? "SMOKE"))
(define cloud (exists? "CLOUD"))

(define T (if smoke 32 (if cloud 512 256)))    ; context length
(define V 128)                                 ; vocab = ASCII
(define D (if smoke 32 (if cloud 512 256)))    ; model width
(define H (if smoke 2 8))                      ; attention heads
(define HD (if smoke 16 (if cloud 64 32)))     ; head dim = D / H
(define L (if smoke 2 (if cloud 12 6)))        ; transformer blocks
(define B (if smoke 8 64))                     ; batch size
```

`train.lisp` — four edits:

1. Config lines:

```lisp
(define steps (if smoke 50 (if cloud 100000 50000)))
(define sample-every (if smoke 25 (if cloud 5000 1000)))
```

(`log-every`, `ckpt-every`, Adam hyperparameters unchanged.)

2. Data load (replace the `(define ids ...)` line):

```lisp
(define ids (if (exists? "corpus.bin")
                (read-corpus "corpus.bin")
                (array (text->ids (read-file "corpus.txt")) "int32")))
```

(`length`, `slice` work on mx.arrays: Python `len`/slicing.)

3. `get-batch` gains int32 casts:

```lisp
(define (get-batch src nmax)     ; -> (xb yb), each (B T) int32
  (begin
    (define pos (+ (reshape (randint 0 nmax (list B)) (list B 1)) tvec))
    (list (int32 (take src pos)) (int32 (take src (+ pos 1))))))
```

4. Compiled val + split sample cadence: after `(define step-fn ...)` add
`(define val-fn (jit batch-loss))`, change `val-loss`'s body to use
`(val-fn p (car vb) (nth vb 1))` in place of `(batch-loss ...)`, and split
the checkpoint `when` into two:

```lisp
        (when (= (mod step ckpt-every) 0)
          (save-ckpt "ckpt.npz" step best-val p2 m2 v2))
        (when (= (mod step sample-every) 0)
          (display (sample-text p2 150 0.5)))
```

- [ ] **Step 4: Verify GREEN**

Run: `/tmp/mlx-venv/bin/python -m unittest test_mlx_lisp -v` — all pass (both TestTraining subprocess tests ~40s each).
Run: `python3 -m unittest test_lisp` — 23 OK.

- [ ] **Step 5: Commit**

```bash
git add mlx_lisp.py model.lisp train.lisp test_mlx_lisp.py
git commit -m "$(cat <<'EOF'
Add CLOUD config tier and GB-scale corpus.bin data path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: build_stack_corpus.py

**Files:**
- Create: `build_stack_corpus.py`
- Create: `test_build_stack_corpus.py`

**Interfaces:**
- Produces: `corpus.bin` (shuffled ASCII bytes) when run on a box with `datasets` + HF token (env `HF_TOKEN`). Pure functions `ascii_filter(text) -> bytes` and `shuffle_docs(docs, seed) -> list` are unit-testable without network.

- [ ] **Step 1: Write the failing tests**

Create `test_build_stack_corpus.py`:

```python
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
```

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest test_build_stack_corpus`
Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Implement**

Create `build_stack_corpus.py`:

```python
#!/usr/bin/env python3
"""Build corpus.bin: shuffled, ASCII-filtered Lisp-family code from The Stack.

Run on the training box (needs `pip install datasets` and an HF token with
access to bigcode/the-stack-dedup, via the HF_TOKEN env var):

    HF_TOKEN=... python3 build_stack_corpus.py --max-gb 5 --out corpus.bin
"""

import argparse
import random

LANGS = ["common-lisp", "scheme", "emacs-lisp", "racket", "clojure"]
KEEP = frozenset({9, 10} | set(range(32, 127)))
MIN_DOC_BYTES = 100


def ascii_filter(text: str) -> bytes:
    return bytes(b for b in text.encode("utf-8", "ignore") if b in KEEP)


def shuffle_docs(docs: list, seed: int) -> list:
    random.Random(seed).shuffle(docs)
    return docs


def collect(max_bytes: int):
    from datasets import load_dataset
    docs, total = [], 0
    per_lang = max_bytes // len(LANGS)
    for lang in LANGS:
        got = 0
        ds = load_dataset("bigcode/the-stack-dedup",
                          data_dir=f"data/{lang}", split="train",
                          streaming=True)
        for row in ds:
            b = ascii_filter(row["content"])
            if len(b) < MIN_DOC_BYTES:
                continue
            docs.append(b + b"\n\n")
            got += len(b) + 2
            if got >= per_lang:
                break
        print(f"{lang}: {got / 1e6:.1f} MB", flush=True)
        total += got
    return docs, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=5.0)
    ap.add_argument("--out", default="corpus.bin")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    docs, total = collect(int(args.max_gb * 1e9))
    shuffle_docs(docs, args.seed)
    with open(args.out, "wb") as f:
        for d in docs:
            f.write(d)
    print(f"wrote {args.out}: {total / 1e9:.2f} GB in {len(docs)} docs (shuffled)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m unittest test_build_stack_corpus` — all pass (no network used).

- [ ] **Step 5: Commit**

```bash
git add build_stack_corpus.py test_build_stack_corpus.py
git commit -m "$(cat <<'EOF'
Add The Stack corpus builder (streamed, ASCII-filtered, doc-shuffled)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Lambda helper scripts

**Files:**
- Create: `cloud/lambda.sh`
- Create: `cloud/setup.sh`

**Interfaces:**
- Produces: `cloud/lambda.sh {types|ls|keys|launch REGION SSHKEY|terminate ID}` (reads `~/.lambda-api-key`); `cloud/setup.sh` (run on the instance from the repo root, creates `~/venv` with mlx[cuda]).

- [ ] **Step 1: Write cloud/lambda.sh**

```sh
#!/bin/sh
# Minimal Lambda Cloud API helper. Requires ~/.lambda-api-key (chmod 600).
set -e
KEY=$(cat "$HOME/.lambda-api-key")
API=https://cloud.lambdalabs.com/api/v1
case "$1" in
  types)     curl -su "$KEY:" "$API/instance-types" | python3 -m json.tool ;;
  ls)        curl -su "$KEY:" "$API/instances" | python3 -m json.tool ;;
  keys)      curl -su "$KEY:" "$API/ssh-keys" | python3 -m json.tool ;;
  launch)    [ -n "$3" ] || { echo "usage: $0 launch REGION SSHKEY" >&2; exit 1; }
             curl -su "$KEY:" -X POST -H 'Content-Type: application/json' \
               -d "{\"region_name\":\"$2\",\"instance_type_name\":\"gpu_1x_a100\",\"ssh_key_names\":[\"$3\"]}" \
               "$API/instance-operations/launch" | python3 -m json.tool ;;
  terminate) [ -n "$2" ] || { echo "usage: $0 terminate INSTANCE_ID" >&2; exit 1; }
             curl -su "$KEY:" -X POST -H 'Content-Type: application/json' \
               -d "{\"instance_ids\":[\"$2\"]}" \
               "$API/instance-operations/terminate" | python3 -m json.tool ;;
  *) echo "usage: $0 types|ls|keys|launch REGION SSHKEY|terminate ID" >&2; exit 1 ;;
esac
```

- [ ] **Step 2: Write cloud/setup.sh**

```sh
#!/bin/sh
# One-time setup on a fresh Lambda instance. Run from the repo root.
set -e
python3 -m venv "$HOME/venv"
"$HOME/venv/bin/pip" install -q --upgrade pip
"$HOME/venv/bin/pip" install -q "mlx[cuda]" numpy datasets
"$HOME/venv/bin/python" -c "import mlx.core as mx; print('mlx device:', mx.default_device())"
```

- [ ] **Step 3: Verify**

`sh -n cloud/lambda.sh && sh -n cloud/setup.sh` (parse-check both), `chmod +x cloud/*.sh`, and run `./cloud/lambda.sh` with no args — expect the usage line, exit 1. Do NOT call the real API in this task (no key assumed present yet).

- [ ] **Step 4: Commit**

```bash
git add cloud/lambda.sh cloud/setup.sh
git commit -m "$(cat <<'EOF'
Add Lambda Cloud API helper and instance setup scripts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4 (controller, operational): Launch + Phase-0 gate

Pre-req: `~/.lambda-api-key` exists (chmod 600). If missing, STOP and ask the user to create it. Push the branch first so the box can clone the code.

- [ ] `./cloud/lambda.sh keys` → pick the user's SSH key name (verify the matching private key exists in `~/.ssh`; if none, STOP and ask).
- [ ] `./cloud/lambda.sh types` → find a region with `gpu_1x_a100` capacity; `./cloud/lambda.sh launch REGION SSHKEY` → note instance id; poll `./cloud/lambda.sh ls` until status `active` + IP assigned.
- [ ] `ssh ubuntu@IP` (StrictHostKeyChecking=accept-new): `git clone https://github.com/dpalmqvist/gpt-lisp && cd gpt-lisp && sh cloud/setup.sh && git checkout <branch>` — confirm `mlx device:` prints a GPU/CUDA device.
- [ ] Phase-0 gate: scp the first 5MB of the laptop's `corpus.txt` to the box as `corpus.txt`; on box: `~/venv/bin/python -m unittest test_mlx_lisp -v` (all non-skipped pass) and a SMOKE train run (`touch SMOKE && ~/venv/bin/python mlx_lisp.py train.lisp`; loss must decrease; then `rm -f SMOKE ckpt*.npz`).
- [ ] Gate verdict: green → Task 5. Any CUDA-missing-op failure → STOP, report to user with the exact error (JAX fallback is a new decision).

### Task 5 (controller, operational): Corpus + training launch

Pre-req: `~/.hf-token` exists locally. If missing, STOP and ask.

- [ ] `scp ~/.hf-token ubuntu@IP:~/.hf-token && ssh ubuntu@IP chmod 600 .hf-token`
- [ ] On box in tmux: `HF_TOKEN=$(cat ~/.hf-token) ~/venv/bin/python build_stack_corpus.py --max-gb 5` (30-90 min; poll). Verify `corpus.bin` ≥ 3GB.
- [ ] `rm -f corpus.txt SMOKE && touch CLOUD`; launch in tmux: `PYTHONUNBUFFERED=1 ~/venv/bin/python mlx_lisp.py train.lisp > train.log 2>&1`
- [ ] Verify liftoff: config line shows `smoke #f`, corpus-chars ≥ 3e9, `start-step 1 of 100000`; first losses 3-7 and falling. Estimate steps/sec and report projected duration + cost to the user.
- [ ] Controller monitors via periodic ssh polling of `train.log` (milestones every 10k steps, tracebacks, completion, process-exit; also confirm the instance is still billing-active via `./cloud/lambda.sh ls`).

### Task 6 (controller, operational): Retrieve, verify, terminate, document

- [ ] On completion (or user-directed early stop): `scp ubuntu@IP:gpt-lisp/ckpt-best.npz ./cloud-best.npz` (~460MB; verify size + step via a tempdir `sample.lisp` run with `CLOUD` flag: copy model.lisp+sample.lisp+cloud-best.npz→ckpt-best.npz into a tempdir, `touch CLOUD`, run with `/tmp/mlx-venv/bin/python`).
- [ ] Report final val trajectory + a local showcase sample to the user.
- [ ] Terminate: `./cloud/lambda.sh terminate ID`; confirm via `./cloud/lambda.sh ls` that billing stopped. (Pre-approved in the spec.)
- [ ] README: add a short "Cloud training (Lambda + MLX CUDA)" subsection pointing at `cloud/` and `build_stack_corpus.py`; commit.
- [ ] Final suites green; whole-branch review; integration via superpowers:finishing-a-development-branch.
