# Lambda Cloud Run — Design

**Date:** 2026-08-05
**Goal:** Train a ~38M-parameter version of the Lisp GPT on ~5GB of
Lisp-family code from The Stack, on a Lambda Cloud 1x A100, using the same
MLX Lisp codebase via MLX's CUDA backend.

## Decisions (from brainstorming)

- Budget: 1x A100, ~$15-30 total (12-24h ceiling; estimate ~4-8h actual)
- Corpus: The Stack (deduplicated), Lisp-family subsets, streamed on-box
- Model: ~38M params (CLOUD config tier)
- Access: Lambda Cloud API key supplied by the user in `~/.lambda-api-key`
  (chmod 600, never pasted into chat); HF token in `~/.hf-token` likewise
- User pre-approved: instance termination after the final checkpoint is
  downloaded and verified locally
- Tree experiments: explicitly skipped (user decision)

## 1. Backend + Phase-0 gate

`pip install "mlx[cuda]"` on the instance. The CUDA backend is young, so
everything gates on Phase 0 run on the box: full `test_mlx_lisp` suite plus
a SMOKE training run (against a small corpus.txt scp'd from the laptop).
Any unsupported primitive → STOP and report; fallback (JAX port) is a
separate user decision, out of scope here.

## 2. Config tiers

`model.lisp`/`train.lisp` gain a `CLOUD` flag file alongside `SMOKE`:

| | SMOKE | LOCAL (default) | CLOUD |
|---|---|---|---|
| T | 32 | 256 | 512 |
| D | 32 | 256 | 512 |
| H | 2 | 8 | 8 |
| HD | 16 | 32 | 64 |
| L | 2 | 6 | 12 |
| B | 8 | 64 | 64 |
| steps | 50 | 50000 | 100000 |
| sample-every | 25 | 1000 | 5000 |

CLOUD ≈ 38M params. 100k steps × B×T = ~3.3B chars ≈ one epoch over the
corpus — overfitting unlikely by construction. lr/Adam unchanged (3e-4).
Sampling moves to its own `sample-every` cadence (checkpoint cadence stays
1000); generation and val at D=512/L=12 are too slow to run interpreted
every 1000 steps, so `val-loss` switches to a `(jit batch-loss)` compiled
function.

## 3. GB-scale data path

- New primitive `read-corpus`: `np.fromfile(path, uint8)` → `mx.array` —
  no per-char Python list. Corpus stays uint8 on device (~5GB of A100 HBM);
  batches are gathered then cast via a new `int32` primitive.
- `train.lisp` prefers `corpus.bin` when present, else falls back to the
  existing `corpus.txt` text path. Laptop workflow unchanged.
- `build_stack_corpus.py` (runs on the box): streams
  `bigcode/the-stack-dedup` `data/{common-lisp,scheme,emacs-lisp,racket,clojure}`,
  ASCII-filters each document (bytes 9, 10, 32-126), drops docs <100 bytes,
  caps per-language bytes at max-gb/5, **shuffles documents with a fixed
  seed** (fixes run 2's val-split-is-one-language flaw), writes `corpus.bin`.
  Default cap 5GB.

## 4. Orchestration

- `cloud/lambda.sh`: minimal curl helper over the Lambda Cloud API
  (types/ls/keys/launch/terminate) reading `~/.lambda-api-key`.
- `cloud/setup.sh`: on-instance venv + `mlx[cuda]` + numpy + datasets.
- Flow: launch A100 → clone the public repo on the box → setup → Phase-0
  gate → build corpus (HF token via env) → `touch CLOUD` → launch training
  under `PYTHONUNBUFFERED=1` in tmux → controller monitors via ssh log
  polling → on completion scp `ckpt-best.npz` (~460MB) to the laptop →
  verify it samples locally (CLOUD flag + legacy-compatible sample.lisp) →
  terminate instance (pre-approved) → docs + merge.
- Crash safety: existing resume + atomic checkpoints; a mid-run instance
  failure loses ≤1000 steps if we relaunch.

## 5. Out of scope

JAX fallback port, multi-GPU, BPE/sexpr tokenization, lr schedules,
tree-structured models (user-skipped).
