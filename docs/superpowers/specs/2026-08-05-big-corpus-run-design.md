# Big-Corpus Training Run — Design

**Date:** 2026-08-05
**Goal:** Rerun the 50,000-step training on a ~60MB corpus so the 4.8M-param
model stops overfitting (run 1 on 2.6MB hit its val minimum at ~7% of the
run), and land the run-protecting fixes from the final branch review first.

## Background

Run 1 (2026-08-05): best val 1.23 at step ~4000, overfit thereafter, stopped
at step ~13,500 by user decision. Best checkpoint preserved. The final
whole-branch review parked three Important fixes as "cannot land mid-run";
training is now stopped, so they land before the next long run.

## 1. Hardening (parked review fixes, now due)

- **Atomic checkpoint writes:** `save-tree` writes `<path>.tmp.npz` then
  `os.replace`s onto `<path>`. A crash mid-save can no longer corrupt an
  existing checkpoint; sampling while training becomes genuinely safe.
- **`best-val` persisted:** the checkpoint tree becomes
  `(list step-array best-array params m v)` (best-array = 1-element float32).
  Resume restores `best-val`, so a restart can never overwrite
  `ckpt-best.npz` with a worse model. `sample.lisp` reads params at index 2,
  with a length check (5 = new format, 4 = legacy) so archived run-1
  checkpoints stay loadable.
- **`sample.lisp` guard:** friendly message instead of a traceback when no
  checkpoint exists.
- **Loss-decrease test (spec debt from run 1):** `test_mlx_lisp.py` gains a
  test that copies `model.lisp` + `train.lisp` + a ~200KB corpus slice into a
  tempdir, runs a SMOKE training subprocess there, and asserts the logged
  train loss decreases and checkpoints appear. Skipped when `corpus.txt` is
  absent.
- **Tests stop touching the repo root:** `TestModel` runs in a tempdir cwd
  (model.lisp copied in; SMOKE created there; exception-safe cleanup via
  addCleanup).
- **Doc cleanups:** `train.lisp`/`sample.lisp` header comments say `python3`
  (matching README); README Tests section lists `test_mlx_lisp` too.

## 2. Corpus (~60MB)

`fetch_corpus.sh` adds three shallow clones: GNU Emacs (sparse checkout of
`lisp/` only, via `--filter=blob:none --sparse`; ~40MB Elisp), SBCL (~12MB
Common Lisp), Guile from Savannah (~10MB Scheme; if Savannah is unreachable,
skip it — the corpus is still ~55MB). The `find` gains `-name '*.el'`. Same
ASCII filter, same V=128 vocab, val split still the last 5% (~3MB).
Expected `corpus.txt`: 50–70MB.

## 3. Fresh start + launch

Archive run 1 (`mv ckpt-best.npz run1-best.npz`, keep `ckpt-best.backup.npz`
as-is or delete), delete `ckpt.npz`, rebuild `corpus.txt`, launch fresh with
`PYTHONUNBUFFERED=1` per the runbook. Model config and 50,000 steps
unchanged: at B=64/T=256 that is ~13–15 epochs over 60MB (vs ~85 over run
1's corpus by its stop point). Expected duration ~6.5h at run 1's measured
~0.45s/step.

## 4. Monitoring

Same controller monitor as run 1: 5k-step milestones, Python tracebacks
only (generated samples legitimately contain the word "error"), completion,
trainer-process-exit. Controller reports val trajectory; `ckpt-best`
captures the val minimum automatically wherever it lands.

## 5. Out of scope

Model scaling, new primitives beyond the atomic save, RNG-stream
checkpointing (still deferred), corpus deduplication.
