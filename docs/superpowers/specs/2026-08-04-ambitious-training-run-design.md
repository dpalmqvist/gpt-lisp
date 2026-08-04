# Ambitious Training Run — Design

**Date:** 2026-08-04
**Goal:** Scale gpt-lisp from a 190-char toy run to an overnight-capable training
run on a real downloaded Lisp/Scheme corpus, keeping the model and training loop
entirely in Lisp. The existing `gpt.lisp` minimal demo stays untouched.

## Decisions (from brainstorming)

- Scope: corpus + architecture + optimizer + jit ("all of it")
- Corpus: downloaded Lisp/Scheme source (~2–4MB), not the repo's own files
- Budget: multi-hour/overnight runs acceptable; must include smoke mode,
  checkpointing, and resume so a bug or crash cannot waste the night
- Purity: model and training loop in Lisp; new capabilities become one-line
  MLX primitives in `mlx_lisp.py` (approach C: three small Lisp files)

## 1. Corpus pipeline — `fetch_corpus.sh`

- Shallow-clone into gitignored `corpus_repos/`:
  - `norvig/paip-lisp` (Common Lisp code from *Paradigms of AI Programming*)
  - `ashinn/chibi-scheme` (clean, sizable Scheme library tree)
- Concatenate all `.lisp` and `.scm` files, strip all bytes outside printable
  ASCII plus newline and tab (preserves the V=128 ASCII vocab), write
  `corpus.txt` (~2–4MB).
- `corpus.txt`, `corpus_repos/`, and `*.npz` are gitignored; the script is the
  reproducible artifact.
- Train/val split: last 5% of the text is validation.

## 2. New primitives — `mlx_lisp.py`

One line each, same style as the existing table:

| Lisp name        | Implementation                                        | Used for |
|------------------|-------------------------------------------------------|----------|
| `randint`        | `mx.random.randint(lo, hi, shape)`                    | minibatch offsets |
| `transpose-axes` | `mx.transpose(a, axes)`                               | multi-head attention |
| `pow`            | `a ** b`                                              | Adam bias correction |
| `save-tree`      | `mx.savez(path, **dict(tree_flatten(tree)))`          | checkpoints |
| `load-tree`      | `tree_unflatten(list(mx.load(path).items()))`         | checkpoints |
| `read-file`      | file contents as Lisp `String`                        | corpus loading |
| `exists?`        | `os.path.exists`                                      | resume + smoke flag |
| `load`           | tokenize/evaluate a `.lisp` file into the current env | three-file split |
| `seed`           | `mx.random.seed`                                      | reproducibility |

Minibatch window gathering needs no new primitive:
`(take flat-ids (+ (reshape offsets (list B 1)) (array (range T) "int32")))`
yields a (B,T) batch; labels are the same with offsets+1.

## 3. Model — `model.lisp`

- Config at top: `T=256`, `D=256`, `H=8` heads (head dim 32), `L=6` blocks,
  `V=128`, `B=64` → ~4.8M parameters.
- Multi-head causal self-attention: q/k/v/out projections D→D; reshape
  (B,T,D)→(B,T,H,32), `transpose-axes` to (B,H,T,32); batched matmul for
  scores (B,H,T,T); the (T,T) causal mask broadcasts over batch and heads;
  softmax, weighted sum, transpose/reshape back, output projection.
- Layernorm with learned scale and shift (gamma, beta) at every LN site
  (2 per block + final).
- MLP: D → 4D → D with relu.
- Residual connections and 1/sqrt(nin) init as in `gpt.lisp`.
- File contains only config and pure functions — `(gpt p ids)` and
  `(batch-loss p xb yb)` — no training state, so both `train.lisp` and
  `sample.lisp` just `(load "model.lisp")`.

## 4. Training — `train.lisp`

- `(load "model.lisp")`, read `corpus.txt`, split train/val.
- Compiled step: `(define step-fn (jit (vgrad batch-loss)))` — `mx.compile`
  traces the Lisp interpreter once, then replays the compiled graph at native
  speed each step. Loss takes `(p xb yb)` so batches are inputs, not closures.
- Optimizer: Adam (lr 3e-4, beta1 0.9, beta2 0.99, eps 1e-8, bias-corrected)
  implemented as a generic tree-map over `(params m v)` in Lisp, like the
  existing `sgd`.
- Loop: tail-recursive; each step samples B random offsets, gathers (xb, yb),
  computes loss+grads via `step-fn`, applies Adam, forces the tree.
- Checkpointing: the whole training state `(step params m v)` is one tree,
  saved to `ckpt.npz` every 1000 steps, and to `ckpt-best.npz` whenever val
  loss improves. On startup, if `ckpt.npz` exists training resumes from it —
  a crash costs at most 1000 steps.
- Logging: every 250 steps print step / train loss / val loss (mean over 4
  fixed val batches sampled once with a fixed seed); every 1000 steps also
  print a ~150-char sample at temp 0.5.
- Smoke mode: if a file named `SMOKE` exists, config shrinks (T=32, D=32,
  H=2, L=2, B=8, 50 steps) for a ~1-minute end-to-end pipeline test.
- Real run: 50,000 steps (estimated 1–2h compiled; overnight budget is
  headroom, and val loss will plateau on a ~3MB corpus well before that).

## 5. Sampling — `sample.lisp`

`(load "model.lisp")`, `load-tree` `ckpt-best.npz` (falling back to
`ckpt.npz`), generate ~500 chars from the prompt `"(define "` at temp 0.6,
print. Safe to run while training continues, since checkpoints are complete
snapshots.

## 6. Testing & rollout

- `test_mlx_lisp.py`, skipped automatically when MLX is unavailable (run with
  `/tmp/mlx-venv/bin/python`): save/load tree round-trip, `load` include,
  randint/gather shapes, and a tiny end-to-end training run asserting the
  loss decreases.
- Rollout order: primitives (TDD) → smoke run → launch real run in the
  background → README update → commit/push.
- `gpt.lisp` is not modified.
