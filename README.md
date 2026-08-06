# gpt-lisp

A character-level GPT written and trained **entirely in Lisp** — running on a
minimal Scheme-flavored interpreter written in Python, with GPU arrays and
autodiff provided by [MLX](https://github.com/ml-explore/mlx).

Two facts make the ML layer nearly free:

1. Lisp lists *are* Python lists → `mx.array` accepts them directly.
2. Lisp lambdas *are* Python callables → `mx.grad` / `mx.compile` accept them
   directly. Autodiff of Lisp code costs one lambda.

**New here? [docs/MANUAL.md](docs/MANUAL.md) is the step-by-step guide** to
setup, training (laptop or cloud GPU), and sampling.

## Files

- `lisp.py` — the interpreter: reader, evaluator with tail-call optimization,
  macros (`defmacro` + quasiquote), and a REPL.
- `mlx_lisp.py` — MLX array/autodiff primitives layered on the interpreter
  (`matmul`, `softmax`, `grad`, `jit`, ...).
- `gpt.lisp` — a 2-block, single-head, ~50k-parameter transformer trained on a
  tiny corpus of Lisp source, then sampled autoregressively.
- `test_lisp.py` — interpreter test suite.

## Run

```sh
pip install mlx            # Apple silicon

python3 lisp.py            # plain Lisp REPL
python3 mlx_lisp.py        # GPU Lisp REPL
python3 mlx_lisp.py gpt.lisp   # train the Lisp GPT and watch it write Lisp
```

Example output:

```
("step" 400 "loss" 0.103826604783535)
("final loss" 0.09020739048719406)
"--- the Lisp LLM writes Lisp: ---"
"(define (fib n) (if (< n 2) n (+ 1) (fifififififact ..."
```

## Scaled training run

The ambitious version: a ~4.8M-parameter, 6-block, 8-head GPT trained on real
Lisp — Norvig's *Paradigms of AI Programming* code, chibi-scheme, SBCL,
Guile, and GNU Emacs's Elisp tree (~100MB of ASCII source).

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

## Cloud training (Lambda + MLX CUDA)

The same codebase trains unchanged on NVIDIA GPUs via MLX's CUDA backend.
A ~38M-param model (`touch CLOUD`: D=512, 12 blocks, T=512) trained 100k
steps on ~2.1GB of The Stack's Lisp-family code (Common Lisp, Scheme,
Emacs Lisp, Racket, Clojure) reaches val loss ~0.48 in ~8h on one A100
(~$18 on Lambda Cloud).

- `build_stack_corpus.py` — streams + shuffles + ASCII-filters the corpus
  to `corpus.bin` (needs an HF token with the-stack-dedup access).
- `cloud/lambda.sh` — minimal Lambda Cloud API helper (launch/terminate).
- `cloud/setup.sh` — on-instance setup (`mlx[cuda]` + CUDA headers).

The first full run (2026-08-06) went 4.85 → **0.48** val loss in one pass:
a Lisp program, interpreting a neural network, learning to write Lisp.
Sampled from its best checkpoint:

```lisp
(define (dump-node-slots node)
    (if (node? node)
        node
        (dump-dprod-nodes (node-prod (node-prod node))
                          (dump-dprod-dprod-nodes node))))
```

Balanced parens, `cond`/`let*`/`if` idioms, predicate/accessor naming
conventions, cross-definition references — the flaws left are semantic,
not syntactic. To sample from a trained checkpoint locally: put it next
to `model.lisp` as `ckpt-best.npz`, `touch CLOUD` (matching the config
it was trained with), and run `python3 mlx_lisp.py sample.lisp`.

## Side quest: Collatz to 10^12

`collatz.lisp` verifies the Collatz conjecture for every n ≤ 10^12, entirely
in the same Lisp. It shows each n ≥ 3 drops below its own start (strong
induction closes the argument), in three tiers: a mod-2^16 residue sieve
computed vectorized on the GPU rules out ~97% of numbers wholesale; survivors
run as ~8.7M-lane int64 GPU batches with periodic stream compaction; and the
rare lanes whose glide peaks would overflow int64 (they reach ~10^19) retire
to a scalar recheck on the interpreter, whose numbers are Python bignums.

```sh
python3 mlx_lisp.py collatz.lisp   # ~168M numbers/s on an M-series laptop,
                                   # full 10^12 in under 2 h; checkpoints and
                                   # resumes via collatz-ckpt.npz
touch SMOKE                        # (first) for a seconds-long 10^7 smoke run
```

Design notes: `docs/superpowers/specs/2026-08-06-collatz-design.md`.

## Tests

```sh
python3 -m unittest test_lisp        # interpreter (no mlx needed)
python3 -m unittest test_mlx_lisp    # MLX layer + model + training smoke
python3 -m unittest test_collatz     # sieve soundness, glide kernel, smoke
```
