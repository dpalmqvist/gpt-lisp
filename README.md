# gpt-lisp

A character-level GPT written and trained **entirely in Lisp** — running on a
minimal Scheme-flavored interpreter written in Python, with GPU arrays and
autodiff provided by [MLX](https://github.com/ml-explore/mlx).

Two facts make the ML layer nearly free:

1. Lisp lists *are* Python lists → `mx.array` accepts them directly.
2. Lisp lambdas *are* Python callables → `mx.grad` / `mx.compile` accept them
   directly. Autodiff of Lisp code costs one lambda.

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

## Tests

```sh
python3 -m unittest test_lisp        # interpreter (no mlx needed)
python3 -m unittest test_mlx_lisp    # MLX layer + model + training smoke
```
