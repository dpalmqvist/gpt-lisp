#!/usr/bin/env python3
"""GPU-flavored Lisp: MLX array/autodiff primitives layered on lisp.py.

Two facts make this extension nearly free:
  1. Lisp lists ARE Python lists  -> mx.array accepts them directly.
  2. Lisp lambdas (Procedure) ARE Python callables -> mx.grad/compile/vmap
     accept them directly. Autodiff of Lisp code costs one lambda.
"""

import functools
import operator as op
import sys

import mlx.core as mx

from lisp import standard_env, evaluate, tokenize, parse, repl, String


def _reduce(f):
    return lambda *a: functools.reduce(f, a)


def gpu_env():
    env = standard_env()
    env.update({
        # array-safe arithmetic (works elementwise on tensors, broadcasts)
        "+": _reduce(op.add), "-": lambda a, *b: -a if not b else functools.reduce(op.sub, b, a),
        "*": _reduce(op.mul), "/": _reduce(op.truediv),

        # constructors
        "array":   lambda x, *dt: mx.array(x, getattr(mx, dt[0]) if dt else mx.float32),
        "zeros":   lambda shape: mx.zeros(shape),
        "ones":    lambda shape: mx.ones(shape),
        "arange":  lambda *a: mx.arange(*a, dtype=mx.float32),
        "randn":   lambda shape: mx.random.normal(shape),
        "uniform": lambda shape, lo, hi: mx.random.uniform(lo, hi, shape),

        # linear algebra & shaping
        "matmul": mx.matmul, "@": mx.matmul,
        "transpose": mx.transpose,
        "reshape": lambda a, shape: mx.reshape(a, shape),
        "shape": lambda a: list(a.shape),

        # reductions & elementwise ops
        "sum": mx.sum, "mean": mx.mean, "argmax": mx.argmax,
        "sin": mx.sin, "cos": mx.cos, "abs": mx.abs,
        "exp": mx.exp, "log": mx.log, "sqrt": mx.sqrt, "square": mx.square,
        "tanh": mx.tanh, "maximum": mx.maximum,
        "relu": lambda x: mx.maximum(x, 0.0),
        "sigmoid": mx.sigmoid,
        "softmax": lambda x: mx.softmax(x, axis=-1),

        # === the payoff: function transformations over Lisp lambdas ===
        "grad":     lambda f: mx.grad(f),                # d f / d arg0
        "gradN":    lambda f, *ns: mx.grad(f, argnums=list(ns) or [0]),
        "vgrad":    lambda f: mx.value_and_grad(f),
        "jit":      lambda f: mx.compile(f),             # fuse into one kernel
        "force":    lambda *xs: (mx.eval(*xs), xs[-1])[1],  # MLX is lazy

        "item": lambda a: a.item(),
        "device": lambda: str(mx.default_device()),

        # --- transformer-grade primitives ---
        "take":    lambda a, idx: mx.take(a, idx, axis=0),      # embedding lookup
        "swap":    lambda a: mx.swapaxes(a, -1, -2),            # K^T for attention
        "meank":   lambda x: mx.mean(x, axis=-1, keepdims=True),
        "sumlast": lambda x: mx.sum(x, axis=-1),
        "log-softmax": lambda x: x - mx.logsumexp(x, axis=-1, keepdims=True),
        "onehot":  lambda ids, n: (ids[..., None] == mx.arange(n)).astype(mx.float32),
        "causal-mask": lambda t: mx.where(
            mx.arange(t)[:, None] >= mx.arange(t)[None, :], 0.0, -1e9),
        "sample":  lambda logits, temp: mx.random.categorical(logits / temp),
        "at":      lambda a, *ix: a[tuple(ix)],
        "force-tree": lambda t: (mx.eval(t), t)[1],   # force a whole param tree

        # --- text as ASCII ids, and small list utilities ---
        "text->ids": lambda s: [ord(c) for c in s],
        "ids->text": lambda ids: String("".join(chr(int(i)) for i in ids)),
        "range":   lambda *a: list(range(*a)),
        "slice":   lambda xs, i, j: xs[i:j],
        "nth":     lambda xs, i: xs[i],
    })
    return env


def run_file(path: str):
    with open(path) as f:
        tokens = tokenize(f.read())
    env = gpu_env()
    while tokens:
        evaluate(parse(tokens), env)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        repl(gpu_env(), banner=f"gpu lisp on {mx.default_device()} — Ctrl-D to exit")
