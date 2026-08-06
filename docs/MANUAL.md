# gpt-lisp Manual

Step-by-step instructions for setting up, training, and sampling the Lisp
GPT. Every command here was used in real runs; the Troubleshooting section
lists every failure we actually hit.

## 1. Setup

### macOS (Apple silicon)

```sh
git clone https://github.com/dpalmqvist/gpt-lisp && cd gpt-lisp
python3 -m venv .venv
.venv/bin/pip install mlx numpy
```

Use `.venv/bin/python` for every command below. Check it works:

```sh
.venv/bin/python -c "import mlx.core as mx; print(mx.default_device())"
# Device(gpu, 0)
```

### Linux + NVIDIA GPU

```sh
git clone https://github.com/dpalmqvist/gpt-lisp && cd gpt-lisp
sh cloud/setup.sh        # creates ~/venv with mlx[cuda] + CUDA headers
```

Use `~/venv/bin/python` for every command below.

### Try the REPL and the mini demo

```sh
.venv/bin/python mlx_lisp.py            # GPU-Lisp REPL, Ctrl-D to exit
.venv/bin/python mlx_lisp.py gpt.lisp   # self-contained ~50k-param demo (~2 min)
```

## 2. Pick a model size

The model configuration is chosen by flag files in the working directory
(read once at process start):

| flag file | tier | params | context | steps | good for |
|---|---|---|---|---|---|
| `SMOKE` | tiny | ~100k | 32 | 50 | 1-minute pipeline test |
| *(none)* | LOCAL | ~4.8M | 256 | 50,000 | laptop training (hours) |
| `CLOUD` | big | ~38M | 512 | 100,000 | A100-class GPU (~8h) |

`SMOKE` wins if both files exist. **The flag must match the checkpoint**:
a checkpoint trained under `CLOUD` only loads with the `CLOUD` flag
present (the array shapes must agree).

## 3. Build a training corpus

Two options — both produce printable-ASCII-only data (the model's
vocabulary is the 128 ASCII codes):

**Option A — `corpus.txt` (~100MB, no accounts needed):**

```sh
./fetch_corpus.sh
```

Clones five well-known Lisp/Scheme/Elisp repos into `corpus_repos/`
(kept for re-runs — don't delete) and writes `corpus.txt`.

**Option B — `corpus.bin` (~2GB, The Stack):**

Requires a Hugging Face account: visit
https://huggingface.co/datasets/bigcode/the-stack-dedup and click
"Agree and access repository", then create a read token.

```sh
HF_TOKEN=hf_... .venv/bin/python build_stack_corpus.py --max-gb 5
```

Streams the Common Lisp, Scheme, Emacs Lisp, Racket, and Clojure subsets,
shuffles documents, and writes `corpus.bin` (auto-capped at 2^31-1 bytes,
MLX's array-size limit). `train.lisp` prefers `corpus.bin` when both exist.

## 4. Smoke-test the pipeline (1 minute)

Always do this before a long run:

```sh
touch SMOKE
.venv/bin/python mlx_lisp.py train.lisp
rm -f SMOKE ckpt.npz ckpt-best.npz
```

Expected: a config line, loss starting near 4.85 (= ln 128, uniform
guessing) and clearly decreasing by step 50, then a gibberish sample.

## 5. Train

```sh
nohup env PYTHONUNBUFFERED=1 .venv/bin/python mlx_lisp.py train.lisp > train.log 2>&1 &
tail -f train.log
```

`PYTHONUNBUFFERED=1` matters: without it Python buffers stdout under
nohup and the log stays empty for hours while training runs fine.

What you'll see in the log:

- a config line: device, corpus size, start step, best-val so far
- every 250 steps: `("step" N "train" X "val" Y)` — val is averaged over
  4 fixed batches, so it's comparable across restarts
- every 1000 steps (5000 under `CLOUD`): a 150-char sample at temp 0.5

Checkpoints (written atomically — a crash can't corrupt them):

- `ckpt.npz` — every 1000 steps; the resume point
- `ckpt-best.npz` — whenever validation loss improves; what you sample from

**Stopping and resuming:** kill the process anytime (`kill <pid>`); at
most 1000 steps are lost. Rerunning `train.lisp` with `ckpt.npz` present
prints `resuming from ckpt.npz` and continues — step count, best-val, and
Adam state all survive the restart. For a fresh start, delete both
checkpoint files first.

Hyperparameters (edit at the top of `train.lisp` / `model.lisp`): Adam
lr 3e-4, beta1 0.9, beta2 0.99; batch 64; last 5% of the corpus is the
validation split.

## 6. Train on a cloud GPU (Lambda Cloud)

Everything above works unchanged on NVIDIA via MLX's CUDA backend.
The extra steps:

```sh
# once: put your API key in place (never commit it anywhere)
printf '%s' 'YOUR_LAMBDA_KEY' > ~/.lambda-api-key && chmod 600 ~/.lambda-api-key

./cloud/lambda.sh keys                 # SSH key names on your account
./cloud/lambda.sh types                # regions with capacity
./cloud/lambda.sh launch REGION SSHKEY # boots a 1x A100
./cloud/lambda.sh ls                   # find the instance IP
```

On the instance:

```sh
git clone https://github.com/dpalmqvist/gpt-lisp && cd gpt-lisp
sh cloud/setup.sh
# smoke-test (section 4), then build the corpus (section 3, option B)
touch CLOUD
nohup env PYTHONUNBUFFERED=1 ~/venv/bin/python mlx_lisp.py train.lisp > train.log 2>&1 &
```

When training finishes, from your own machine:

```sh
scp ubuntu@IP:gpt-lisp/ckpt-best.npz ./cloud-best.npz
./cloud/lambda.sh terminate INSTANCE_ID     # stop paying!
./cloud/lambda.sh ls                        # confirm it's gone
```

Reference numbers from the first full run: 100k steps on a 2.1GB corpus
took ~8h on one `gpu_1x_a100_sxm4` (~$18), reaching val loss 0.48.

## 7. Run inference

`sample.lisp` loads the best checkpoint and generates 500 characters from
the prompt `"(define "` at temperature 0.6:

```sh
.venv/bin/python mlx_lisp.py sample.lisp
```

It prefers `ckpt-best.npz`, falls back to `ckpt.npz`, and prints a
friendly message if neither exists. Remember the flag rule from section 2
— sampling a `CLOUD`-trained checkpoint needs `touch CLOUD` first.

To sample a downloaded checkpoint without touching your training state,
use a scratch directory:

```sh
D=$(mktemp -d)
cp model.lisp sample.lisp $D/ && cp cloud-best.npz $D/ckpt-best.npz
touch $D/CLOUD
(cd $D && /path/to/repo/.venv/bin/python /path/to/repo/mlx_lisp.py sample.lisp)
```

Custom prompt, length, or temperature — write a tiny Lisp file (higher
temperature = more varied, lower = more repetitive):

```lisp
; my-sample.lisp
(load "model.lisp")
(define st (load-tree "ckpt-best.npz"))
(define params (nth st 2))
(display (ids->text (gen params (text->ids "(defmacro ") 300 0.9)))
```

```sh
.venv/bin/python mlx_lisp.py my-sample.lisp
```

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `train.log` empty but training runs | stdout buffering — relaunch with `PYTHONUNBUFFERED=1`; resume is automatic |
| `cuda_bf16.h` not found (NVIDIA) | `pip install nvidia-cuda-runtime-cu12` (cloud/setup.sh already does) |
| `DatasetNotFoundError ... gated dataset` | accept the terms on the the-stack-dedup page with the token's account |
| `OverflowError: Shape dimension ... 32-bit` | corpus file > 2^31-1 bytes; rebuild (the builder now caps automatically) or `truncate -s 2147483647 corpus.bin` |
| Out-of-memory when sampling during training (NVIDIA) | the trainer holds the GPU pool — sample from a copy of the checkpoint on another machine, or read the in-log samples |
| Shape mismatch loading a checkpoint | flag file doesn't match the tier the checkpoint was trained with (section 2) |
| Loss stuck at ~4.85 | corpus not found or unreadable — check the config line's `corpus-chars` |

## 9. Tests

```sh
python3 -m unittest test_lisp                  # interpreter (no mlx needed)
python3 -m unittest test_build_stack_corpus    # corpus builder logic (no network)
.venv/bin/python -m unittest test_mlx_lisp     # MLX layer + model + training (~1 min)
```
