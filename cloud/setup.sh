#!/bin/sh
# One-time setup on a fresh Lambda instance. Run from the repo root.
set -e
python3 -m venv "$HOME/venv"
"$HOME/venv/bin/pip" install -q --upgrade pip
"$HOME/venv/bin/pip" install -q "mlx[cuda]" nvidia-cuda-runtime-cu12 numpy datasets
"$HOME/venv/bin/python" -c "import mlx.core as mx; print('mlx device:', mx.default_device())"
