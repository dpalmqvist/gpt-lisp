# Collatz conjecture tester in GPU Lisp — design

2026-08-06. Goal: a program in this repo's Lisp (`mlx_lisp.py` dialect) that
verifies the Collatz conjecture for every n up to N = 10^12, using the GPU as
massively parallel lanes. Written autonomously (no interactive brainstorm was
possible); decisions and their reasons are recorded here instead.

## What "verify" means (soundness model)

Running every trajectory down to 1 is both wasteful and impossible in int64
(full-trajectory peaks for n ≤ 10^12 exceed 2^63). Instead use strong
induction, the standard trick of every serious Collatz search:

> If 1 and 2 reach 1, and every n ≥ 3 eventually **drops below its own start**
> (its "glide" terminates), then every n ≤ N reaches 1.

Proof: a minimal counterexample n₀ would glide to some m < n₀, which by
minimality reaches 1. So each lane only iterates until `value < start` —
~10 steps on average instead of ~200 — and batch order doesn't matter.

Steps use the shortcut map T(n) = n/2 (even) | (3n+1)/2 (odd). (3n+1)/2 > n,
so no drop is missed by combining the two halvings.

## Architecture: three tiers, all in the one Lisp

1. **Residue sieve (GPU, once at startup).** For M = 2^16, a number
   n = M·k + r glides within 16 steps *for every k ≥ 1* iff the affine form
   (a, b) — value a·k + b, starting (M, r) — reaches a state with a < M and
   b ≤ r. Only ~3% of residues survive (2114 of 65536 under the conservative criterion, vs the classical 1729); the rest need no testing.
   The sieve is computed vectorized: 65536 lanes, 16 steps, exact int64
   (values stay < 3^16·2^16 < 2^42). The elimination criterion (a < M AND
   b ≤ r) is deliberately conservative — sound for all k ≥ 1; any residue it
   keeps unnecessarily just gets tested, never skipped.
   Survivor extraction: argmax/clear loop (~2.1k iterations, startup only).

2. **Glide sweep (GPU, the bulk).** k-chunks ascending. Candidates built by
   broadcast: n₀ = M·(k₀ + iota(KC)) ⊕ survivors → (KC × S) lanes, flattened.
   Vectorized step with 0/1 int64 masks (no data-dependent branches):
   `stepped = odd·((3v+1)//2) + (1−odd)·(v//2)`, `v' = where(active, stepped, v)`.
   A lane deactivates when v' < n₀ (verified) or when v > SAFE = (2^63−2)//3
   (frozen and escalated, see tier 3). Every 16 steps, still-active lanes are
   **stream-compacted** (argsort by activity, gather, slice to the live count)
   so stragglers don't drag full-width kernels. Batch ends when no lane is
   active or a step cap (4096 ≫ any known glide in range) trips — capped lanes
   also escalate to tier 3, so the cap cannot cause a miss.

3. **Bignum escalation (scalar Lisp, rare).** Lanes that near int64 overflow
   (glide peaks for n ≤ 10^12 reach ~10^19; path record n = 8 528 817 511)
   are re-verified by a plain recursive Lisp glide function — the tree-walking
   interpreter's numbers are Python ints, i.e. arbitrary precision, and its
   trampoline makes the recursion safe. The GPT-Lisp thesis in miniature:
   GPU arrays for the mass, interpreter bignums for the stragglers.

Base case: n ∈ [3, 2^20) is verified brute-force on GPU with no sieve (one
batch, covers the sieve's k ≥ 1 requirement); the sweep starts at k = 2^20/M.
Any lane that fails to glide within all caps and also fails the scalar recheck
would be printed loudly as a counterexample candidate — that's the discovery
path, not an error path.

## Program shape

- `collatz.lisp` — single driver in the style of `train.lisp`: definitions,
  then config, then run. `SMOKE` file → N = 10^7 (seconds); else N = 10^12.
- Progress line per chunk group: frontier n, lanes/s, ETA, overflow count.
- Checkpoint/resume via `save-tree`/`load-tree` (`collatz-ckpt.npz`, atomic),
  saved every 50 chunks, following the training-loop idiom.
- New primitives in `mlx_lisp.py`, one "collatz-grade" section in the existing
  style, all one-liners: `//` (floor division — works on both mx arrays and
  Python bignums, which is what makes tier 3 share code with tier 2), `int64`
  (cast, mirrors `int32`), `where`, `argsort`, `iota` (int64 arange — float32
  `arange` corrupts indices past 2^24), `time` (for throughput/ETA).

## Sizing

KC = 4096 k's/chunk × 2114 survivors ≈ 8.7M lanes/batch, ~5 int64 arrays
≈ 350 MB. ~3.7k chunks for 10^12. Measured on this machine (M-series, MLX
GPU): ~172M numbers/s, ~97 min for the full 10^12. The program checkpoints,
so the run is resumable and correctness does not depend on finishing in one
sitting.

## Testing

- Sieve soundness: mod-16 survivors must be exactly {7, 11, 15}; every
  eliminated mod-256 residue spot-checked to actually glide for k = 1..3.
- Vector step vs. known values (27 → 41, 6 → 3, ...).
- Scalar glide checker on 27 (glide 96 in T-steps ~59) and on a path-record
  number whose peak exceeds int64.
- End-to-end SMOKE subprocess run in a temp cwd: verifies [3, 10^7], asserts
  zero failures and clean checkpoint write.
- Primitive unit tests in the `test_mlx_lisp.py` style.

## Rejected alternatives

- **Full trajectories to 1**: needs >64-bit everywhere, ~20× more steps.
- **Hardcoded sieve residue table**: error-prone to transcribe; computing it
  on-device is cheap, self-verifying, and more in the repo's spirit.
- **mod 4 sieve only (pure Lisp, no new primitives)**: 2× more candidates and
  still needs `//`/`where`/`int64` for correctness — the extra primitives are
  the cost of doing it right, not of doing it big.
- **Python-side driver**: the request is a program *in this Lisp*; Python
  contributes only the six primitives above.
