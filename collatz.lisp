; ============================================================
; collatz.lisp — hyper-parallel Collatz conjecture verifier.
;
; Verifies every n <= N reaches 1 by strong induction: 1 and 2 reach 1
; directly, and every n >= 3 is shown to drop below its own start
; (a minimal counterexample would glide to a smaller verified number).
; Steps use the shortcut map T(n) = n/2 (even) | (3n+1)/2 (odd);
; (3n+1)/2 > n, so combining the halvings never skips a drop.
;
; Three tiers, one language:
;   1. a mod-2^16 residue sieve, computed vectorized on the GPU, proves
;      ~97% of numbers glide without ever enumerating them;
;   2. surviving candidates run as millions of int64 GPU lanes, with
;      periodic stream compaction so stragglers don't drag full-width
;      kernels;
;   3. lanes whose values near int64 overflow (glide peaks reach ~10^19
;      below 10^12) retire to a scalar recheck on the tree-walking
;      interpreter, whose numbers are Python bignums.
;
; Run:        python3 mlx_lisp.py collatz.lisp        (N = 10^12)
; Smoke test: touch SMOKE first                       (N = 10^7)
; Resumes from collatz-ckpt.npz if present; delete it to restart.
; ============================================================

(defmacro (when test body) `(if ,test ,body))

; --- config ---
(define smoke (exists? "SMOKE"))
(define N (if smoke 10000000 1000000000000))
(define SK 16)                        ; sieve modulus bits
(define M (pow 2 SK))                 ; sieve modulus
(define BASE (pow 2 20))              ; brute-force floor: no sieve below this
(define KC 4096)                      ; k's per chunk -> KC*survivors lanes
(define SAFE (// (- (pow 2 63) 2) 3)) ; 3v+1 stays in int64 iff v <= SAFE
(define MAXSTEP 4096)                 ; >> longest glide below 10^12 (~1000)
(define COMPACT-EVERY 16)             ; steps between live-lane compactions
(define CKPT-EVERY 50)                ; chunks between checkpoints
(define LOG-EVERY 10)                 ; chunks between progress lines
(define CKPT "collatz-ckpt.npz")

(define failures 0)                   ; would-be counterexamples (expect 0)
(define escalated 0)                  ; lanes re-verified on bignums

; --- tier 3: scalar glide check, exact at any size ---
(define (scalar-step n)
  (if (= (mod n 2) 0) (// n 2) (// (+ (* 3 n) 1) 2)))

(define (scalar-glides? n0)           ; does n0's trajectory drop below n0?
  (begin
    (define (run v steps)
      (if (< v n0) #t
          (if (> steps 1000000) #f    ; conjecture-shattering if ever hit
              (run (scalar-step v) (+ steps 1)))))
    (run (scalar-step n0) 1)))

(define (recheck n)                   ; lanes the GPU couldn't finish land here
  (begin
    (set! escalated (+ escalated 1))
    (when (not (scalar-glides? n))
      (begin
        (set! failures (+ failures 1))
        (display (list "COUNTEREXAMPLE-CANDIDATE" n))))))

; --- pull the (few) flagged lane values back to interpreter land ---
(define (extract-values vals mask)    ; values of vals where int64 mask = 1
  (begin
    (define idx (iota (length vals)))
    (define (go m acc)
      (if (= (item (sum m)) 0) acc
          (begin
            (define i (item (argmax m)))
            (go (* m (- 1 (int64 (= idx i))))
                (append acc (list (item (at vals i))))))))
    (go mask (list))))

; --- tier 1: residue sieve, vectorized over all M residues at once ---
; State (a, b) stands for the whole family {a*k + b, k >= 1}; start (M, r).
; While fewer than SK halvings have happened, a is even, so parity of
; a*k + b is the parity of b and one T-step is uniform across the family.
; Residue r is eliminated once a < M and b <= r: from then on
; a*k + b <= a*k + r < M*k + r for every k >= 1 — a guaranteed glide.
; Conservative (standard sieves drop the b <= r condition) but sound.
(define (sieve-elim sk)               ; -> int64 (2^sk,) mask, 1 = eliminated
  (begin
    (define m (pow 2 sk))
    (define r (iota m))
    (define (go a b elim step)
      (if (= step sk) elim
          (begin
            (define odd (mod b 2))
            (define a2 (// (* (+ 1 (* 2 odd)) a) 2))
            (define b2 (+ (* odd (// (+ (* 3 b) 1) 2))
                          (* (- 1 odd) (// b 2))))
            (define hit (* (int64 (< a2 m)) (int64 (<= b2 r))))
            (go a2 b2 (maximum elim hit) (+ step 1)))))
    (go (+ (* 0 r) m) r (* 0 r) 0)))

(define (sieve-survivors sk)          ; -> residues that still need testing
  (extract-values (iota (pow 2 sk)) (- 1 (sieve-elim sk))))

; --- tier 2: the glide kernel ---
; All lanes step together under 0/1 int64 masks. A lane retires when its
; value drops below its start; a lane about to overflow (odd value > SAFE)
; is handed to tier 3 and frozen — its wrapped products are discarded by
; the where(). Every COMPACT-EVERY steps live lanes are compacted (argsort
; by liveness + gather) so a few stragglers don't drag full-width kernels.
(define (glide-lanes n0)              ; n0 int64 (B,), every entry >= 3
  (begin
    (define (run v starts live step)
      (begin
        (define odd (mod v 2))
        (define ovf (* live odd (int64 (> v SAFE))))
        (when (> (item (sum ovf)) 0)
          (map recheck (extract-values starts ovf)))
        (define live2 (- live ovf))
        (define stepped (+ (* odd (// (+ (* 3 v) 1) 2))
                           (* (- 1 odd) (// v 2))))
        (define v2 (where (= live2 1) stepped v))
        (define live3 (- live2 (* live2 (int64 (< v2 starts)))))
        (define cnt (item (sum live3)))
        (if (= cnt 0) #t
            (if (> step MAXSTEP)
                (begin                ; give up on the rest, bignum-check them
                  (map recheck (extract-values starts live3))
                  #t)
                (if (= (mod step COMPACT-EVERY) 0)
                    (begin            ; keep only the live lanes
                      (define keep (slice (argsort (- 1 live3)) 0 cnt))
                      (define starts2 (take starts keep))
                      (run (take v2 keep) starts2
                           (+ (* 0 starts2) 1) (+ step 1)))
                    (run v2 starts live3 (+ step 1)))))))
    (run n0 n0 (+ (* 0 n0) 1) 1)))

; --- candidate construction and the ascending sweep ---
; (surv, NS, surv-arr are defined in the run section below; verify-chunk
; looks them up at call time)
(define (verify-chunk k0 kn)          ; lanes M*k + r, k in [k0, k0+kn)
  (glide-lanes
    (reshape (+ (reshape (* M (+ k0 (iota kn))) (list kn 1)) surv-arr)
             (list (* kn NS)))))

(define kmin (// BASE M))             ; M*kmin = BASE, so k >= 1 always holds
(define kmax (// N M))

(define (save-progress next-k)
  (save-tree CKPT (list (array (list next-k) "int64")
                        (array (list escalated) "int64"))))

(define t0 (time))
(define (sweep k k-start chunks)
  (if (> k kmax)
      (display (list "sweep done. chunks" chunks "escalated" escalated
                     "failures" failures))
      (begin
        (define kn (min KC (+ 1 (- kmax k))))
        (verify-chunk k kn)
        (define k2 (+ k kn))
        (when (= (mod chunks LOG-EVERY) 0)
          (begin
            (define rate (/ (* M (- k2 k-start)) (max 0.001 (- (time) t0))))
            (display (list "n" (* M k2) "Mnum/s" (floor (/ rate 1000000))
                           "eta-min" (floor (/ (max 0 (- N (* M k2)))
                                               (* rate 60)))
                           "esc" escalated))))
        (when (= (mod chunks CKPT-EVERY) 0) (save-progress k2))
        (sweep k2 k-start (+ chunks 1)))))

; --- run ---
(define surv (sieve-survivors SK))
(define NS (length surv))
(define surv-arr (array surv "int64"))

(display (list "device" (device) "N" N "smoke" smoke
               "sieve-mod" M "survivors" NS "lanes/chunk" (* KC NS)))

(glide-lanes (+ 3 (iota (- BASE 3)))) ; base: [3, BASE) brute force, no sieve
(display (list "base verified to" BASE))

(define resume-k
  (if (exists? CKPT)
      (begin
        (define st (load-tree CKPT))
        (set! escalated (item (nth st 1)))
        (display (list "resuming at k" (item (car st))))
        (item (car st)))
      kmin))

(sweep resume-k resume-k 1)

(if (= failures 0)
    (display (list "VERIFIED: every n <=" (- (* M (+ kmax 1)) 1) "reaches 1"))
    (display (list "FAILURES" failures "— see COUNTEREXAMPLE-CANDIDATE lines")))
