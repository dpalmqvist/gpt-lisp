; ============================================================
; train.lisp — minibatched Adam training with checkpoints/resume.
; Run: python3 mlx_lisp.py train.lisp
; Smoke test: touch SMOKE first (tiny model, 50 steps).
; ============================================================

(load "model.lisp")
(defmacro (when test body) `(if ,test ,body))

(define steps (if smoke 50 (if cloud 100000 50000)))
(define lr 0.0003)
(define b1 0.9)
(define b2 0.99)
(define eps 0.00000001)
(define log-every (if smoke 10 250))
(define ckpt-every (if smoke 25 1000))
(define sample-every (if smoke 25 (if cloud 5000 1000)))
(define nval-batch 4)

; --- data: char ids, last 5% is validation ---
(define ids (if (exists? "corpus.bin")
                (read-corpus "corpus.bin")
                (array (text->ids (read-file "corpus.txt")) "int32")))
(define n (length ids))
(define nval (floor (* 0.05 n)))
(define ntrain (- n nval))
(define train-ids (slice ids 0 ntrain))
(define val-ids (slice ids ntrain n))
(define max-off (- ntrain T))

(define tvec (array (range T) "int32"))
(define (get-batch src nmax)     ; -> (xb yb), each (B T) int32
  (begin
    (define pos (+ (reshape (randint 0 nmax (list B)) (list B 1)) tvec))
    (list (int32 (take src pos)) (int32 (take src (+ pos 1))))))

(seed 1234)                      ; fixed val batches for comparable losses
(define val-batches
  (map (lambda (i) (get-batch val-ids (- nval T))) (range nval-batch)))
(seed 42)

(define (val-loss p)
  (/ (apply + (map (lambda (vb) (item (val-fn p (car vb) (nth vb 1))))
                   val-batches))
     (* 1.0 nval-batch)))

; --- Adam as generic tree maps, like gpt.lisp's sgd ---
(define (tree-map2 f a b)
  (if (list? a) (map (lambda (x y) (tree-map2 f x y)) a b) (f a b)))

(define (tree-map3 f a b c)
  (if (list? a) (map (lambda (x y z) (tree-map3 f x y z)) a b c) (f a b c)))

(define (tree-zeros p) (if (list? p) (map tree-zeros p) (* 0.0 p)))

(define (adam-step p g m v t)    ; -> (p m v) updated
  (begin
    (define m2 (tree-map2 (lambda (mi gi) (+ (* b1 mi) (* (- 1.0 b1) gi))) m g))
    (define v2 (tree-map2 (lambda (vi gi) (+ (* b2 vi) (* (- 1.0 b2) (square gi)))) v g))
    (define c1 (- 1.0 (pow b1 t)))
    (define c2 (- 1.0 (pow b2 t)))
    (define p2 (tree-map3
                 (lambda (pi mi vi)
                   (- pi (* lr (/ (/ mi c1) (+ (sqrt (/ vi c2)) eps)))))
                 p m2 v2))
    (list p2 m2 v2)))

; --- compiled train step: traced through the interpreter ONCE ---
(define step-fn (jit (vgrad batch-loss)))
(define val-fn (jit batch-loss))

(define (save-ckpt path step best p m v)
  (save-tree path (list (array (list step) "int32")
                        (array (list best) "float32") p m v)))

(define (train-loop p m v step)
  (if (> step steps)
      p
      (begin
        (define bpair (get-batch train-ids max-off))
        (define lg (step-fn p (car bpair) (nth bpair 1)))
        (define pmv (adam-step p (nth lg 1) m v step))
        (define p2 (car pmv))
        (define m2 (nth pmv 1))
        (define v2 (nth pmv 2))
        (force-tree (list p2 m2 v2))
        (when (= (mod step log-every) 0)
          (begin
            (define vl (val-loss p2))
            (display (list "step" step "train" (item (car lg)) "val" vl))
            (when (< vl best-val)
              (begin
                (set! best-val vl)
                (save-ckpt "ckpt-best.npz" step vl p2 m2 v2)))))
        (when (= (mod step ckpt-every) 0)
          (save-ckpt "ckpt.npz" step best-val p2 m2 v2))
        (when (= (mod step sample-every) 0)
          (display (sample-text p2 150 0.5)))
        (train-loop p2 m2 v2 (+ step 1)))))

; --- init or resume ---
(define state0
  (if (exists? "ckpt.npz")
      (begin
        (display "resuming from ckpt.npz")
        (define st (load-tree "ckpt.npz"))
        (if (= (length st) 5)
            (list (item (car st)) (item (nth st 1))
                  (nth st 2) (nth st 3) (nth st 4))
            (list (item (car st)) 999.0
                  (nth st 1) (nth st 2) (nth st 3))))
      (begin
        (define p (init-params))
        (list 0 999.0 p (tree-zeros p) (tree-zeros p)))))

(define best-val (nth state0 1))

(display (list "device" (device) "corpus-chars" n "smoke" smoke
               "start-step" (+ (car state0) 1) "of" steps
               "best-val" best-val))

(define final
  (train-loop (nth state0 2) (nth state0 3) (nth state0 4)
              (+ (car state0) 1)))

(display (list "done. final val loss" (val-loss final)))
(display (sample-text final 300 0.5))
