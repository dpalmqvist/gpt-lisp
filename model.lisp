; ============================================================
; model.lisp — config + architecture for the scaled Lisp GPT.
; Pure functions only; no training state. Loaded by train.lisp
; and sample.lisp. A file named SMOKE selects the tiny config.
; ============================================================

(define smoke (exists? "SMOKE"))

(define T (if smoke 32 256))     ; context length
(define V 128)                   ; vocab = ASCII
(define D (if smoke 32 256))     ; model width
(define H (if smoke 2 8))        ; attention heads
(define HD (if smoke 16 32))     ; head dim = D / H
(define L (if smoke 2 6))        ; transformer blocks
(define B (if smoke 8 64))       ; batch size

; --- parameters ---
(define (linear nin nout)
  (list (* (randn (list nin nout)) (/ 1.0 (sqrt (* 1.0 nin))))
        (zeros (list nout))))

(define (ln-params) (list (ones (list D)) (zeros (list D))))

(define (block-params)           ; ln1 q k v out ln2 up down
  (list (ln-params) (linear D D) (linear D D) (linear D D) (linear D D)
        (ln-params) (linear D (* 4 D)) (linear (* 4 D) D)))

(define (init-params)            ; 0: tok emb, 1: pos emb, 2..L+1: blocks,
  (append                        ; L+2: final ln, L+3: head
    (list (* (randn (list V D)) 0.02)
          (* (randn (list T D)) 0.02))
    (map (lambda (i) (block-params)) (range L))
    (list (ln-params) (linear D V))))

; --- architecture ---
(define (dense p x) (+ (matmul x (car p)) (nth p 1)))

(define (ln p x)                 ; layernorm with learned scale + shift
  (begin
    (define xc (- x (meank x)))
    (+ (* (car p) (/ xc (sqrt (+ (meank (square xc)) 0.00001))))
       (nth p 1))))

(define (heads x)                ; (B T D) -> (B H T HD)
  (transpose-axes
    (reshape x (list (nth (shape x) 0) (nth (shape x) 1) H HD))
    (list 0 2 1 3)))

(define (unheads x)              ; (B H T HD) -> (B T D)
  (begin
    (define y (transpose-axes x (list 0 2 1 3)))
    (reshape y (list (nth (shape y) 0) (nth (shape y) 1) D))))

(define (attn p x m)             ; p: (q k v out) — multi-head causal
  (begin
    (define q (heads (dense (nth p 0) x)))
    (define k (heads (dense (nth p 1) x)))
    (define v (heads (dense (nth p 2) x)))
    (define scores (+ (/ (matmul q (swap k)) (sqrt (* 1.0 HD))) m))
    (dense (nth p 3) (unheads (matmul (softmax scores) v)))))

(define (block p x m)            ; p: (ln1 q k v out ln2 up down)
  (begin
    (define h (+ x (attn (slice p 1 5) (ln (car p) x) m)))
    (+ h (dense (nth p 7) (relu (dense (nth p 6) (ln (nth p 5) h)))))))

(define (run-blocks p i x m)
  (if (= i L) x (run-blocks p (+ i 1) (block (nth p (+ 2 i)) x m) m)))

(define (gpt p ids)
  (begin
    (define t (nth (shape ids) 1))
    (define x (+ (take (car p) ids)
                 (take (nth p 1) (array (range t) "int32"))))
    (define xf (run-blocks p 0 x (causal-mask t)))
    (dense (nth p (+ L 3)) (ln (nth p (+ L 2)) xf))))

(define (batch-loss p xb yb)     ; mean cross-entropy over batch
  (- (mean (sumlast (* (log-softmax (gpt p xb)) (onehot yb V))))))

; --- generation ---
(define (last-k xs k)
  (if (<= (length xs) k) xs (slice xs (- (length xs) k) (length xs))))

(define (gen p ctx n temp)
  (if (= n 0)
      ctx
      (begin
        (define logits (gpt p (array (list (last-k ctx T)) "int32")))
        (define nxt (item (sample (at logits 0 -1) temp)))
        (gen p (append ctx (list nxt)) (- n 1) temp))))

(define (sample-text p n temp)
  (ids->text (gen p (text->ids "(define ") n temp)))
