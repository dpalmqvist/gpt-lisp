; ============================================================
; nanoGPT.lisp — a character-level transformer, entirely in Lisp,
; trained on Lisp source code. Architecture: GPT.
;   token emb + pos emb -> 2 x [LayerNorm -> Attention -> residual
;                               LayerNorm -> MLP       -> residual]
;   -> LayerNorm -> vocab projection
; ============================================================

; --- training corpus: the model learns to write Lisp ---
(define corpus "(define (fact n) (if (= n 0) 1 (* n (fact (- n 1))))) (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))) (define (len xs) (if (null? xs) 0 (+ 1 (len (cdr xs))))) ")

(define T 32)     ; context length
(define V 128)    ; vocab = ASCII
(define D 32)     ; model width

; --- every T-char window predicts the next chars ---
(define ids (text->ids corpus))
(define nwin (- (length ids) T))
(define Xids (array (map (lambda (i) (slice ids i (+ i T))) (range 0 nwin 2)) "int32"))
(define Yids (array (map (lambda (i) (slice ids (+ i 1) (+ i T 1))) (range 0 nwin 2)) "int32"))

; --- parameters: one nested list. grad returns the same tree. ---
(define (linear nin nout)
  (list (* (randn (list nin nout)) (/ 1.0 (sqrt (* 1.0 nin)))) (zeros (list nout))))

(define (block-params)                       ; q k v out | mlp-up mlp-down
  (list (linear D D) (linear D D) (linear D D) (linear D D)
        (linear D (* 4 D)) (linear (* 4 D) D)))

(define params
  (list (* (randn (list V D)) 0.1)           ; 0: token embeddings
        (* (randn (list T D)) 0.1)           ; 1: position embeddings
        (block-params)                        ; 2: transformer block 1
        (block-params)                        ; 3: transformer block 2
        (linear D V)))                        ; 4: head -> logits

; --- the architecture ---
(define (dense p x) (+ (matmul x (car p)) (car (cdr p))))

(define (ln x)                               ; layernorm (no learned scale)
  (/ (- x (meank x))
     (sqrt (+ (meank (square (- x (meank x)))) 0.00001))))

(define (attn p x m)                         ; single-head causal attention
  (begin
    (define q (dense (nth p 0) x))
    (define k (dense (nth p 1) x))
    (define v (dense (nth p 2) x))
    (define scores (+ (/ (matmul q (swap k)) (sqrt (* 1.0 D))) m))
    (dense (nth p 3) (matmul (softmax scores) v))))

(define (block p x m)
  (begin
    (define h (+ x (attn p (ln x) m)))
    (+ h (dense (nth p 5) (relu (dense (nth p 4) (ln h)))))))

(define (gpt p ids)
  (begin
    (define t (nth (shape ids) 1))
    (define x (+ (take (nth p 0) ids)
                 (take (nth p 1) (array (range t) "int32"))))
    (define m (causal-mask t))
    (dense (nth p 4) (ln (block (nth p 3) (block (nth p 2) x m) m)))))

; --- cross-entropy: -mean log p(correct next char) ---
(define (loss p)
  (- (mean (sumlast (* (log-softmax (gpt p Xids)) (onehot Yids V))))))

(define dloss (grad loss))

; --- training: generic tree SGD, tail-recursive loop ---
(define (sgd p g lr)
  (if (list? p)
      (map (lambda (pi gi) (sgd pi gi lr)) p g)
      (- p (* lr g))))

(defmacro (when test body) `(if ,test ,body))

(define (train p n steps lr)
  (if (= n 0)
      p
      (begin
        (when (= (mod n 100) 0)
          (display (list "step" (- steps n) "loss" (item (loss p)))))
        (train (force-tree (sgd p (dloss p) lr)) (- n 1) steps lr))))

(display (list "device:" (device) "windows:" nwin "params: ~50k"))
(display (list "initial loss" (item (loss params)) "(uniform would be" (log 128.0) ")"))

(define trained (train params 600 600 0.35))
(display (list "final loss" (item (loss trained))))

; --- generation: autoregressive sampling, one char at a time ---
(define (last-k xs k)
  (if (<= (length xs) k) xs (slice xs (- (length xs) k) (length xs))))

(define (gen p ctx n temp)
  (if (= n 0)
      ctx
      (begin
        (define logits (gpt p (array (list (last-k ctx T)) "int32")))
        (define nxt (item (sample (at logits 0 -1) temp)))
        (gen p (append ctx (list nxt)) (- n 1) temp))))

(display "--- the Lisp LLM writes Lisp: ---")
(display (ids->text (gen trained (text->ids "(define ") 120 0.25)))
