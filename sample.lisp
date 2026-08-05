; sample.lisp — generate from the best checkpoint.
; Run: python3 mlx_lisp.py sample.lisp
(load "model.lisp")

(define ckpt (if (exists? "ckpt-best.npz") "ckpt-best.npz"
             (if (exists? "ckpt.npz") "ckpt.npz" #f)))
(if (equal? ckpt #f)
    (display "no checkpoint found - run train.lisp first")
    (begin
      (define st (load-tree ckpt))
      (define params (if (= (length st) 5) (nth st 2) (nth st 1)))
      (display (list "checkpoint" ckpt "step" (item (car st))))
      (display (sample-text params 500 0.6))))
