; sample.lisp — generate from the best checkpoint.
; Run: /tmp/mlx-venv/bin/python mlx_lisp.py sample.lisp
(load "model.lisp")

(define ckpt (if (exists? "ckpt-best.npz") "ckpt-best.npz" "ckpt.npz"))
(define st (load-tree ckpt))
(display (list "checkpoint" ckpt "step" (item (car st))))
(display (sample-text (nth st 1) 500 0.6))
