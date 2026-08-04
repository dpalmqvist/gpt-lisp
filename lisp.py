#!/usr/bin/env python3
"""A minimal Scheme-flavored LISP interpreter with a REPL."""

import functools
import math
import re
import operator as op
import sys

Symbol = str
Number = (int, float)


class String(str):
    """Distinct from Symbol so evaluate() treats it as a literal."""


# ---------- Reader (tokenize + parse) ----------

TOKEN_RE = re.compile(r'''\s*(;[^\n]*|"(?:\\.|[^"\\])*"|,@|[()'`,]|[^\s()'`,;]+)''')
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def tokenize(src: str):
    tokens, pos = [], 0
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            break
        pos = m.end()
        tok = m.group(1)
        # a token starting with " that the string pattern didn't fully match
        # can only be an unterminated string grabbed by the atom fallback
        if tok.startswith('"') and not STRING_RE.fullmatch(tok):
            raise SyntaxError("unterminated string")
        if not tok.startswith(";"):  # skip comments
            tokens.append(tok)
    return tokens


READER_MACROS = {"'": "quote", "`": "quasiquote",
                 ",": "unquote", ",@": "unquote-splicing"}


def parse(tokens):
    if not tokens:
        raise SyntaxError("unexpected EOF")
    tok = tokens.pop(0)
    if tok == "(":
        lst = []
        while tokens and tokens[0] != ")":
            lst.append(parse(tokens))
        if not tokens:
            raise SyntaxError("missing )")
        tokens.pop(0)  # discard ")"
        return lst
    if tok == ")":
        raise SyntaxError("unexpected )")
    if tok in READER_MACROS:
        return [READER_MACROS[tok], parse(tokens)]
    return atom(tok)


ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def atom(tok: str):
    if tok.startswith('"'):
        # decode left-to-right so \\n stays backslash+n instead of newline
        body = re.sub(r"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)),
                      tok[1:-1])
        return String(body)
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return Symbol(tok)


# ---------- Environment ----------

class Env(dict):
    def __init__(self, params=(), args=(), outer=None):
        if len(params) != len(args):
            raise TypeError(f"expected {len(params)} arguments, got {len(args)}")
        super().__init__(zip(params, args))
        self.outer = outer

    def find(self, name):
        if name in self:
            return self
        if self.outer is None:
            raise NameError(f"unbound symbol: {name}")
        return self.outer.find(name)


def standard_env() -> Env:
    env = Env()
    env.update(vars(math))  # sin, cos, sqrt, pi, ...
    env.update({
        "+": lambda *a: (sum(a) if all(isinstance(x, Number) for x in a)
                         else functools.reduce(op.add, a)),
        "-": lambda a, *b: -a if not b else a - sum(b),
        "*": lambda *a: math.prod(a),
        "/": lambda a, *b: a / math.prod(b) if b else 1 / a,
        ">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le, "=": op.eq,
        "abs": abs, "min": min, "max": max, "not": op.not_,
        "eq?": op.is_, "equal?": op.eq,
        "car": lambda x: x[0],
        "cdr": lambda x: x[1:],
        "cons": lambda a, b: [a] + list(b),
        "list": lambda *a: list(a),
        "list?": lambda x: isinstance(x, list),
        "null?": lambda x: x == [],
        "number?": lambda x: isinstance(x, Number),
        "symbol?": lambda x: isinstance(x, Symbol),
        "procedure?": callable,
        "length": len, "mod": op.mod,
        "append": lambda *a: sum(map(list, a), []),
        "map": lambda f, *xs: list(map(f, *xs)),  # variadic: (map f xs ys)
        "apply": lambda f, xs: f(*xs),
        "display": lambda x: print(lisp_str(x)),
        "begin": lambda *a: a[-1],
        "#t": True, "#f": False,
    })
    return env


# ---------- Evaluator ----------

class Procedure:
    """A user-defined lambda."""

    def __init__(self, params, body, env):
        if not isinstance(params, list):
            raise SyntaxError("parameters must be a list of symbols")
        self.params, self.body, self.env = params, body, env

    def __call__(self, *args):
        return evaluate(self.body, Env(self.params, args, self.env))


class Macro(Procedure):
    """Like Procedure, but receives UNEVALUATED forms and returns code."""


def implicit_begin(forms):
    """A body of several forms behaves as (begin ...); one form stays bare
    so tail-call handling sees it directly."""
    return forms[0] if len(forms) == 1 else ["begin", *forms]


def expand_qq(x, env):
    """Expand a quasiquoted template: `(a ,b ,@c) builds a list, evaluating
    only the unquoted parts."""
    if not isinstance(x, list) or not x:
        return x
    if x[0] == "unquote":
        return evaluate(x[1], env)
    out = []
    for item in x:
        if isinstance(item, list) and item and item[0] == "unquote-splicing":
            out.extend(evaluate(item[1], env))
        else:
            out.append(expand_qq(item, env))
    return out


def evaluate(x, env: Env):
    while True:
        if isinstance(x, String):
            return x
        if isinstance(x, Symbol):
            return env.find(x)[x]
        if not isinstance(x, list):      # constant literal
            return x
        if not x:
            return x
        head = x[0]
        if head == "quote":              # (quote exp)
            return x[1]
        if head == "quasiquote":         # `template with , and ,@ holes
            return expand_qq(x[1], env)
        if head == "defmacro":           # (defmacro (name args...) body...)
            name, params = x[1][0], x[1][1:]
            env[name] = Macro(params, implicit_begin(x[2:]), env)
            return None
        if head == "if":                 # (if test conseq alt) — tail position
            _, test, conseq, *alt = x
            if evaluate(test, env) is not False:
                x = conseq
            elif alt:
                x = alt[0]
            else:
                return None
            continue
        if head == "define":             # (define name exp) | (define (f args) body...)
            if isinstance(x[1], list):
                name, params = x[1][0], x[1][1:]
                env[name] = Procedure(params, implicit_begin(x[2:]), env)
            else:
                env[x[1]] = evaluate(x[2], env)
            return None
        if head == "set!":               # (set! name exp)
            env.find(x[1])[x[1]] = evaluate(x[2], env)
            return None
        if head == "lambda":             # (lambda (args) body...)
            return Procedure(x[1], implicit_begin(x[2:]), env)
        if head == "begin":              # (begin exp...) — last expr is tail position
            for exp in x[1:-1]:
                evaluate(exp, env)
            if len(x) == 1:
                return None
            x = x[-1]
            continue
        # procedure call
        proc = evaluate(head, env)
        if type(proc) is Macro:          # expand with RAW forms, then eval result
            x = proc(*x[1:])
            continue
        args = [evaluate(arg, env) for arg in x[1:]]
        if type(proc) is Procedure:  # tail call: loop instead of recursing
            # (exact type check: mx.grad's wrapper forwards __class__,
            #  which makes isinstance() lie and would bypass the wrapper)
            x, env = proc.body, Env(proc.params, args, proc.env)
            continue
        return proc(*args)


# ---------- Printer + REPL ----------

def lisp_str(x) -> str:
    if x is True:
        return "#t"
    if x is False:
        return "#f"
    if isinstance(x, list):
        return "(" + " ".join(map(lisp_str, x)) + ")"
    if isinstance(x, String):
        return f'"{x}"' 
    if isinstance(x, Procedure):
        return "#<procedure>"
    if callable(x):
        return "#<builtin>"
    return str(x)


def balanced(src: str) -> bool:
    """Ready to evaluate? Counts paren *tokens*, so parens inside strings
    and comments don't confuse the REPL."""
    try:
        tokens = tokenize(src)
    except SyntaxError:
        return False  # unterminated string: keep reading input
    return tokens.count("(") <= tokens.count(")")


def repl(env=None, banner="minimal lisp — Ctrl-D to exit"):
    env = env or standard_env()
    print(banner)
    buf = ""
    while True:
        try:
            buf += input("... " if buf else "lisp> ") + " "
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            buf = ""
            continue
        if not buf.strip() or not balanced(buf):
            continue  # keep reading multi-line input
        try:
            tokens = tokenize(buf)
            while tokens:
                result = evaluate(parse(tokens), env)
                if result is not None:
                    print(lisp_str(result))
        except Exception as e:
            print(f"error: {e}")
        buf = ""


if __name__ == "__main__":
    if len(sys.argv) > 1:  # run a file: python lisp.py program.lisp
        with open(sys.argv[1]) as f:
            tokens = tokenize(f.read())
        env = standard_env()
        while tokens:
            evaluate(parse(tokens), env)
    else:
        repl()
