#!/bin/sh
# Build corpus.txt: ASCII Lisp/Scheme source from two well-known repos.
set -e
mkdir -p corpus_repos
[ -d corpus_repos/paip-lisp ] || \
    git clone --depth 1 https://github.com/norvig/paip-lisp corpus_repos/paip-lisp
[ -d corpus_repos/chibi-scheme ] || \
    git clone --depth 1 https://github.com/ashinn/chibi-scheme corpus_repos/chibi-scheme
find corpus_repos \( -name '*.lisp' -o -name '*.scm' \) -type f | sort | xargs cat \
    | LC_ALL=C tr -cd '\11\12\40-\176' > corpus.txt
wc -c corpus.txt
