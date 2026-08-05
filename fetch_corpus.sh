#!/bin/sh
# Build corpus.txt: ASCII Lisp/Scheme/Elisp source from well-known repos.
set -e
mkdir -p corpus_repos
clone() { [ -d "corpus_repos/$2" ] || git clone --depth 1 "$1" "corpus_repos/$2"; }
clone https://github.com/norvig/paip-lisp paip-lisp
clone https://github.com/ashinn/chibi-scheme chibi-scheme
clone https://github.com/sbcl/sbcl sbcl
clone https://git.savannah.gnu.org/git/guile.git guile || echo "warning: guile clone failed, continuing without it"
if [ ! -d corpus_repos/emacs ]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/emacs-mirror/emacs corpus_repos/emacs
    git -C corpus_repos/emacs sparse-checkout set lisp
fi
find corpus_repos \( -name '*.lisp' -o -name '*.scm' -o -name '*.el' \) -type f \
    | sort | xargs cat | LC_ALL=C tr -cd '\11\12\40-\176' > corpus.txt
wc -c corpus.txt
