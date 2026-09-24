#!/bin/zsh
# 5-fold Transformer (no descriptions): two folds at a time, then fold 4.
cd "$(dirname "$0")"
run() { SEQ_THREADS=5 .venv/bin/python -u seq_model.py --no_desc --fold $1 2>&1 | grep -E "^VALID|held-out macro-F1:|Traceback" | sed "s/^/fold $1: /"; }
{ run 0 & run 1 & wait; run 2 & run 3 & wait; run 4; echo exit_folds; } > outputs/folds.log 2>&1
