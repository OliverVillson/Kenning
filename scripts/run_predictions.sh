#!/usr/bin/env bash
# Theory predictions (kenning/predictions.py) on a GPU, after scripts/run_h200.sh.
#
#   bash scripts/run_predictions.sh quick   # 4 spans per group: check timings first
#   bash scripts/run_predictions.sh         # the full analysis
#
# It adds two probe passes the predictions need and run_h200.sh does not do:
# 2 and 4 vectors per kenning at one length (P6b), and a random-token control
# (P6a, P6b). Then it runs the analysis on the saved continue kennings. Every
# stage resumes. GPU time is not measured yet: each span prints its seconds,
# so run quick first and scale.
set -euo pipefail

MODEL="${MODEL:-ibm-granite/granite-4.2-3b}"
NAME="$(basename "$MODEL")"
TRACES="${TRACES:-data/traces-$NAME.jsonl}"

if [[ "${1:-full}" == "quick" ]]; then
  RUN="${RUN:-runs/$NAME-quick}"
  CONTROL="${CONTROL:-runs/$NAME-quick-control}"
  LENGTHS="32,128"; SAMPLES=8; STEPS=300; SPANS=4; KNEE_LENGTH=128
else
  RUN="${RUN:-runs/$NAME}"
  CONTROL="${CONTROL:-runs/$NAME-control}"
  LENGTHS="32,64,128,256,512"; SAMPLES=16; STEPS=600; SPANS=8; KNEE_LENGTH=128
fi

echo "== P6b: more vectors per kenning at one length"
python -m kenning probe --model "$MODEL" --out "$RUN" --mode continue --source traces --data "$TRACES" \
  --lengths "$KNEE_LENGTH" --n-vectors 2,4 --n-samples "$SAMPLES" --batch 8 --steps "$STEPS" --save-vectors

echo "== P6: random-token control (epsilon of an incompressible span, not a usefulness score)"
python -m kenning probe --model "$MODEL" --out "$CONTROL" --mode continue --source random \
  --lengths "$LENGTHS" --n-vectors 1,2,4 --n-samples "$SAMPLES" --batch 8 --steps "$STEPS"

echo "== predictions"
python -m kenning predictions --model "$MODEL" --run "$RUN" --control-run "$CONTROL" --max-spans "$SPANS"

echo "== done: $RUN/predictions.md"
