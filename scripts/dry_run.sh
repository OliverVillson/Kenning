#!/usr/bin/env bash
# The whole of step 1 on a laptop CPU with nothing to download: train a toy
# byte-level model on this repository's own text, then run every command on it.
# Takes a few minutes. The numbers mean nothing about a real model; the point
# is to see every stage work before paying for GPU time.
set -euo pipefail

OUT="${OUT:-runs/dry-run}"
TEXT="${TEXT:-docs}"

python -m kenning toy --text "$TEXT" --out "$OUT/toy-model" --steps "${TOY_STEPS:-400}"
python -m kenning inspect --model "$OUT/toy-model" --out "$OUT/model.json" > /dev/null
python -m kenning traces --model "$OUT/toy-model" --out "$OUT/traces.jsonl" \
  --limit 3 --samples-per-prompt 2 --max-new-tokens 400
python -m kenning repeats --model "$OUT/toy-model" --source text --data "$TEXT"
python -m kenning probe --model "$OUT/toy-model" --out "$OUT/probe" --mode both \
  --source text --data "$TEXT" --lengths 8,16,32 --n-samples 4 --batch 4 \
  --steps 200 --cont-len 16 --n-train 4 --n-val 2
python -m kenning probe --model "$OUT/toy-model" --out "$OUT/probe" --mode continue \
  --source text --data "$TEXT" --lengths 8,16,32 --n-samples 4 --batch 4 \
  --steps 200 --cont-len 16 --n-train 4 --n-val 2 --positions gap
cat "$OUT/probe/summary.md"
