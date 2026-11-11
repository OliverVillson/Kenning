#!/usr/bin/env bash
# Step 1 on a GPU: inspect the model, generate its reasoning traces, run the
# capacity probe, write the report.
#
#   bash scripts/run_h200.sh quick   # about 8 chunks per length, 3 lengths: check timings first
#   bash scripts/run_h200.sh         # the full run
#
# Every stage resumes where it stopped, so the script is safe to re-run after
# an interruption. Override MODEL, RUN or TRACES in the environment if needed.
set -euo pipefail

MODEL="${MODEL:-ibm-granite/granite-4.2-3b}"
NAME="$(basename "$MODEL")"
TRACES="${TRACES:-data/traces-$NAME.jsonl}"

if [[ "${1:-full}" == "quick" ]]; then
  RUN="${RUN:-runs/$NAME-quick}"
  PROMPT_LIMIT=12
  RECON_LENGTHS="32,128,512"
  CONT_LENGTHS="32,128"
  SAMPLES=8
  RECON_STEPS=500
  CONT_STEPS=300
else
  RUN="${RUN:-runs/$NAME}"
  PROMPT_LIMIT=0
  RECON_LENGTHS="32,64,128,256,512,1024"
  CONT_LENGTHS="32,64,128,256,512"
  SAMPLES=16
  RECON_STEPS=1500
  CONT_STEPS=600
fi

echo "== 0. model facts and self-check"
python -m kenning inspect --model "$MODEL" --out "$RUN/model.json"

echo "== 1. the model's own reasoning traces"
python -m kenning traces --model "$MODEL" --out "$TRACES" --limit "$PROMPT_LIMIT" \
  --samples-per-prompt 2 --max-new-tokens 2048

echo "== 1b. how much of the reasoning repeats itself"
python -m kenning repeats --model "$MODEL" --data "$TRACES" --out "$RUN/repeats.json"

echo "== 2. reconstruct: storage ceiling of one vector"
python -m kenning probe --model "$MODEL" --out "$RUN" --mode reconstruct \
  --source traces --data "$TRACES" --lengths "$RECON_LENGTHS" \
  --n-samples "$SAMPLES" --batch "$SAMPLES" --steps "$RECON_STEPS" --save-vectors

echo "== 3. continue: how much of that later text can use (renumbered positions)"
python -m kenning probe --model "$MODEL" --out "$RUN" --mode continue \
  --source traces --data "$TRACES" --lengths "$CONT_LENGTHS" \
  --n-samples "$SAMPLES" --batch 8 --steps "$CONT_STEPS" --positions renumber

echo "== 4. continue again with gap positions, to choose between the two"
python -m kenning probe --model "$MODEL" --out "$RUN" --mode continue \
  --source traces --data "$TRACES" --lengths "$CONT_LENGTHS" \
  --n-samples "$SAMPLES" --batch 8 --steps "$CONT_STEPS" --positions gap

echo "== done: $RUN/summary.md and $RUN/capacity.png"
cat "$RUN/summary.md"
