# Kenning

Language models that coin their own tokens.

A kenning is a token the model mints for itself while it works: one or a few
vectors that stand in for a span of its context or reasoning, kept in a table
next to the text they came from. The model can read a kenning in place of the
span, emit it again as a single token, and expand it back to the exact text.
The plan, the terms and how this relates to published work are in
[docs/DESIGN.md](docs/DESIGN.md).

**Status: step 1 of 5.** This repository currently contains the capacity
probe, which measures how much text a kenning can stand in for in a frozen
model before any training is done. First target: IBM Granite 4.2 3B.

## What the probe measures

For each chunk of text it optimises a few input vectors by gradient descent,
with the model frozen, and asks two questions:

- **Reconstruct:** can the model reproduce the chunk exactly from the
  kenning? This is the storage ceiling.
- **Continue:** does the kenning make the model predict what comes after the
  chunk the way the full chunk would? This is scored on real text the
  kenning was never trained on, so it measures use, not memorisation.

Both are upper bounds on what a trained writer network can reach.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Use a recent `transformers`; the Granite 4.2 model card states the minimum
version it needs.

## Check everything works, with nothing to download

```bash
pytest                      # about 20 seconds on a laptop CPU
bash scripts/dry_run.sh     # a few minutes: trains a toy model and runs every stage on it
```

The dry run's numbers say nothing about a real model. It exists so that
crashes happen on a laptop and not on rented hardware.

## Run on a GPU

```bash
bash scripts/run_h200.sh quick    # small pass: check timings and that scores move
bash scripts/run_h200.sh          # full run
```

Each stage prints its time per batch and resumes if interrupted. Outputs
land in `runs/granite-4.2-3b/`:

| File | Contents |
|---|---|
| `summary.md` | Tables for both modes and the headline capacity. |
| `capacity.png` | The two curves against chunk length. |
| `results.jsonl` | One row per chunk with every measured number. |
| `model.json` | Hidden size, tied or separate embeddings, self-check results. |
| `repeats.json` | How much of the model's reasoning repeats itself. |
| `vectors-*.pt` | The optimised kennings, for later analysis. |

`docs/DESIGN.md` section 5 lists the thresholds that decide what happens
next.

## Commands

```bash
python -m kenning inspect --model ibm-granite/granite-4.2-3b
python -m kenning traces  --model ... --out data/traces.jsonl
python -m kenning repeats --model ... --data data/traces.jsonl
python -m kenning probe   --model ... --out runs/x --source traces --data data/traces.jsonl
python -m kenning report  --run runs/x
python -m kenning toy     --text docs --out runs/toy-model
```

`--model` takes a Hugging Face id, a local path, or `tiny-random` (a small
untrained model built in memory). `python -m kenning probe --help` lists
every option.

## Tuning notes

- If reconstruct accuracy stalls well below 1.0 at short lengths, try
  `--lr 0.003` and `--lr 0.03`. The right value depends on the scale of the
  model's embeddings, which `inspect` prints.
- If optimisation is unstable in 16-bit precision, add `--dtype float32`.
  A 3B model in 32-bit takes about 15 GB.
- `--n-vectors 1,4` compares one-vector and four-vector kennings in one run.
- `--source random` probes random token ids: raw storage with no language
  structure to lean on. Only reconstruct mode is meaningful there.
- Out-of-memory errors halve the batch and retry automatically.
