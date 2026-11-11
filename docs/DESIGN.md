# Kenning: design

Status: step 1 (capacity probe) is implemented in this repository. Everything
after step 1 is a plan, and section 3 explains why that plan changed after a
check of published work.

## 1. The idea

A kenning is an Old Norse figure of speech that replaces a phrase with one
coined compound ("whale-road" for the sea). Here it is a token that a
language model coins for itself while it works.

The model reads a span of its own context or reasoning, decides it is worth
keeping, and mints a kenning for it: one or a few vectors in the model's
input embedding space, registered in a table next to the text they stand
for. From then on the model can

- **read** the kenning in place of the span, so the context gets shorter,
- **emit** the kenning again as a single token whenever it wants to refer to
  that span, so recurring ideas cost one token each time,
- **expand** the kenning back to its exact source text when it needs a
  detail the vector lost.

The first target model is IBM Granite 4.2 3B (Apache 2.0, reasoning inside
`<think>` tags). A release would be named after its base, for example
`Kenning-1-Granite-4.2-3B`.

## 2. Terms

| Term | Meaning |
|---|---|
| Kenning | A minted token: read vector(s), optionally an emit vector, and the source text it stands for. |
| Mint | The act of creating a kenning from a span. |
| Writer | The part that turns a span into the kenning's vectors. In the final system this is the model itself plus a small projector network. |
| Reader | The model consuming kennings in its context. |
| Table | The per-context (later per-project) store of kennings: id, vectors, source text. |
| Expand | Replacing a kenning in context with its source text. |
| Frozen | Weights that are not trained. |
| LoRA | Low-rank adaptation: a cheap way to fine-tune a model by training small added matrices instead of all weights. |
| KL divergence | A measure of how far one probability distribution is from another. Here: how far the model's next-token predictions with a kenning are from its predictions with the full text. |
| Cross-entropy (CE) | How surprised the model is by the actual next token, in nats (natural-log units) per token. Lower is better. |
| RoPE | Rotary position embeddings: the model knows token positions by rotating vectors inside attention. Only relative distance matters. |
| CoT | Chain of thought: the model's written-out reasoning. |

## 3. Published work, and what it leaves open

Checked on 2026-11-11 with a handful of searches, reading abstracts and
summaries rather than full papers. Treat this as a map, not a literature
review, and re-read the papers before citing them.

| Work | What it does | How it differs from Kenning |
|---|---|---|
| Kuratov et al. 2025, "Cramming 1568 tokens into a single vector" ([arXiv:2502.13063](https://arxiv.org/abs/2502.13063)) | Optimises one input vector per text with the model frozen; reports a single vector letting Llama-3-8B reproduce 1,568 tokens. | Measures capacity only. Our step 1 reuses the method. |
| zip2zip ([arXiv:2506.01084](https://arxiv.org/abs/2506.01084)) | Merges recurring token sequences into new "hypertokens" at inference time using LZW (a classic dictionary compression algorithm); a small encoder builds their embeddings. | New tokens are surface n-grams chosen by a fixed rule, not concepts chosen by the model. Reports a collapse on arithmetic (GSM8K) and that only about a quarter of hypertokens get reused. |
| LightThinker ([arXiv:2502.15589](https://arxiv.org/abs/2502.15589)) | Trains a model to compress finished thought steps into a few "gist" tokens built from hidden states, then drops the original text. | Gists are read once as context. They are not emitted again, not expandable. |
| Selective Latent Thinking ([arXiv:2605.25745](https://arxiv.org/abs/2605.25745)) | Compresses only reasoning spans a confidence gate accepts; keeps numbers as text. | Same single-use limit. Shows that *which* spans are compressed decides accuracy. |
| Abstract-CoT ([arXiv:2604.22709](https://arxiv.org/abs/2604.22709)) | Reasons in a fixed codebook of reserved tokens (64 in the main runs) learned in post-training; reports up to 11.6x fewer reasoning tokens on MATH-500. Includes results on Granite-4.0-Micro (3B). | Vocabulary is fixed after training. Tokens have no text definition. |
| CIRF ([arXiv:2605.28292](https://arxiv.org/abs/2605.28292)) | Clusters reasoning steps into 32 to 256 reusable "functional" tokens offline, then fine-tunes the model to reason in them. | Fixed vocabulary; tokens do not decode to text. |
| Shorthand for Thought ([arXiv:2604.26355](https://arxiv.org/abs/2604.26355)) | Adds 250 frequent multi-word "supertokens" mined from reasoning traces to the vocabulary. | Fixed vocabulary built once. Useful training lessons, see section 7. |
| Token Assorted ([arXiv:2502.03275](https://arxiv.org/abs/2502.03275)), CCoT ([arXiv:2412.13171](https://arxiv.org/abs/2412.13171)), CODI ([arXiv:2502.21074](https://arxiv.org/abs/2502.21074)), CoLaR ([arXiv:2505.16552](https://arxiv.org/abs/2505.16552)), Coconut | Other ways to replace written reasoning with latent (non-text) tokens or vectors. | Compress or replace reasoning; none mint reusable named tokens at inference time. |
| Generation with Dynamic Vocabulary ([arXiv:2410.08481](https://arxiv.org/abs/2410.08481)) | Lets a model emit whole text phrases as single units, embedded on the fly by a phrase encoder. | Candidate phrases are supplied from outside the model, not chosen by it; aimed at fluent generation, not reasoning memory. |

**What this means.** The two halves of the original idea each exist
separately: making new tokens for recurring token chains at inference time
(zip2zip), and compressing finished reasoning into vectors and dropping the
text (LightThinker, Selective Latent Thinking). Reasoning in a small learned
vocabulary of reusable abstract tokens also exists (Abstract-CoT, CIRF), but
with a vocabulary frozen at training time.

What I did not find is the combination that makes a kenning a kenning:

1. minted **at inference time**,
2. for a span the **model chose** because of what it means,
3. **emitted again** later as a token,
4. **anchored to text** it can be expanded back into,
5. kept in a table that can **outlive the context**.

That combination is the project. A release that only did step compression
would be a reimplementation of LightThinker on a different model.

## 4. System sketch

```
context:  ... tokens ... [K17] ... tokens ... <mint> ... [K17] ... <expand>[K17] ...

table:    id   read vector(s)   emit vector   source text
          K17  r_17             w_17          "the tank fills at 300 L/h net of the leak"
```

- **Read path.** A kenning in the context is fed to the model as its read
  vector(s) instead of a row of the embedding table.
- **Mint.** The model emits `<mint>` after a span. The hidden state there
  goes through a projector to produce the read vector, and through a second
  projector to produce the emit vector. Granite 4.2 3B is reported to keep
  input and output embeddings separate, so the two cannot be shared; `kenning
  inspect` prints whether that is true for the loaded weights.
- **Emit path.** Next-token scores are computed over the normal vocabulary
  and over the table's emit vectors together, so a kenning competes with
  ordinary tokens.
- **Expand.** `<expand>` followed by a kenning makes the runtime splice the
  source text back into the context. This is what makes the system lossless
  even though each vector is lossy.
- **Positions.** When a span is replaced, the following tokens either keep
  their old position numbers ("gap") or are renumbered. Step 1 measures
  which one the model tolerates better.

## 5. Build order

Each step ends in a number that decides whether to continue.

| Step | What | Trains | Decides |
|---|---|---|---|
| 0 | `kenning inspect`: confirm hidden size, tied or separate embeddings, that embeddings can be fed directly. | nothing | Whether the code paths are valid for this model. |
| 1 | **Capacity probe** (this branch). Optimise one kenning per chunk with the model frozen. | only the kenning vectors | How many tokens a kenning can hold, and how much of that later reasoning can use. Sets chunk length and vectors per kenning. |
| 1b | Reuse study: how often do ideas actually recur in Granite's reasoning? (`kenning repeats` gives the exact-repeat lower bound.) | nothing | Whether emit-again reuse has enough to work with. |
| 2 | Writer: `<mint>` hidden state through a projector becomes the kenning. Loss: the model reading the kenning should predict what follows as the frozen model does reading the full span. | projectors + LoRA | How close a one-pass writer gets to the step 1 ceiling. |
| 3 | **Kenning 1**: mint, read, emit-again and expand inside one context. | projectors + LoRA (or partial unfreezing, see section 7) | Accuracy against context length, compared with plain reasoning, text summarisation, and single-use gists. |
| 4 | Viewer: kennings shown as chips that expand to their source text. | nothing | Demo. |
| 5 | Kenning 2: tables that persist across contexts, merging and eviction. | to be designed | |

### Step 1 thresholds (proposed before seeing any result)

Reconstruct mode, one vector:

- lossless capacity of 128 tokens or more: one vector per kenning is enough
  for a reasoning step.
- 32 to 128: plan for 2 to 4 vectors per kenning.
- under 32: single-vector kennings are not viable on this model.

Continue mode, scored on the held-out real continuation:

- 0.8 or more of the chunk's effect recovered at 128 tokens: go to step 2
  with chunks of about that size.
- 0.5 to 0.8: go, with shorter chunks or more vectors per kenning.
- under 0.5 at 64 tokens: the model cannot use what a vector stores. Stop
  and investigate before training anything.

Positions: use whichever of gap and renumber recovers more. If they are
within 0.02, use gap, because cached attention state stays valid.

## 6. Step 1 in detail

The model is frozen. For each chunk of k tokens we optimise n input vectors
by gradient descent (Adam, learning rate 0.01, at most 1,000 steps by
default; these are starting values to tune).

**Reconstruct.** The model reads `start | kenning | chunk` and is scored on
predicting each chunk token. Accuracy 1.0 means decoding from the kenning
alone gives the chunk back exactly. Also reported: bits of information the
kenning supplied (the drop in cross-entropy, summed over the chunk).

**Continue.** This is the measurement that matters for Kenning. Storage is
useless if later reasoning cannot use it, and the published results above
show that the two come apart.

- Teacher: the frozen model reading `start | chunk | continuation`.
- Student: the same model reading `start | kenning | continuation`.
- The kenning is trained to match the teacher's next-token distributions
  (KL divergence) on 8 continuations sampled from the teacher.
- The best step is picked on 4 more sampled continuations.
- The score is taken on the real continuation from the source text, which
  the kenning never saw. A kenning that merely memorised its training
  continuations scores badly here.

Headline: `recovered = 1 - KL(teacher, student) / KL(teacher, no context)`.
1.0 means the kenning replaces the chunk perfectly; 0.0 means it is no
better than deleting the chunk.

**Baseline for free.** Vectors start as the mean of the chunk's token
embeddings. The score before any optimisation is reported as "mean of
embeddings": what you get by simply adding the chunk's tokens together.

**Text sources.** `traces` (Granite's own reasoning on 40 built-in
problems; the relevant case), `text` (any files), `random` (random token
ids, a control with no structure to exploit).

**Both numbers are upper bounds.** Each kenning here gets hundreds of
optimisation steps on its own chunk. The step 2 writer gets one forward
pass.

**Also recorded.** The length of each optimised vector relative to a normal
token embedding, which shows whether good kennings sit far outside the
ordinary embedding distribution.

## 7. Lessons taken from published work

- **Keep numbers as text.** zip2zip reports GSM8K accuracy for one model
  falling from 0.82 to 0.15, and Selective Latent Thinking reports that
  compressing arbitrary spans cut accuracy from 54.7 to 17.3 on one
  benchmark while keeping numbers explicit preserved most of it. Kennings
  should not swallow digits, and evaluation must include arithmetic.
- **Initialise from token embeddings.** Shorthand for Thought reports that
  new embeddings started from the mean of their parts trained, and random
  starts did not.
- **LoRA may not be enough.** The same paper reports that LoRA made the
  model use new tokens without shortening its output, and that unfreezing a
  few layers was needed. Budget for trying both in step 3.
- **Measure reuse, not just creation.** zip2zip reports about a quarter of
  its new tokens ever being used again. A kenning that is minted and never
  emitted again is only compression.
- **Fixed small vocabularies work.** Abstract-CoT reports large compression
  on a 3B Granite model with 64 reserved tokens. That is the baseline a
  growing vocabulary has to justify itself against.

## 8. Decisions for Oliver

1. **Scope of Kenning 1.** Recommended: include emit-again and expand, for
   the reason in section 3. The alternative (step compression only) is
   quicker but already published.
2. **Benchmarks.** Maths and code reasoning with exact-match answers.
   Granite 4.2 3B is reported not to have had the agentic training stage of
   its larger siblings, so agent tasks are a poor first demo.
3. **Licence for this repository.** It is public with no licence file,
   which means nobody may legally reuse the code. Apache 2.0 would match
   the base model.
4. **What is public when.** Results committed to this repository are
   visible immediately.

## 9. Risks

- **The reading side.** One vector cannot be attended into the way a span
  of tokens can. Step 1's continue mode exists to put a number on this.
- **Reuse may be rare.** If ideas seldom recur inside one reasoning trace,
  emit-again only pays off with persistence (step 5).
- **Crowding.** Many similar kennings will be confused at emit time.
- **Staleness.** Vectors written for one set of weights drift if the model
  is fine-tuned afterwards.
- **Compounding loss.** Kennings minted from spans that already contain
  kennings accumulate error.
- **Custom inference.** The released model will not run in standard tools
  without this repository's generation loop.
