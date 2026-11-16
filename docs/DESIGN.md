# Kenning: design

Status: step 1 (capacity probe) is implemented in this repository. Everything
after step 1 is a plan, and section 3 explains why that plan changed after a
check of published work (re-checked against full texts on 2026-11-16).

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

Re-checked on 2026-11-16 against the full text of each paper (arXiv HTML or
PDF), replacing the abstract-only check of 2026-11-11. Two kinds of
statement are kept apart below:

- **Read**: stated in the paper, with the section, table or equation given.
  Where a fetch was cut short and a passage was only seen through an
  automated summary, the cell says *(summary only)*; re-read those before
  citing them.
- **Inferred**: our judgement about what the paper does not do, or how it
  compares. These are marked *Inferred* and are claims about absence, which
  a later version of a paper can overturn.

Each row is scored against the five kenning properties: **P1** minted at
inference time, **P2** for a span the model chose by content, **P3** emitted
again as an output token, **P4** anchored to source text it expands back
into, **P5** stored in a table that outlives the context. Y = yes, ~ =
partly, N = no (from the text read).

### 3.1 The papers named in the 2026-11-11 table

| Work | P1 | P2 | P3 | P4 | P5 | What the paper says (read) | Corrections to the earlier table |
|---|---|---|---|---|---|---|---|
| Kuratov et al., "Cramming 1568 Tokens into a Single Vector and Back Again" ([arXiv:2502.13063](https://arxiv.org/abs/2502.13063)) | Y | N | N | ~ | N | One input vector optimised per text with the model frozen (§3, Fig. 2). Llama-3.1-8B reconstructs up to 1,568 tokens from one vector at 0.99 token accuracy on PG-19 text; 792 for random word sequences (§4.1, Table 1). Capacity is set "not by the length of the input but by the amount of uncertainty to be reduced" (Abstract, §4.3). No downstream use is tested; the Limitations section names it as open. Vector norms are not reported. | Model is Llama-3.1-8B, not Llama-3-8B. *Inferred:* our continue mode (using the vector as context, not just reconstructing) is a measurement this paper does not make. |
| zip2zip ([arXiv:2506.01084](https://arxiv.org/abs/2506.01084)) | Y | N | Y | Y | N | Hypertokens are created during each inference session; the codebook "is initialized as an empty set" per session (§2.1) and need not persist "across inference calls" (footnote 3). Merges follow the LZW rule (§2.1). A trained hyper-encoder builds input embeddings, and a second one output embeddings when weights are untied (§2.2). Generation uses "a joint softmax over the union of the base vocabulary and the hyper vocabulary" (§2.3), and LZW decoding is lossless (§2.1). Phi-3.5-4B GSM8K: 0.82 base, 0.15 zip2zip (Table 4, §3.5, "significant degradation"); Phi-3-14B 0.84 to 0.25. "Only about 25% of hypertokens are reused during generation" (§5, no supporting table). | The earlier table implied hypertokens are not emitted. They are: zip2zip already has P1, P3 and P4 at the level of surface token strings. "Collapse" is our word; the paper says "significant degradation", for Phi-3.5-4B. |
| LightThinker ([arXiv:2502.15589](https://arxiv.org/abs/2502.15589)) | Y | ~ | N | N | N | Gist ("cache") tokens hold hidden states of a finished thought; the original text is discarded (Abstract, §3.2). Segment boundaries come from a rule, every 6 tokens or at "\n\n" (§3.2, §4.1; Limitations item 6), while the model learns when to trigger compression (Appendix A.1). Cache tokens are excluded from the training targets (§3.2). "Our cache tokens do not aim for lossless compression" (§3.3). Qwen2.5-7B: accuracy 62.90 to 62.80, peak tokens down 70% (Table 1, §4.2); Llama3.1-8B loses about 6 points. Errors are "not sufficiently sensitive to numerical values" (§4.5). | Span choice is rule-based boundaries with learned timing, not model-chosen by meaning. Otherwise confirmed. |
| Selective Latent Thinking ([arXiv:2605.25745](https://arxiv.org/abs/2605.25745), v2) | Y | Y | N | N | N | A learned confidence gate accepts the longest prefix whose per-token scores pass a threshold (§3.1, Eq. 1); accepted spans become one latent "injected back into the primary LLM as a proxy embedding" (§3.1), input only, "within a single reasoning trajectory" (§1). Abstract: compressing arbitrary spans cuts Qwen3-4B GSM-Hard accuracy from 54.7% to 17.3%, while keeping numerical tokens explicit retains 51.8%. | The 17.3% is a random-span baseline (2 to 5 tokens), not the method, and the comparison is not length-matched *(summary only for the table location)*. The benchmark is GSM-Hard. This paper does have P2: the gate chooses spans by content. |
| Abstract-CoT, "Thinking Without Words" ([arXiv:2604.22709](https://arxiv.org/abs/2604.22709), v2) | N | N | Y | N | N | M = 64 reserved tokens in the main runs (§4.1), introduced in post-training (§1). The model generates them under constrained decoding (§3.2). Their frequencies follow a power law, "indicating re-use over a learned 'reasoning language'" (§4.3, Fig. 4). Tokens are "semantically uninformative" (§1); no decoding to text. Granite-4.0-Micro (3B) on MATH-500 (Table 1): base 71.4% at 1,043 tokens, SFT+RL 73.8% at 1,587, Abstract-CoT 74.4% at 153. | Confirmed: 64 tokens, Granite-4.0-Micro included. The "11.6x" is Qwen3-8B against SFT+RL (1,671 / 144); for Granite the ratio is 6.8x against the untuned base model and 10.4x against SFT+RL (computed from Table 1). |
| CIRF ([arXiv:2605.28292](https://arxiv.org/abs/2605.28292)) | N | N | Y | N | N | Reasoning steps are split at delimiters such as "Step k" (§3.2), clustered offline with a VQ-VAE, and the vocabulary is extended "with K new special tokens" (§3.2) that the fine-tuned model emits (Eq. 13 to 14). No text decoder is defined; the VQ-VAE decoder reconstructs embeddings (§3.2, Eq. 10). GSM8K 71.2% against CODI 25.5% (§4.2). | "32 to 256" is not supported: Table 2 tests K = 32, 128, 512 and 2,048. Say "the paper defines no decoder to text" rather than "tokens do not decode". |
| Shorthand for Thought ([arXiv:2604.26355](https://arxiv.org/abs/2604.26355), v7) | N | N | Y | Y | N | K = 250 cross-word merges mined from the model's own traces, each "one new vocabulary entry" (§4.1); a fixed extension of the vocabulary. New rows are initialised as the mean of their parts; random starts "failed to train stably", reported "qualitatively rather than as a controlled ablation" (§4.1). Traces shorten by 8.1% on average (§1). | See section 7 for the LoRA claim, which is weaker than we stated. |
| Generation with Dynamic Vocabulary ([arXiv:2410.08481](https://arxiv.org/abs/2410.08481)) | Y | N | Y | ~ | N | Candidate phrases are retrieved from external documents per test sample (DPR plus FAISS, §3.1, Appendix B) and embedded by a phrase encoder (§2.2). Phrases are scored in the same softmax as tokens (Eq. 1). The method "relies on external techniques, such as a retriever" (Limitations). How emitted phrases map back to text is not spelled out. | Confirmed. Note it has P1 and P3: a per-input vocabulary whose new entries are emitted. |

Token Assorted, CCoT, CODI, CoLaR and Coconut were not re-read; the
2026-11-11 one-line summary of them stands as unverified.

### 3.2 New work found on 2026-11-16

A fresh search (about ten queries for inference-time tokens, dynamic
vocabularies, expandable compression and persistent latent memory) found the
papers below. Each was read past the abstract, but the later sections of
most came through automated summaries, so treat these rows as one pass.

| Work | P1 | P2 | P3 | P4 | P5 | What it does (read) |
|---|---|---|---|---|---|---|
| "Large Language Model as Token Compressor and Decompressor" ([arXiv:2603.25340](https://arxiv.org/abs/2603.25340)) | N | ~ | Y | ~ | N | A LoRA-tuned model compresses text into a variable number of Z-tokens from a learned codebook added to the vocabulary (§1, §3.1), and decodes them back (BLEU-4 99.31 at 4x, Table 1). The codebook is trained; Z-tokens are not tied to source spans. |
| TokMem ([arXiv:2510.00444](https://arxiv.org/abs/2510.00444)) | N | ~ | Y | N | ~ | One trained memory token per procedure, shared "between the input embedding layer and the LM head" (§2.2), so the model emits it to recall the procedure. "No direct textual form" (§2.2); the bank is trained, not written at inference. |
| LCM, "Lossless Context Management" ([arXiv:2605.04050](https://arxiv.org/abs/2605.04050)) | ~ | N | N | Y | Y | An engine keeps an immutable store and a tree of text summaries; "any summary can be replaced by the original content via the lcm_expand tool" (§2), in "a persistent store" (§2.1). Summaries are text, chosen by the engine, not tokens. (The arXiv stamp, 14 Feb 2026, does not match the id; check before citing.) |
| "End-to-End Context Compression at Scale" ([arXiv:2606.09659](https://arxiv.org/abs/2606.09659), v2) | Y | ~ | N | ~ | N | An encoder maps each fixed block of N tokens to one latent the decoder reads (§3). An expand tool returns a segment's original text, and generated tokens are compacted online *(§7, summary only)*. |
| ASTC, "In-Model Context Compression with Agentic Soft Tokens" ([arXiv:2610.41291](https://arxiv.org/abs/2610.41291)) | Y | Y | N | N | ~ | Training-free: a frozen model merges predictable spans of tool output into mean-pooled soft tokens in its cache, keeping exact values as hard tokens (§4.1). Input only. |
| LensVLM ([arXiv:2605.07019](https://arxiv.org/abs/2605.07019)) | ~ | Y | N | Y | N | Text is rendered to images; the model picks images to expand, and an Expand tool returns their source text (§3.2). Expansion is a tool call, not a token. |
| Also checked, further away | | | | | | SEED ([arXiv:2611.06084](https://arxiv.org/abs/2611.06084)), Memorilla ([arXiv:2610.21375](https://arxiv.org/abs/2610.21375)), FlashMem ([arXiv:2601.05505](https://arxiv.org/abs/2601.05505)), Metacognitive Reuse ([arXiv:2509.13237](https://arxiv.org/abs/2509.13237), named text "behaviors" in a growing handbook), CLSR ([arXiv:2606.29354](https://arxiv.org/abs/2606.29354), invented text symbol systems). None has P3, none has P4 at token level. |

Not yet opened, worth a look before a public claim: MemGen, MemOS,
Cartridges, ToolGen, MemoryLLM and M+ (persistent-memory systems named in
the papers above).

### 3.3 What this means

**No paper read has all five properties** (read, as of 2026-11-16). The
highest any reaches is three. But every property on its own, and several
pairs, are already published, so the claim has to be narrower than
"kennings are new":

- **P1 + P3 + P4 together are published** for surface strings: zip2zip
  mints tokens per session, emits them through a joint softmax and decodes
  them losslessly. Generation with Dynamic Vocabulary has P1 + P3 with
  retrieved phrases. "A model emits tokens it did not have at training time"
  is not new.
- **P2 is published for input-only compression**: Selective Latent Thinking
  (learned gate) and ASTC (frozen model, predictability) choose spans by
  content.
- **Expanding back to source text is published** as a tool or runtime step:
  LCM, LensVLM, the end-to-end compression paper. P4 alone is not new.
- **Learned tokens in the output head with persistence (P3 + P5)** exist in
  TokMem, but are trained offline per procedure.

*Inferred:* what remains open is the join of the two lines of work. Every
system that emits new tokens (zip2zip, Dynamic Vocabulary, Abstract-CoT,
CIRF, Shorthand, TokMem, Z-tokens) gets its spans from a fixed rule, a
retriever or training. Every system where the model chooses spans by
content (Selective Latent Thinking, ASTC, LightThinker's timing) only reads
the result once as input. **The narrowed claim is: a vector that the model
mints at inference for a span it chose, which then competes with the
vocabulary at the output and stays anchored to its source text in a table
that can outlive the context.** P2 + P3 at once is the core; P4 and P5 are
what make it safe and useful, but are not by themselves new.

Consequences for the plan:

1. Scope decision 1 in section 8 stands, and is stronger: emit-again is the
   part that separates Kenning from published work, so a Kenning 1 without
   it would add nothing new.
2. A release must say "to our knowledge, as of <date>" and cite zip2zip,
   Selective Latent Thinking, LCM and TokMem as the nearest work on each
   side, rather than claim each property is new.
3. zip2zip becomes a required baseline alongside Abstract-CoT: it is the
   published system with emit-again, so a kenning has to beat surface
   n-gram tokens at reuse, not just exist.

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

Re-checked against full texts on 2026-11-16; see section 3 for sources.

- **Keep numbers as text.** zip2zip reports Phi-3.5-4B GSM8K accuracy
  falling from 0.82 to 0.15 (Table 4), and Selective Latent Thinking reports
  that compressing random spans cut Qwen3-4B GSM-Hard accuracy from 54.7% to
  17.3% while keeping numbers explicit kept 51.8% (Abstract). LightThinker
  attributes its maths errors to compression that is "not sufficiently
  sensitive to numerical values" (§4.5). *Inferred:* Cramming explains why;
  capacity is set by the uncertainty to be removed, and digits carry the
  most. Kennings should not swallow digits, and evaluation must include
  arithmetic.
- **Initialise from token embeddings.** Shorthand for Thought initialises
  new rows as the mean of their parts and reports that random starts
  "failed to train stably", but only "qualitatively rather than as a
  controlled ablation" (§4.1). Weak evidence; cheap to follow.
- **LoRA may not be enough (weak evidence).** Shorthand for Thought's
  comparison of LoRA, embedding-only training and unfreezing a few layers is
  in its Appendix D, Table 8, which we only saw through a summary: on one
  model (QwQ-32B) LoRA changed output length by -1.0% to +3.4%, while
  unfreezing shortened it by 2.3% to 10.6%. One model and three benchmarks;
  it does not show unfreezing is needed. Budget for trying both in step 3.
- **Measure reuse, not just creation.** zip2zip states that only about 25%
  of hypertokens are reused during generation (§5), without a supporting
  table. A kenning that is minted and never emitted again is only
  compression.
- **Fixed small vocabularies work.** Abstract-CoT reaches 74.4% on
  MATH-500 with 153 reasoning tokens on Granite-4.0-Micro (3B), against
  71.4% at 1,043 tokens for the untuned model (Table 1), with 64 reserved
  tokens. That is the baseline a growing vocabulary has to justify itself
  against.
- **Span choice decides accuracy.** Selective Latent Thinking and ASTC both
  protect hard-to-predict content and compress only what the model finds
  predictable. *Inferred:* the mint decision should be gated the same way,
  and step 1's per-chunk results should be read by span type.

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
