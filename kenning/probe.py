"""The capacity probe: how much text can one input vector stand in for?

The model is frozen throughout. For each chunk of text we optimise a small
set of input vectors (a candidate "kenning") by gradient descent and measure
what the model can do when it reads the kenning instead of the chunk.

Two modes answer two different questions:

reconstruct
    Can the model reproduce the chunk, token for token, from the kenning?
    This is the storage ceiling. It follows the per-sample optimisation idea
    of Kuratov et al. 2025 ("Cramming 1568 tokens into a single vector").

continue
    Does the kenning make the model predict what comes *after* the chunk the
    way the full chunk would? This is the question that matters for Kenning:
    storage is useless if later reasoning cannot use it. The kenning is
    trained on continuations sampled from the model itself and scored on the
    real continuation, which it never saw, so memorising cannot inflate the
    number.

Both give upper bounds: a learned writer network (step 2 of the plan) has to
produce its vector in one forward pass and will do worse than a vector that
was optimised for hundreds of steps on one sample.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models import LM

LN2 = math.log(2.0)


@dataclass
class ProbeConfig:
    n_vectors: int = 1  # vectors per kenning
    steps: int = 1000  # maximum optimisation steps per batch
    lr: float = 0.01
    init: str = "mean"  # "mean" of the chunk's token embeddings, or "random"
    seed: int = 0
    # reconstruct mode
    acc_target: float = 1.0  # stop a batch once every sample reaches this accuracy
    plateau_steps: int = 300  # stop if nothing improved for this many steps (0 = never)
    # continue mode
    cont_len: int = 64  # tokens of continuation scored after the chunk
    n_train: int = 8  # sampled continuations the kenning is trained on
    n_val: int = 4  # sampled continuations used only to pick the best step
    eval_every: int = 25
    patience: int = 8  # evaluations without improvement before stopping
    positions: str = "renumber"  # "renumber" or "gap", see _position_ids


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def _logits(lm: LM, embeds: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
    kwargs = {"inputs_embeds": embeds}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return lm.model(**kwargs).logits


def _start_ids(lm: LM, batch: int) -> torch.Tensor:
    return torch.full((batch, 1), lm.start_id, device=lm.device, dtype=torch.long)


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy per sample in nats. logits [B, T, V], targets [B, T]."""
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(1)


def _require_finite(value: torch.Tensor, what: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{what} is not finite. Try --dtype float32 or a lower --lr; results so far are saved."
        )


def init_vectors(lm: LM, tok_emb: torch.Tensor, n: int, how: str, gen: torch.Generator) -> torch.Tensor:
    """Starting point for the kenning vectors, shape [B, n, d], float32.

    "mean": split the chunk into n consecutive segments and average each
    segment's token embeddings, rescaled to the length of a typical token
    embedding. The step-0 score of this start is reported as the
    "mean of embeddings" baseline: it is what you get by simply adding the
    chunk's tokens together.
    "random": Gaussian noise with the spread of the embedding table.
    """
    emb = tok_emb.float()
    B, k, d = emb.shape
    if how == "mean":
        segments = torch.tensor_split(emb, n, dim=1) if k >= n else [emb] * n
        v = torch.stack([s.mean(1) for s in segments], dim=1)
        typical = emb.norm(dim=-1).mean(1)
        v = v * (typical[:, None, None] / v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    elif how == "random":
        std = lm.embeddings.weight.float().std()
        v = torch.randn(B, n, d, generator=gen).to(lm.device) * std
    else:
        raise ValueError(f"unknown init {how!r}")
    return v.detach().clone()


def _norm_ratio(vec: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
    """Length of the kenning vectors relative to a typical token embedding in the chunk."""
    return vec.float().norm(dim=-1).mean(1) / tok_emb.float().norm(dim=-1).mean(1).clamp_min(1e-8)


# --------------------------------------------------------------------------
# reconstruct mode
# --------------------------------------------------------------------------


def reconstruct_logits(lm: LM, vec: torch.Tensor, tok_emb: torch.Tensor) -> torch.Tensor:
    """Logits that predict each chunk token when the model reads [start, kenning, chunk].

    The sequence is  start | v_1 .. v_n | t_1 .. t_{k-1}.  The output at the
    last kenning vector predicts t_1 and the output at t_j predicts t_{j+1},
    so the k predictions sit at indices n .. n+k-1.
    """
    B, k, _ = tok_emb.shape
    n = vec.shape[1]
    start = lm.embeddings(_start_ids(lm, B))
    x = torch.cat([start, vec.to(lm.dtype), tok_emb[:, :-1]], dim=1)
    return _logits(lm, x)[:, n : n + k].float()


def probe_reconstruct(lm: LM, ctx: torch.Tensor, cfg: ProbeConfig):
    """Optimise one kenning per chunk so the frozen model reproduces the chunk.

    ctx: [B, k] token ids. Returns (rows, vectors) where rows is one dict per
    sample and vectors is the best kenning found for each, [B, n, d] on CPU.

    Teacher-forced accuracy of 1.0 means greedy decoding from the kenning
    reproduces the chunk exactly, because every greedy step then sees the
    same prefix it was scored on.
    """
    t0 = time.time()
    ctx = ctx.to(lm.device)
    B, k = ctx.shape
    n = cfg.n_vectors
    gen = torch.Generator().manual_seed(cfg.seed)

    with torch.no_grad():
        tok_emb = lm.embeddings(ctx)
        start = lm.embeddings(_start_ids(lm, B))
        # Baseline: how well the model predicts the chunk with no kenning at all.
        ce_base = _cross_entropy(_logits(lm, torch.cat([start, tok_emb[:, :-1]], dim=1)).float(), ctx)

    vec = torch.nn.Parameter(init_vectors(lm, tok_emb, n, cfg.init, gen))
    opt = torch.optim.Adam([vec], lr=cfg.lr, betas=(0.9, 0.9))

    best_acc = torch.full((B,), -1.0, device=lm.device)
    best_ce = torch.full((B,), float("inf"), device=lm.device)
    best_vec = vec.detach().clone()
    first_hit = torch.full((B,), -1, dtype=torch.long, device=lm.device)
    acc_init = ce_init = None
    last_improvement = 0
    step = 0

    for step in range(cfg.steps + 1):
        logits = reconstruct_logits(lm, vec, tok_emb)
        ce = _cross_entropy(logits, ctx)
        _require_finite(ce, "reconstruction loss")
        with torch.no_grad():
            acc = (logits.argmax(-1) == ctx).float().mean(1)
            if step == 0:
                acc_init, ce_init = acc.clone(), ce.detach().clone()
            better = (acc > best_acc) | ((acc == best_acc) & (ce < best_ce * (1 - 1e-3)))
            if better.any():
                last_improvement = step
                best_acc = torch.where(better, acc, best_acc)
                best_ce = torch.where(better, ce.detach(), best_ce)
                best_vec[better] = vec.detach()[better]
            newly = (acc >= cfg.acc_target) & (first_hit < 0)
            first_hit[newly] = step
        done = bool((acc >= cfg.acc_target).all())
        stalled = cfg.plateau_steps > 0 and step - last_improvement >= cfg.plateau_steps
        if done or stalled or step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        ce.sum().backward()
        opt.step()

    ratio = _norm_ratio(best_vec, tok_emb)
    seconds = (time.time() - t0) / B
    rows = []
    for b in range(B):
        rows.append(
            {
                "mode": "reconstruct",
                "k": k,
                "n_vectors": n,
                "accuracy": best_acc[b].item(),
                "lossless": bool(best_acc[b].item() >= 1.0),
                "ce": best_ce[b].item(),
                "ce_no_kenning": ce_base[b].item(),
                # Total information the kenning supplied about the chunk, in bits.
                "bits_gained": (ce_base[b].item() - best_ce[b].item()) * k / LN2,
                "accuracy_init": acc_init[b].item(),
                "ce_init": ce_init[b].item(),
                "steps_run": step,
                "steps_to_target": int(first_hit[b].item()),
                "vector_norm_ratio": ratio[b].item(),
                "seconds": seconds,
            }
        )
    return rows, best_vec.float().cpu()


# --------------------------------------------------------------------------
# continue mode
# --------------------------------------------------------------------------


@torch.no_grad()
def sample_continuations(lm: LM, prefix: torch.Tensor, m: int, gen: torch.Generator):
    """Sample m tokens after each prefix from the model's own distribution.

    Plain ancestral sampling at temperature 1 with no truncation, so the
    samples follow the teacher distribution exactly. Returns the sampled
    tokens [N, m] and the teacher's log-probabilities at every sampled
    position [N, m, V], which become the training targets.
    """
    out = lm.model(input_ids=prefix, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1].float()
    tokens, logps = [], []
    for t in range(m):
        logp = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(logp.exp().cpu(), 1, generator=gen).to(lm.device)
        tokens.append(nxt)
        logps.append(logp)
        if t < m - 1:
            out = lm.model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1].float()
    return torch.cat(tokens, dim=1), torch.stack(logps, dim=1)


def _position_ids(lm: LM, mode: str, k: int, n: int, m: int, batch: int) -> torch.Tensor | None:
    """Positions for the sequence  start | v_1 .. v_n | x_1 .. x_{m-1}.

    "renumber": 0, 1, 2, ... as if the chunk had never existed.
    "gap": the kenning takes the last n positions of the chunk it replaces and
        the continuation keeps the positions it had with the full chunk in
        place, leaving a hole where the rest of the chunk was.
    """
    if mode == "renumber":
        return None
    if mode != "gap":
        raise ValueError(f"unknown positions mode {mode!r}")
    if not lm.accepts_position_ids:
        raise ValueError(f"{type(lm.model).__name__}.forward takes no position_ids; use --positions renumber")
    if n > k:
        raise ValueError("gap positions need n_vectors <= chunk length")
    pos = [0] + list(range(k - n + 1, k + 1)) + list(range(k + 1, k + m))
    return torch.tensor(pos, device=lm.device).unsqueeze(0).expand(batch, -1)


def continue_logq(lm: LM, vec: torch.Tensor, conts: torch.Tensor, k: int, positions: str) -> torch.Tensor:
    """Student log-probabilities over each continuation token given only the kenning.

    vec [B, n, d]; conts [B, R, m] (R continuations per chunk). The output at
    the last kenning vector predicts x_1, so the m predictions sit at indices
    n .. n+m-1. Returns [B*R, m, V].
    """
    B, R, m = conts.shape
    n = vec.shape[1]
    start = lm.embeddings(_start_ids(lm, B * R))
    cont_emb = lm.embeddings(conts.reshape(B * R, m)[:, :-1])
    x = torch.cat([start, vec.to(lm.dtype).repeat_interleave(R, dim=0), cont_emb], dim=1)
    logits = _logits(lm, x, _position_ids(lm, positions, k, n, m, B * R))[:, n : n + m].float()
    return F.log_softmax(logits, dim=-1)


def _kl(teacher_logp: torch.Tensor, student_logq: torch.Tensor, B: int) -> torch.Tensor:
    """KL(teacher || student) in nats per token, averaged per chunk. Inputs [B*R, m, V]."""
    p = teacher_logp.exp()
    kl = (p * (teacher_logp.clamp_min(-1e4) - student_logq)).sum(-1)
    return kl.view(B, -1).mean(1)


def _token_ce(logq: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the actual tokens, nats per token per sample. logq [B, m, V], tokens [B, m]."""
    return -logq.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).mean(1)


def probe_continue(lm: LM, ctx: torch.Tensor, cont: torch.Tensor, cfg: ProbeConfig):
    """Optimise one kenning per chunk so the model continues as if it had read the chunk.

    ctx [B, k] is the chunk, cont [B, m] the real text that follows it.

    Teacher: the frozen model reading  start | chunk | continuation.
    Student: the same model reading    start | kenning | continuation.
    The kenning is trained to minimise KL(teacher || student) on continuations
    sampled from the teacher, the best step is chosen on further sampled
    continuations, and the reported score is on the real continuation.

    Headline number per sample:
        recovered = 1 - KL(teacher || student) / KL(teacher || no context)
    1.0 means the kenning replaces the chunk perfectly for predicting what
    follows; 0.0 means it is no better than deleting the chunk.
    """
    t0 = time.time()
    ctx, cont = ctx.to(lm.device), cont.to(lm.device)
    B, k = ctx.shape
    m = cont.shape[1]
    n, R_train, R_val = cfg.n_vectors, cfg.n_train, cfg.n_val
    if R_val < 1 or R_train < 1:
        raise ValueError("continue mode needs n_train >= 1 and n_val >= 1")
    R = R_train + R_val
    gen = torch.Generator().manual_seed(cfg.seed)

    with torch.no_grad():
        tok_emb = lm.embeddings(ctx)
        start_ids = _start_ids(lm, B)
        prefix = torch.cat([start_ids, ctx], dim=1)

        # Teacher targets on sampled continuations.
        sampled, sampled_logp = sample_continuations(lm, prefix.repeat_interleave(R, dim=0), m, gen)
        V = sampled_logp.shape[-1]
        sampled = sampled.view(B, R, m)
        sampled_logp = sampled_logp.view(B, R, m, V)
        x_train, x_val = sampled[:, :R_train], sampled[:, R_train:]
        lp_train = sampled_logp[:, :R_train].reshape(B * R_train, m, V)
        lp_val = sampled_logp[:, R_train:].reshape(B * R_val, m, V)
        del sampled_logp

        # Teacher on the real continuation. The last chunk token is at index k
        # (start is index 0) and predicts the first continuation token.
        full_ids = torch.cat([prefix, cont[:, :-1]], dim=1)
        lp_true = F.log_softmax(lm.model(input_ids=full_ids).logits[:, k : k + m].float(), dim=-1)
        ce_full = _token_ce(lp_true, cont)

        # No-context baseline: the model sees the continuation with the chunk deleted.
        def no_context(conts: torch.Tensor) -> torch.Tensor:
            flat = conts.reshape(-1, m)
            ids = torch.cat([_start_ids(lm, flat.shape[0]), flat[:, :-1]], dim=1)
            return F.log_softmax(lm.model(input_ids=ids).logits.float(), dim=-1)

        none_true = no_context(cont)
        kl_none = _kl(lp_true, none_true, B)
        ce_none = _token_ce(none_true, cont)
        kl_none_val = _kl(lp_val, no_context(x_val), B)
        del none_true

    vec = torch.nn.Parameter(init_vectors(lm, tok_emb, n, cfg.init, gen))
    opt = torch.optim.Adam([vec], lr=cfg.lr, betas=(0.9, 0.9))

    inf = torch.full((B,), float("inf"), device=lm.device)
    best = {"kl_val": inf.clone(), "kl": inf.clone(), "ce": inf.clone()}
    best_step = torch.zeros(B, dtype=torch.long, device=lm.device)
    best_vec = vec.detach().clone()
    init = {}
    kl_train = inf.clone()
    bad_evals = 0
    step = 0

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                kl_val = _kl(lp_val, continue_logq(lm, vec, x_val, k, cfg.positions), B)
                _require_finite(kl_val, "validation KL")
                logq_true = continue_logq(lm, vec, cont.unsqueeze(1), k, cfg.positions)
                kl_true = _kl(lp_true, logq_true, B)
                ce_true = _token_ce(logq_true, cont)
                del logq_true
                if step == 0:
                    init = {"kl": kl_true.clone(), "ce": ce_true.clone(), "kl_val": kl_val.clone()}
                improved = kl_val < best["kl_val"] * (1 - 1e-3)
                if improved.any():
                    bad_evals = 0
                    best["kl_val"] = torch.where(improved, kl_val, best["kl_val"])
                    best["kl"] = torch.where(improved, kl_true, best["kl"])
                    best["ce"] = torch.where(improved, ce_true, best["ce"])
                    best_step[improved] = step
                    best_vec[improved] = vec.detach()[improved]
                else:
                    bad_evals += 1
            if bad_evals >= cfg.patience or step == cfg.steps:
                break
        kl_train = _kl(lp_train, continue_logq(lm, vec, x_train, k, cfg.positions), B)
        opt.zero_grad(set_to_none=True)
        kl_train.sum().backward()
        opt.step()
        kl_train = kl_train.detach()

    def recovered(kl: torch.Tensor, none: torch.Tensor) -> torch.Tensor:
        return 1.0 - kl / none.clamp_min(1e-8)

    ratio = _norm_ratio(best_vec, tok_emb)
    seconds = (time.time() - t0) / B
    rows = []
    for b in range(B):
        gain_possible = ce_none[b].item() - ce_full[b].item()
        rows.append(
            {
                "mode": "continue",
                "k": k,
                "cont_len": m,
                "n_vectors": n,
                "positions": cfg.positions,
                # Held-out real continuation, at the step with the best validation KL.
                "kl_kenning": best["kl"][b].item(),
                "kl_no_context": kl_none[b].item(),
                "recovered": recovered(best["kl"], kl_none)[b].item(),
                "kl_init": init["kl"][b].item(),
                "recovered_init": recovered(init["kl"], kl_none)[b].item(),
                "ce_full_context": ce_full[b].item(),
                "ce_kenning": best["ce"][b].item(),
                "ce_no_context": ce_none[b].item(),
                "ce_gain_possible": gain_possible,
                "ce_gain_kenning": ce_none[b].item() - best["ce"][b].item(),
                # Sampled validation continuations (more tokens, so less noisy, but used to pick the step).
                "kl_val": best["kl_val"][b].item(),
                "kl_val_init": init["kl_val"][b].item(),
                "kl_val_no_context": kl_none_val[b].item(),
                "recovered_val": recovered(best["kl_val"], kl_none_val)[b].item(),
                "kl_train_last": kl_train[b].item(),
                "best_step": int(best_step[b].item()),
                "steps_run": step,
                "vector_norm_ratio": ratio[b].item(),
                "seconds": seconds,
            }
        )
    return rows, best_vec.float().cpu()
