"""Check the predictions of the kenning theory against saved probe vectors.

The theory ("Kenning Mathematics", section 7) makes testable predictions about
kennings found by the capacity probe. This module turns each one into a
measurement with a confirm / refute rule. The rules live in ``THRESHOLDS`` and
were fixed before any Granite result existed; ``thresholds_digest()`` is
printed in every report so a reader can check they did not move afterwards.

Vocabulary used below (all from the theory notes):

L_s(z)
    The probe's continue loss: KL(teacher || student) in nats per token,
    averaged over continuations sampled from the teacher. Teacher = the frozen
    model reading ``start | chunk``; student = the same model reading
    ``start | z``.
G_s(z) ("G-hat")
    The Gauss-Newton matrix (Fisher information) of L_s at z: the average over
    sampled contexts of J^T S(p) J, where J is the Jacobian of the student's
    logits with respect to z and S(p) = diag(p) - p p^T. It is never formed:
    every use goes through G_s(z) @ delta, computed as one Jacobian-vector
    product (JVP, forward mode), a cheap softmax step and one
    vector-Jacobian product (VJP, ordinary backprop).
f_u'(0)
    The slope of the loss in rho = 1/||z|| at rho = 0, for a fixed direction
    u. Computed exactly by scaling the kenning position's sublayer outputs by
    sigma (see ``_sublayer_scale``) and differentiating at sigma = 0, which is
    the "pinned model" of the theory. Never emulated with a huge norm.

Verdicts: CONFIRMED, REFUTED, INCONCLUSIVE (between the two thresholds, or too
few spans), NOT_TESTABLE (step 1 does not produce the data the claim needs)
and NOT_RUN (skipped on purpose; the reason is given).
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import data
from .models import LM, BYTE_TOKENIZER_MARKER, TINY_RANDOM
from .probe import _position_ids, _start_ids, continue_logits, sample_continuations

CONFIRMED, REFUTED, INCONCLUSIVE = "CONFIRMED", "REFUTED", "INCONCLUSIVE"
NOT_TESTABLE, NOT_RUN = "NOT_TESTABLE", "NOT_RUN"

# --------------------------------------------------------------------------
# Thresholds. Fixed on 16 November 2026, before any Granite result existed.
# Changing a value changes thresholds_digest(); say why in the commit.
# --------------------------------------------------------------------------

THRESHOLDS: dict = {
    # Fewer spans than this in a test gives INCONCLUSIVE, whatever the numbers.
    "min_spans": 6,
    # P1a radial profile at fixed direction: the loss is flat (changes by at
    # most this share of its total range) below 0.5 x the smallest and above
    # 2 x the largest context update norm ||S_l||.
    "p1a_flat_share_confirm": 0.10,
    "p1a_flat_share_refute": 0.30,
    # P1b: the sign of f'(0) at the pinned optimum predicts whether the best
    # norm is finite (f' < 0) or infinite (f' >= 0). Share of spans agreeing.
    "p1b_agree_confirm": 0.75,
    "p1b_agree_refute": 0.50,
    "p1b_interior_margin": 0.02,  # finite r must beat the pinned loss by this relative margin
    # P2 plain gradient descent from the saved kenning: where f'(0) > 0 the
    # norm grows (share of those spans) ...
    "p2a_grow_confirm": 0.75,
    "p2a_grow_refute": 0.50,
    # ... and, where f'(0) > 0, r^3 grows linearly at slope 3 * lr * f'(0):
    # share of those spans with measured / predicted slope in [1/2, 2].
    "p2b_ratio_band": [0.5, 2.0],
    "p2b_share_confirm": 0.60,
    "p2b_share_refute": 0.30,
    # P4a fast spectral decay of G_s(z*): effective rank tr(G) / lambda_max as
    # a share of the number of free dimensions n * d.
    "p4a_eff_rank_share_confirm": 0.10,
    "p4a_eff_rank_share_refute": 0.30,
    # P4b substituting the Euclidean nearest token raises L by at least this
    # many times as much as substituting the first token of the G-Lasso path
    # (median over spans).
    "p4b_ratio_confirm": 2.0,
    "p4b_ratio_refute": 1.0,
    # P4c cosine distance tracks the behavioural distance D better than
    # Euclidean distance (Spearman correlation gap at least this) and only
    # moderately (cosine correlation below the cap).
    "p4c_gap_confirm": 0.05,
    "p4c_cos_cap": 0.90,
    # P5a margin of the mean writer over the Jensen-Shannon bound grows with
    # order sensitivity: Spearman(margin, JSD).
    "p5a_rho_confirm": 0.30,
    "p5a_rho_refute": 0.0,
    # P5b centred-and-rescaled mean beats the raw mean and loses to deletion:
    # share of spans with each ordering.
    "p5b_share_confirm": 0.70,
    "p5b_share_refute": 0.30,
    # P6a epsilon_1 against chunk length k, slope of log-log fit.
    "p6a_natural_slope_confirm": 0.80,  # natural text: sub-linear
    "p6a_natural_slope_refute": 1.00,
    "p6a_random_slope_band": [0.80, 1.25],  # random tokens: near-linear
    # P6b epsilon_n against n: early drop per vector / late drop per vector.
    "p6b_knee_confirm": 3.0,  # natural text has a knee
    "p6b_knee_refute": 1.5,
    "p6b_linear_band": [0.5, 2.0],  # random tokens are near-linear
    # P6c position-preserving (gap) ids give lower epsilon than renumbered.
    "p6c_share_confirm": 0.80,
    "p6c_share_refute": 0.50,
    # P7 one Gauss-Newton step from the mean-of-embeddings start: median share
    # of the gap to the optimised kenning that it recovers, and median error
    # of its predicted loss relative to the starting loss.
    "p7_recovered_confirm": 0.50,
    "p7_recovered_refute": 0.20,
    "p7_pred_err_confirm": 0.20,
    # P8 emission residual Var(rho)(1 - R^2): multi-token words over open
    # spans of the same length (ratio of medians) ...
    "p8a_ratio_confirm": 0.25,
    "p8a_ratio_refute": 0.75,
    # ... and its growth with span length is at most quadratic (log-log slope).
    "p8b_slope_confirm": 2.0,
    "p8b_slope_refute": 2.5,
    # P9 (partial) writer outputs must be free to scale: optimised kennings sit
    # far above token norms (median norm ratio).
    "p9_norm_ratio_confirm": 3.0,
    "p9_norm_ratio_refute": 1.5,
}


def thresholds_digest() -> str:
    """First 12 hex digits of the SHA-256 of the thresholds, for reports."""
    blob = json.dumps(THRESHOLDS, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Settings:
    """Analysis sizes. Defaults suit Granite on one H200; the dry run shrinks them."""

    teacher_samples: int = 8  # continuations sampled from the teacher per span
    max_spans: int = 8  # spans per (k, positions) group in the per-span tests
    eval_batch: int = 4  # candidate vectors per forward pass
    lanczos_steps: int = 30
    trace_probes: int = 8
    cg_steps: int = 40
    sweep_points: int = 17
    sphere_steps: int = 100  # P1b: Adam steps on the direction at each fixed norm
    gd_steps: int = 200  # P2: plain gradient descent steps
    pair_spans: int = 8  # P4c: kennings per group compared pairwise
    emission_contexts: int = 400
    emission_context_len: int = 32
    emission_spans: int = 12  # per category and length
    seed: int = 0
    skip: tuple = ()
    cont_len: int = 64  # read from the run's config by analyse()


# --------------------------------------------------------------------------
# spans, behaviours and the loss
# --------------------------------------------------------------------------


@dataclass
class Span:
    chunk: torch.Tensor  # [k] token ids
    z: torch.Tensor  # [n, d] optimised kenning, float32
    k: int
    n: int
    positions: str
    sample: int
    row: dict = field(default_factory=dict)  # the matching results.jsonl row, if any

    @property
    def key(self) -> str:
        return f"k{self.k}-n{self.n}-{self.positions}-{self.sample}"


@dataclass
class Behaviour:
    """The span's own behaviour: continuations sampled from the teacher."""

    conts: torch.Tensor  # [R, m]
    logp: torch.Tensor  # [R, m, V] teacher log-probabilities on those continuations


def load_spans(run_dir: str) -> list[Span]:
    """Continue-mode kennings saved by ``kenning probe --save-vectors``."""
    rows = {}
    path = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[(r["mode"], r["k"], r["n_vectors"], r.get("positions", "-"), r["sample"])] = r
    spans = []
    for f in sorted(glob.glob(os.path.join(run_dir, "vectors-continue-*.pt"))):
        blob = torch.load(f, map_location="cpu")
        m = re.match(r"vectors-continue-k(\d+)-n(\d+)-(\w+)-\d+\.pt", os.path.basename(f))
        k, n, positions = int(m.group(1)), int(m.group(2)), m.group(3)
        for j, sample in enumerate(blob["samples"]):
            spans.append(
                Span(
                    chunk=blob["chunks"][j].long(),
                    z=blob["vectors"][j].float(),
                    k=k,
                    n=n,
                    positions=positions,
                    sample=int(sample),
                    row=rows.get(("continue", k, n, positions, int(sample)), {}),
                )
            )
    return spans


def run_args(run_dir: str) -> dict:
    """Arguments of the most recent probe call in the run."""
    configs = sorted(glob.glob(os.path.join(run_dir, "config-*.json")))
    if not configs:
        return {}
    return json.load(open(configs[-1])).get("args", {})


@torch.no_grad()
def teacher_behaviour(lm: LM, chunk: torch.Tensor, R: int, m: int, gen: torch.Generator) -> Behaviour:
    prefix = torch.cat([_start_ids(lm, 1)[0], chunk.to(lm.device)]).unsqueeze(0).repeat(R, 1)
    conts, logp = sample_continuations(lm, prefix, m, gen)
    return Behaviour(conts=conts, logp=logp)


def _logits_for(lm: LM, span: Span, Z: torch.Tensor, beh: Behaviour) -> torch.Tensor:
    """Student logits for P candidate kennings Z [P, n', d] on the behaviour's contexts: [P*R, m, V]."""
    P = Z.shape[0]
    conts = beh.conts.unsqueeze(0).expand(P, -1, -1)
    return continue_logits(lm, Z, conts, span.k, span.positions)


def _kl_rows(teacher_logp: torch.Tensor, logits: torch.Tensor, P: int) -> torch.Tensor:
    """Per-candidate KL(teacher || student), nats per token. teacher [R, m, V], logits [P*R, m, V]."""
    logq = F.log_softmax(logits, dim=-1)
    tl = teacher_logp.repeat(P, 1, 1)
    kl = (tl.exp() * (tl.clamp_min(-1e4) - logq)).sum(-1)
    return kl.view(P, -1).mean(1)


def loss_fn(lm: LM, span: Span, beh: Behaviour, Z: torch.Tensor) -> torch.Tensor:
    """L_s for each candidate in Z [P, n', d], differentiable. Returns [P]."""
    return _kl_rows(beh.logp, _logits_for(lm, span, Z.to(lm.device), beh), Z.shape[0])


@torch.no_grad()
def losses(lm: LM, span: Span, beh: Behaviour, Z: torch.Tensor, batch: int) -> torch.Tensor:
    return torch.cat([loss_fn(lm, span, beh, Z[i : i + batch]) for i in range(0, len(Z), batch)]).cpu()


def grad_fn(lm: LM, span: Span, beh: Behaviour, z: torch.Tensor) -> tuple[float, torch.Tensor]:
    zz = z.to(lm.device).clone().requires_grad_(True)
    L = loss_fn(lm, span, beh, zz.unsqueeze(0))[0]
    (g,) = torch.autograd.grad(L, zz)
    return L.item(), g.detach()


# --------------------------------------------------------------------------
# Jacobian-vector products and the Gauss-Newton product
# --------------------------------------------------------------------------


def use_eager_attention(lm: LM) -> None:
    """Fused attention kernels have no forward-mode derivative; plain attention does."""
    setter = getattr(lm.model, "set_attn_implementation", None)
    if setter is not None:
        try:
            setter("eager")
            return
        except Exception:  # pragma: no cover - depends on the model class
            pass
    if hasattr(lm.model.config, "_attn_implementation"):
        lm.model.config._attn_implementation = "eager"


def jvp(fn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """J v for J = d fn / d x, without forming J.

    Forward mode first. Some kernels (fused attention, some state-space
    layers) have no forward-mode rule; then fall back to the double-backward
    identity J v = d/dw <J^T w, v>, which costs two backward passes.
    """
    try:
        return torch.func.jvp(fn, (x,), (v,))[1]
    except (NotImplementedError, RuntimeError):
        return jvp_double_backward(fn, x, v)


def jvp_double_backward(fn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """J v through two reverse-mode passes: J v = d/dw <J^T w, v> at any w."""
    xx = x.detach().clone().requires_grad_(True)
    y = fn(xx)
    w = torch.zeros_like(y, requires_grad=True)
    (g,) = torch.autograd.grad((y * w).sum(), xx, create_graph=True)
    (jv,) = torch.autograd.grad((g * v).sum(), w)
    return jv


def gn_product(lm: LM, span: Span, beh: Behaviour, z: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """G_s(z) @ delta for each delta in deltas [Q, n, d]. Returns [Q, n, d].

    JVP w = J delta  ->  S(p) w = p*w - p<p,w>  ->  VJP J^T (S w), averaged
    over the R*m sampled positions so that G_s is the Gauss-Newton Hessian of
    the per-token loss.
    """
    Q = deltas.shape[0]
    z = z.to(lm.device)
    deltas = deltas.to(lm.device)
    Zq = z.unsqueeze(0).expand(Q, -1, -1).contiguous()
    fn = lambda Z: _logits_for(lm, span, Z, beh)  # noqa: E731
    eta = jvp(fn, Zq, deltas)
    with torch.no_grad():
        p = F.softmax(fn(Zq), dim=-1)
        s_eta = p * eta - p * (p * eta).sum(-1, keepdim=True)
    Zg = Zq.clone().requires_grad_(True)
    (out,) = torch.autograd.grad((fn(Zg) * s_eta).sum(), Zg)
    R, m = beh.conts.shape
    return out.detach() / (R * m)


def gn_operator(lm, span, beh, z, batch):
    def op(V: torch.Tensor) -> torch.Tensor:  # V [Q, n*d] flat
        shape = (V.shape[0],) + tuple(z.shape)
        outs = [gn_product(lm, span, beh, z, V[i : i + batch].view(-1, *z.shape)) for i in range(0, len(V), batch)]
        return torch.cat(outs).view(V.shape[0], -1)

    return op


def lanczos_top(op, dim: int, steps: int, gen: torch.Generator, device) -> torch.Tensor:
    """Ritz values of a symmetric operator by Lanczos with full reorthogonalisation, descending."""
    steps = min(steps, dim)
    q = torch.randn(dim, generator=gen).to(device)
    q = q / q.norm()
    Qs, alphas, betas = [q], [], []
    for i in range(steps):
        w = op(Qs[-1].unsqueeze(0))[0]
        a = torch.dot(w, Qs[-1])
        alphas.append(a)
        basis = torch.stack(Qs)
        w = w - basis.T @ (basis @ w)
        w = w - basis.T @ (basis @ w)
        b = w.norm()
        if i == steps - 1 or b < 1e-10 * max(1.0, abs(a.item())):
            break
        betas.append(b)
        Qs.append(w / b)
    T = torch.diag(torch.stack(alphas))
    for i, b in enumerate(betas[: len(alphas) - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    return torch.linalg.eigvalsh(T.double().cpu()).flip(0).float()


def gn_trace(lm: LM, span: Span, beh: Behaviour, z: torch.Tensor, probes: int, gen: torch.Generator) -> float:
    """tr G_s(z), unbiased, from sampled score vectors (theory section 6).

    Draw c ~ p independently at every position and take g = J^T (onehot(c) - p).
    Cross-position terms vanish in expectation, so E||g||^2 = sum_j tr(J_j^T S J_j).
    One VJP per probe.
    """
    total = 0.0
    R, m = beh.conts.shape
    for _ in range(probes):
        zz = z.to(lm.device).clone().unsqueeze(0).requires_grad_(True)
        logits = _logits_for(lm, span, zz, beh)
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)
            draws = torch.multinomial(p.reshape(-1, p.shape[-1]).cpu(), 1, generator=gen).view(p.shape[:-1])
            cot = F.one_hot(draws.to(p.device), p.shape[-1]).float() - p
        (g,) = torch.autograd.grad((logits * cot).sum(), zz)
        total += g.pow(2).sum().item()
    return total / probes / (R * m)


def conjugate_gradient(op, b: torch.Tensor, steps: int, tol: float = 1e-4) -> torch.Tensor:
    """Solve op(x) = b for a positive semi-definite op, starting from 0."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = torch.dot(r, r)
    b_norm = b.norm()
    for _ in range(steps):
        Ap = op(p.unsqueeze(0))[0]
        pAp = torch.dot(p, Ap)
        if pAp <= 0:
            break
        a = rr / pAp
        x = x + a * p
        r = r - a * Ap
        rr_new = torch.dot(r, r)
        if rr_new.sqrt() <= tol * b_norm:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


# --------------------------------------------------------------------------
# the norm: sublayer scaling, the pinned model and f'(0)
# --------------------------------------------------------------------------

SUBLAYER_NAMES = (
    "self_attn", "attention", "attn", "mlp", "feed_forward", "mamba", "mixer", "linear_attn",
    "block_sparse_moe", "shared_mlp", "moe",
)  # fmt: skip


def decoder_layers(model) -> list:
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"cannot find the decoder layers of {type(model).__name__}")


def sublayers(model) -> list:
    """Modules whose output is added to the residual stream, one or more per layer."""
    found = []
    for layer in decoder_layers(model):
        mods = [m for name, m in layer.named_children() if name in SUBLAYER_NAMES]
        if not mods:
            names = [name for name, _ in layer.named_children()]
            raise ValueError(f"no known sublayer in {type(layer).__name__} (children: {names})")
        found.extend(mods)
    return found


@contextlib.contextmanager
def _sublayer_scale(lm: LM, sigma: torch.Tensor, n: int):
    """Multiply every sublayer's output at the kenning positions (1..n) by sigma [B].

    With the kenning fed as R0 * u, the residual there becomes
    c R0 u + sigma S_l = sigma (c (R0 / sigma) u + S_l), and RMSNorm removes the
    overall factor, so the model computes L((R0 / sigma) u) up to the RMSNorm
    epsilon. sigma = 0 is the theory's pinned model: the kenning position
    presents sqrt(d) gamma_l * u at every sublayer.
    """

    def hook(_module, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        B, T = x.shape[0], x.shape[1]
        s = sigma.to(x.dtype).view(-1, 1, 1)
        if s.shape[0] != B:
            s = s.repeat_interleave(B // s.shape[0], dim=0)
        ones = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
        scale = torch.cat([ones, s.expand(B, n, 1), torch.ones(B, T - 1 - n, 1, device=x.device, dtype=x.dtype)], 1)
        y = x * scale
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y

    handles = [m.register_forward_hook(hook) for m in sublayers(lm.model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def scaled_loss(lm: LM, span: Span, beh: Behaviour, U: torch.Tensor, sigma: torch.Tensor, R0: float) -> torch.Tensor:
    """L_s((R0 / sigma) u) for each unit direction U [P, n, d] and sigma [P]; differentiable."""
    with _sublayer_scale(lm, sigma, span.n):
        return loss_fn(lm, span, beh, U * R0)


def fprime0(lm: LM, span: Span, beh: Behaviour, u: torch.Tensor, R0: float) -> tuple[float, float]:
    """(L at infinite norm along u, f_u'(0)), with f_u(rho) = L_s(u / rho) and u of unit norm."""
    sigma = torch.zeros(1, device=lm.device, requires_grad=True)
    L = scaled_loss(lm, span, beh, u.to(lm.device).unsqueeze(0), sigma, R0)[0]
    (g,) = torch.autograd.grad(L, sigma)
    # rho = sigma / R0, so dL/drho = R0 dL/dsigma.
    return L.item(), g.item() * R0


@torch.no_grad()
def residual_updates(lm: LM, span: Span, z: torch.Tensor) -> tuple[float, list[float]]:
    """Norm of the residual stream at the (last) kenning position entering layer 0, and ||S_l|| after each layer."""
    layers = decoder_layers(lm.model)
    seen = {}

    def pre(idx):
        def f(_m, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            seen.setdefault(("in", idx), x[0, span.n].float().clone())

        return f

    def post(idx):
        def f(_m, _a, out):
            x = out[0] if isinstance(out, tuple) else out
            seen[("out", idx)] = x[0, span.n].float().clone()

        return f

    hs = [l.register_forward_pre_hook(pre(i), with_kwargs=True) for i, l in enumerate(layers)]
    hs += [l.register_forward_hook(post(i)) for i, l in enumerate(layers)]
    try:
        x = torch.cat([lm.embeddings(_start_ids(lm, 1)), z.to(lm.device, lm.dtype).unsqueeze(0)], dim=1)
        pos = _position_ids(lm, span.positions, span.k, span.n, 1, 1)
        kwargs = {"inputs_embeds": x}
        if pos is not None:
            kwargs["position_ids"] = pos[:, : 1 + span.n]
        lm.model(**kwargs)
    finally:
        for h in hs:
            h.remove()
    x0 = seen[("in", 0)]
    S = [(seen[("out", i)] - x0).norm().item() for i in range(len(layers))]
    return x0.norm().item(), S


def _unit(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def optimise_direction(lm, span, beh, u0, R0, r, steps, lr=0.01):
    """Best loss over unit directions at a fixed norm r (r = None: infinite norm, the pinned model)."""
    w = torch.nn.Parameter(u0.clone().to(lm.device))
    opt = torch.optim.Adam([w], lr=lr)
    sigma = torch.zeros(1, device=lm.device)
    best, best_u = float("inf"), u0.clone()
    for _ in range(steps + 1):
        u = _unit(w)
        if r is None:
            L = scaled_loss(lm, span, beh, u.unsqueeze(0), sigma, R0)[0]
        else:
            L = loss_fn(lm, span, beh, (r * u).unsqueeze(0))[0]
        if L.item() < best:
            best, best_u = L.item(), u.detach().cpu().clone()
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
    return best, best_u


# --------------------------------------------------------------------------
# small statistics helpers
# --------------------------------------------------------------------------


def _median(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(xs, ys) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    rx, ry = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def linfit(xs, ys) -> tuple[float, float, float]:
    """Least-squares slope, intercept and R^2."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - icpt - slope * x) ** 2 for x, y in zip(xs, ys))
    return slope, icpt, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def verdict_high(value, confirm, refute, n=None):
    """Higher is the predicted direction."""
    if n is not None and n < THRESHOLDS["min_spans"]:
        return INCONCLUSIVE
    if value != value:
        return INCONCLUSIVE
    return CONFIRMED if value >= confirm else REFUTED if value <= refute else INCONCLUSIVE


def verdict_low(value, confirm, refute, n=None):
    """Lower is the predicted direction."""
    if n is not None and n < THRESHOLDS["min_spans"]:
        return INCONCLUSIVE
    if value != value:
        return INCONCLUSIVE
    return CONFIRMED if value <= confirm else REFUTED if value >= refute else INCONCLUSIVE


def _combine(*verdicts):
    if any(v == REFUTED for v in verdicts):
        return REFUTED
    if all(v == CONFIRMED for v in verdicts):
        return CONFIRMED
    return INCONCLUSIVE


def result(pid, claim, verdict, value=None, rule="", n=None, details=None, note=""):
    return {
        "id": pid, "claim": claim, "verdict": verdict, "value": value, "rule": rule,
        "spans": n, "details": details or {}, "note": note,
    }  # fmt: skip


# --------------------------------------------------------------------------
# a ledger, so an interrupted analysis resumes
# --------------------------------------------------------------------------


class Ledger:
    """Per-span measurements, appended to a JSONL file and reused on the next run."""

    def __init__(self, path: str):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r["test"], r["key"])] = r["value"]

    def get(self, test: str, key: str, compute):
        if (test, key) in self.rows:
            return self.rows[(test, key)]
        value = compute()
        self.rows[(test, key)] = value
        with open(self.path, "a") as f:
            f.write(json.dumps({"test": test, "key": key, "value": value}) + "\n")
        return value


# --------------------------------------------------------------------------
# per-span measurements
# --------------------------------------------------------------------------


def measure_radial(lm, span, beh, st: Settings) -> dict:
    """P1a: L along r at the saved direction, and the context update norms."""
    u = _unit(span.z)
    R0 = span.z.norm(dim=-1).mean().item()
    r_resid, S = residual_updates(lm, span, span.z)
    c = r_resid / span.z[-1].norm().item()  # any embedding multiplier the model applies
    s_lo, s_hi = min(S[1:] or S), max(S)
    lo, hi = 0.05 * s_lo / c, 20.0 * s_hi / c  # in embedding units
    rs = [lo * (hi / lo) ** (i / (st.sweep_points - 1)) for i in range(st.sweep_points)]
    Ls = losses(lm, span, beh, torch.stack([r * u for r in rs]), st.eval_batch).tolist()
    with torch.no_grad():
        Ls.append(scaled_loss(lm, span, beh, u.unsqueeze(0).to(lm.device), torch.zeros(1, device=lm.device), R0).item())
    finite = Ls[:-1]
    rng = max(Ls) - min(Ls)
    low = [L for r, L in zip(rs, finite) if r * c <= 0.5 * s_lo]
    high = [L for r, L in zip(rs, finite) if r * c >= 2.0 * s_hi] + [Ls[-1]]
    share = lambda xs: (max(xs) - min(xs)) / rng if rng > 1e-6 and len(xs) >= 2 else float("nan")  # noqa: E731
    return {
        "r": rs, "loss": finite, "loss_inf": Ls[-1], "r_saved": R0, "embed_multiplier": c,
        "S_norms": S, "range": rng, "low_flat_share": share(low), "high_flat_share": share(high),
        "low_points": len(low), "high_points": len(high),
    }  # fmt: skip


def measure_interior(lm, span, beh, st: Settings) -> dict:
    """P1b: optimise the direction at a few fixed norms and at infinity; f'(0) at the pinned optimum."""
    R0 = span.z.norm(dim=-1).mean().item()
    u0 = _unit(span.z)
    L_inf, u_inf = optimise_direction(lm, span, beh, u0, R0, None, st.sphere_steps)
    _, fp = fprime0(lm, span, beh, u_inf, R0)
    finite = {}
    for mult in (0.25, 1.0, 4.0, 16.0):
        finite[str(mult)], _ = optimise_direction(lm, span, beh, u0, R0, mult * R0, st.sphere_steps)
    best_finite = min(finite.values())
    interior = best_finite < L_inf * (1 - THRESHOLDS["p1b_interior_margin"]) - 1e-6
    return {"loss_inf_opt": L_inf, "fprime0_at_pinned_opt": fp, "loss_finite_opt": finite,
            "interior": bool(interior), "predicted_interior": bool(fp < 0),
            "agree": bool(interior == (fp < 0))}  # fmt: skip


def measure_gd(lm, span, beh, st: Settings, lam_max: float) -> dict:
    """P2: plain gradient descent from the saved kenning at a fixed step 1 / lambda_max."""
    lr = 1.0 / max(lam_max, 1e-12)
    z = span.z.clone().to(lm.device)
    norms, Ls = [], []
    for _ in range(st.gd_steps + 1):
        L, g = grad_fn(lm, span, beh, z)
        norms.append(z.norm(dim=-1).mean().item())
        Ls.append(L)
        z = z - lr * g
    u = _unit(z.detach().cpu())
    _, fp_end = fprime0(lm, span, beh, u, norms[-1])
    _, fp_start = fprime0(lm, span, beh, _unit(span.z), norms[0])
    half = len(norms) // 2
    ts = list(range(half, len(norms)))
    slope, _, r2 = linfit(ts, [norms[t] ** 3 for t in ts])
    pred = 3 * lr * fp_end
    return {
        "lr": lr, "r_start": norms[0], "r_end": norms[-1], "loss_start": Ls[0], "loss_end": Ls[-1],
        "fprime0_start": fp_start, "fprime0_end": fp_end, "r3_slope": slope, "r3_r2": r2,
        "r3_slope_predicted": pred, "slope_ratio": slope / pred if pred != 0 else float("nan"),
        "norm_grew": norms[-1] > norms[0],
    }  # fmt: skip


def measure_spectrum(lm, span, beh, st: Settings, gen) -> dict:
    """P4a: lambda_max by Lanczos and the trace by sampled scores, both through G-products only."""
    op = gn_operator(lm, span, beh, span.z, st.eval_batch)
    ritz = lanczos_top(op, span.z.numel(), st.lanczos_steps, gen, lm.device)
    tr = gn_trace(lm, span, beh, span.z, st.trace_probes, gen)
    lam = ritz[0].item()
    return {"lambda_max": lam, "trace": tr, "eff_rank": tr / lam if lam > 0 else float("nan"),
            "dim": span.z.numel(), "ritz_top": ritz[:10].tolist()}  # fmt: skip


def measure_readout(lm, span, beh, st: Settings) -> dict:
    """P4b: Euclidean nearest token against the first token of the G-Lasso path (n = 1 only)."""
    z = span.z[0].to(lm.device)
    E = lm.embeddings.weight.float()
    v_e = int(torch.cdist(z.unsqueeze(0), E).argmin())
    Gz = gn_product(lm, span, beh, span.z, span.z.unsqueeze(0))[0, 0]
    v_g = int((E @ Gz).abs().argmax())
    cands = torch.stack([E[v_e], E[v_g]]).unsqueeze(1)
    Gc = gn_product(lm, span, beh, span.z, cands)[:, 0]
    alpha = [(torch.dot(c[0], Gz) / torch.dot(c[0], g)).item() for c, g in zip(cands, Gc)]
    Z = torch.stack([span.z, E[v_e].unsqueeze(0), alpha[0] * E[v_e].unsqueeze(0), alpha[1] * E[v_g].unsqueeze(0)])
    L = losses(lm, span, beh, Z, st.eval_batch).tolist()
    d_e, d_e_scaled, d_g = L[1] - L[0], L[2] - L[0], L[3] - L[0]
    return {
        "loss_kenning": L[0], "token_euclid": v_e, "token_lasso": v_g, "alpha_lasso": alpha[1],
        "dloss_euclid": d_e, "dloss_euclid_scaled": d_e_scaled, "dloss_lasso": d_g,
        "ratio": d_e / d_g if d_g > 0 else float("nan"),
        "ratio_scaled": d_e_scaled / d_g if d_g > 0 else float("nan"),
    }  # fmt: skip


def measure_gauss_newton(lm, span, beh, st: Settings) -> dict:
    """P7: one Gauss-Newton step from the mean-of-embeddings start the probe used."""
    from .probe import init_vectors

    with torch.no_grad():
        tok = lm.embeddings(span.chunk.to(lm.device).unsqueeze(0))
        z0 = init_vectors(lm, tok, span.n, "mean", torch.Generator())[0].float()
    L0, g = grad_fn(lm, span, beh, z0)
    op = gn_operator(lm, span, beh, z0, st.eval_batch)
    b = -g.flatten()
    delta = conjugate_gradient(op, b, st.cg_steps)
    pred = L0 - 0.5 * torch.dot(b, delta).item()
    D = delta.view_as(z0)
    Z = torch.stack([z0 + t * D for t in (1.0, 0.5, 0.25, 0.125)] + [span.z.to(lm.device)])
    L = losses(lm, span, beh, Z, st.eval_batch).tolist()
    L1, Lstar = L[0], L[-1]
    gap = L0 - Lstar
    return {
        "loss_start": L0, "loss_step": L1, "loss_step_best_of_halvings": min(L[:4]), "loss_optimised": Lstar,
        "loss_predicted": pred, "recovered": (L0 - L1) / gap if gap > 1e-9 else float("nan"),
        "pred_error": abs(pred - Lstar) / max(L0, 1e-9),
    }  # fmt: skip


def deletion_loss(lm, span, beh) -> float:
    empty = torch.zeros(1, 0, span.z.shape[-1])
    return losses(lm, span, beh, empty, 1)[0].item()


def measure_shuffle(lm, span, st: Settings, gen) -> dict:
    """P5: order-blind mean writer on a span and a shuffled copy; centred mean; deletion."""
    from .probe import init_vectors

    m = int(st.cont_len)
    R = st.teacher_samples
    perm = torch.randperm(span.k, generator=gen)
    s, sp = span.chunk, span.chunk[perm]
    one = Span(chunk=s, z=span.z[:1], k=span.k, n=1, positions="renumber", sample=span.sample)
    two = Span(chunk=sp, z=span.z[:1], k=span.k, n=1, positions="renumber", sample=span.sample)
    P, Pp = teacher_behaviour(lm, s, R, m, gen), teacher_behaviour(lm, sp, R, m, gen)
    with torch.no_grad():
        tok = lm.embeddings(s.to(lm.device).unsqueeze(0)).float()
        z_mean = init_vectors(lm, tok, 1, "mean", torch.Generator())[0]
        centroid = lm.embeddings.weight.float().mean(0)
        zc = tok[0].mean(0) - centroid
        zc = (zc / zc.norm() * tok[0].norm(dim=-1).mean()).unsqueeze(0)

        def seq_logp(beh_from: Behaviour, chunk) -> torch.Tensor:
            """log P_chunk(c) per sampled continuation c of beh_from (sequence level)."""
            Rn = beh_from.conts.shape[0]
            prefix = torch.cat([_start_ids(lm, 1)[0], chunk.to(lm.device)]).unsqueeze(0).repeat(Rn, 1)
            ids = torch.cat([prefix, beh_from.conts[:, :-1]], dim=1)
            lp = F.log_softmax(lm.model(input_ids=ids).logits[:, chunk.numel() : chunk.numel() + m].float(), -1)
            return lp.gather(-1, beh_from.conts.unsqueeze(-1)).squeeze(-1).sum(1)

        # JSD(P, P') = 1/2 E_P log(P/M) + 1/2 E_P' log(P'/M), M = (P + P') / 2, per token.
        a_p, a_q = seq_logp(P, s), seq_logp(P, sp)
        b_p, b_q = seq_logp(Pp, s), seq_logp(Pp, sp)
        log_m_a = torch.logaddexp(a_p, a_q) - math.log(2)
        log_m_b = torch.logaddexp(b_p, b_q) - math.log(2)
        jsd = 0.5 * ((a_p - log_m_a).mean() + (b_q - log_m_b).mean()).item() / m
        L_s = losses(lm, one, P, z_mean.unsqueeze(0), 1)[0].item()
        L_sp = losses(lm, two, Pp, z_mean.unsqueeze(0), 1)[0].item()
        L_c = losses(lm, one, P, zc.unsqueeze(0), 1)[0].item()
    L_del = deletion_loss(lm, one, P)
    return {
        "jsd": jsd, "loss_mean": L_s, "loss_mean_shuffled": L_sp, "margin": max(L_s, L_sp) - jsd,
        "bound_holds": (L_s + L_sp) >= 2 * jsd, "loss_centred": L_c, "loss_deleted": L_del,
        "centred_beats_mean": L_c < L_s, "centred_loses_to_deletion": L_c > L_del,
        "mean_worse_than_deletion": L_s > L_del,
    }  # fmt: skip


@torch.no_grad()
def sample_student(lm, span: Span, z: torch.Tensor, R: int, m: int, gen) -> torch.Tensor:
    """R continuations of m tokens sampled from the model reading start | z (no cache; short)."""
    conts = torch.zeros(R, 1, dtype=torch.long, device=lm.device)
    for t in range(m):
        logits = continue_logits(lm, z.to(lm.device).unsqueeze(0), conts.unsqueeze(0), span.k, span.positions)
        p = F.softmax(logits[:, t].float(), dim=-1)
        nxt = torch.multinomial(p.cpu(), 1, generator=gen).to(lm.device)
        conts = torch.cat([conts[:, :t], nxt, torch.zeros_like(nxt)], dim=1)
    return conts[:, :m]


def measure_pairs(lm, spans: list[Span], st: Settings, gen) -> dict:
    """P4c: behavioural distance D between kennings of one group against cosine and Euclidean distance."""
    m = int(st.cont_len)
    behs = []
    for sp in spans:
        conts = sample_student(lm, sp, sp.z, st.teacher_samples, m, gen)
        with torch.no_grad():
            logp = F.log_softmax(continue_logits(lm, sp.z.to(lm.device).unsqueeze(0), conts.unsqueeze(0), sp.k, sp.positions), -1)
        behs.append(Behaviour(conts=conts, logp=logp))
    n = len(spans)
    Z = torch.stack([sp.z for sp in spans])
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        D[i] = losses(lm, spans[i], behs[i], Z, st.eval_batch).tolist()
    jeff, cos, euc = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            jeff.append(0.5 * (D[i][j] + D[j][i]))
            a, b = spans[i].z.flatten(), spans[j].z.flatten()
            cos.append(1 - F.cosine_similarity(a, b, dim=0).item())
            euc.append((a - b).norm().item())
    return {"jeffreys": jeff, "cosine_distance": cos, "euclidean_distance": euc,
            "spearman_cosine": spearman(cos, jeff), "spearman_euclid": spearman(euc, jeff)}  # fmt: skip


# --------------------------------------------------------------------------
# P8 emission, which needs no kenning at all
# --------------------------------------------------------------------------


def _capture_final_hidden(lm: LM):
    store = {}
    head = lm.model.get_output_embeddings()

    def hook(_m, inp):
        store["h"] = inp[0].detach()

    return store, head.register_forward_pre_hook(hook)


def emission_spans(lm: LM, streams, st: Settings, gen):
    """Multi-token words from the corpus, and open-ended windows of 2, 4, 8 and 16 tokens."""
    text = " ".join(lm.decode(s.tolist()) for s in streams)
    counts = defaultdict(int)
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text):
        counts[w] += 1
    words = {}
    for w, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        ids = lm.encode(" " + w)
        if ids and not lm.decode(ids[:1]).strip():
            ids = ids[1:]  # a lone leading space token (byte tokenizer)
        if 2 <= len(ids) <= 16:
            words.setdefault(len(ids), []).append(ids)
    word_spans = [(len(ids), ids) for k in sorted(words) for ids in words[k][: st.emission_spans]]
    open_spans = []
    long = [s for s in streams if len(s) > 64]
    for k in (2, 4, 8, 16):
        for _ in range(st.emission_spans):
            s = long[int(torch.randint(len(long), (1,), generator=gen))]
            t = int(torch.randint(len(s) - k, (1,), generator=gen))
            open_spans.append((k, s[t : t + k].tolist()))
    return word_spans, open_spans


@torch.no_grad()
def measure_emission(lm: LM, streams, st: Settings, gen) -> dict:
    """P8: residual of the best linear emission logit, Var(rho_s)(1 - R^2), held out.

    rho_s(X) = log P(s_2..s_k | X, s_1). It is regressed on the final hidden
    state h(X) by ridge regression on half the contexts and scored on the other
    half, so the residual is not flattered when there are fewer contexts than
    hidden dimensions.
    """
    Lx = st.emission_context_len
    long = [s for s in streams if len(s) > Lx + 1]
    ctxs = []
    for _ in range(st.emission_contexts):
        s = long[int(torch.randint(len(long), (1,), generator=gen))]
        t = int(torch.randint(Lx, len(s), (1,), generator=gen))
        ctxs.append(s[t - Lx : t])
    X = torch.stack(ctxs).to(lm.device)
    X = torch.cat([_start_ids(lm, len(X)), X], dim=1)
    store, handle = _capture_final_hidden(lm)
    try:
        H = []
        for i in range(0, len(X), 64):
            lm.model(input_ids=X[i : i + 64])
            H.append(store["h"][:, -1].float())
        H = torch.cat(H)
    finally:
        handle.remove()
    half = len(H) // 2
    Htr, Hte = H[:half], H[half:]
    mu = Htr.mean(0)
    A = Htr - mu
    lam = 1e-2 * A.pow(2).sum() / A.shape[1] + 1e-6
    ridge = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=A.device), A.T)

    def residual(ids):
        k = len(ids)
        s = torch.tensor(ids, device=lm.device).unsqueeze(0).expand(len(X), -1)
        rho = []
        for i in range(0, len(X), 64):
            seq = torch.cat([X[i : i + 64], s[i : i + 64]], dim=1)
            lp = F.log_softmax(lm.model(input_ids=seq).logits.float(), -1)
            T = X.shape[1]
            tail = lp[:, T : T + k - 1].gather(-1, s[i : i + 64, 1:].unsqueeze(-1)).squeeze(-1).sum(1)
            rho.append(tail)
        rho = torch.cat(rho)
        ytr, yte = rho[:half], rho[half:]
        w = ridge @ (ytr - ytr.mean())
        pred = (Hte - mu) @ w + ytr.mean()
        return {"k": k, "var": yte.var().item(), "residual": (yte - pred).pow(2).mean().item()}

    word_spans, open_spans = emission_spans(lm, streams, st, gen)
    return {
        "words": [dict(residual(ids), text=lm.decode(ids)) for _, ids in word_spans],
        "open": [residual(ids) for _, ids in open_spans],
        "contexts": len(X), "hidden": H.shape[1],
    }  # fmt: skip


# --------------------------------------------------------------------------
# results.jsonl-only tests (P6, P9)
# --------------------------------------------------------------------------


def _eps_table(rows):
    """Median epsilon (kl_val, nats per token) per (positions, n, k) from continue rows."""
    groups = defaultdict(list)
    for r in rows:
        if r["mode"] == "continue":
            groups[(r["positions"], r["n_vectors"], r["k"])].append(r["kl_val"])
    return {key: _median(v) for key, v in groups.items()}, groups


def check_p6(rows, control_rows) -> list[dict]:
    T = THRESHOLDS
    out = []
    table, groups = _eps_table(rows)
    ks = sorted({k for (pos, n, k) in table if pos == "renumber" and n == 1})

    def slope_for(tab, positions="renumber"):
        pts = [(math.log(k), math.log(tab[(positions, 1, k)])) for k in ks if tab.get((positions, 1, k), 0) > 0]
        return linfit([p[0] for p in pts], [p[1] for p in pts])[0], len(pts)

    s_nat, npts = slope_for(table)
    v = verdict_low(s_nat, T["p6a_natural_slope_confirm"], T["p6a_natural_slope_refute"]) if npts >= 3 else INCONCLUSIVE
    det = {"slope_natural": s_nat, "lengths": ks, "eps1_by_k": {k: table.get(("renumber", 1, k)) for k in ks}}
    if control_rows:
        ctab, _ = _eps_table(control_rows)
        cks = sorted({k for (pos, n, k) in ctab if pos == "renumber" and n == 1})
        pts = [(math.log(k), math.log(ctab[("renumber", 1, k)])) for k in cks if ctab[("renumber", 1, k)] > 0]
        s_rand = linfit([p[0] for p in pts], [p[1] for p in pts])[0]
        lo, hi = T["p6a_random_slope_band"]
        v_rand = (CONFIRMED if lo <= s_rand <= hi else REFUTED) if len(pts) >= 3 else INCONCLUSIVE
        det.update(slope_random=s_rand, eps1_random_by_k={k: ctab.get(("renumber", 1, k)) for k in cks})
        v = _combine(v, v_rand)
        note = ""
    else:
        note = "No random-token control run given (--control-run), so only the natural-text half is judged."
        v = INCONCLUSIVE if v == CONFIRMED else v
    out.append(result("P6a", "epsilon_1 grows sub-linearly in k for natural text and near-linearly for random tokens",
                      v, s_nat, f"log-log slope natural <= {T['p6a_natural_slope_confirm']} (refute >= "
                      f"{T['p6a_natural_slope_refute']}); random in {T['p6a_random_slope_band']}", npts, det, note))

    def knee(tab):
        ratios = {}
        for k in sorted({k for (_, _, k) in tab}):
            ns = sorted(n for (pos, n, kk) in tab if pos == "renumber" and kk == k)
            if len(ns) < 3:
                continue
            e = [tab[("renumber", n, k)] for n in ns]
            early = (e[0] - e[1]) / (ns[1] - ns[0])
            late = (e[-2] - e[-1]) / (ns[-1] - ns[-2])
            ratios[k] = early / late if late > 0 else (float("inf") if early > 0 else float("nan"))
        return ratios

    kn = knee(table)
    if kn:
        r = _median(list(kn.values()))
        v = verdict_high(r, T["p6b_knee_confirm"], T["p6b_knee_refute"])
        det = {"early_over_late_by_k": kn}
        if control_rows:
            kc = knee(_eps_table(control_rows)[0])
            if kc:
                rc = _median(list(kc.values()))
                lo, hi = T["p6b_linear_band"]
                v = _combine(v, CONFIRMED if lo <= rc <= hi else REFUTED)
                det["random_early_over_late_by_k"] = kc
        out.append(result("P6b", "epsilon_n against n has a knee for natural text and is near-linear for random tokens",
                          v, r, f"median early/late drop per vector >= {T['p6b_knee_confirm']} (refute <= "
                          f"{T['p6b_knee_refute']}); random in {T['p6b_linear_band']}", len(kn), det))
    else:
        out.append(result("P6b", "epsilon_n against n has a knee for natural text", NOT_RUN,
                          note="Needs continue runs with at least three values of --n-vectors at one length."))

    pairs = []
    by_key = defaultdict(dict)
    for r in rows:
        if r["mode"] == "continue":
            by_key[(r["k"], r["n_vectors"], r["sample"])][r["positions"]] = r["kl_val"]
    for (k, n, s), d in by_key.items():
        if "gap" in d and "renumber" in d:
            pairs.append((k, d["renumber"], d["gap"]))
    if pairs:
        share = sum(g < rn for _, rn, g in pairs) / len(pairs)
        gain = [(rn - g) / max(rn, 1e-9) for _, rn, g in pairs]
        rho_k = spearman([p[0] for p in pairs], gain)
        v = verdict_high(share, T["p6c_share_confirm"], T["p6c_share_refute"], len(pairs))
        out.append(result("P6c", "position-preserving (gap) ids lower epsilon for every span, most for long ones",
                          v, share, f"share of spans with gap < renumber >= {T['p6c_share_confirm']} (refute <= "
                          f"{T['p6c_share_refute']})", len(pairs), {"spearman_k_vs_relative_gain": rho_k}))
    else:
        out.append(result("P6c", "gap ids lower epsilon", NOT_RUN, note="Needs both --positions gap and renumber runs."))
    out.append(result("P6d", "epsilon rises with prefix diversity and horizon", NOT_TESTABLE,
                      note="Step 1 uses one fixed prefix (the start token) and one horizon per run."))
    return out


def check_p9(rows) -> dict:
    T = THRESHOLDS
    ratios = [r["vector_norm_ratio"] for r in rows if r["mode"] == "continue"]
    med = _median(ratios)
    v = verdict_high(med, T["p9_norm_ratio_confirm"], T["p9_norm_ratio_refute"], len(ratios))
    return result("P9", "writer outputs need norms far above token norms (the writer half needs step 2)", v, med,
                  f"median kenning/token norm ratio >= {T['p9_norm_ratio_confirm']} (refute <= {T['p9_norm_ratio_refute']})",
                  len(ratios), note="Only the norm sub-claim is testable before a writer exists.")


# --------------------------------------------------------------------------
# the whole analysis
# --------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        head = subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--", "."], text=True,
                                        stderr=subprocess.DEVNULL)
        return head.strip()[:12] + ("+uncommitted-changes" if dirty.strip() else "")
    except Exception:
        return None


def is_toy(name: str) -> bool:
    return name == TINY_RANDOM or os.path.exists(os.path.join(name, BYTE_TOKENIZER_MARKER))


def analyse(lm: LM, run_dir: str, st: Settings, control_dir: str | None = None, out_dir: str | None = None,
            log=print) -> dict:
    """Run every prediction check. Returns the report dict and writes predictions.{json,md}."""
    t0 = time.time()
    T = THRESHOLDS
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    ledger = Ledger(os.path.join(out_dir, "predictions-spans.jsonl"))
    use_eager_attention(lm)
    args = run_args(run_dir)
    st.cont_len = int(args.get("cont_len", 64))
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "results.jsonl")) if l.strip()]
    control_rows = None
    if control_dir:
        control_rows = [json.loads(l) for l in open(os.path.join(control_dir, "results.jsonl")) if l.strip()]
    spans = load_spans(run_dir)
    single = [s for s in spans if s.n == 1]
    groups = defaultdict(list)
    for s in single:
        if len(groups[(s.k, s.positions)]) < st.max_spans:
            groups[(s.k, s.positions)].append(s)
    chosen = [s for g in groups.values() for s in g]
    log(f"[predictions] {len(spans)} saved continue kennings, {len(chosen)} single-vector spans analysed "
        f"in {len(groups)} groups; thresholds {thresholds_digest()}")
    skip = set(st.skip)
    per = defaultdict(list)
    for i, span in enumerate(chosen):
        gen = torch.Generator().manual_seed(st.seed + 1009 * i)
        beh = teacher_behaviour(lm, span.chunk, st.teacher_samples, st.cont_len, gen)
        key = span.key
        ts = time.time()
        if "P1" not in skip:
            per["P1a"].append(ledger.get("P1a", key, lambda: measure_radial(lm, span, beh, st)))
            per["P1b"].append(ledger.get("P1b", key, lambda: measure_interior(lm, span, beh, st)))
        spec = None
        if not {"P4", "P2"} <= skip:
            spec = ledger.get("P4a", key, lambda: measure_spectrum(lm, span, beh, st, gen))
            per["P4a"].append(spec)
        if "P2" not in skip:
            per["P2"].append(ledger.get("P2", key, lambda: measure_gd(lm, span, beh, st, spec["lambda_max"])))
        if "P4" not in skip:
            per["P4b"].append(ledger.get("P4b", key, lambda: measure_readout(lm, span, beh, st)))
        if "P7" not in skip:
            per["P7"].append(ledger.get("P7", key, lambda: measure_gauss_newton(lm, span, beh, st)))
        if "P5" not in skip and span.positions == "renumber":
            per["P5"].append(ledger.get("P5", key, lambda: measure_shuffle(lm, span, st, gen)))
        log(f"[predictions] span {i + 1}/{len(chosen)} {key}: {time.time() - ts:.0f}s")
    if "P4" not in skip:
        for (k, positions), g in sorted(groups.items()):
            if len(g) >= 3:
                gen = torch.Generator().manual_seed(st.seed + 17 * k)
                gk = f"k{k}-{positions}"
                per["P4c"].append(ledger.get("P4c", gk, lambda: measure_pairs(lm, g[: st.pair_spans], st, gen)))
    emission = None
    if "P8" not in skip:
        source, path = args.get("source", "text"), args.get("data")
        streams = data.token_streams(lm, source, path, seed=st.seed)
        emission = ledger.get("P8", "all", lambda: measure_emission(lm, streams, st, torch.Generator().manual_seed(st.seed)))

    res = []
    # P1a
    if per["P1a"]:
        lows = [d["low_flat_share"] for d in per["P1a"] if d["low_points"] >= 2]
        highs = [d["high_flat_share"] for d in per["P1a"]]
        worst = max(_median(lows), _median(highs))
        v = verdict_low(worst, T["p1a_flat_share_confirm"], T["p1a_flat_share_refute"], len(per["P1a"]))
        res.append(result("P1a", "at fixed direction the loss is flat below early-layer and above late-layer residual "
                          "norms, changing in between", v, worst,
                          f"larger of the two median plateau variations, as a share of the loss range, <= "
                          f"{T['p1a_flat_share_confirm']} (refute >= {T['p1a_flat_share_refute']})", len(per["P1a"]),
                          {"median_low_share": _median(lows), "median_high_share": _median(highs),
                           "spans_with_low_points": len(lows)}))
    else:
        res.append(result("P1a", "radial profile", NOT_RUN))
    if per["P1b"]:
        agree = sum(d["agree"] for d in per["P1b"]) / len(per["P1b"])
        v = verdict_high(agree, T["p1b_agree_confirm"], T["p1b_agree_refute"], len(per["P1b"]))
        res.append(result("P1b", "the sign of f'(0) at the pinned optimum predicts whether the best norm is finite or "
                          "infinite", v, agree, f"share agreeing >= {T['p1b_agree_confirm']} (refute <= "
                          f"{T['p1b_agree_refute']})", len(per["P1b"]),
                          {"interior_share": sum(d["interior"] for d in per["P1b"]) / len(per["P1b"]),
                           "fprime_negative_share": sum(d["predicted_interior"] for d in per["P1b"]) / len(per["P1b"])},
                          f"Directions optimised for {st.sphere_steps} Adam steps per norm, so 'finite' is approximate."))
    else:
        res.append(result("P1b", "interior or infinite optimum", NOT_RUN))
    # P2
    if per["P2"]:
        up = [d for d in per["P2"] if d["fprime0_start"] > 0]
        down = [d for d in per["P2"] if d["fprime0_start"] <= 0]
        grow = sum(d["norm_grew"] for d in up) / len(up) if up else float("nan")
        va = verdict_high(grow, T["p2a_grow_confirm"], T["p2a_grow_refute"], len(up))
        res.append(result("P2a", "under plain gradient descent the norm grows when f'(0) > 0", va, grow,
                          f"share of those spans whose norm grew >= {T['p2a_grow_confirm']} (refute <= "
                          f"{T['p2a_grow_refute']})", len(up),
                          {"fprime_positive_share": len(up) / len(per["P2"]),
                           "norm_grew_share_where_fprime_negative": (sum(d["norm_grew"] for d in down) / len(down))
                           if down else float("nan")},
                          "f'(0) is taken at the saved direction. The theory says nothing about spans with f'(0) <= 0; "
                          "their growth share is shown for information."))
        pos = [d for d in per["P2"] if d["fprime0_end"] > 0]
        lo, hi = T["p2b_ratio_band"]
        share = sum(lo <= d["slope_ratio"] <= hi for d in pos) / len(pos) if pos else float("nan")
        vb = verdict_high(share, T["p2b_share_confirm"], T["p2b_share_refute"], len(pos))
        res.append(result("P2b", "where f'(0) > 0, r^3 grows linearly in steps at slope 3 x lr x f'(0) (r ~ t^(1/3))", vb,
                          share, f"share of those spans with measured/predicted slope in [{lo}, {hi}] >= "
                          f"{T['p2b_share_confirm']} (refute <= {T['p2b_share_refute']})", len(pos),
                          {"median_slope_ratio": _median([d["slope_ratio"] for d in pos]),
                           "median_r3_r2": _median([d["r3_r2"] for d in pos])}))
    else:
        res.append(result("P2a", "norm drift under gradient descent", NOT_RUN))
    res.append(result("P2c", "a spherical parametrisation at fixed large r reaches the same loss in far fewer steps",
                      NOT_RUN, note="Needs a second optimiser in the probe; deferred so step 1 runs stay comparable."))
    res.append(result("P3", "as r grows, prefix-to-first-token mutual information falls to zero and bounds L from below",
                      NOT_TESTABLE, note="Step 1 has a single fixed prefix (the start token), so I_s(a; c_1) = 0 "
                      "identically. Step 2 should save kennings read after varied prefixes."))
    # P4
    if per["P4a"]:
        shares = [d["eff_rank"] / d["dim"] for d in per["P4a"]]
        med = _median(shares)
        v = verdict_low(med, T["p4a_eff_rank_share_confirm"], T["p4a_eff_rank_share_refute"], len(shares))
        res.append(result("P4a", "the spectrum of G_s(z*) decays fast: tr(G)/lambda_max << n d", v, med,
                          f"median effective rank / (n d) <= {T['p4a_eff_rank_share_confirm']} (refute >= "
                          f"{T['p4a_eff_rank_share_refute']})", len(shares),
                          {"median_eff_rank": _median([d["eff_rank"] for d in per["P4a"]]), "dim": per["P4a"][0]["dim"]}))
    if per["P4b"]:
        ratios = [d["ratio"] for d in per["P4b"]]
        med = _median(ratios)
        v = verdict_high(med, T["p4b_ratio_confirm"], T["p4b_ratio_refute"], len(ratios))
        res.append(result("P4b", "substituting the Euclidean nearest token raises L much more than the first G-Lasso "
                          "token", v, med, f"median dL(Euclid)/dL(Lasso) >= {T['p4b_ratio_confirm']} (refute <= "
                          f"{T['p4b_ratio_refute']})", len(ratios),
                          {"median_ratio_euclid_scaled": _median([d["ratio_scaled"] for d in per["P4b"]]),
                           "same_token_share": sum(d["token_euclid"] == d["token_lasso"] for d in per["P4b"]) / len(ratios)},
                          "Lasso token uses the span's own metric G_s(z*) and its G-optimal coefficient."))
    if per["P4c"]:
        jeff, cos, euc = [], [], []
        for d in per["P4c"]:
            jeff += d["jeffreys"]
            cos += d["cosine_distance"]
            euc += d["euclidean_distance"]
        rc, re_ = spearman(cos, jeff), spearman(euc, jeff)
        gap = rc - re_
        if len(jeff) < T["min_spans"] or gap != gap:
            v = INCONCLUSIVE
        elif gap <= 0:
            v = REFUTED
        elif gap >= T["p4c_gap_confirm"] and rc < T["p4c_cos_cap"]:
            v = CONFIRMED
        else:
            v = INCONCLUSIVE
        res.append(result("P4c", "cosine distance tracks behavioural distance better than Euclidean, only moderately well",
                          v, gap, f"Spearman(cos) - Spearman(Euclid) >= {T['p4c_gap_confirm']} with Spearman(cos) < "
                          f"{T['p4c_cos_cap']}; refute if <= 0", len(jeff),
                          {"spearman_cosine": rc, "spearman_euclid": re_}, "Pairs within each (k, positions) group."))
    # P5
    if per["P5"]:
        d5 = per["P5"]
        rho = spearman([d["jsd"] for d in d5], [d["margin"] for d in d5])
        v = verdict_high(rho, T["p5a_rho_confirm"], T["p5a_rho_refute"], len(d5))
        res.append(result("P5a", "the mean writer's loss on a span and its shuffle exceeds the JSD bound by a margin that "
                          "grows with order sensitivity", v, rho, f"Spearman(margin, JSD) >= {T['p5a_rho_confirm']} "
                          f"(refute <= {T['p5a_rho_refute']})", len(d5),
                          {"bound_holds_share": sum(d["bound_holds"] for d in d5) / len(d5),
                           "median_jsd": _median([d["jsd"] for d in d5]), "median_margin": _median([d["margin"] for d in d5])},
                          "The bound itself is a theorem; bound_holds_share below 1 is Monte Carlo noise."))
        beat = sum(d["centred_beats_mean"] for d in d5) / len(d5)
        lose = sum(d["centred_loses_to_deletion"] for d in d5) / len(d5)
        v = _combine(verdict_high(beat, T["p5b_share_confirm"], T["p5b_share_refute"], len(d5)),
                     verdict_high(lose, T["p5b_share_confirm"], T["p5b_share_refute"], len(d5)))
        res.append(result("P5b", "a centred-and-rescaled mean beats the raw mean but still loses to deletion", v,
                          min(beat, lose), f"both shares >= {T['p5b_share_confirm']} (refute if either <= "
                          f"{T['p5b_share_refute']})", len(d5),
                          {"centred_beats_mean_share": beat, "centred_loses_to_deletion_share": lose,
                           "mean_worse_than_deletion_share": sum(d["mean_worse_than_deletion"] for d in d5) / len(d5)}))
    res += check_p6(rows, control_rows)
    # P7
    if per["P7"]:
        rec = _median([d["recovered"] for d in per["P7"]])
        err = _median([d["pred_error"] for d in per["P7"]])
        v = verdict_high(rec, T["p7_recovered_confirm"], T["p7_recovered_refute"], len(per["P7"]))
        if v == CONFIRMED and not err <= T["p7_pred_err_confirm"]:
            v = INCONCLUSIVE
        res.append(result("P7", "one Gauss-Newton step from a reasonable start recovers most of the gap, and predicts the "
                          "converged loss", v, rec, f"median share of gap recovered >= {T['p7_recovered_confirm']} and "
                          f"median prediction error <= {T['p7_pred_err_confirm']} (refute if recovered <= "
                          f"{T['p7_recovered_refute']})", len(per["P7"]),
                          {"median_pred_error": err,
                           "median_recovered_best_of_halvings": _median([(d["loss_start"] - d["loss_step_best_of_halvings"])
                                                                         / max(d["loss_start"] - d["loss_optimised"], 1e-9)
                                                                         for d in per["P7"]])},
                          "Start is the probe's mean-of-embeddings vector, the only writer step 1 has; the theory "
                          "asks for a 'reasonable' start, so a failure here is weak evidence."))
    # P8
    if emission:
        words, opn = emission["words"], emission["open"]
        ks = sorted({w["k"] for w in words} & {o["k"] for o in opn})
        ratios = []
        for k in ks:
            wm = _median([w["residual"] for w in words if w["k"] == k])
            om = _median([o["residual"] for o in opn if o["k"] == k])
            if om > 0:
                ratios.append(wm / om)
        r = _median(ratios)
        v = verdict_low(r, T["p8a_ratio_confirm"], T["p8a_ratio_refute"], sum(w["k"] in ks for w in words))
        res.append(result("P8a", "the emission residual is near zero for multi-token words and large for open spans", v, r,
                          f"median over matched lengths of word/open residual <= {T['p8a_ratio_confirm']} (refute >= "
                          f"{T['p8a_ratio_refute']})", sum(w["k"] in ks for w in words),
                          {"matched_lengths": ks, "contexts": emission["contexts"], "hidden": emission["hidden"]}))
        oks = sorted({o["k"] for o in opn})
        pts = [(math.log(k - 1), math.log(_median([o["residual"] for o in opn if o["k"] == k]))) for k in oks
               if _median([o["residual"] for o in opn if o["k"] == k]) > 0]
        slope = linfit([p[0] for p in pts], [p[1] for p in pts])[0]
        v = verdict_low(slope, T["p8b_slope_confirm"], T["p8b_slope_refute"]) if len(pts) >= 3 else INCONCLUSIVE
        res.append(result("P8b", "the emission residual grows with span length at most quadratically", v, slope,
                          f"log-log slope against k-1 <= {T['p8b_slope_confirm']} (refute >= {T['p8b_slope_refute']})",
                          len(opn), {"median_residual_by_k": {k: _median([o["residual"] for o in opn if o["k"] == k]) for k in oks}}))
    res.append(check_p9(rows))

    report = {
        "model": lm.name, "toy_model": is_toy(lm.name), "run": os.path.abspath(run_dir),
        "control_run": os.path.abspath(control_dir) if control_dir else None,
        "thresholds": THRESHOLDS, "thresholds_digest": thresholds_digest(), "git_commit": _git_commit(),
        "settings": {k: v for k, v in vars(st).items()}, "spans_analysed": len(chosen),
        "seconds": time.time() - t0, "results": res, "per_span": dict(per),
    }  # fmt: skip
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    with open(os.path.join(out_dir, "predictions.md"), "w") as f:
        f.write(to_markdown(report))
    return report


def _fmt(x):
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.3g}"
    return str(x)


def to_markdown(report: dict) -> str:
    lines = [f"# Theory predictions: {report['model']}", ""]
    if report["toy_model"]:
        lines += ["**Toy model.** These numbers check that the analysis runs. They say nothing reliable about Granite.", ""]
    lines += [
        f"Thresholds digest `{report['thresholds_digest']}` (fixed in `kenning/predictions.py` before any Granite "
        f"result). Code commit `{report['git_commit']}`. {report['spans_analysed']} single-vector spans, "
        f"{report['seconds']:.0f} s.", "",
        "| ID | Prediction | Verdict | Value | Rule | Spans |", "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in report["results"]:
        lines.append(f"| {r['id']} | {r['claim']} | **{r['verdict']}** | {_fmt(r['value'])} | {r['rule']} | {_fmt(r['spans'])} |")
    lines += ["", "## Notes and details", ""]
    for r in report["results"]:
        bits = [f"{k} = {_fmt(v)}" for k, v in r["details"].items() if not isinstance(v, (dict, list))]
        if r["note"] or bits:
            lines.append(f"- **{r['id']}**: " + "; ".join(([r["note"]] if r["note"] else []) + bits))
    return "\n".join(lines) + "\n"
