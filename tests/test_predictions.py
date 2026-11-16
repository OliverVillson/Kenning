"""Checks of the theory-prediction machinery on the tiny random model: CPU only, no downloads."""

import json
import os

import pytest
import torch

from kenning import cli, models
from kenning import predictions as pr


@pytest.fixture(scope="module")
def lm():
    lm = models.load(models.TINY_RANDOM, device="cpu")
    pr.use_eager_attention(lm)
    return lm


def _span(lm, k=6, n=1, positions="renumber", seed=0, z=None):
    gen = torch.Generator().manual_seed(seed)
    chunk = torch.randint(0, 256, (k,), generator=gen)
    if z is None:
        z = torch.randn(n, lm.hidden_size, generator=gen) * 3 * lm.embeddings.weight.std()
    return pr.Span(chunk=chunk, z=z.float(), k=k, n=z.shape[0], positions=positions, sample=0)


def _beh(lm, span, R=3, m=5, seed=1):
    return pr.teacher_behaviour(lm, span.chunk, R, m, torch.Generator().manual_seed(seed))


def test_jvp_paths_agree_with_finite_differences(lm):
    span = _span(lm)
    beh = _beh(lm, span)
    fn = lambda Z: pr._logits_for(lm, span, Z, beh)  # noqa: E731
    z = span.z.unsqueeze(0)
    v = torch.randn_like(z)
    v = v / v.norm()
    fwd = pr.jvp(fn, z, v)
    dbl = pr.jvp_double_backward(fn, z, v)
    eps = 1e-3 * z.norm().item()
    with torch.no_grad():
        fd = (fn(z + eps * v) - fn(z - eps * v)) / (2 * eps)
    assert torch.allclose(fwd, dbl, atol=1e-4)
    assert (fwd - fd).abs().max() < 2e-2 * fd.abs().max()


@pytest.mark.parametrize("positions", ["renumber", "gap"])
def test_gauss_newton_product_is_the_hessian_at_a_perfect_fit(lm, positions):
    """With one vector per chunk token set to the chunk itself, L = 0, so the Hessian is exactly G."""
    span = _span(lm, k=4, positions=positions)
    span = pr.Span(chunk=span.chunk, z=lm.embeddings(span.chunk).detach().float(), k=4, n=4, positions=positions, sample=0)
    beh = _beh(lm, span)
    assert pr.losses(lm, span, beh, span.z.unsqueeze(0), 1).item() < 1e-5
    d = torch.randn_like(span.z)
    d = d / d.norm()
    Gd = pr.gn_product(lm, span, beh, span.z, d.unsqueeze(0))[0]
    eps = 1e-2 * span.z.norm().item()
    _, gp = pr.grad_fn(lm, span, beh, span.z + eps * d)
    _, gm = pr.grad_fn(lm, span, beh, span.z - eps * d)
    fd = (gp - gm) / (2 * eps)
    assert (Gd - fd).norm() < 0.05 * fd.norm()


def test_gauss_newton_product_is_symmetric_and_positive(lm):
    span = _span(lm)
    beh = _beh(lm, span)
    a, b = torch.randn(2, 1, lm.hidden_size).unbind(0)
    Ga, Gb = pr.gn_product(lm, span, beh, span.z, torch.stack([a, b]))
    assert abs(torch.dot(a.flatten(), Gb.flatten()) - torch.dot(Ga.flatten(), b.flatten())) < 1e-4 * Ga.norm() * b.norm()
    assert torch.dot(a.flatten(), Ga.flatten()) >= 0


def test_lanczos_and_trace_match_the_explicit_matrix(lm):
    span = _span(lm)
    beh = _beh(lm, span)
    d = lm.hidden_size
    op = pr.gn_operator(lm, span, beh, span.z, 16)
    G = op(torch.eye(d))
    eig = torch.linalg.eigvalsh(0.5 * (G + G.T).double())
    ritz = pr.lanczos_top(op, d, 30, torch.Generator().manual_seed(0), lm.device)
    assert abs(ritz[0].item() - eig[-1].item()) < 1e-3 * eig[-1].item()
    tr = pr.gn_trace(lm, span, beh, span.z, 64, torch.Generator().manual_seed(0))
    assert abs(tr - torch.trace(G).item()) < 0.25 * torch.trace(G).item()


def test_conjugate_gradient_solves_a_small_system():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    x = pr.conjugate_gradient(lambda V: V @ A.T, b, 10)
    assert torch.allclose(A @ x, b, atol=1e-5)


def test_sublayer_scaling_reproduces_the_norm_and_its_pinned_limit(lm):
    span = _span(lm)
    beh = _beh(lm, span)
    R0 = span.z.norm().item()
    u = pr._unit(span.z)
    with torch.no_grad():
        plain = pr.losses(lm, span, beh, torch.stack([R0 * u, 2 * R0 * u, 50 * R0 * u]), 3)
        s = pr.scaled_loss(lm, span, beh, u.unsqueeze(0).expand(2, -1, -1), torch.tensor([1.0, 0.5]), R0)
    assert abs(s[0] - plain[0]) < 1e-5  # the hooks are the identity at sigma = 1
    assert abs(s[1] - plain[1]) < 1e-3  # and halve the norm's effect at sigma = 1/2
    L_inf, fp = pr.fprime0(lm, span, beh, u, R0)
    rho = 1 / (50 * R0)
    assert abs((plain[2].item() - L_inf) - fp * rho) < 0.2 * abs(fp * rho) + 1e-5  # first-order expansion in 1/r


def test_residual_updates_report_the_layers(lm):
    span = _span(lm)
    r, S = pr.residual_updates(lm, span, span.z)
    assert len(S) == len(pr.decoder_layers(lm.model))
    assert abs(r - span.z.norm().item()) < 1e-4  # this model has no embedding multiplier
    assert all(s > 0 for s in S)


def test_statistics_helpers():
    assert pr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    slope, icpt, r2 = pr.linfit([0, 1, 2], [1, 3, 5])
    assert (slope, icpt, r2) == pytest.approx((2.0, 1.0, 1.0))
    assert pr.verdict_high(0.9, 0.8, 0.5) == pr.CONFIRMED
    assert pr.verdict_high(0.6, 0.8, 0.5) == pr.INCONCLUSIVE
    assert pr.verdict_low(0.9, 0.1, 0.3) == pr.REFUTED
    assert pr.verdict_high(0.9, 0.8, 0.5, n=1) == pr.INCONCLUSIVE  # too few spans


def test_thresholds_are_pinned():
    """Changing a threshold must be a deliberate, visible act: update this digest in the same commit."""
    assert pr.thresholds_digest() == "414efb66ebad"


def test_predictions_cli_end_to_end_and_resumes(tmp_path):
    text = tmp_path / "text"
    text.mkdir()
    (text / "a.txt").write_text("the quick brown fox jumps over the lazy dog while kenning words wander. " * 60)
    run = str(tmp_path / "run")
    common = ["--model", models.TINY_RANDOM, "--device", "cpu", "--out", run, "--source", "text", "--data", str(text),
              "--n-samples", "3", "--batch", "3", "--steps", "10", "--cont-len", "6", "--n-train", "2", "--n-val", "2",
              "--eval-every", "5", "--save-vectors", "--mode", "continue", "--lengths", "6,12"]  # fmt: skip
    assert cli.main(["probe", *common]) == 0
    assert cli.main(["probe", *common, "--positions", "gap"]) == 0
    assert cli.main(["probe", *common[:-2], "--lengths", "6", "--n-vectors", "2,3"]) == 0
    argv = ["predictions", "--model", models.TINY_RANDOM, "--device", "cpu", "--run", run, "--teacher-samples", "2",
            "--max-spans", "3", "--lanczos-steps", "5", "--trace-probes", "2", "--cg-steps", "5", "--sphere-steps", "3",
            "--gd-steps", "6", "--pair-spans", "3", "--emission-contexts", "40", "--emission-spans", "2"]  # fmt: skip
    assert cli.main(argv) == 0
    rep = json.load(open(os.path.join(run, "predictions.json")))
    ids = {r["id"] for r in rep["results"]}
    assert {"P1a", "P1b", "P2a", "P2b", "P3", "P4a", "P4b", "P4c", "P5a", "P5b", "P6a", "P6b", "P6c", "P7", "P8a", "P9"} <= ids
    allowed = {pr.CONFIRMED, pr.REFUTED, pr.INCONCLUSIVE, pr.NOT_TESTABLE, pr.NOT_RUN}
    assert all(r["verdict"] in allowed for r in rep["results"])
    assert rep["thresholds_digest"] == pr.thresholds_digest() and rep["toy_model"]
    assert "Toy model" in open(os.path.join(run, "predictions.md")).read()
    ledger = open(os.path.join(run, "predictions-spans.jsonl")).read()
    assert cli.main(argv) == 0  # second run reuses every per-span measurement
    assert open(os.path.join(run, "predictions-spans.jsonl")).read() == ledger
