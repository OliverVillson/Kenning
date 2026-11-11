"""Tests run on a tiny randomly initialised model: no downloads, CPU only, a few seconds."""

import json
import os

import pytest
import torch
import torch.nn.functional as F

from kenning import cli, data, models, probe, report, traces


@pytest.fixture(scope="module")
def lm():
    return models.load(models.TINY_RANDOM, device="cpu")


def _ids(lm, shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, generator=gen)


def test_self_check_passes(lm):
    check = models.self_check(lm)
    assert check["embeds_match_ids"]
    assert check["position_ids_supported"] and check["positions_match"]


def test_reconstruct_logits_line_up_with_plain_model(lm):
    """A kenning that is just a token embedding must behave exactly like that token."""
    ctx = _ids(lm, (2, 10))
    token = torch.tensor([[65], [66]])
    with torch.no_grad():
        got = probe.reconstruct_logits(lm, lm.embeddings(token).float(), lm.embeddings(ctx))
        ids = torch.cat([torch.full((2, 1), lm.start_id), token, ctx[:, :-1]], dim=1)
        want = lm.model(input_ids=ids).logits[:, 1:11]
    assert got.shape == (2, 10, lm.vocab_size)
    assert torch.allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("positions", ["gap", "renumber"])
def test_continue_matches_teacher_when_kenning_is_the_chunk(lm, positions):
    """With one vector per chunk token, set to the chunk's own embeddings, student equals teacher."""
    k, m = 9, 6
    ctx, cont = _ids(lm, (2, k), 1), _ids(lm, (2, m), 2)
    with torch.no_grad():
        got = probe.continue_logq(lm, lm.embeddings(ctx).float(), cont.unsqueeze(1), k, positions)
        ids = torch.cat([torch.full((2, 1), lm.start_id), ctx, cont[:, :-1]], dim=1)
        want = F.log_softmax(lm.model(input_ids=ids).logits[:, k : k + m], dim=-1)
    assert torch.allclose(got, want, atol=1e-4)


def test_gap_positions_keep_the_continuation_where_it_was(lm):
    pos = probe._position_ids(lm, "gap", k=10, n=2, m=4, batch=1)
    assert pos.tolist() == [[0, 9, 10, 11, 12, 13]]
    assert probe._position_ids(lm, "renumber", k=10, n=2, m=4, batch=1) is None


def test_sampled_continuations_report_the_teacher_distribution(lm):
    prefix = torch.cat([torch.full((3, 1), lm.start_id), _ids(lm, (3, 7), 3)], dim=1)
    tokens, logp = probe.sample_continuations(lm, prefix, 5, torch.Generator().manual_seed(0))
    assert tokens.shape == (3, 5) and logp.shape == (3, 5, lm.vocab_size)
    with torch.no_grad():
        ids = torch.cat([prefix, tokens[:, :-1]], dim=1)
        want = F.log_softmax(lm.model(input_ids=ids).logits[:, 7:12], dim=-1)
    assert torch.allclose(logp, want, atol=1e-4)


def test_reconstruct_improves_on_its_starting_point(lm):
    cfg = probe.ProbeConfig(steps=120, lr=0.02, plateau_steps=0)
    rows, vecs = probe.probe_reconstruct(lm, _ids(lm, (3, 6), 4), cfg)
    assert vecs.shape == (3, 1, lm.hidden_size)
    for r in rows:
        assert r["ce"] < r["ce_init"]
        assert r["accuracy"] >= r["accuracy_init"]
        assert r["bits_gained"] > 0


def test_continue_reports_finite_scores_and_never_gets_worse_than_its_start(lm):
    cfg = probe.ProbeConfig(steps=60, lr=0.02, cont_len=8, n_train=4, n_val=2, eval_every=10, patience=10)
    rows, _ = probe.probe_continue(lm, _ids(lm, (2, 12), 5), _ids(lm, (2, 8), 6), cfg)
    for r in rows:
        assert all(v == v and abs(v) != float("inf") for v in r.values() if isinstance(v, float))
        assert r["kl_no_context"] >= 0 and r["kl_kenning"] >= 0
        assert r["kl_val"] <= r["kl_val_init"]  # the best step is never worse than step 0
        assert r["best_step"] >= 0


def test_chunks_are_deterministic_and_do_not_overlap():
    streams = [torch.arange(0, 1000), torch.arange(5000, 5400)]
    ctx, cont = data.make_chunks(streams, k=30, m=10, n=12, seed=3)
    again, _ = data.make_chunks(streams, k=30, m=10, n=12, seed=3)
    assert ctx.shape == (12, 30) and cont.shape == (12, 10)
    assert torch.equal(ctx, again)
    assert torch.equal(cont[:, 0], ctx[:, -1] + 1)  # continuation follows its chunk
    tokens = torch.cat([ctx, cont], dim=1).flatten().tolist()
    assert len(tokens) == len(set(tokens))  # no token used twice
    both_docs = {int(t) >= 5000 for t in ctx[:, 0]}
    assert both_docs == {True, False}  # spread across documents


def test_chunks_longer_than_any_document_give_nothing():
    ctx, cont = data.make_chunks([torch.arange(50)], k=64, m=8, n=4)
    assert ctx.shape == (0, 64) and cont.shape == (0, 8)


def test_split_reasoning_handles_the_tag_variants():
    assert traces.split_reasoning("<think>a b</think>answer") == "a b"
    assert traces.split_reasoning("a b</think>answer") == "a b"  # tag opened by the chat template
    assert traces.split_reasoning("<think>still thinking") == "still thinking"
    assert traces.split_reasoning("plain answer") is None


def test_cli_runs_end_to_end_and_resumes(tmp_path):
    text = tmp_path / "text"
    text.mkdir()
    (text / "a.txt").write_text("the quick brown fox jumps over the lazy dog. " * 40)
    run = str(tmp_path / "run")
    argv = [
        "probe", "--model", models.TINY_RANDOM, "--device", "cpu", "--out", run,
        "--source", "text", "--data", str(text), "--lengths", "8,16", "--n-samples", "3",
        "--batch", "2", "--steps", "20", "--cont-len", "8", "--n-train", "2", "--n-val", "2",
        "--eval-every", "10", "--save-vectors",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    rows = report.load_rows(run)
    assert len(rows) == 2 * 2 * 3  # modes x lengths x samples
    assert {r["mode"] for r in rows} == {"reconstruct", "continue"}
    assert os.path.exists(os.path.join(run, "summary.md"))
    summary = json.load(open(os.path.join(run, "summary.json")))
    assert len(summary["reconstruct"]) == 2 and len(summary["continue"]) == 2

    assert cli.main(argv) == 0  # second run finds everything done
    assert len(report.load_rows(run)) == len(rows)


def test_traces_and_inspect_commands(tmp_path):
    out = str(tmp_path / "traces.jsonl")
    argv = ["traces", "--model", models.TINY_RANDOM, "--device", "cpu", "--out", out,
            "--limit", "2", "--samples-per-prompt", "2", "--max-new-tokens", "12"]  # fmt: skip
    assert cli.main(argv) == 0
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 4 and {r["prompt_id"] for r in rows} == {0, 1}
    assert cli.main(argv) == 0  # resumes without duplicating
    assert len(open(out).readlines()) == 4
    assert cli.main(["inspect", "--model", models.TINY_RANDOM, "--device", "cpu"]) == 0
