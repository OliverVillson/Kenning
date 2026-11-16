"""Command line: python -m kenning {inspect,traces,toy,probe,repeats,report}."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from . import data, models, probe, report, reuse, toy, traces


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help='Hugging Face id, local path, or "tiny-random"')
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])


def cmd_inspect(args) -> int:
    lm = models.load(args.model, args.device, args.dtype)
    info = models.describe(lm)
    print(json.dumps(info, indent=2))
    if args.out:
        models.save_json(info, args.out)
    return 0 if info["self_check"]["embeds_match_ids"] else 1


def cmd_traces(args) -> int:
    lm = models.load(args.model, args.device, args.dtype)
    prompts = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    if args.limit:
        prompts = (prompts or traces.PROMPTS)[: args.limit]
    n = traces.generate_traces(
        lm,
        args.out,
        prompts=prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking=not args.no_thinking,
        seed=args.seed,
    )
    print(f"[traces] wrote {n} new traces to {args.out}")
    return 0


def cmd_toy(args) -> int:
    loss = toy.train_toy(args.text, args.out, steps=args.steps, seed=args.seed)
    print(f"[toy] saved to {args.out} (final loss {loss:.3f} nats/byte)")
    return 0


def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["mode"], r["k"], r["n_vectors"], r.get("positions", "-"), r["sample"]))
    return keys


def cmd_probe(args) -> int:
    lm = models.load(args.model, args.device, args.dtype)
    info = models.describe(lm)
    check = info["self_check"]
    if not check["embeds_match_ids"]:
        print(
            "[probe] ABORT: feeding embeddings gives different logits from feeding token ids "
            f"(max difference {check['embeds_max_diff']:.4g}). The probe would measure the wrong thing.",
            file=sys.stderr,
        )
        return 1
    if args.positions == "gap" and not check.get("position_ids_supported"):
        print("[probe] ABORT: this model takes no position_ids; use --positions renumber", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    run_args = {key: value for key, value in vars(args).items() if key != "func"}
    models.save_json({"args": run_args, "model": info}, os.path.join(args.out, f"config-{int(time.time())}.json"))
    results_path = os.path.join(args.out, "results.jsonl")
    done = _done_keys(results_path)
    streams = data.token_streams(lm, args.source, args.data, seed=args.seed)
    total_tokens = sum(len(s) for s in streams)
    print(f"[probe] {lm.name} on {lm.device} ({info['dtype']}), hidden size {lm.hidden_size}; "
          f"{len(streams)} documents, {total_tokens} tokens from source '{args.source}'", flush=True)

    modes = ["reconstruct", "continue"] if args.mode == "both" else [args.mode]
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

    with open(results_path, "a") as out:
        for mode in modes:
            positions = args.positions if mode == "continue" else "-"
            m = args.cont_len if mode == "continue" else 0
            for n in _ints(args.n_vectors):
                for k in _ints(args.lengths):
                    ctx, cont = data.make_chunks(streams, k, m, args.n_samples, args.seed)
                    tag = f"[{mode} k={k} n={n}]"
                    if len(ctx) == 0:
                        print(f"{tag} skipped: no document has {k + m} tokens", flush=True)
                        continue
                    if len(ctx) < args.n_samples:
                        print(f"{tag} only {len(ctx)} of {args.n_samples} samples available", flush=True)
                    pending = [i for i in range(len(ctx)) if (mode, k, n, positions, i) not in done]
                    batch = args.batch
                    i = 0
                    while i < len(pending):
                        idx = pending[i : i + batch]
                        cfg = probe.ProbeConfig(
                            n_vectors=n,
                            steps=args.steps,
                            lr=args.lr,
                            init=args.init,
                            seed=args.seed + idx[0],
                            cont_len=args.cont_len,
                            n_train=args.n_train,
                            n_val=args.n_val,
                            eval_every=args.eval_every,
                            patience=args.patience,
                            positions=args.positions,
                            plateau_steps=args.plateau_steps,
                        )
                        try:
                            if mode == "reconstruct":
                                rows, vecs = probe.probe_reconstruct(lm, ctx[idx], cfg)
                            else:
                                rows, vecs = probe.probe_continue(lm, ctx[idx], cont[idx], cfg)
                        except oom:
                            if batch == 1:
                                raise
                            batch = max(1, batch // 2)
                            torch.cuda.empty_cache()
                            print(f"{tag} out of memory, retrying with batch {batch}", flush=True)
                            continue
                        for j, row in zip(idx, rows):
                            row.update(sample=j, source=args.source, model=lm.name, positions=positions)
                            out.write(json.dumps(row) + "\n")
                        out.flush()
                        if args.save_vectors:
                            name = f"vectors-{mode}-k{k}-n{n}-{positions}-{idx[0]}.pt"
                            torch.save({"samples": idx, "vectors": vecs, "chunks": ctx[idx]}, os.path.join(args.out, name))
                        if mode == "reconstruct":
                            note = (f"accuracy {sum(r['accuracy'] for r in rows) / len(rows):.3f}, "
                                    f"lossless {sum(r['lossless'] for r in rows)}/{len(rows)}")
                        else:
                            kl = sum(r["kl_kenning"] for r in rows)
                            none = sum(r["kl_no_context"] for r in rows)
                            note = f"recovered {1 - kl / max(none, 1e-8):.3f}"
                        print(f"{tag} samples {idx[0]}-{idx[-1]}: {note}, {rows[0]['steps_run']} steps, "
                              f"{rows[0]['seconds'] * len(rows):.0f}s", flush=True)
                        i += len(idx)

    summary = report.write_report(args.out, title=lm.name)
    print(f"[probe] wrote {results_path}, summary.md" + (", capacity.png" if summary["plot"] else ""))
    for key, value in summary["headlines"].items():
        print(f"[probe] {key} = {value}")
    return 0


def cmd_repeats(args) -> int:
    tok = models.load_tokenizer(args.model)
    docs = [tok.encode(t, add_special_tokens=False) for t in data.read_texts(args.source, args.data)]
    rows = reuse.repeat_table(docs, _ints(args.lengths))
    print(f"{len(docs)} documents, {sum(len(d) for d in docs)} tokens")
    print("span length (tokens) | share of tokens inside a span seen earlier in the same document")
    for r in rows:
        print(f"{r['span_tokens']:>20} | {r['repeated_share']:.3f}")
    if args.out:
        models.save_json({"documents": len(docs), "rows": rows}, args.out)
    return 0


def cmd_report(args) -> int:
    summary = report.write_report(args.run, title=args.title)
    print(open(os.path.join(args.run, "summary.md")).read())
    if not summary["plot"]:
        print("(install matplotlib to get capacity.png)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kenning", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print the model facts the design depends on and self-check the probe paths")
    _add_model_args(p)
    p.add_argument("--out", default=None, help="also write the facts to this JSON file")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("traces", help="generate the model's own reasoning traces as probe text")
    _add_model_args(p)
    p.add_argument("--out", required=True, help="JSONL file to append to")
    p.add_argument("--prompts-file", default=None, help="one prompt per line (default: built-in problems)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N prompts")
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-thinking", action="store_true", help="ask the chat template to disable thinking")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_traces)

    p = sub.add_parser("toy", help="train a tiny byte-level model on local text, for dry runs only")
    p.add_argument("--text", required=True, help="text file or directory to train on")
    p.add_argument("--out", required=True, help="directory to save the model to")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_toy)

    p = sub.add_parser("probe", help="run the capacity probe")
    _add_model_args(p)
    p.add_argument("--out", required=True, help="run directory; an existing run is resumed")
    p.add_argument("--mode", default="both", choices=["reconstruct", "continue", "both"])
    p.add_argument("--source", default="traces", choices=["traces", "text", "random"])
    p.add_argument("--data", default=None, help="traces JSONL (source=traces) or text file/directory (source=text)")
    p.add_argument("--lengths", default="16,32,64,128,256,512", help="chunk lengths in tokens")
    p.add_argument("--n-vectors", default="1", help="vectors per kenning, comma separated to compare")
    p.add_argument("--n-samples", type=int, default=16, help="chunks per length")
    p.add_argument("--batch", type=int, default=8, help="chunks optimised at once (halved on out-of-memory)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--init", default="mean", choices=["mean", "random"])
    p.add_argument("--plateau-steps", type=int, default=300, help="reconstruct: stop after this many steps without improvement")
    p.add_argument("--cont-len", type=int, default=64, help="continue: continuation tokens scored")
    p.add_argument("--n-train", type=int, default=8, help="continue: sampled continuations trained on")
    p.add_argument("--n-val", type=int, default=4, help="continue: sampled continuations for picking the best step")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--patience", type=int, default=8, help="continue: evaluations without improvement before stopping")
    p.add_argument("--positions", default="renumber", choices=["renumber", "gap"])
    p.add_argument("--save-vectors", action="store_true", help="save the optimised kennings for later analysis")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("repeats", help="how much of the text repeats itself (lower bound on reuse)")
    p.add_argument("--model", required=True, help="model whose tokenizer to use")
    p.add_argument("--source", default="traces", choices=["traces", "text"])
    p.add_argument("--data", required=True)
    p.add_argument("--lengths", default="4,8,16,32")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_repeats)

    p = sub.add_parser("report", help="rebuild summary.md, summary.json and capacity.png for a run")
    p.add_argument("--run", required=True)
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
