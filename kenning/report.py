"""Turn a run's results.jsonl into a summary table, a JSON file and a chart."""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict

# Chart colours: a categorical palette checked for colour-blind separation on
# the light surface, used in this fixed order (one colour per series, never
# reassigned). Text and grid use neutral ink so colour only ever marks a series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
BASELINE = "#8a897f"


def load_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "results.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-sample rows into one entry per (mode, vectors, positions, length)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["n_vectors"], r.get("positions", "-"), r["k"])].append(r)

    reconstruct, cont = [], []
    for (mode, n, positions, k), g in sorted(groups.items()):
        if mode == "reconstruct":
            hits = [r["steps_to_target"] for r in g if r["steps_to_target"] >= 0]
            reconstruct.append(
                {
                    "n_vectors": n,
                    "k": k,
                    "samples": len(g),
                    "accuracy_median": _median([r["accuracy"] for r in g]),
                    "accuracy_mean": _mean([r["accuracy"] for r in g]),
                    "lossless_fraction": _mean([1.0 if r["lossless"] else 0.0 for r in g]),
                    "accuracy_mean_of_embeddings": _mean([r["accuracy_init"] for r in g]),
                    "bits_gained_mean": _mean([r["bits_gained"] for r in g]),
                    "steps_to_lossless_median": _median(hits) if hits else None,
                    "vector_norm_ratio": _mean([r["vector_norm_ratio"] for r in g]),
                    "seconds_per_sample": _mean([r["seconds"] for r in g]),
                }
            )
        else:
            kl_none = sum(r["kl_no_context"] for r in g)
            gain = sum(r["ce_gain_possible"] for r in g)
            val_none = sum(r["kl_val_no_context"] for r in g)
            cont.append(
                {
                    "n_vectors": n,
                    "positions": positions,
                    "k": k,
                    "samples": len(g),
                    # Ratios of sums, so chunks where context barely matters cannot blow up the average.
                    "recovered": 1 - sum(r["kl_kenning"] for r in g) / kl_none if kl_none > 0 else float("nan"),
                    "recovered_median": _median([r["recovered"] for r in g]),
                    "recovered_mean_of_embeddings": 1 - sum(r["kl_init"] for r in g) / kl_none
                    if kl_none > 0
                    else float("nan"),
                    "recovered_val": 1 - sum(r["kl_val"] for r in g) / val_none if val_none > 0 else float("nan"),
                    "ce_gain_fraction": sum(r["ce_gain_kenning"] for r in g) / gain if gain > 0 else float("nan"),
                    "kl_no_context_mean": _mean([r["kl_no_context"] for r in g]),
                    "best_step_median": _median([r["best_step"] for r in g]),
                    "vector_norm_ratio": _mean([r["vector_norm_ratio"] for r in g]),
                    "seconds_per_sample": _mean([r["seconds"] for r in g]),
                }
            )

    headlines = {}
    for n in sorted({e["n_vectors"] for e in reconstruct}):
        ok = [e["k"] for e in reconstruct if e["n_vectors"] == n and e["lossless_fraction"] >= 0.5]
        headlines[f"lossless_capacity_tokens_n{n}"] = max(ok) if ok else 0
    return {"reconstruct": reconstruct, "continue": cont, "headlines": headlines}


def _fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.{digits}f}"
    return str(x)


def to_markdown(summary: dict, title: str) -> str:
    lines = [f"# Capacity probe: {title}", ""]
    if summary["reconstruct"]:
        lines += [
            "## Reconstruct: can the model reproduce the chunk from the kenning?",
            "",
            "Accuracy is the share of chunk tokens predicted correctly; 1.00 means the chunk comes back exactly.",
            "\"Lossless\" is the share of samples that reached 1.00. \"Mean of embeddings\" is the accuracy of",
            "simply averaging the chunk's token embeddings, before any optimisation.",
            "",
            "| vectors | chunk tokens | samples | accuracy (median) | lossless | mean of embeddings | bits stored | steps to lossless | norm vs token | s/sample |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in summary["reconstruct"]:
            lines.append(
                f"| {e['n_vectors']} | {e['k']} | {e['samples']} | {_fmt(e['accuracy_median'])} | "
                f"{_fmt(e['lossless_fraction'])} | {_fmt(e['accuracy_mean_of_embeddings'])} | "
                f"{_fmt(e['bits_gained_mean'], 0)} | {_fmt(e['steps_to_lossless_median'], 0)} | "
                f"{_fmt(e['vector_norm_ratio'], 1)}x | {_fmt(e['seconds_per_sample'], 1)} |"
            )
        lines.append("")
        for key, value in summary["headlines"].items():
            n = key.rsplit("_n", 1)[-1]
            lines.append(
                f"**Lossless capacity with {n} vector(s): {value} tokens** "
                "(largest chunk length where at least half the samples were reproduced exactly)."
            )
        lines.append("")
    if summary["continue"]:
        lines += [
            "## Continue: does the kenning stand in for the chunk when predicting what follows?",
            "",
            "\"Recovered\" is the share of the chunk's effect on the next tokens that the kenning restores,",
            "scored on the real continuation, which the kenning was never trained on. 1.00 means the kenning",
            "replaces the chunk perfectly; 0.00 means it is no better than deleting the chunk. Negative means",
            "worse than deleting it. \"Validation\" is the same measure on sampled continuations.",
            "",
            "| vectors | positions | chunk tokens | samples | recovered | recovered (median) | mean of embeddings | validation | best step | norm vs token | s/sample |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in summary["continue"]:
            lines.append(
                f"| {e['n_vectors']} | {e['positions']} | {e['k']} | {e['samples']} | {_fmt(e['recovered'])} | "
                f"{_fmt(e['recovered_median'])} | {_fmt(e['recovered_mean_of_embeddings'])} | "
                f"{_fmt(e['recovered_val'])} | {_fmt(e['best_step_median'], 0)} | "
                f"{_fmt(e['vector_norm_ratio'], 1)}x | {_fmt(e['seconds_per_sample'], 1)} |"
            )
        lines.append("")
    lines += [
        "## How to use these numbers",
        "",
        "Both modes are upper bounds: each kenning here was optimised by gradient descent on its own",
        "chunk, which a one-pass writer network cannot match. See docs/DESIGN.md for the thresholds",
        "that decide the next step.",
        "",
    ]
    return "\n".join(lines)


def _series_label(n: int, positions: str, show_positions: bool) -> str:
    label = f"{n} vector" + ("" if n == 1 else "s")
    return f"{label}, {positions}" if show_positions and positions != "-" else label


def plot(summary: dict, path: str, title: str) -> bool:
    """Two small charts side by side, each with its own single axis. Returns False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
    except ImportError:
        return False

    panels = [p for p in ("reconstruct", "continue") if summary[p]]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, panel in zip(axes[0], panels):
        entries = summary[panel]
        ax.set_facecolor(SURFACE)
        keys = sorted({(e["n_vectors"], e.get("positions", "-")) for e in entries})
        show_positions = len({p for _, p in keys}) > 1
        lengths = sorted({e["k"] for e in entries})
        metric = "accuracy_median" if panel == "reconstruct" else "recovered"

        if panel == "continue":
            # Reference line: the untrained mean-of-embeddings start, for the first series only.
            n0, p0 = keys[0]
            base = [e for e in entries if (e["n_vectors"], e["positions"]) == (n0, p0)]
            ax.plot(
                [e["k"] for e in base],
                [e["recovered_mean_of_embeddings"] for e in base],
                color=BASELINE,
                linewidth=1.5,
                marker="o",
                markersize=4,
                label="mean of embeddings, no optimisation",
                zorder=2,
            )
            ax.axhline(0, color=GRID, linewidth=1, zorder=1)

        ends = []
        for i, (n, positions) in enumerate(keys[: len(SERIES)]):
            pts = [e for e in entries if (e["n_vectors"], e.get("positions", "-")) == (n, positions)]
            xs, ys = [e["k"] for e in pts], [e[metric] for e in pts]
            ax.plot(
                xs,
                ys,
                color=SERIES[i],
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=_series_label(n, positions, show_positions),
                zorder=3,
            )
            ends.append((xs[-1], ys[-1]))

        # Label the last point of each line with its value, unless two labels
        # would sit on top of each other; then the legend and table carry it.
        values = sorted(y for _, y in ends)
        if all(b - a >= 0.08 for a, b in zip(values, values[1:])):
            for x, y in ends:
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(8, 0),
                            va="center", fontsize=9, color=TEXT)

        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(k) for k in lengths])
        ax.minorticks_off()
        ax.margins(x=0.12)
        ax.set_xlabel("Chunk length (tokens)", color=TEXT_MUTED, fontsize=10)
        if panel == "reconstruct":
            ax.set_ylim(-0.03, 1.08)
            ax.set_title("Reproduced exactly?", color=TEXT, fontsize=12, loc="left", pad=22)
            ax.text(0, 1.03, "Median share of chunk tokens the model gets back from the kenning",
                    transform=ax.transAxes, color=TEXT_MUTED, fontsize=9)
        else:
            lo = min([0.0] + [e["recovered"] for e in entries] + [e["recovered_mean_of_embeddings"] for e in entries])
            ax.set_ylim(min(-0.03, lo - 0.12), 1.08)
            ax.set_title("Usable in place of the chunk?", color=TEXT, fontsize=12, loc="left", pad=22)
            ax.text(0, 1.03, "Share of the chunk's effect on the next tokens that the kenning restores",
                    transform=ax.transAxes, color=TEXT_MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=9)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        if len(ax.get_legend_handles_labels()[0]) > 1:
            legend = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.27), frameon=False, fontsize=9,
                               borderaxespad=0, handlelength=2.2)
            for text in legend.get_texts():
                text.set_color(TEXT)

    fig.suptitle(f"Kenning capacity probe: {title}", color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


def write_report(run_dir: str, title: str | None = None) -> dict:
    rows = load_rows(run_dir)
    title = title or (rows[0].get("model", "run") if rows else "run")
    if os.path.exists(title):
        title = os.path.basename(os.path.normpath(title))
    summary = summarise(rows)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(to_markdown(summary, title))
    summary["plot"] = plot(summary, os.path.join(run_dir, "capacity.png"), title)
    return summary
