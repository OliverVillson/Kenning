"""Where probe text comes from, and how it is cut into chunks."""

from __future__ import annotations

import glob
import json
import os

import torch

from .models import LM

TEXT_EXTENSIONS = (".txt", ".md", ".py", ".rst", ".tex")


def read_texts(source: str, path: str | None) -> list[str]:
    """Return a list of documents.

    source="traces": a JSONL file written by ``kenning traces``; the model's own
        reasoning is used when present, otherwise the full completion.
    source="text":   a text file, or a directory searched recursively.
    """
    if source == "traces":
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"traces file not found: {path!r} (run `kenning traces` first)")
        texts = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    texts.append(row.get("reasoning") or row.get("text") or "")
        return [t for t in texts if t.strip()]
    if source == "text":
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"text path not found: {path!r}")
        files = [path] if os.path.isfile(path) else sorted(
            p
            for p in glob.glob(os.path.join(path, "**", "*"), recursive=True)
            if os.path.isfile(p) and p.endswith(TEXT_EXTENSIONS)
        )
        texts = []
        for p in files:
            with open(p, errors="replace") as f:
                texts.append(f.read())
        return [t for t in texts if t.strip()]
    raise ValueError(f"unknown source {source!r}")


def token_streams(lm: LM, source: str, path: str | None, random_tokens: int = 200_000, seed: int = 0):
    """Tokenise the source into a list of 1-D LongTensors, one per document.

    source="random" returns uniformly random token ids: a control with no
    structure for the model to exploit, so it shows raw storage capacity.
    """
    if source == "random":
        gen = torch.Generator().manual_seed(seed)
        return [torch.randint(0, lm.vocab_size, (random_tokens,), generator=gen)]
    return [torch.tensor(ids, dtype=torch.long) for ids in map(lm.encode, read_texts(source, path)) if ids]


def make_chunks(streams, k: int, m: int, n: int, seed: int = 0):
    """Cut up to ``n`` windows of ``k`` context tokens followed by ``m`` continuation tokens.

    Windows never cross a document boundary and never overlap. They are spread
    across documents round-robin so one long document cannot dominate. The
    result is deterministic for a given seed, which is what makes resuming a
    run safe.

    Returns (ctx [n', k], cont [n', m]) with n' <= n.
    """
    need = k + m
    gen = torch.Generator().manual_seed(seed + 7919 * k + 31 * m)
    per_doc = []
    for s in streams:
        slots = len(s) // need
        if slots == 0:
            continue
        slack = len(s) - slots * need
        offset = int(torch.randint(0, slack + 1, (1,), generator=gen)) if slack else 0
        order = torch.randperm(slots, generator=gen).tolist()
        per_doc.append([s[offset + i * need : offset + (i + 1) * need] for i in order])

    windows = []
    depth = 0
    while len(windows) < n and any(depth < len(d) for d in per_doc):
        for d in per_doc:
            if depth < len(d) and len(windows) < n:
                windows.append(d[depth])
        depth += 1

    if not windows:
        return torch.empty(0, k, dtype=torch.long), torch.empty(0, m, dtype=torch.long)
    w = torch.stack(windows)
    return w[:, :k].contiguous(), w[:, k:].contiguous()
