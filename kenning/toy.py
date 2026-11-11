"""Train a tiny byte-level model on local text.

This is only for dry runs: it gives the probe a model that has learned real
structure, in a few minutes on a laptop CPU and with nothing to download, so
bugs show up before any GPU time is spent. Its numbers say nothing about a
real model.
"""

from __future__ import annotations

import json
import math
import os

import torch

from .data import read_texts
from .models import BYTE_TOKENIZER_MARKER, ByteTokenizer, build_tiny_model


def train_toy(
    text_path: str,
    out_dir: str,
    steps: int = 1500,
    batch: int = 32,
    seq_len: int = 256,
    lr: float = 3e-3,
    hidden: int = 96,
    layers: int = 3,
    seed: int = 0,
) -> float:
    """Train on every text file under ``text_path`` and save to ``out_dir``. Returns the final loss."""
    torch.manual_seed(seed)
    raw = bytearray("\n\n".join(read_texts("text", text_path)).encode("utf-8"))
    data = torch.frombuffer(raw, dtype=torch.uint8).long()
    if len(data) < seq_len + 2:
        raise ValueError(f"not enough text under {text_path!r} to train on")
    model = build_tiny_model(hidden=hidden, layers=layers, seed=seed).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bos = torch.full((batch, 1), ByteTokenizer.bos_token_id, dtype=torch.long)
    loss_value = float("nan")
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        starts = torch.randint(0, len(data) - seq_len, (batch,))
        x = torch.cat([bos, torch.stack([data[s : s + seq_len] for s in starts])], dim=1)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_value = loss.item()
        if step % 100 == 0 or step == steps - 1:
            print(f"[toy] step {step:5d}  loss {loss_value:.3f} nats/byte", flush=True)
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, BYTE_TOKENIZER_MARKER), "w") as f:
        json.dump({"tokenizer": "bytes", "trained_on": os.path.abspath(text_path), "steps": steps}, f)
    return loss_value
