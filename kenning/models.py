"""Model loading, a no-download tiny model, and model inspection."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any

import torch

TINY_RANDOM = "tiny-random"
BYTE_TOKENIZER_MARKER = "kenning_byte_tokenizer.json"


class ByteTokenizer:
    """Byte-level tokenizer with no files to download.

    Ids 0..255 are raw bytes. It exists so the whole pipeline can be tested
    without network access (unit tests, smoke tests, the toy model).
    """

    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    vocab_size = 259
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.bos_token_id] + ids if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size


@dataclass
class LM:
    """A loaded, frozen language model plus what the probe needs to know about it."""

    model: Any
    tok: Any
    name: str
    device: torch.device
    dtype: torch.dtype
    start_id: int  # token placed at position 0 of every probe sequence

    @property
    def embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        return self.embeddings.weight.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.weight.shape[0]

    @property
    def accepts_position_ids(self) -> bool:
        return "position_ids" in inspect.signature(self.model.forward).parameters

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype: str | None = None) -> torch.dtype:
    """bfloat16 on CUDA (fast, fits easily), float32 elsewhere (safe gradients)."""
    if dtype and dtype != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def build_tiny_model(hidden: int = 96, layers: int = 3, heads: int = 4, seed: int = 0):
    """A small randomly initialised Llama-style model over bytes."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=ByteTokenizer.vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=8192,
        bos_token_id=ByteTokenizer.bos_token_id,
        eos_token_id=ByteTokenizer.eos_token_id,
        pad_token_id=ByteTokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def load_tokenizer(name: str):
    """Just the tokenizer, for analyses that never run the model."""
    if name == TINY_RANDOM or os.path.exists(os.path.join(name, BYTE_TOKENIZER_MARKER)):
        return ByteTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def load(name: str, device: str | None = None, dtype: str | None = None) -> LM:
    """Load a model by Hugging Face id, local path, or the name "tiny-random".

    The model comes back frozen and in eval mode: the probe only ever trains
    the kenning vectors, never the model.
    """
    dev = pick_device(device)
    dt = pick_dtype(dev, dtype)

    if name == TINY_RANDOM:
        model, tok = build_tiny_model(), ByteTokenizer()
    elif os.path.exists(os.path.join(name, BYTE_TOKENIZER_MARKER)):
        from transformers import AutoModelForCausalLM

        model, tok = AutoModelForCausalLM.from_pretrained(name), ByteTokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        try:  # newer transformers call the argument "dtype", older ones "torch_dtype"
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)

    model = model.to(device=dev, dtype=dt).eval()
    model.requires_grad_(False)

    start_id = next(
        (i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) if i is not None), 0
    )
    return LM(model=model, tok=tok, name=name, device=dev, dtype=dt, start_id=int(start_id))


@torch.no_grad()
def self_check(lm: LM, atol: float | None = None) -> dict:
    """Verify the two code paths the probe relies on before spending GPU time.

    1. Feeding token embeddings through ``inputs_embeds`` must give the same
       logits as feeding token ids. If a model rescales embeddings internally
       this fails, and every probe number would be wrong.
    2. If the model accepts ``position_ids``, passing 0..T-1 explicitly must
       match the default. The "gap" position mode depends on it.
    """
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, min(lm.vocab_size, 250), (2, 12), generator=gen).to(lm.device)
    ids[:, 0] = lm.start_id
    by_ids = lm.model(input_ids=ids).logits.float()
    by_emb = lm.model(inputs_embeds=lm.embeddings(ids)).logits.float()
    scale = by_ids.abs().max().item()
    tol = atol if atol is not None else (1e-3 if lm.dtype == torch.float32 else 5e-2) * max(scale, 1.0)
    emb_diff = (by_ids - by_emb).abs().max().item()
    out = {"embeds_match_ids": emb_diff <= tol, "embeds_max_diff": emb_diff, "tolerance": tol}

    if lm.accepts_position_ids:
        pos = torch.arange(ids.shape[1], device=lm.device).unsqueeze(0).expand_as(ids)
        by_pos = lm.model(input_ids=ids, position_ids=pos).logits.float()
        pos_diff = (by_ids - by_pos).abs().max().item()
        out.update(position_ids_supported=True, positions_match=pos_diff <= tol, positions_max_diff=pos_diff)
    else:
        out.update(position_ids_supported=False)
    return out


@torch.no_grad()
def describe(lm: LM) -> dict:
    """Facts about the model that the Kenning design depends on."""
    model, cfg = lm.model, lm.model.config
    emb_in = model.get_input_embeddings().weight
    out_layer = model.get_output_embeddings()
    emb_out = out_layer.weight if out_layer is not None else None
    tied = emb_out is not None and (
        emb_in.data_ptr() == emb_out.data_ptr()
        or (emb_in.shape == emb_out.shape and torch.equal(emb_in, emb_out))
    )
    row_norms = emb_in.float().norm(dim=1)
    norm_layers = sorted({type(m).__name__ for n, m in model.named_modules() if "norm" in n.lower()})
    info = {
        "name": lm.name,
        "model_class": type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "hidden_size": lm.hidden_size,
        "vocab_size": lm.vocab_size,
        "layers": getattr(cfg, "num_hidden_layers", None),
        "attention_heads": getattr(cfg, "num_attention_heads", None),
        "kv_heads": getattr(cfg, "num_key_value_heads", None),
        "max_positions": getattr(cfg, "max_position_embeddings", None),
        "tied_embeddings": bool(tied),
        "norm_layers": norm_layers,
        "embedding_row_norm_mean": row_norms.mean().item(),
        "embedding_row_norm_std": row_norms.std().item(),
        "embedding_value_std": emb_in.float().std().item(),
        "has_chat_template": bool(getattr(lm.tok, "chat_template", None)),
        "start_token_id": lm.start_id,
        "device": str(lm.device),
        "dtype": str(lm.dtype).replace("torch.", ""),
        "self_check": self_check(lm),
    }
    return info


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
