# -*- coding: utf-8 -*-
"""Standalone ONNX exporter: checkpoint -> encoder + decoder-step graphs.

Exports a trained checkpoint into the ONNX pair consumed by the inference
service's ONNX backend (../inference, BACKEND=onnx): the service then runs
greedy decoding under ONNX Runtime without importing torch.

I/O contract (dynamic batch and sequence axes everywhere):
    encoder.onnx      src_tokens int64 (B,S)      -> memory float32 (B,S,D)
    decoder_step.onnx ys int64 (B,T), memory (B,S,D),
                      src_tokens int64 (B,S)      -> logits float32 (B,vocab)

Design notes:
- The PAD mask is computed INSIDE each graph from pad_idx, baked in from the
  checkpoint config (data_prep always trains spm with pad/unk/bos/eos =
  0/1/2/3, so the config value is authoritative).
- The decoder step takes the whole generated sequence `ys` and returns the
  next-token logits for the last position only -- no look-ahead mask needed
  because `ys` only ever contains past tokens (the same invariant
  train/decoding.py relies on).
- No KV cache, mirroring decoding.py: each step recomputes the decoder stack
  over the growing sequence, O(T^2) per sequence.

Usage:
    python export_onnx.py --checkpoint <ckpt>.pt [--output-dir DIR] [--opset 17]
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from model import Transformer

logger = logging.getLogger("export")

# Opset 17: supported by any ONNX Runtime >= 1.14.
DEFAULT_OPSET = 17

# Model params read from the checkpoint's embedded config, with the
# paper-base defaults as fallback.
_CFG_KEYS = {
    "d_model": 512,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_len": 512,
    "tie_weights": True,
    "pad_idx": 0,
}


# -------------------------
# Export wrappers
# -------------------------
class EncoderWrapper(nn.Module):
    """encoder(src_tokens) with the pad mask derived inside the graph."""

    def __init__(self, transformer: Transformer, pad_idx: int):
        super().__init__()
        self.encoder = transformer.encoder
        self.pad_idx = pad_idx

    def forward(self, src_tokens: torch.Tensor) -> torch.Tensor:
        src_mask = src_tokens == self.pad_idx
        return self.encoder(src_tokens, src_mask)


class DecoderStepWrapper(nn.Module):
    """One decode step: (ys, memory, src_tokens) -> next-token logits."""

    def __init__(self, transformer: Transformer, pad_idx: int):
        super().__init__()
        self.decoder = transformer.decoder
        self.out = transformer.out
        self.pad_idx = pad_idx

    def forward(self, ys: torch.Tensor, memory: torch.Tensor,
                src_tokens: torch.Tensor) -> torch.Tensor:
        memory_mask = src_tokens == self.pad_idx
        # no look-ahead mask: ys only contains already-emitted tokens
        h = self.decoder(ys, memory, tgt_mask=None, memory_mask=memory_mask)
        return self.out(h[:, -1])


# -------------------------
# Model reconstruction
# -------------------------
def load_model_from_checkpoint(checkpoint_path: Path):
    """Rebuild the Transformer from a training checkpoint.

    The checkpoint payload is primitives + tensors only (train/checkpoint.py),
    so weights_only=True -- torch's safe loader -- works.
    Returns (model, config_dict).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                      weights_only=True)

    cfg = ckpt.get("config") or {}
    if "vocab_size" not in cfg:
        raise ValueError(
            f"Checkpoint config lacks vocab_size (saved by"
            f" train/checkpoint.py?): "
            f"{checkpoint_path}"
        )
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ValueError(f"Checkpoint has no model_state: {checkpoint_path}")

    params = {k: cfg.get(k, default) for k, default in _CFG_KEYS.items()}
    model = Transformer(
        src_vocab=cfg["vocab_size"],
        tgt_vocab=cfg["vocab_size"],
        d_model=params["d_model"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        max_len=params["max_len"],
        tie_weights=params["tie_weights"],
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, cfg


# -------------------------
# Export
# -------------------------
def export_checkpoint_to_onnx(
    checkpoint_path,
    output_dir=None,
    opset: int = DEFAULT_OPSET,
    validate: bool = True,
):
    """Export encoder.onnx + decoder_step.onnx next to (or into) the checkpoint.

    Returns (encoder_path, decoder_path).
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = (Path(output_dir) if output_dir is not None
                  else checkpoint_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_from_checkpoint(checkpoint_path)
    pad_idx = int(cfg.get("pad_idx", 0))
    vocab_size = int(cfg["vocab_size"])
    d_model = model.encoder.embed.embedding_dim

    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder_step.onnx"

    # dummy inputs (values irrelevant; shapes drive the dynamic-axis export)
    lo = max(4, vocab_size // 2)
    hi = max(8, vocab_size)
    dummy_src = torch.randint(lo, hi, (2, 5), dtype=torch.long)
    dummy_ys = torch.randint(lo, hi, (2, 3), dtype=torch.long)
    dummy_memory = torch.randn(2, 5, d_model)

    logger.info("Exporting encoder -> %s (opset %d)", encoder_path, opset)
    torch.onnx.export(
        EncoderWrapper(model, pad_idx),
        (dummy_src,),
        str(encoder_path),
        input_names=["src_tokens"],
        output_names=["memory"],
        dynamic_axes={
            "src_tokens": {0: "batch", 1: "src_seq"},
            "memory": {0: "batch", 1: "src_seq"},
        },
        opset_version=opset,
    )

    logger.info("Exporting decoder step -> %s (opset %d)", decoder_path, opset)
    torch.onnx.export(
        DecoderStepWrapper(model, pad_idx),
        (dummy_ys, dummy_memory, dummy_src),
        str(decoder_path),
        input_names=["ys", "memory", "src_tokens"],
        output_names=["logits"],
        dynamic_axes={
            "ys": {0: "batch", 1: "tgt_seq"},
            "memory": {0: "batch", 1: "src_seq"},
            "src_tokens": {0: "batch", 1: "src_seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )

    if validate:
        _validate(encoder_path, decoder_path, dummy_src, dummy_ys)

    return encoder_path, decoder_path


def _validate(encoder_path: Path, decoder_path: Path, dummy_src, dummy_ys):
    """Structure check (onnx) + one ORT round-trip, both optional."""
    try:
        import onnx
    except ImportError:
        logger.warning("onnx package not installed; skipping structure check")
    else:
        for p in (encoder_path, decoder_path):
            onnx.checker.check_model(onnx.load(str(p)))
        logger.info("onnx.checker passed for both graphs")

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping smoke run")
        return

    providers = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(encoder_path), providers=providers)
    dec = ort.InferenceSession(str(decoder_path), providers=providers)
    memory = enc.run(None, {"src_tokens": dummy_src.numpy()})[0]
    logits = dec.run(
        None,
        {"ys": dummy_ys.numpy(), "memory": memory,
         "src_tokens": dummy_src.numpy()},
    )[0]
    logger.info("ORT smoke run OK: memory %s, logits %s",
                memory.shape, logits.shape)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Export a trained pt->en Transformer checkpoint to ONNX "
                    "(encoder + decoder-step graphs for inference/)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="training checkpoint (.pt) produced by train/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="destination dir (default: checkpoint dir)")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                        help="ONNX opset")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip onnx.checker / ORT smoke run")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    encoder_path, decoder_path = export_checkpoint_to_onnx(
        args.checkpoint, args.output_dir,
        opset=args.opset, validate=not args.no_validate,
    )
    print(f"ONNX export completed:\n  {encoder_path}\n  {decoder_path}")


if __name__ == "__main__":
    main()
