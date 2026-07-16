"""SentencePiece BPE vocabulary training.

Trains a single shared pt/en model with a fixed special-token layout:
pad=0, unk=1, bos=2, eos=3.

Only SentencePiece is supported on purpose: it has first-class
Windows/Linux wheels, and its C++ trainer is fast on CPU.
"""

import logging
from pathlib import Path

import sentencepiece as spm

logger = logging.getLogger(__name__)

MODEL_PREFIX = "spm_en_pt"


def train_spm(
    input_file: Path,
    out_dir: Path,
    vocab_size: int = 32_000,
) -> Path:
    """Train a BPE model; returns the path to the .model file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / MODEL_PREFIX

    logger.info(
        "Training SentencePiece BPE (vocab=%d) on %s", vocab_size, input_file
    )
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        character_coverage=1.0,
    )

    model_path = Path(f"{prefix}.model")
    logger.info("Vocabulary trained: %s", model_path)
    return model_path
