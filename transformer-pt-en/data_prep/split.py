"""Deterministic train/valid/test splitting and file writing."""

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.95
VALID_RATIO = 0.025
# test gets the remainder (0.025)


def shuffle_split(
    pairs: list[tuple[str, str]],
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Shuffle deterministically and split into (train, valid, test)."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    t = int(TRAIN_RATIO * n)
    v = int(VALID_RATIO * n)
    return shuffled[:t], shuffled[t:t + v], shuffled[t + v:]


def save_parallel(
    pairs: list[tuple[str, str]],
    out_dir: Path,
    prefix: str,
) -> None:
    """Write line-aligned <prefix>.en / <prefix>.pt files."""
    en_path = out_dir / f"{prefix}.en"
    pt_path = out_dir / f"{prefix}.pt"

    with open(en_path, "w", encoding="utf-8") as fe, \
         open(pt_path, "w", encoding="utf-8") as fp:
        for e, p in pairs:
            fe.write(e + "\n")
            fp.write(p + "\n")

    logger.info("Wrote %d pairs -> %s / %s", len(pairs), en_path, pt_path)


def merge_and_shuffle(
    en_file: Path,
    pt_file: Path,
    out_file: Path,
    seed: int,
) -> None:
    """Merge both languages into one shuffled file (tokenizer training input).

    Training the vocabulary on both sides of the corpus yields a single
    shared pt/en BPE model, which halves embedding parameters and lets the
    model tie encoder/decoder vocabularies.
    """
    with open(en_file, encoding="utf-8") as fe, \
         open(pt_file, encoding="utf-8") as fp:
        lines = list(fe) + list(fp)

    random.Random(seed).shuffle(lines)

    with open(out_file, "w", encoding="utf-8") as fo:
        for line in lines:
            fo.write(line.rstrip() + "\n")

    logger.info("Merged %d lines -> %s", len(lines), out_file)
