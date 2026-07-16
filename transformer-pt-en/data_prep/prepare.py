"""Data-prep entry point: OPUS corpus -> clean pairs -> splits -> BPE vocab.

Usage (local CPU):
    python prepare.py --dataset EMEA --vocab_size 32000

Outputs under data/<dataset>/vocab/:
    train.en / train.pt / valid.* / test.*   line-aligned splits (95/2.5/2.5)
    train.merged                             both languages, shuffled (vocab input)
    spm_en_pt.model / .vocab                 shared SentencePiece BPE

Note: the .en/.pt split files are plain TEXT (English / Portuguese),
not PyTorch checkpoints.
"""

import logging
import sys
from pathlib import Path

from clean import FastTextLangID, clean_parallel_corpus
from download import (
    download_corpus,
    ensure_lid_model,
    extract_corpus,
    read_lines,
)
from prep_config import DATASETS, PrepConfig
from split import merge_and_shuffle, save_parallel, shuffle_split
from train_vocab import train_spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_raw_pairs(cfg: PrepConfig) -> tuple[list[str], list[str]]:
    """Download (cached), extract and read the line-aligned en/pt corpus."""
    zip_path = download_corpus(DATASETS[cfg.dataset], cfg.corpus_dir)
    extract_dir = extract_corpus(zip_path, cfg.corpus_dir / "raw")

    # wikimedia zips extract to "wikimedia.en-pt.{en,pt}"; others keep
    # their registry name as the file stem
    stem = "wikimedia" if cfg.dataset.startswith("wikimedia") else cfg.dataset

    en = read_lines(extract_dir / f"{stem}.en-pt.en")
    pt = read_lines(extract_dir / f"{stem}.en-pt.pt")

    if len(en) != len(pt):
        raise RuntimeError(
            f"Corpus sides differ in length: en={len(en)} pt={len(pt)}"
        )
    logger.info("Loaded %d raw sentence pairs from %s", len(en), cfg.dataset)
    return en, pt


def main() -> None:
    # Windows consoles may default to a legacy code page; corpus text is
    # UTF-8 with diacritics, so make stdout tolerant instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = PrepConfig.from_cli()
    logger.info("Config: %s", cfg)

    en, pt = load_raw_pairs(cfg)

    lid_path: Path = ensure_lid_model(cfg.models_dir)
    lid = FastTextLangID(lid_path)

    logger.info("Cleaning sentence pairs...")
    clean_en, clean_pt = clean_parallel_corpus(
        en,
        pt,
        lid=lid,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        lang_score_thresh=cfg.lang_score_thresh,
        min_ratio=cfg.min_ratio,
        max_ratio=cfg.max_ratio,
        max_punct_ratio=cfg.max_punct_ratio,
        batch_size=cfg.batch_size,
    )
    if not clean_en:
        raise RuntimeError("No valid sentence pairs after cleaning")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, valid_pairs, test_pairs = shuffle_split(
        list(zip(clean_en, clean_pt)), seed=cfg.seed
    )
    logger.info(
        "Split: train=%d valid=%d test=%d",
        len(train_pairs), len(valid_pairs), len(test_pairs),
    )

    save_parallel(train_pairs, cfg.out_dir, "train")
    save_parallel(valid_pairs, cfg.out_dir, "valid")
    save_parallel(test_pairs, cfg.out_dir, "test")

    merged = cfg.out_dir / "train.merged"
    merge_and_shuffle(
        cfg.out_dir / "train.en",
        cfg.out_dir / "train.pt",
        merged,
        seed=cfg.seed,
    )

    logger.info("Training tokenizer...")
    train_spm(merged, cfg.out_dir, vocab_size=cfg.vocab_size)

    logger.info("Done. Vocab dir: %s", cfg.out_dir)


if __name__ == "__main__":
    main()
