"""Corpus registry and configuration for the data-preparation pipeline.

The pipeline builds a shared Portuguese/English SentencePiece BPE vocabulary
from OPUS parallel corpora. Runs entirely on local CPU.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path


# =========================================================
# Dataset registry (OPUS, Moses format, en-pt direction)
# =========================================================
DATASETS = {
    "Books": "https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-pt.txt.zip",  # 1,404 sentences
    "TED2013": ("https://object.pouta.csc.fi/OPUS-TED2013/v1.1/"
                "moses/en-pt.txt.zip"),  # 155,995 sentences
    "KDE4": "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-pt.txt.zip",  # 229,418 sentences
    "TED2020": ("https://object.pouta.csc.fi/OPUS-TED2020/v1/"
                "moses/en-pt.txt.zip"),  # 328,791 sentences
    "wikimedia21": ("https://object.pouta.csc.fi/OPUS-wikimedia/v20210402/"
                    "moses/en-pt.txt.zip"),  # 506,289 sentences
    "wikimedia23": ("https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/"
                    "moses/en-pt.txt.zip"),  # 821,837 sentences
    "EMEA": "https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-pt.txt.zip",  # 1,082,144 sentences
}

# fastText supervised language-identification model (lid.176).
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MODEL_SIZE = 131_266_198
LID_MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"


@dataclass
class PrepConfig:
    """All tunables for one data-prep run."""

    # Corpus selection
    dataset: str = "EMEA"
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Vocabulary
    vocab_size: int = 32_000

    # Sentence-pair filters
    min_len: int = 3            # min words per sentence
    max_len: int = 1024         # max words per sentence
    lang_score_thresh: float = 0.8   # min fastText LID confidence
    min_ratio: float = 0.5      # min en/pt word-count ratio
    max_ratio: float = 2.0      # max en/pt word-count ratio
    max_punct_ratio: float = 0.3

    # Pipeline behaviour
    batch_size: int = 4096      # LID prediction batch size
    seed: int = 42

    # Derived paths (filled in __post_init__)
    corpus_dir: Path = field(init=False)   # raw + cleaned corpus files
    out_dir: Path = field(init=False)      # splits + trained tokenizer
    models_dir: Path = field(init=False)   # downloaded LID model

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.corpus_dir = self.data_dir / self.dataset
        self.out_dir = self.corpus_dir / "vocab"
        self.models_dir = self.data_dir / "models"

    @staticmethod
    def from_cli(argv=None) -> "PrepConfig":
        parser = argparse.ArgumentParser(
            description="Download, clean and split an OPUS en-pt corpus, "
                        "then train a shared SentencePiece BPE vocabulary."
        )
        parser.add_argument("--dataset", choices=sorted(DATASETS), default="EMEA")
        parser.add_argument("--data_dir", type=Path, default=Path("data"))
        parser.add_argument("--vocab_size", type=int, default=32_000)
        parser.add_argument("--min_len", type=int, default=3)
        parser.add_argument("--max_len", type=int, default=1024)
        parser.add_argument("--lang_score_thresh", type=float, default=0.8)
        parser.add_argument("--min_ratio", type=float, default=0.5)
        parser.add_argument("--max_ratio", type=float, default=2.0)
        parser.add_argument("--max_punct_ratio", type=float, default=0.3)
        parser.add_argument("--batch_size", type=int, default=4096)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(argv)
        return PrepConfig(**vars(args))
