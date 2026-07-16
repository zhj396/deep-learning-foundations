"""Parallel-corpus cleaning: language ID, noise filters, deduplication.

Cleaning order is cheap-checks-first so the expensive fastText LID model
only sees pairs that already passed length/ASCII heuristics:

    empty/length -> ASCII heuristic (English side) -> batched fastText LID
    -> noise filters (URL/HTML/punctuation) -> length-ratio -> xxhash dedup

The LID predictor is injectable (any object exposing
``predict(list[str]) -> (langs, scores)``), which keeps the pipeline
unit-testable without fastText installed.
"""

import logging
import re
from typing import Iterable

import xxhash

logger = logging.getLogger(__name__)


# =========================================================
# fastText language identifier
# =========================================================
class FastTextLangID:
    """Thin batched wrapper around the fastText lid.176 model."""

    def __init__(self, model_path: str):
        try:
            import fasttext  # lazy: heavy dep, only needed to load the LID model
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "fastText is required for corpus cleaning but is not installed.\n"
                "Install the prebuilt-wheel package (covers current Python\n"
                "versions incl. 3.13; only data_prep/ needs fastText):\n"
                "    pip install fasttext-predict\n"
                "or:\n"
                "    conda install -c conda-forge fasttext"
            ) from e
        self.model = fasttext.load_model(str(model_path))

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        # One line per call: the multiline return contract differs across
        # fasttext builds (labels-only vs labels+probs); the single-line
        # form (labels, probs) is stable everywhere.
        labels: list[str] = []
        scores: list[float] = []
        for text in texts:
            line_labels, line_probs = self.model.predict(text)
            labels.append(line_labels[0] if line_labels else "")
            scores.append(float(line_probs[0]) if line_probs else 0.0)
        langs = [label.removeprefix("__label__") for label in labels]
        return langs, scores


# =========================================================
# Noise filters
# =========================================================
URL_RE = re.compile(r"https?://|www\.")
HTML_RE = re.compile(r"<[^>]+>")


def is_noisy(text: str, max_punct_ratio: float = 0.3) -> bool:
    """True if the text contains URLs, HTML tags, or too much punctuation."""
    if URL_RE.search(text):
        return True
    if HTML_RE.search(text):
        return True

    punct = sum(not c.isalnum() and not c.isspace() for c in text)
    if punct / max(1, len(text)) > max_punct_ratio:
        return True

    return False


def length_ratio_ok(en: str, pt: str, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
    """True if the en/pt word-count ratio is within [min_ratio, max_ratio].

    Wildly different lengths usually mean mis-aligned sentence pairs.
    """
    len_en, len_pt = len(en.split()), len(pt.split())
    if len_pt == 0:
        return False
    return min_ratio <= len_en / len_pt <= max_ratio


# =========================================================
# Deduplication
# =========================================================
def pair_hash(en: str, pt: str) -> int:
    """Order-sensitive 64-bit hash of an (en, pt) pair."""
    h = xxhash.xxh64()
    h.update(en.encode("utf-8"))
    h.update(b"|||")
    h.update(pt.encode("utf-8"))
    return h.intdigest()


# =========================================================
# Main cleaning pipeline
# =========================================================
def clean_parallel_corpus(
    en: Iterable[str],
    pt: Iterable[str],
    lid,
    min_len: int = 3,
    max_len: int = 1024,
    lang_score_thresh: float = 0.8,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    max_punct_ratio: float = 0.3,
    batch_size: int = 4096,
) -> tuple[list[str], list[str]]:
    """Filter a raw parallel corpus down to clean, deduplicated pairs.

    Args:
        en, pt: line-aligned source/target sentences.
        lid: predictor with ``predict(list[str]) -> (langs, scores)``
            (e.g. FastTextLangID).
        Remaining args: filter thresholds, see module docstring.

    Returns:
        (clean_en, clean_pt) line-aligned lists.
    """
    seen: set[int] = set()
    clean_en: list[str] = []
    clean_pt: list[str] = []

    buf_en: list[str] = []
    buf_pt: list[str] = []

    stats = {"total": 0, "basic": 0, "ascii": 0, "lid": 0, "noise": 0,
             "ratio": 0, "dup": 0}

    def flush() -> None:
        if not buf_en:
            return

        en_langs, en_scores = lid.predict(buf_en)
        pt_langs, pt_scores = lid.predict(buf_pt)

        for e, p, le, lp, se, sp in zip(
            buf_en, buf_pt, en_langs, pt_langs, en_scores, pt_scores
        ):
            # language + confidence
            if le != "en" or lp != "pt" or se < lang_score_thresh or sp < lang_score_thresh:
                stats["lid"] += 1
                continue

            # noise (URL / HTML / punctuation ratio)
            if is_noisy(e, max_punct_ratio) or is_noisy(p, max_punct_ratio):
                stats["noise"] += 1
                continue

            # alignment sanity: en/pt length ratio
            if not length_ratio_ok(e, p, min_ratio, max_ratio):
                stats["ratio"] += 1
                continue

            # exact-duplicate pairs
            h = pair_hash(e, p)
            if h in seen:
                stats["dup"] += 1
                continue
            seen.add(h)

            clean_en.append(e)
            clean_pt.append(p)

        buf_en.clear()
        buf_pt.clear()

    for e, p in zip(en, pt):
        stats["total"] += 1
        e = e.strip()
        p = p.strip()

        # cheap checks first: empty lines and word-count bounds
        if not e or not p:
            stats["basic"] += 1
            continue
        if not (min_len <= len(e.split()) <= max_len):
            stats["basic"] += 1
            continue
        if not (min_len <= len(p.split()) <= max_len):
            stats["basic"] += 1
            continue

        # fast heuristic: the English side should be (near-)pure ASCII,
        # which removes most mis-labelled lines before paying for LID
        if not e.isascii():
            stats["ascii"] += 1
            continue

        buf_en.append(e)
        buf_pt.append(p)

        if len(buf_en) >= batch_size:
            flush()

    flush()

    kept = len(clean_en)
    logger.info(
        "Cleaning: %d in -> %d kept (%.1f%%) | dropped: basic=%d ascii=%d "
        "lid=%d noise=%d ratio=%d dup=%d",
        stats["total"], kept,
        100.0 * kept / max(1, stats["total"]),
        stats["basic"], stats["ascii"], stats["lid"], stats["noise"],
        stats["ratio"], stats["dup"],
    )
    return clean_en, clean_pt
