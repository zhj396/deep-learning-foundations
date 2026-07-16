"""Download helpers: OPUS corpus zips and the fastText LID model.

Uses only the standard library (urllib + zipfile) so the pipeline runs
unchanged on Windows, Linux and macOS -- no wget/unzip binaries required.
"""

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

from prep_config import LID_MODEL_SHA256, LID_MODEL_SIZE, LID_MODEL_URL

logger = logging.getLogger(__name__)


def _download(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Download url to dest with a simple progress log."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    logger.info("Downloading %s -> %s", url, dest)

    def _report(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 2000 == 0:
            done = min(blocks * block_size, total)
            logger.info("  ... %d%%", 100 * done // total)

    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    tmp.replace(dest)

    actual = dest.stat().st_size
    if expected_size is not None and actual != expected_size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Size mismatch for {dest}: expected {expected_size} bytes, got {actual}"
        )
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_lid_model(models_dir: Path) -> Path:
    """Return the path to fastText lid.176.bin, downloading it if needed.

    A fresh download is verified by file size and sha256 so a truncated
    or corrupted download is never silently used. A previously verified
    file is trusted on a size check alone (re-hashing 126 MB on every
    run is not worth it).
    """
    path = models_dir / "lid.176.bin"
    if path.exists() and path.stat().st_size == LID_MODEL_SIZE:
        logger.info("LID model already present: %s", path)
        return path

    _download(LID_MODEL_URL, path, expected_size=LID_MODEL_SIZE)

    digest = _sha256(path)
    if digest != LID_MODEL_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for lid.176.bin: expected {LID_MODEL_SHA256}, got {digest}"
        )
    logger.info("LID model verified (sha256 OK): %s", path)
    return path


def _valid_zip(path: Path) -> bool:
    """True if path is a readable zip archive (zipfile central directory)."""
    try:
        return path.is_file() and zipfile.is_zipfile(path)
    except OSError:
        return False


def download_corpus(url: str, corpus_dir: Path) -> Path:
    """Download an OPUS moses zip into corpus_dir (cached between runs).

    A cached zip is only trusted after a zip-integrity probe; a
    truncated or corrupted file is re-downloaded.
    """
    zip_path = corpus_dir / "data.zip"
    if zip_path.exists():
        if _valid_zip(zip_path):
            logger.info("Corpus zip already present: %s", zip_path)
            return zip_path
        logger.warning("Invalid cached zip, re-downloading: %s", zip_path)
        zip_path.unlink()
    return _download(url, zip_path)


def extract_corpus(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the corpus zip into extract_dir (atomic)."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("Corpus already extracted: %s", extract_dir)
        return extract_dir

    # extract to a temp dir first, then move into place, so an
    # interrupted extraction never leaves a partial extract_dir that a
    # later run would trust
    tmp_dir = extract_dir.with_name(extract_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    logger.info("Extracting %s -> %s", zip_path, extract_dir)
    try:
        tmp_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        if extract_dir.exists():
            # only reachable for an empty leftover dir (non-empty dirs
            # returned early); rmdir raises rather than delete data
            extract_dir.rmdir()
        tmp_dir.replace(extract_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return extract_dir


def read_lines(path: Path) -> list[str]:
    """Read a text file as UTF-8 lines with trailing newlines removed."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
