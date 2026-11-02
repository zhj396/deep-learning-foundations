"""Parallel-corpus dataset and batch collation.

Special-token ids live on the dataset INSTANCE (read once from the
processor) rather than in module-level globals: instance state is safe
with DataLoader worker processes and supports processors that use
non-default ids.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """Line-aligned src/tgt text files encoded with a SentencePiece-like processor.

    The processor is duck-typed: anything exposing ``encode(text,
    out_type=int) -> list[int]``, ``decode(ids) -> str`` and the callables
    ``pad_id/unk_id/bos_id/eos_id`` works, so callers can inject a fake.
    """

    def __init__(
        self,
        src_file,
        tgt_file,
        sp_processor=None,
        sp_model_path: Optional[str] = None,
    ):
        self.src = Path(src_file).read_text(encoding="utf-8").splitlines()
        self.tgt = Path(tgt_file).read_text(encoding="utf-8").splitlines()
        if len(self.src) != len(self.tgt):
            raise ValueError(
                f"Source and target files differ in length: "
                f"{len(self.src)} vs {len(self.tgt)}"
            )

        if sp_processor is not None:
            self.sp = sp_processor
        elif sp_model_path is not None:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        else:
            raise ValueError("Either sp_processor or sp_model_path must be provided")

        self.pad_idx = self._special_id("pad_id", 0)
        self.unk_idx = self._special_id("unk_id", 1)
        self.bos_idx = self._special_id("bos_id", 2)
        self.eos_idx = self._special_id("eos_id", 3)

    def _special_id(self, name: str, fallback: int) -> int:
        getter = getattr(self.sp, name, None)
        if getter is None:
            return fallback
        idx = getter()
        return fallback if idx is None or idx < 0 else idx

    def encode(self, text: str) -> List[int]:
        """Text -> [BOS] + ids + [EOS]."""
        ids = self.sp.encode(text.strip(), out_type=int)
        return [self.bos_idx] + ids + [self.eos_idx]

    def decode(self, ids: List[int]) -> str:
        """Ids -> text, dropping special tokens."""
        specials = {self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx}
        return self.sp.decode([i for i in ids if i not in specials])

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.encode(self.src[idx]), self.encode(self.tgt[idx])


class Collator:
    """Pad (and optionally truncate) encoded pairs into rectangular tensors.

    All state lives on the instance (pad/eos ids, max_len) rather than in
    module globals, which is safe to ship to DataLoader worker processes.
    """

    def __init__(self, pad_idx: int, eos_idx: int, max_len: Optional[int] = None):
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

    def _truncate(self, ids: List[int]) -> List[int]:
        if self.max_len is not None and len(ids) > self.max_len:
            # keep room for EOS so decoding still terminates
            return ids[: self.max_len - 1] + [self.eos_idx]
        return list(ids)

    def _pad(self, seqs: List[List[int]]) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        padded = [s + [self.pad_idx] * (width - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_batch = [self._truncate(src) for src, _ in batch]
        tgt_batch = [self._truncate(tgt) for _, tgt in batch]
        return self._pad(src_batch), self._pad(tgt_batch)
