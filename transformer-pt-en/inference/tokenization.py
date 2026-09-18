# -*- coding: utf-8 -*-
"""Tokenization contract for the translation service.

Pure sentencepiece: NO torch import, so ONNX-backend deployments stay
framework-free.

Mirrors train/dataset.py exactly:
    encode(text) -> [BOS] + ids + [EOS]
        (training wrapped the source side the same way, so inference must)
    decode(ids)  -> text, cut at the first EOS, special tokens stripped
        (mirrors TranslationDataset.decode + the strip() in evaluate.compute_bleu)

Special-token ids are read from the processor itself -- never hardcoded --
with the same fallback logic as TranslationDataset._special_id.
"""

from typing import Dict, List


def _special_id(processor, name: str, fallback: int) -> int:
    """Read a special-token id from a SentencePiece-like processor."""
    getter = getattr(processor, name, None)
    if getter is None:
        return fallback
    idx = getter()
    return fallback if idx is None or idx < 0 else idx


class Translator:
    """Text <-> token-id conversion around a SentencePiece model.

    The processor is duck-typed (encode/decode + *_id callables) so callers
    can inject a fake.
    """

    def __init__(self, sp_model_path=None, processor=None):
        if processor is not None:
            self.sp = processor
        elif sp_model_path is not None:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        else:
            raise ValueError("Either sp_model_path or processor must be provided")

        self.pad_idx = _special_id(self.sp, "pad_id", 0)
        self.unk_idx = _special_id(self.sp, "unk_id", 1)
        self.bos_idx = _special_id(self.sp, "bos_id", 2)
        self.eos_idx = _special_id(self.sp, "eos_id", 3)

    def encode(self, text: str) -> List[int]:
        """Text -> [BOS] + ids + [EOS] (identical to training-time src encoding)."""
        ids = self.sp.encode(text.strip(), out_type=int)
        return [self.bos_idx] + ids + [self.eos_idx]

    def decode(self, ids) -> str:
        """Ids -> text: stop at the first EOS, drop all special tokens."""
        specials = {self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx}
        kept = []
        for t in ids:
            t = int(t)
            if t == self.eos_idx:
                break
            if t not in specials:
                kept.append(t)
        return self.sp.decode(kept)

    def decode_params(self) -> Dict[str, int]:
        """Special ids as kwargs for backends.DecodeParams."""
        return {"bos_idx": self.bos_idx, "eos_idx": self.eos_idx, "pad_idx": self.pad_idx}
