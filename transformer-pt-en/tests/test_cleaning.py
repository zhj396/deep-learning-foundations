"""Corpus-cleaning filters with a fake LID predictor (no fastText needed)."""

import pytest

pytest.importorskip("xxhash")

from clean import clean_parallel_corpus, is_noisy, length_ratio_ok, pair_hash


def test_is_noisy():
    assert is_noisy("see http://example.com for details")
    assert is_noisy("click www.example.com now")
    assert is_noisy("<b>bold</b> text here")
    assert is_noisy("!!!...???###%%%")
    assert not is_noisy("this is a perfectly clean sentence")


def test_length_ratio_boundaries():
    # ratios exactly at the bounds are kept
    assert length_ratio_ok("a b c d", "x y", min_ratio=0.5, max_ratio=2.0)  # 2.0
    assert length_ratio_ok("a b", "x y z w", min_ratio=0.5, max_ratio=2.0)       # 0.5
    assert not length_ratio_ok("a b c d e", "x y", min_ratio=0.5, max_ratio=2.0)  # 2.5
    assert not length_ratio_ok("a b", "", min_ratio=0.5, max_ratio=2.0)          # empty pt


def test_pair_hash_deterministic_and_order_sensitive():
    h1 = pair_hash("hello world", "ola mundo")
    assert h1 == pair_hash("hello world", "ola mundo")
    assert h1 != pair_hash("ola mundo", "hello world")


class FakeLID:
    """Language predictor keyed by exact text; unknown text -> ('en', 0.99)."""

    def __init__(self, overrides):
        self.overrides = overrides

    def predict(self, texts):
        langs, scores = [], []
        for t in texts:
            lang, score = self.overrides.get(t, ("en", 0.99))
            langs.append(lang)
            scores.append(score)
        return langs, scores


def test_clean_parallel_corpus_filters():
    en = [
        "the cat sat on the mat",                        # keep
        "keep this clean pair",                          # keep
        "keep this clean pair",                          # duplicate -> drop
        "bonjour pas anglais du tout ici",               # wrong language -> drop
        "http://spam.com is here now",                   # noisy URL -> drop
        "low confidence sentence right here",            # LID score < 0.8 -> drop
        "short",                                         # below min_len -> drop
        "one two three four five six seven eight nine",  # length ratio -> drop
    ]
    pt = [
        "o gato sentou no tapete",
        "mantenha este par limpo",
        "mantenha este par limpo",
        "bonjour pas anglais du tout ici",
        "http://spam.com esta aqui agora",
        "frase de baixa confianca aqui",
        "curto",
        "palavra um dois",
    ]
    lid = FakeLID({
        "o gato sentou no tapete": ("pt", 0.99),
        "mantenha este par limpo": ("pt", 0.99),
        "http://spam.com esta aqui agora": ("pt", 0.99),
        "frase de baixa confianca aqui": ("pt", 0.99),
        "curto": ("pt", 0.99),
        "palavra um dois": ("pt", 0.99),
        "bonjour pas anglais du tout ici": ("fr", 0.99),
        "low confidence sentence right here": ("en", 0.5),
    })

    clean_en, clean_pt = clean_parallel_corpus(
        en, pt, lid=lid,
        min_len=3, max_len=1024,
        lang_score_thresh=0.8,
        batch_size=2,  # force multiple flushes; dedup must span batches
    )

    assert clean_en == ["the cat sat on the mat", "keep this clean pair"]
    assert clean_pt == ["o gato sentou no tapete", "mantenha este par limpo"]


def test_clean_parallel_corpus_empty_input():
    lid = FakeLID({})
    clean_en, clean_pt = clean_parallel_corpus([], [], lid=lid)
    assert clean_en == []
    assert clean_pt == []
