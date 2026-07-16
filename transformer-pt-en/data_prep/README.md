# Data Prep: OPUS Corpus → Shared pt/en BPE Vocabulary

CPU-only pipeline that turns a raw OPUS parallel corpus into the cleaned
splits and a shared pt/en SentencePiece vocabulary for training.

```
OPUS zip (moses en-pt)          data/<corpus>/raw/            data/<corpus>/vocab/
┌────────────────────┐  extract ┌──────────────┐  clean+split ┌──────────────────────┐
│ prepare.py         │────────▶ │ *.en-pt.en   │────────────▶ │ train/valid/test.en  │
│  download (urllib) │          │ *.en-pt.pt   │              │ train/valid/test.pt  │
│  lid.176.bin (126MB│          └──────────────┘              │ train.merged         │
│  sha256-verified)  │                                        │ spm_en_pt.model/.vocab│
└────────────────────┘                                        └──────────────────────┘
```

## Corpora (`--dataset`)

| Name | Sentences | Domain |
|---|---|---|
| `Books` | 1,404 | literature |
| `TED2013` | 155,995 | talks |
| `KDE4` | 229,418 | software docs |
| `TED2020` | 328,791 | talks |
| `wikimedia21` | 506,289 | wikimedia |
| `wikimedia23` | 821,837 | wikimedia |
| `EMEA` (default) | 1,082,144 | medical/legal EU texts |

All URLs point at the OPUS mirror (object.pouta.csc.fi), Moses format,
`en-pt` direction. Downloads are cached under `data/` (gitignored): the
fastText LID model is verified by size + sha256, corpus zips by a
zip-integrity probe (re-downloaded if invalid).

## Cleaning rules (cheap checks first)

1. **Basic** — non-empty, word count within `[--min_len, --max_len]`
2. **ASCII heuristic** — English side must be ASCII (kills most mislabelled
   lines before paying for LID)
3. **fastText LID** — batched (`--batch_size 4096`) `lid.176` predictions;
   both sides must match en/pt with confidence ≥ `--lang_score_thresh 0.8`
4. **Noise** — URLs, HTML tags, punctuation ratio > `--max_punct_ratio 0.3`
5. **Alignment** — en/pt word-count ratio within `[--min_ratio, --max_ratio]`
6. **Dedup** — exact duplicate pairs via xxhash64

The LID predictor is injectable (`clean.clean_parallel_corpus(lid=...)`), so
the pipeline is unit-testable without fastText installed.

## Usage

```bash
pip install -r requirements.txt
python prepare.py --dataset EMEA --vocab_size 32000
# first run also downloads lid.176.bin (~126 MB) into data/models/
```

Output: `data/<corpus>/vocab/{train,valid,test}.{en,pt}` (95/2.5/2.5 split,
deterministic via `--seed`), `train.merged`, and `spm_en_pt.model/.vocab`
(BPE, shared pt+en vocab, pad/unk/bos/eos = 0/1/2/3).

## Notes

- **`.pt` files are Portuguese text**, not PyTorch checkpoints (OPUS convention).
- **fastText**: `requirements.txt` pins `fasttext-predict`, which ships
  prebuilt wheels for current Python versions (incl. 3.13) and exposes the
  same `load_model()`/`predict()` surface — this pipeline only needs LID
  prediction, not fastText training. If it is unavailable on your platform,
  `conda install -c conda-forge fasttext` provides the same API.
- **SentencePiece only**: SentencePiece has first-class Windows/Linux
  wheels and its C++ trainer is fast on CPU.
- No `wget`/`unzip` binaries and no HuggingFace `datasets` dependency —
  downloads/extraction use `urllib` + `zipfile`, so the pipeline behaves
  identically on Windows, Linux and macOS.
