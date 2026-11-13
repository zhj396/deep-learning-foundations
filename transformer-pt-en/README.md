# Transformer pt→en: From Raw Corpus to BLEU-driven Training

A from-scratch reproduction of *Attention Is All You Need* (Vaswani et al.,
2017) for Portuguese→English machine translation — covering the full path
from raw OPUS parallel corpora through BLEU-driven multi-GPU DDP training
to a pluggable serving stack (PyTorch / ONNX Runtime). Nothing is imported
from `torch.nn.Transformer`; attention, masking, positional encoding, label
smoothing, the Noam schedule, and greedy/beam decoding are all implemented
directly.

```
┌─────────────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────────┐
│ data_prep/  (local CPU)     │    │ data/<corpus>/vocab/ │    │ train/  (Kaggle T4 x2, DDP)  │
│  OPUS en-pt zip (7 corpora) │    │  train.en / train.pt │    │  spm encode -> Transformer   │
│  fastText LID >= 0.8        │───▶│  valid.* / test.*    │───▶│  Noam + AMP + grad-accum     │
│  noise/ratio/dedup filters  │    │  (95 / 2.5 / 2.5)    │    │  BLEU controller:            │
│  SentencePiece BPE 16k      │    │  spm_en_pt.model     │    │   LS decay, LR scale, early  │
│  (shared pt+en vocab)       │    │  train.merged        │    │  stop -> checkpoints/*.pt    │
└─────────────────────────────┘    └──────────────────────┘    │  + training_history.png      │
                                                               └──────────────────────────────┘

                                              artifacts/<model>/
┌──────────────────────────────┐   export     ┌─────────────────────────┐     ┌────────────────────────────┐
│ train/checkpoints/           │────────────▶ │ checkpoint.pt           │────▶│ inference/  (local CPU)    │
│  best_model_bleu_epoch_N.pt  │    copy      │ encoder.onnx            │     │  Flask UI + REST + CLI     │
│  latest_model.pt ────────────┼────────────▶ │ decoder_step.onnx       │     │  torch | ONNX backends     │
└──────────────────────────────┘  + spm model │ spm_en_pt.model         │     │  dynamic batching          │
                                              └─────────────────────────┘     └────────────────────────────┘
```

## Architecture

The paper's architecture at a scale matched to the data: 6+6 Pre-LN blocks,
head_dim 64, tied decoder-embedding ↔ output projection, one shared
SentencePiece BPE vocabulary for pt+en — **19,268,224 parameters**.

| d_model | layers | heads | d_ff | max_len | dropout | label smoothing | optimizer |
|---|---|---|---|---|---|---|---|
| 256 | 6+6 | 4 | 1024 | 128 | 0.1 | 0.1 (decaying) | Adam β=(0.9, 0.98), ε=1e-9 + Noam |

### Deviations from the paper

| Deviation | Rationale |
|---|---|
| **Pre-LN** blocks + **final LayerNorm** instead of Post-LN | Pre-LN trains stably without fragile warmup tuning (Xiong et al., 2020); the terminal LN is standard Pre-LN practice (GPT-2, fairseq) and matters here because the output projection is tied to the decoder embedding |
| **Shared 16k spm BPE vocab** for pt+en, tied embed↔out | Paper used separate ~37k wordpiece vocabs; a shared vocab is the modern norm for same-alphabet pairs and halves embedding parameters. The size is set by the corpus: past ~16k the added rows are phrases this corpus rarely (or never) contains, so their embeddings — shared with the output projection — stay near their initialization |
| **Model width**: d_model 256 / d_ff 1024, ~19M parameters | d_model 512 was trained on ~4.5M pairs; on a few hundred thousand pairs that capacity is mostly spent fitting noise. 256 puts the model near one parameter per ten training tokens while keeping the paper's head_dim (64) and the 6+6 depth |
| **Noam warmup** `--warmup_steps 1500`, not 4000 | Noam counts optimizer *updates*, and one update here covers `batch_size × world_size × accumulation_steps` = 512 sentences, so a 30-epoch run is ~15k updates. 4000 was ~4% of the paper's schedule and would be ~26% of this one; Pre-LN tolerates the shorter ramp |
| **Dataset**: OPUS en-pt (EMEA / TED / wikimedia), not WMT'14 En↔De | BLEU is **not comparable** to the paper's 27.3/28.4 — the gap analysis is qualitative (curve shape, effect of data scale and step count) |
| **Dropout** applied to embeddings **+** positional encoding | The paper's P_drop regularizes the sum; dropping before PE would leave positions unregularized |
| **Scale**: T4×2 + gradient accumulation, far fewer than 100k–300k steps | Hobby compute budget; effective batch raised via `--accumulation_steps` |

Faithful to the paper: scaled dot-product attention, multi-head
split/concat, sinusoidal positional encoding, Noam LR formula, Adam
hyperparameters, label smoothing 0.1, residual dropout placement inside
sublayers.

## Results

Selection — best checkpoint, label-smoothing and LR decays, early stop —
reads a fixed **validation** probe; the **test** split is decoded once
after training, on the checkpoint the probe picked, so the quoted BLEU is
not the maximum of a search over it. Seed 41. *Table fills in as training
runs complete.*

| Metric | Value |
|---|---|
| Test BLEU (full split, greedy) | _TBD_ |
| Test BLEU (full split, beam 4 / length penalty 0.6) | _TBD_ |
| Best validation probe BLEU (the selection signal) | _TBD_ |
| Best val loss / token acc | _TBD_ |
| Epochs to best | _TBD_ |
| Reference: paper base model (WMT'14 En→De) | 27.3 (not comparable — different data/scale) |

## Quick Start

```bash
# 1) Data prep (local CPU; first run downloads lid.176.bin, ~126 MB)
cd data_prep
pip install -r requirements.txt
python prepare.py --dataset EMEA --vocab_size 16000
# -> data/EMEA/vocab/{train,valid,test}.{en,pt} + spm_en_pt.model

# 2) Train (Kaggle T4 x2: upload the vocab dir as a Dataset input)
cd ../train
pip install -r requirements.txt
python main.py --data_vocab <vocab_dir> --epochs 30 --use_amp \
    --batch_size 64 --accumulation_steps 4
# >1 CUDA device -> automatic mp.spawn DDP (NCCL); 1 GPU/CPU -> single process

# CPU smoke run
python main.py --data_vocab <vocab_dir> --no_cuda \
    --d_model 128 --num_layers 2 --epochs 1

# Resume / beam-search probes
python main.py --data_vocab <vocab_dir> --resume
python main.py --data_vocab <vocab_dir> --use_beam --beam_size 4 \
    --length_penalty 0.6 --bleu_interval 2 --bleu_batches 20

# 3) Tests (CPU-only, no downloads, seconds)
cd ..
python -m pytest tests -q

# 4) Serve (local CPU)
mkdir -p artifacts/base
cp <vocab_dir>/spm_en_pt.model artifacts/base/
cp train/checkpoints/best_model_bleu_epoch_N.pt artifacts/base/checkpoint.pt
python train/export_onnx.py --checkpoint artifacts/base/checkpoint.pt  # optional ONNX path
cd inference
pip install -r requirements.txt
BACKEND=torch python app.py          # http://127.0.0.1:5000 (BACKEND=onnx: torch-free)
python translate.py -e "o gato esta aqui"
```

Details: [`data_prep/README.md`](data_prep/README.md) ·
[`train/README.md`](train/README.md) · [`inference/README.md`](inference/README.md)

## Implementation notes

- **BLEU-driven controller** — every N epochs the ranks decode strided
  shards of a fixed **validation** probe and rank 0 scores the union with
  sacrebleu; a gain must clear `bleu_min_delta` to count, since corpus BLEU
  over a few hundred sentences has a noise band of its own. Stagnation
  halves label smoothing (after 2 misses) and the LR scale (floor
  `lr_min_scale`), patience triggers early stop. The LR scale lives
  *inside* the Noam schedule, so it survives subsequent steps; controller
  state goes to all ranks in one broadcast and a failed shard is voted
  through a MIN-reduce, so all ranks exit together. Ranks idle during the
  probe block in those collectives, so `--bleu_max_len` caps its generation
  length and the process-group timeout sits above a whole probe.
- **Selection vs reporting** — the test split never enters the training
  loop: one pass after training decodes all of it at full `max_len` using
  the best-validation checkpoint and writes `checkpoints/test_bleu.json`.
  Reporting the maximum of a probe series scored on test would be a
  selection-biased estimate rather than a generalization one.
- **DDP-safe checkpointing** — rank-0 atomic saves (tmp + `os.replace`),
  tensors normalized to CPU, payload restricted to primitives + tensors so
  `torch.load(weights_only=True)` (torch ≥ 2.6 default) works by
  construction; every rank loads weights independently to avoid NCCL
  deadlocks; model-state mismatch on resume is fatal by design.
- **Gradient accumulation** skips the all-reduce on non-boundary steps via
  `model.no_sync()`.
- **Decoding without KV cache** (documented limitation): the full decoder
  stack re-runs over the growing sequence each step — O(t²), adequate at
  max_len=128. Caching `past_key_values` is the natural next step for
  edge-deployment latency.
- **Batching** via a `Collator` instance (pad/truncate with per-dataset
  special-token ids — no module-level mutable globals, safe under
  DataLoader workers).
- **Data cleaning** is cheap-checks-first: length/ASCII heuristics run
  before the batched fastText LID model, then URL/HTML/punctuation filters,
  en/pt length-ratio alignment check, and xxhash64 exact-duplicate removal.
  The LID predictor is injectable, so the pipeline is unit-tested without
  fastText installed.
- **Torch-free serving path** — `train/export_onnx.py` splits the model into
  `encoder.onnx` + `decoder_step.onnx` (dynamic batch/seq axes, pad mask
  baked into the graphs); the ONNX Runtime backend re-implements the greedy
  loop over numpy and is unit-tested for **exact token parity** with the
  torch decoder. Model hyperparameters are rebuilt from the config embedded
  in the checkpoint — no side-car config to drift.
- **Dynamic batching** — a worker thread with per-item error handling merges
  sentences from concurrent requests into single decode calls (same contract
  as `cifar10/inference`); autoregressive decoding amortises especially well
  over batches, since B sentences share every decoder step.

## Platform notes

- **Kaggle**: pip installs are session-scoped; do not reinstall the
  preinstalled CUDA torch. `checkpoints/` lands in `/kaggle/working`
  (ephemeral) — save best/latest as notebook outputs.
- **Windows**: data_prep needs fastText; if `fasttext-wheel` has no wheel
  for your Python (e.g. 3.12), use a 3.10/3.11 venv or conda-forge. Only
  data_prep needs it — train/ and tests never import it.
- `.pt` split files are **Portuguese text** (OPUS convention), not PyTorch
  checkpoints.
- `torch.load` defaults to `weights_only=True` since torch 2.6; the
  checkpoint payload is primitives + tensors only, so the safe path is the
  default path.

## Environment

- Training: Kaggle Tesla T4 ×2 (DDP/NCCL) · Data prep / serving / tests: local CPU
- Python ≥ 3.10, PyTorch ≥ 2.1, sentencepiece, sacrebleu; serving: flask,
  onnxruntime (torch-free with `BACKEND=onnx`)
