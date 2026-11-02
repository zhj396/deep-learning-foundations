# Train: BLEU-driven Transformer (pt → en)

From-scratch PyTorch reproduction of *Attention Is All You Need*
(Vaswani et al., 2017) for Portuguese→English translation, with a
BLEU-driven training controller. Runs on a single CPU/GPU process or
multi-GPU DDP (developed on Kaggle Tesla T4 ×2).

## Modules

| File | Responsibility |
|---|---|
| `main.py` | entry point; `mp.spawn` DDP when >1 CUDA device |
| `config.py` | dataclass config, generated CLI, JSON/YAML persistence |
| `dataset.py` | line-aligned text dataset + `Collator` (pad/truncate) |
| `layers.py` | positional encoding, attention, Pre-LN enc/dec layers |
| `model.py` | `Transformer` (+ final LayerNorms, tied embed↔out) |
| `loss.py` | label-smoothing KL loss, token accuracy |
| `scheduler.py` | Noam schedule with integrated plateau `lr_scale` |
| `decoding.py` | greedy + beam search (fairseq-style length penalty) |
| `evaluate.py` | token-weighted eval epoch (DDP-reduced), sacrebleu |
| `bleu_control.py` | BLEU-driven controller (LS/LR decay, early stop) |
| `checkpoint.py` | DDP-safe atomic save, hardened `weights_only` load |
| `train.py` | epoch loop + full-run orchestration |
| `utils.py` | seeding, rank-aware logging, barriers, probe sharding, plots |

## Usage

```bash
pip install -r requirements.txt

# full run (auto-DDP when multiple GPUs are visible)
python main.py --data_vocab <vocab_dir> --epochs 30 --use_amp \
    --batch_size 64 --accumulation_steps 4

# CPU smoke test (tiny model, one epoch; the full-split test pass would
# dominate a smoke run, so it is switched off here)
python main.py --data_vocab <vocab_dir> --no_cuda \
    --d_model 128 --num_layers 2 --epochs 1 --no-final_test

# resume from the latest checkpoint / from an explicit file
python main.py --data_vocab <vocab_dir> --resume
python main.py --data_vocab <vocab_dir> --resume_ckpt checkpoints/best_model_bleu_epoch_12.pt

# beam-search BLEU probes, evaluated more often on more batches
python main.py --data_vocab <vocab_dir> --use_beam --beam_size 4 \
    --length_penalty 0.6 --bleu_interval 2 --bleu_batches 20
```

`<vocab_dir>` is `data_prep`'s output directory containing
`{train,valid,test}.{en,pt}` and `spm_en_pt.model`. Every `Config` field
is a CLI flag (`--help` lists them); booleans accept `--flag/--no-flag`.

## BLEU-driven controller

Every `--bleu_interval` epochs the ranks split a fixed probe slice —
`--bleu_batches × --batch_size` **validation** sentences, handed out in
strided shards — decode it in parallel (greedy or beam), and rank 0
computes sacrebleu over the union. A gain has to clear
`--bleu_min_delta` (default 0.1) to count, because corpus BLEU over a few
hundred sentences wobbles by more than that on its own. On stagnation the
controller:

1. halves **label smoothing** after 2 consecutive misses (floor 0),
2. halves the **LR scale** after every miss (floor `--lr_min_scale`) —
   applied *inside* the Noam schedule, so it survives subsequent steps,
3. **early-stops** after `--patience` misses, keeping the best-BLEU
   checkpoint (`checkpoints/best_model_bleu_epoch_*.pt`).

Controller state is broadcast to all ranks in one collective, so DDP
workers exit together. A rank whose shard fails to decode votes that
through a MIN-reduce instead of deciding alone, so every rank aborts on
the same epoch.

## Selection vs reporting

Everything that *chooses* — the best checkpoint, the label-smoothing and
LR decays, early stopping — reads the validation probe
(`--bleu_split valid`, the default). The **test** split is decoded exactly
once, after the loop, on the checkpoint the probe selected (falling back
to the weights as they stand when no best checkpoint was written), with
the full `--max_len` budget rather than the probe cap, and the result goes
to the log and `checkpoints/test_bleu.json`. That keeps the quoted number
out of the search that picked the weights: a maximum over 15 probes on the
test split would be a selection-biased estimate, not a generalization one.

`--no-final_test` skips the pass (it is `--bleu_batches`-independent and
covers the whole split, so it is the slower of the two decodes).
`--bleu_split test` exists to measure that bias deliberately, not as a
default.

## Probe cost and the collective timeout

Decoding re-runs the decoder stack over the growing sequence (no KV
cache), so one sentence costs O(length²): the probe is quadratically
sensitive to `--bleu_max_len`, linearly to `--beam_size` and to the
sentence count, and split evenly by the number of ranks. Everything rank 0
does alone — sacrebleu, checkpoint rotation — happens while the other
ranks block inside the probe's collectives, so `--timeout` (the
process-group watchdog's limit, default 1800 s) must stay above the whole
probe, not just the decode. `--use_amp` covers both decode passes, and
they inherit that precision in their BLEU numbers.

## Checkpoints & resume

`checkpoints/latest_model.pt` is written every epoch (rank 0, atomic
tmp→replace, tensors on CPU); it carries model + optimizer + Noam state +
AMP scaler + controller state + history + resolved config. Loads default
to `torch.load(weights_only=True)`; full-pickle loading is opt-in via
`--trust_checkpoint`. Model-state mismatch on resume is fatal by design —
silently continuing on random weights is the worst failure mode.

## Kaggle notes

- The pip installs are **session-scoped**; keep them in the notebook's
  first cell. Do not reinstall the preinstalled CUDA torch.
- `checkpoints/` lands in `/kaggle/working` (ephemeral) — save the
  best/latest checkpoint as notebook output or re-attach it next session.
- DDP (NCCL + `mp.spawn`) triggers automatically when 2 GPUs are enabled.
- Probes are part of the session budget, not free: each one decodes
  `--bleu_batches × --batch_size` sentences autoregressively, so
  `--bleu_interval` trades control granularity for epochs per session. The
  final test pass adds one whole-split decode to the very end of the run —
  budget it before the session timer, not after it.
