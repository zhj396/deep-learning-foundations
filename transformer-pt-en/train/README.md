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
| `utils.py` | seeding, rank-aware logging, barriers, history plot |

## Usage

```bash
pip install -r requirements.txt

# full run (auto-DDP when multiple GPUs are visible)
python main.py --data_vocab <vocab_dir> --epochs 30 --use_amp \
    --batch_size 64 --accumulation_steps 4

# CPU smoke test (tiny model, one epoch)
python main.py --data_vocab <vocab_dir> --no_cuda \
    --d_model 128 --num_layers 2 --epochs 1

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

Every `--bleu_interval` epochs, rank 0 decodes a fixed probe slice
(`--bleu_batches × --batch_size` test sentences, greedy or beam) and
computes sacrebleu. On stagnation the controller:

1. halves **label smoothing** after 2 consecutive misses (floor 0),
2. halves the **LR scale** after every miss (floor `--lr_min_scale`) —
   applied *inside* the Noam schedule, so it survives subsequent steps,
3. **early-stops** after `--patience` misses, keeping the best-BLEU
   checkpoint (`checkpoints/best_model_bleu_epoch_*.pt`).

Controller state is broadcast to all ranks in one collective, so DDP
workers exit together. The probe BLEU is a *control signal* on a few
hundred sentences — for final numbers, raise `--bleu_batches` or decode
the full test split.

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
