# Inference: pt → en Translation Service

Flask service + CLI serving the trained Transformer through **pluggable
backends** with **dynamic request batching**.

```
                      artifacts/<model>/
                      ┌─────────────────────────┐
train/ ──checkpoint──▶│ checkpoint.pt           │──▶ BACKEND=torch  greedy + beam
        export_onnx ─▶│ encoder.onnx            │──▶ BACKEND=onnx   greedy, torch-FREE
                      │ decoder_step.onnx       │
                      │ spm_en_pt.model         │◀── tokenization (both backends)
                      └─────────────────────────┘
```

## Backends

| | `torch` (default) | `onnx` |
|---|---|---|
| Artifacts | `checkpoint.pt` | `encoder.onnx` + `decoder_step.onnx` |
| Runtime | PyTorch (CPU) | ONNX Runtime (CPU provider) |
| Modes | greedy + beam | greedy only |
| Imports torch? | yes | **never** (lazy imports; torch-free edge path) |
| Decoding | `train/decoding.py` verbatim | numpy loop mirroring it step-for-step |

Model hyperparameters are rebuilt from the config **embedded in the
checkpoint** — no side-car config to keep in sync.

## Quick start

```bash
pip install -r requirements.txt

# 1) put a trained checkpoint + tokenizer into the artifacts dir
mkdir -p ../artifacts/base
cp <vocab_dir>/spm_en_pt.model ../artifacts/base/
cp <ckpt_dir>/best_model_bleu_epoch_N.pt ../artifacts/base/checkpoint.pt

# 2) (optional) export ONNX for the torch-free backend
python ../train/export_onnx.py --checkpoint ../artifacts/base/checkpoint.pt

# 3a) serve
BACKEND=torch python app.py        # http://127.0.0.1:5000  (BACKEND=onnx for ORT)

# 3b) or use the CLI (no Flask)
python translate.py -e "o gato esta aqui"
python translate.py --input test.pt --output test.en --backend onnx
cat sentences.txt | python translate.py --mode beam --beam-size 4
```

curl:

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/translate \
     -H 'Content-Type: application/json' \
     -d '{"text": "o gato esta aqui", "model": "base", "mode": "greedy"}'
```

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Web UI (text + .txt file translation, health pulse, latency badge) |
| `GET /health` | backend, modes, models available/cached, batcher liveness |
| `GET /model-info?model=` | decode signature (params / ONNX I-O shapes) |
| `POST /translate` | JSON `{"text"│"texts", model, mode, max_len, beam_size}` → translations + `latency_ms` |
| `POST /translate_file` | multipart `.txt` (one sentence/line) → line-aligned download + preview |
| `GET /download/<name>` | result file from `/translate_file` |

Env knobs: `BACKEND` (torch│onnx), `ARTIFACTS_DIR`, `HOST`, `PORT`,
`MAX_LEN`, `ORT_THREADS`, `FLASK_DEBUG`.

## Dynamic batching

`batcher.py` runs a worker thread that merges queued sentences from
*concurrent
requests* into single decode calls (5 ms merge window, max batch 32);
decode failures are attached to the affected items and their events are
always set, so clients get an immediate error instead of a hang; `wait()`
has a timeout as a second safety net; dead workers are detected via
`alive` and recreated. Batchers are keyed by `(model, mode, max_len,
beam_size, length_penalty)` so different decode settings never share a
call. Latency-vs-throughput knobs: poll interval, max batch size,
`ORT_THREADS`.

Autoregressive decoding amortises especially well over batches: B sentences
cost roughly the same number of decoder steps as 1, which is exactly why
the merge window pays off here.

## Security notes

Model names are whitelisted (`[a-z0-9_-]`, joined into artifact paths);
uploads go through `secure_filename`, are size-capped, and only `.txt` is
accepted; the Werkzeug debugger is off unless `FLASK_DEBUG=1`
(`threaded=True` is what makes batching observable either way).

## Known limits

- **No KV cache** in either backend (matches `train/decoding.py`): each
  decode step recomputes the decoder stack over the growing sequence,
  O(T²).
- **ONNX backend is greedy-only** — beam search would mean maintaining a
  second decoding implementation for a latency non-starter on edge
  hardware; one decoding implementation per mode keeps the two comparable.
- CPU-only serving (the demo target).
