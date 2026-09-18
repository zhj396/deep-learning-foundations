# Deep Learning Foundations

From-scratch training projects covering the breadth of classical deep learning
practice: convolutional vision models and the Transformer architecture.
Companion to [edge-ai-deployment](https://github.com/zhj396/edge-ai-deployment)
which covers the deployment side (export, quantization, inference backends,
benchmarks).

| Project | Contents |
|---|---|
| [`cifar10/`](cifar10/) | CIFAR-10 classification with four switchable backbones — custom CNN, custom ResNet, ImageNet-transferred ResNet50 / EfficientNetB0 — plus a Flask serving app with pluggable TensorFlow / ONNX Runtime backends |
| [`transformer-pt-en/`](transformer-pt-en/) | Full reproduction of *Attention Is All You Need* (Vaswani et al., 2017) for Portuguese→English machine translation — from-scratch data pipeline (OPUS corpora, fastText cleaning, SentencePiece) through BLEU-driven multi-GPU DDP training, plus a Flask serving app with pluggable PyTorch / ONNX Runtime backends |

## cifar10

- Reproducible pipeline: fixed seed, stratified split, deterministic ops, and
  the resolved configuration archived next to every exported model
- tf.data pipeline with augmentation and MixUp/CutMix; two-phase transfer
  learning; MirroredStrategy on multi-GPU
- Exports to Keras / SavedModel / TFLite / ONNX; serving with dynamic request
  batching and a web UI
- Details, accuracy table, and usage: [`cifar10/README.md`](cifar10/README.md)

## transformer-pt-en

- From-scratch data pipeline: OPUS en↔pt corpora, fastText LID + heuristic
  cleaning, xxhash dedup, 95/2.5/2.5 split, shared 32k SentencePiece BPE
  vocabulary (`data_prep/`, local CPU)
- Encoder–decoder Transformer built from scratch: multi-head attention,
  positional encoding, label smoothing, warmup schedule
- BLEU-driven training on the Portuguese→English dataset (DDP, AMP,
  gradient accumulation), with a gap analysis against the numbers reported
  in the paper
- ONNX export (encoder + decoder-step graphs) and serving with dynamic
  request batching, a web UI, and a CLI; the ONNX Runtime backend never
  imports torch
- CPU-only pytest suite covering masks, causality, loss, decoding,
  checkpointing, corpus cleaning, batching, and torch-vs-ONNX parity
- Details: [`transformer-pt-en/README.md`](transformer-pt-en/README.md)

## Environment

- Training: Kaggle Tesla T4 ×2 · Data prep / serving demos: local CPU
- Python ≥ 3.10, TensorFlow ≥ 2.16 (Keras 3), PyTorch ≥ 2.1, ONNX Runtime ≥ 1.17
