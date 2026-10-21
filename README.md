# Deep Learning Foundations

From-scratch training projects covering convolutional vision models.
Companion to [edge-ai-deployment](https://github.com/zhj396/edge-ai-deployment)
which covers the deployment side (export, quantization, inference backends,
benchmarks).

| Project | Contents |
|---|---|
| [`cifar10/`](cifar10/) | CIFAR-10 classification with four switchable backbones — custom CNN, custom ResNet, ImageNet-transferred ResNet50 / EfficientNetB0 — plus a Flask serving app with pluggable TensorFlow / ONNX Runtime backends |

## cifar10

- Reproducible pipeline: fixed seed, stratified split, deterministic ops, and
  the resolved configuration archived next to every exported model
- tf.data pipeline with augmentation and MixUp/CutMix; two-phase transfer
  learning; MirroredStrategy on multi-GPU
- Exports to Keras / SavedModel / TFLite / ONNX; serving with dynamic request
  batching and a web UI
- Details, accuracy table, and usage: [`cifar10/README.md`](cifar10/README.md)

## Environment

- Training: Kaggle Tesla T4 ×2 · Data prep / serving demos: local CPU
- Python ≥ 3.10, TensorFlow ≥ 2.18 (Keras ≥ 3.10), ONNX Runtime ≥ 1.17
