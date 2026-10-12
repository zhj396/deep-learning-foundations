# -*- coding: utf-8 -*-
"""
CI/local smoke tests - no CIFAR download, no GPU, no pretrained weights.

1. Build the two weight-free backbones (cnn, custom resnet) and run one
   forward pass each: softmax shape (N, 10) with rows summing to 1.
2. Drive the graph-safe MixUp/CutMix augmenter through a small tf.data
   pipeline: image shape preserved, labels blended as one-hot rows, and
   lambda re-sampled per batch (stateless ops + fresh seeds).

The ImageNet backbones (resnet50, efficientnetb0) are deliberately
excluded: building them downloads pretrained weights, which the gate
must not depend on.

Config instantiation anchors generated outputs (checkpoints/, artifacts/,
train.log) to per-test temp directories, leaving the repository tree clean.
"""

import pytest

import numpy as np
import tensorflow as tf

from config import ProjectEnv, parse_args
from data_aug import CutMixMixup
from model import CIFARModel

NUM_CLASSES = 10


@pytest.fixture(autouse=True)
def _throwaway_root(monkeypatch, tmp_path):
    """Anchor config-generated outputs (checkpoints/, artifacts/, train.log)
    in a per-test temp directory instead of the repository tree."""
    monkeypatch.setattr(ProjectEnv, "_detect_root_dir",
                        lambda *args, **kwargs: tmp_path)


def _build_config(model_type: str):
    cfg = parse_args([])
    cfg.model_type = model_type
    cfg.input_shape = None
    cfg.__post_init__()  # re-resolve the input shape for the new model type
    return cfg


def _forward_check(model, input_shape) -> None:
    probs = model(tf.random.uniform((2, *input_shape)), training=False)
    assert probs.shape == (2, NUM_CLASSES)
    assert np.allclose(tf.reduce_sum(probs, axis=1).numpy(), 1.0, atol=1e-5)


def test_cnn_build_and_forward():
    cfg = _build_config("cnn")
    _forward_check(CIFARModel(cfg).build(), tuple(cfg.input_shape))


def test_custom_resnet_build_and_forward():
    cfg = _build_config("resnet")
    _forward_check(CIFARModel(cfg).build(), tuple(cfg.input_shape))


def test_cutmix_mixup_graph_mode():
    augmenter = CutMixMixup(img_size=(32, 32), num_classes=NUM_CLASSES)
    ds = (
        tf.data.Dataset.from_tensor_slices(
            (tf.random.uniform((16, 32, 32, 3)),
             tf.random.uniform((16,), 0, NUM_CLASSES, tf.int32))
        )
        .batch(8, drop_remainder=True)
        .map(augmenter)
    )
    for images, labels in ds.take(2):
        assert images.shape == (8, 32, 32, 3)
        assert labels.shape == (8, NUM_CLASSES)
        # Blended one-hot label rows always sum to 1
        assert np.allclose(tf.reduce_sum(labels, axis=1).numpy(), 1.0, atol=1e-5)


def test_lambda_resampled_inside_traced_graph():
    """An RNG evaluated at trace time would freeze lambda, so a traced
    graph must still produce a new lambda on every call. (Label maxima
    are a weak signal - blended rows hit exactly 1.0 whenever the two
    mixed labels share a class.)"""
    augmenter = CutMixMixup(img_size=(32, 32), num_classes=NUM_CLASSES)

    @tf.function
    def draw():
        return augmenter._sample_lambda(1.0)

    lambdas = {round(float(draw()), 6) for _ in range(5)}
    assert len(lambdas) == 5
