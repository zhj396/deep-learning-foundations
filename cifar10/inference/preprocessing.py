# -*- coding: utf-8 -*-
"""
Preprocessing contract for the CIFAR-10 inference service.

Pure NumPy + PIL: NO TensorFlow import, so ONNX-backend deployments stay
free of any TF dependency.

Every function mirrors the training pipeline (train/dataset.py). The
contract differs per model family:

    cnn / resnet        resize 32x32,  x / 255
    resnet50            resize 224x224, RGB->BGR, subtract caffe mean
                        (= tf.keras.applications.resnet50.preprocess_input)
    efficientnetb0      resize 224x224, feed RAW [0,255]: keras
                        efficientnet.preprocess_input is a documented
                        pass-through placeholder because Rescaling(1/255) +
                        an imagenet stddev correction are baked INTO the
                        model graph (and thus into the exported SavedModel
                        and ONNX files).

Resize uses PIL BILINEAR, matching tf.image.resize for the upsampling the
224x224 models perform (32x32 sources are never downscaled in training).
"""

import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# keras.applications.resnet50 preprocess_input uses caffe-mode preprocessing:
# RGB->BGR, then subtract the per-channel BGR means.
KERAS_CAFFE_BGR_MEAN = np.array([103.939, 116.779, 123.680], dtype=np.float32)


def _preprocess_unit_scale(x: np.ndarray) -> np.ndarray:
    """cnn / resnet: trained with plain x / 255 scaling."""
    return x / 255.0


def _preprocess_caffe_bgr(x: np.ndarray) -> np.ndarray:
    """Equivalent to tf.keras.applications.resnet50.preprocess_input (caffe mode).

    Input x: float32 HWC in [0, 255], RGB order.
    """
    x = x[..., ::-1]                    # RGB -> BGR
    return x - KERAS_CAFFE_BGR_MEAN     # broadcast over H, W


def _preprocess_passthrough(x: np.ndarray) -> np.ndarray:
    """keras EfficientNet: feed raw [0, 255] float32 unchanged.

    The normalization lives inside the model graph, so it is already part of
    the exported SavedModel / ONNX file.
    """
    return x


# model_name -> (input size, preprocessing function)
MODEL_SPECS: Dict[str, Tuple[int, callable]] = {
    "cnn": (32, _preprocess_unit_scale),
    "resnet": (32, _preprocess_unit_scale),
    "resnet50": (224, _preprocess_caffe_bgr),
    "efficientnetb0": (224, _preprocess_passthrough),
}


def preprocess_image(model_name: str, img_bytes: bytes) -> np.ndarray:
    """Decode one image and preprocess it exactly as training did.

    Returns:
        float32 array of shape (1, H, W, 3) ready for the model input.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model type: {model_name}. Available: {sorted(MODEL_SPECS)}"
        )

    size, preprocess_func = MODEL_SPECS[model_name]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    return np.expand_dims(preprocess_func(x), axis=0)
