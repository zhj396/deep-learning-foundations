# -*- coding: utf-8 -*-
"""
Data Augmentation Module:
1. Builds augmentation pipelines from dict / ImageDataGenerator / layer list
2. TrainingAugmentationLayer ensures augmentation runs only during training
3. CutMixMixup: batch-level MixUp/CutMix, safe for tf.data graph tracing
4. Preview visualization and config save/load helpers
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger("train")


@tf.keras.utils.register_keras_serializable(package="Custom")
class TrainingAugmentationLayer(tf.keras.layers.Layer):
    """Wrapper layer that applies augmentation only during the training phase."""

    def __init__(self, model, training_only=True, **kwargs):
        super().__init__(**kwargs)
        self.training_only = training_only

        if isinstance(model, tf.keras.Sequential):
            # Re-wrap so the Random* layers are properly tracked as sublayers
            self.aug_model = tf.keras.Sequential(
                [layer for layer in model.layers], name=f"{model.name}_tracked"
            )
        elif isinstance(model, tf.keras.layers.Layer):
            self.aug_model = model
        else:
            raise TypeError("model must be a tf.keras.Layer or Sequential instance")

    def call(self, inputs, training=False):
        if self.training_only:
            return self.aug_model(inputs, training=training)
        return self.aug_model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "training_only": self.training_only,
            "model": tf.keras.saving.serialize_keras_object(self.aug_model),
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = tf.keras.saving.deserialize_keras_object(config.pop("model"))
        return cls(model=model, **config)


class DataAugmentation:
    """Builder for flexible data augmentation pipelines.

    Accepts a config dict, an ImageDataGenerator, an explicit list of Keras
    layers, or None, and returns a single callable layer.
    """

    def __init__(self, datagen=None, image_size=(32, 32), value_range=(0, 1),
                 summary=False, training_only=False, **kwargs):
        """
        Args:
            datagen: ImageDataGenerator, dict, list of layers, or None
            image_size: (H, W), used to convert pixel shifts to fractions
            value_range: (min, max) of the input pixel range; used by
                RandomBrightness, whose Keras default (0, 255) would not
                match the [0, 1]-scaled images this pipeline augments
            summary: Whether to log the enabled augmentations
            training_only: Whether augmentation runs only in training mode
            kwargs: Additional augmentation parameters
        """
        self.datagen = datagen
        self.image_size = image_size
        self.value_range = value_range
        self.summary = summary
        self.training_only = training_only
        self.kwargs = kwargs
        self.aug_layers: List[tf.keras.layers.Layer] = []
        self.enabled: List[str] = []

    # ============================================================
    # Core Build Method
    # ============================================================
    def build(self) -> tf.keras.layers.Layer:
        """Build and return the final augmentation layer."""
        config = self._parse_config(self.datagen, self.kwargs)
        fill_mode = config.get("fill_mode", "reflect")

        self._add_rotation_layer(config, fill_mode)
        self._add_translation_layer(config, fill_mode)
        self._add_zoom_layer(config, fill_mode)
        self._add_flip_layers(config)
        self._add_shear_layer(config, fill_mode)
        self._add_brightness_layer(config)
        self._add_contrast_layer(config)

        if self.summary:
            self._print_summary(fill_mode)

        sequential = tf.keras.Sequential(self.aug_layers, name="data_augmentation")

        if self._contains_custom_layers(self.aug_layers):
            return TrainingAugmentationLayer(sequential, training_only=self.training_only)
        return sequential

    # ------------------- Individual Augmentation Layers -------------------
    def _add_rotation_layer(self, config: Dict[str, Any], fill_mode: str) -> None:
        """Add a RandomRotation layer (config value in degrees)."""
        rotation = config.get("rotation", 0)
        if rotation:
            rotation_factor = rotation / 360.0  # degrees -> fraction of a full turn
            self.aug_layers.append(layers.RandomRotation(rotation_factor, fill_mode=fill_mode))
            self.enabled.append(f"Rotation({rotation}deg)")

    def _add_translation_layer(self, config: Dict[str, Any], fill_mode: str) -> None:
        """Add a RandomTranslation layer (fractions, or ints interpreted as pixels)."""
        width_shift = config.get("width_shift", 0.0)
        height_shift = config.get("height_shift", 0.0)

        if isinstance(width_shift, int):
            width_shift = width_shift / self.image_size[1]
        if isinstance(height_shift, int):
            height_shift = height_shift / self.image_size[0]

        if width_shift or height_shift:
            self.aug_layers.append(
                layers.RandomTranslation(height_shift, width_shift, fill_mode=fill_mode)
            )
            self.enabled.append(f"Translation(height={height_shift:.3f}, width={width_shift:.3f})")

    def _add_zoom_layer(self, config: Dict[str, Any], fill_mode: str) -> None:
        """Add a RandomZoom layer."""
        zoom = config.get("zoom", None)
        if zoom:
            if isinstance(zoom, (float, int)):
                zoom = (1 - zoom, 1 + zoom)
            height_factor = (zoom[0] - 1.0, zoom[1] - 1.0)
            self.aug_layers.append(
                layers.RandomZoom(height_factor=height_factor, width_factor=height_factor,
                                  fill_mode=fill_mode)
            )
            self.enabled.append(f"Zoom({zoom})")

    def _add_flip_layers(self, config: Dict[str, Any]) -> None:
        """Add horizontal and/or vertical RandomFlip layers."""
        if config.get("horizontal_flip", False):
            self.aug_layers.append(layers.RandomFlip("horizontal"))
            self.enabled.append("Flip(horizontal)")

        if config.get("vertical_flip", False):
            self.aug_layers.append(layers.RandomFlip("vertical"))
            self.enabled.append("Flip(vertical)")

    def _add_shear_layer(self, config: Dict[str, Any], fill_mode: str) -> None:
        """Add a RandomShear layer if available in the current Keras version."""
        shear = config.get("shear", 0)
        if shear and hasattr(layers, "RandomShear"):
            shear_factor = shear * (np.pi / 180)
            self.aug_layers.append(layers.RandomShear(shear_factor, fill_mode=fill_mode))
            self.enabled.append(f"Shear({shear}deg)")

    def _add_brightness_layer(self, config: Dict[str, Any]) -> None:
        """Add a RandomBrightness layer (accepts a (min, max) pair or a float)."""
        brightness = config.get("brightness", None)
        if brightness:
            try:
                min_b, max_b = brightness
                factor = (max_b - min_b) / 2
            except TypeError:
                factor = float(brightness)
            self.aug_layers.append(
                layers.RandomBrightness(factor, value_range=self.value_range))
            self.enabled.append(f"Brightness({brightness})")

    def _add_contrast_layer(self, config: Dict[str, Any]) -> None:
        """Add a RandomContrast layer."""
        contrast = config.get("channel_shift", None) or config.get("contrast", None)
        if contrast:
            factor = contrast if isinstance(contrast, (float, int)) else max(contrast)
            if factor > 1:
                factor /= 255.0
            self.aug_layers.append(layers.RandomContrast(factor))
            self.enabled.append(f"Contrast({factor:.3f})")

    # ============================================================
    # Helper Methods
    # ============================================================
    @staticmethod
    def _contains_custom_layers(layers_list: List[tf.keras.layers.Layer]) -> bool:
        """Return True if any non-standard (non-Random*) layer is present."""
        return any(not layer.__class__.__name__.startswith("Random") for layer in layers_list)

    def _parse_config(self, datagen, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Unified config parser for the different accepted input types."""
        if isinstance(datagen, (list, tuple)):
            for layer in datagen:
                if not isinstance(layer, tf.keras.layers.Layer):
                    raise TypeError("All items in list must be tf.keras.layers.Layer instances")
                self.aug_layers.append(layer)
            return kwargs

        if isinstance(datagen, ImageDataGenerator):
            config = vars(datagen)
        elif isinstance(datagen, dict):
            config = datagen.copy()
        elif datagen is None:
            config = {}
        else:
            raise TypeError("datagen must be ImageDataGenerator, dict, list, or None.")

        config.update(kwargs)
        return config

    def _print_summary(self, fill_mode: str) -> None:
        """Log a summary of the enabled augmentations."""
        if self.aug_layers:
            logger.info("Enabled data augmentations:")
            for i, name in enumerate(self.enabled, 1):
                logger.info(f"  {i:>2}. {name}")
            logger.info(f"  Fill mode: {fill_mode} | total layers: {len(self.aug_layers)}")
        else:
            logger.info("No data augmentation enabled.")

    # ============================================================
    # Visualization & Config Management
    # ============================================================
    def preview(self, image: Union[np.ndarray, tf.Tensor], num_samples: int = 5,
                figsize: Tuple[int, int] = (15, 3)) -> None:
        """Visualize the augmentation effect on a single image."""
        import matplotlib.pyplot as plt

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        if tf.reduce_max(img) > 1.0:
            img = img / 255.0
        img = tf.expand_dims(img, 0)

        aug_layer = self.build()

        plt.figure(figsize=figsize)
        plt.suptitle("Data Augmentation Preview", fontsize=16, fontweight="bold")
        for i in range(num_samples):
            augmented = aug_layer(img, training=True)[0].numpy()
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(np.clip(augmented, 0, 1))
            plt.title(f"Sample {i + 1}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def save_config(self, path: Union[str, Path]) -> None:
        """Save the augmentation configuration to a YAML/JSON file."""
        config = self._parse_config(self.datagen, self.kwargs)
        path = Path(path)

        if path.suffix in {".yaml", ".yml"}:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError("Only .yaml/.yml or .json formats are supported.")

        logger.info(f"Data augmentation config saved to: {path}")

    @classmethod
    def from_config_file(cls, path: Union[str, Path], image_size=(32, 32),
                         summary=False, training_only=False) -> "DataAugmentation":
        """Load an augmentation config from a YAML or JSON file."""
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError("Only .yaml/.yml or .json formats are supported.")

        return cls(datagen=config, image_size=image_size, summary=summary,
                   training_only=training_only)


# ============================================================
# CutMix + MixUp Implementation
# ============================================================
class CutMixMixup:
    """Batch-level MixUp / CutMix augmentation for tf.data pipelines.

    Graph-tracing safety notes:
    - All randomness uses TensorFlow ops (``tf.random.stateless_gamma`` with a
      freshly drawn seed tensor per call). NumPy RNG calls would be frozen at
      trace time and yield the same lambda for every batch.
    - The MixUp-vs-CutMix decision uses ``tf.cond`` instead of a Python ``if``
      on a symbolic tensor (which raises TypeError during ``ds.map`` tracing).

    Expects integer labels of shape ``(batch,)`` and images of shape
    ``(batch, H, W, C)``; returns mixed images and one-hot mixed labels.
    """

    def __init__(self, img_size: Tuple[int, int], alpha: float = 1.0,
                 mixup_alpha: float = 0.2, p_mixup: float = 0.5,
                 num_classes: int = 10):
        self.img_size = img_size
        self.alpha = alpha                # CutMix Beta(alpha, alpha)
        self.mixup_alpha = mixup_alpha    # MixUp Beta(alpha, alpha)
        self.p_mixup = p_mixup
        self.num_classes = num_classes

    @staticmethod
    def _fresh_seed() -> tf.Tensor:
        """Draw a new random seed tensor so each call samples different values."""
        return tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)

    def _sample_lambda(self, alpha: float) -> tf.Tensor:
        """Sample lambda ~ Beta(alpha, alpha) with TF stateless ops (scalar).

        Uses the gamma-ratio identity Beta(a, b) = G1 / (G1 + G2) with
        G1 ~ Gamma(a), G2 ~ Gamma(b) independent, because tf.random has no
        public stateless_beta op. Each draw uses a freshly generated seed, so
        lambda is re-sampled at runtime for every batch (NumPy RNG calls would
        be frozen at graph-trace time instead).
        """
        g1 = tf.random.stateless_gamma((1,), alpha=alpha, seed=self._fresh_seed(),
                                       dtype=tf.float32)
        g2 = tf.random.stateless_gamma((1,), alpha=alpha, seed=self._fresh_seed(),
                                       dtype=tf.float32)
        return tf.reshape(g1 / (g1 + g2 + 1e-12), ())

    def _mix_labels(self, labels: tf.Tensor, index: tf.Tensor,
                    lam: tf.Tensor) -> tf.Tensor:
        """Blend one-hot labels of the original and shuffled batches."""
        y_a = tf.one_hot(labels, self.num_classes, dtype=tf.float32)
        y_b = tf.one_hot(tf.gather(labels, index), self.num_classes, dtype=tf.float32)
        return lam * y_a + (1.0 - lam) * y_b

    def _mixup(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        lam = self._sample_lambda(self.mixup_alpha)
        index = tf.random.shuffle(tf.range(tf.shape(images)[0]))
        mixed_images = lam * images + (1.0 - lam) * tf.gather(images, index)
        return mixed_images, self._mix_labels(labels, index, lam)

    def _cutmix(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        lam = self._sample_lambda(self.alpha)
        batch_size = tf.shape(images)[0]
        index = tf.random.shuffle(tf.range(batch_size))

        h = tf.cast(self.img_size[0], tf.float32)
        w = tf.cast(self.img_size[1], tf.float32)

        # Cut box size: area fraction (1 - lam) of the full image
        cut_rat = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(w * cut_rat, tf.int32)
        cut_h = tf.cast(h * cut_rat, tf.int32)

        cx = tf.random.uniform((), 0, self.img_size[1], dtype=tf.int32)
        cy = tf.random.uniform((), 0, self.img_size[0], dtype=tf.int32)

        x1 = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(w, tf.int32))
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(h, tf.int32))
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(w, tf.int32))
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(h, tf.int32))

        # Binary mask: 1 inside the cut box, 0 elsewhere
        mask = tf.ones((y2 - y1, x2 - x1, tf.shape(images)[-1]), dtype=images.dtype)
        pad_mask = tf.pad(
            mask,
            [[y1, tf.cast(h, tf.int32) - y2], [x1, tf.cast(w, tf.int32) - x2], [0, 0]],
            constant_values=0,
        )
        mask = tf.expand_dims(pad_mask, 0)  # broadcast over the batch dim

        mixed_images = images * (1.0 - mask) + tf.gather(images, index) * mask

        # Adjust lambda for the actual (clipped) box area
        lam_adjusted = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / (w * h)
        return mixed_images, self._mix_labels(labels, index, lam_adjusted)

    def __call__(self, images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly apply MixUp or CutMix to a batch (graph-trace safe)."""
        labels = tf.cast(labels, tf.int32)
        use_mixup = tf.random.uniform(()) < self.p_mixup
        return tf.cond(
            use_mixup,
            lambda: self._mixup(images, labels),
            lambda: self._cutmix(images, labels),
        )


# ===================== Test Function =====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Data Augmentation module...")

    builder = DataAugmentation(
        datagen={
            "rotation": 15,
            "horizontal_flip": True,
            "zoom": 0.2,
            "brightness": (0.8, 1.2),
        },
        summary=True,
    )

    builder.save_config("aug_config.yaml")

    sample_image = tf.random.uniform((32, 32, 3), 0, 1)
    builder.preview(sample_image, num_samples=5)

    # Smoke-test CutMixMixup inside a tf.data map (graph tracing)
    augmenter = CutMixMixup(img_size=(32, 32), num_classes=10)
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform((64, 32, 32, 3)), tf.random.uniform((64,), 0, 10, tf.int32))
    ).batch(8).map(augmenter)
    lambdas = set()
    for imgs, lbls in ds.take(4):
        assert imgs.shape == (8, 32, 32, 3) and lbls.shape == (8, 10)
        lambdas.add(float(tf.reduce_max(lbls).numpy()))
    print(f"CutMixMixup graph-mode smoke test passed; distinct label maxima: {len(lambdas)}")
