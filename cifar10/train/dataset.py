# -*- coding: utf-8 -*-
"""
Dataset Module:
1. Loads CIFAR-10 (local-first: extracted batches -> local archive -> download)
2. Model-specific preprocessing (unit scaling vs. keras.applications preprocess_input)
3. Efficient tf.data pipeline: shuffle raw uint8 BEFORE resize/normalize so the
   shuffle buffer holds 3 KB uint8 images instead of ~600 KB float32 at 224x224
4. Optional augmentation and MixUp/CutMix
5. Dataset statistics and sample visualization
"""

from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import Config
from data_aug import CutMixMixup, DataAugmentation

# keras.application preprocessing functions for transfer-learning backbones.
# They expect uint8-range [0, 255] float input, which is what the raw CIFAR
# arrays provide before any scaling.
PREPROCESS_FUNCTIONS = {
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
}

# ======================================================
# CIFAR-10 raw data source (official URL, tar.gz archive
# with pickled batch files, identical to keras' cache)
# ======================================================
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"
CIFAR10_TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
CIFAR10_TEST_BATCH = "test_batch"


def _unpickle_cifar_batch(path: Path) -> Dict[str, Any]:
    """Load a single CIFAR-10 python batch file (pickle format)."""
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="bytes")
    # Keys are bytes in the raw files (b'data', b'labels'); normalize to str
    return {k.decode() if isinstance(k, bytes) else k: v for k, v in raw.items()}


def _load_cifar10_local(data_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                 Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 from extracted batch files in ``data_dir``.

    Returns the same structure as ``tf.keras.datasets.cifar10.load_data()``:
    ((x_train uint8 NHWC, y_train (N,1) int), (x_test, y_test)).
    """
    def _read_batches(batch_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        for name in batch_names:
            batch = _unpickle_cifar_batch(data_dir / name)
            images.append(np.asarray(batch["data"], dtype=np.uint8))
            labels.extend(np.asarray(batch["labels"], dtype=np.int64).tolist())
        # batch['data'] rows are uint8 vectors in CHW order -> reshape to N x 3 x 32 x 32,
        # then transpose to NHWC (channels-last, keras convention)
        x = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.asarray(labels, dtype=np.uint8).reshape(-1, 1)
        return x, y

    x_train, y_train = _read_batches(CIFAR10_TRAIN_BATCHES)
    x_test, y_test = _read_batches([CIFAR10_TEST_BATCH])
    return (x_train, y_train), (x_test, y_test)


def load_cifar10(data_dir: Path, logger: Any) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                       Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 with a local-first strategy, downloading only if needed.

    Resolution order:
    1. ``data_dir/cifar-10-batches-py/`` — already-extracted batch files
       (fastest; e.g. manually copied or previously extracted)
    2. ``data_dir/cifar-10-python.tar.gz`` — local archive, extracted in place
       (offline-friendly: the tar.gz can be copied between machines)
    3. Online download via ``tf.keras.utils.get_file`` (cached by filename
       into ``~/.keras/datasets``, so the archive is fetched only once);
       it is then copied to ``data_dir`` and extracted in place, so
       subsequent runs resolve through steps 1-2 without network access.
    """
    extracted = data_dir / CIFAR10_EXTRACTED_DIR
    archive = data_dir / CIFAR10_ARCHIVE_NAME

    # 1. Already-extracted batches
    if all((extracted / name).exists() for name in CIFAR10_TRAIN_BATCHES + [CIFAR10_TEST_BATCH]):
        logger.info(f"Loading CIFAR-10 from extracted local files: {extracted}")
        return _load_cifar10_local(extracted)

    # 2. Local archive -> extract once, then load
    if archive.exists():
        logger.info(f"Extracting local CIFAR-10 archive: {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=data_dir)
        return _load_cifar10_local(extracted)

    # 3. Online download via tf.keras.utils.get_file (cached by filename into
    #    ~/.keras/datasets, so the archive is fetched only once); it is then
    #    copied to data_dir and extracted in place, so later runs resolve
    #    through steps 1-2 without network access.
    logger.info(f"No local CIFAR-10 found in {data_dir}; downloading...")
    keras_archive = Path(tf.keras.utils.get_file(
        fname=CIFAR10_ARCHIVE_NAME, origin=CIFAR10_URL))

    # Keep a copy of the archive in data_dir so future runs are offline-first
    shutil.copy2(keras_archive, archive)
    logger.info(f"Archive cached locally for future offline runs: {archive}")

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=data_dir)
    return _load_cifar10_local(extracted)


class CIFARDataLoader:
    """CIFAR-10 loader with model-specific preprocessing and a tf.data pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = cfg.logger

        if cfg.num_classes != 10:
            raise ValueError(f"Only CIFAR-10 (num_classes=10) is supported, got {cfg.num_classes}")

        # ======================================================
        # Load raw dataset (kept as uint8 to save memory; the
        # pipeline casts to float32 per element inside map()).
        # Local-first: data/ under the project root, fall back
        # to the keras cache, then online download.
        # ======================================================
        self.data_dir = Path(cfg.root_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = load_cifar10(self.data_dir, self.logger)
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # ======================================================
        # Resolve preprocessing for the selected model type
        # ======================================================
        self._setup_preprocessing()

        # ======================================================
        # Stratified train/val split (seeded: same split on every
        # run for a given --seed)
        # ======================================================
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train,
            y_train,
            test_size=cfg.val_split_ratio,
            random_state=cfg.seed,
            shuffle=True,
            stratify=y_train,
        )
        self.x_test = x_test
        self.y_test = y_test

        # Built lazily on first access, then reused (see public accessors below)
        self._train_ds: tf.data.Dataset = None
        self._val_ds: tf.data.Dataset = None
        self._test_ds: tf.data.Dataset = None

        self.logger.info(
            f"Dataset ready. Model: {cfg.model_type}, Input shape: {cfg.input_shape} | "
            f"Train: {len(self.x_train)}, Val: {len(self.x_val)}, Test: {len(self.x_test)}"
        )

    def _setup_preprocessing(self) -> None:
        """Select the preprocessing function for the configured model type.

        Standard models (cnn, resnet): simple [0, 1] scaling.
        Transfer models (resnet50, efficientnetb0): keras.applications
        preprocess_input matching the ImageNet weights.
        """
        model_type = self.cfg.model_type.lower()

        if model_type in PREPROCESS_FUNCTIONS:
            self.need_special_preprocess = True
            self.preprocess_func = PREPROCESS_FUNCTIONS[model_type]
            expected = (224, 224, 3)
            if tuple(self.cfg.input_shape) != expected:
                self.logger.warning(
                    f"{model_type} is pretrained at {expected} but input_shape is "
                    f"{self.cfg.input_shape}. Accuracy may degrade; this is allowed "
                    f"for experimentation only."
                )
        elif model_type in ("cnn", "resnet"):
            self.need_special_preprocess = False
            self.preprocess_func = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ======================================================
    # Preprocessing Function
    # ======================================================
    def preprocess(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Cast to float32, resize if needed, and apply model-specific scaling."""
        x = tf.cast(x, tf.float32)

        # Resize only when the target shape differs from the native 32x32
        if self.cfg.input_shape[:2] != (32, 32):
            x = tf.image.resize(x, self.cfg.input_shape[:2])

        if self.need_special_preprocess:
            # Expects [0, 255]-range input (applied before unit scaling):
            # resnet50 converts to BGR and subtracts ImageNet means;
            # efficientnetb0 is a pass-through (normalization lives inside
            # the model graph)
            x = self.preprocess_func(x)
        else:
            x = x / 255.0

        return x, y

    # ======================================================
    # tf.data Pipeline Builder
    # ======================================================
    def _build_dataset(self, x: np.ndarray, y: np.ndarray,
                       training: bool = False) -> tf.data.Dataset:
        """Build the tf.data pipeline.

        Order matters:
        1. slice raw uint8 (small graph constants, low memory)
        2. shuffle BEFORE map so the buffer holds 3 KB uint8 images,
           not 600 KB resized float32 images
        3. preprocess (cast/resize/normalize)
        4. augment (training only, called with training=True so Random*
           layers actually fire inside the graph)
        5. batch (drop_remainder for training -> uniform batch shapes)
        6. MixUp/CutMix (batch level) or one-hot encoding
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y))

        if training:
            ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)

        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        if training and self.cfg.use_augmentation:
            aug_config = dict(self.cfg.augmentation)
            if self.need_special_preprocess and aug_config.get("brightness"):
                # Brightness expects [0, 1]-scaled pixels, but the transfer
                # backbones consume ImageNet-preprocessed values that are not
                # unit-scaled (resnet50: mean-subtracted; efficientnetb0: raw
                # [0, 255] with normalization baked into the graph); applying
                # it would corrupt the input, so it is skipped. Contrast is
                # scale-relative and still applies.
                self.logger.warning(
                    "brightness augmentation requires [0, 1]-scaled inputs and "
                    f"is skipped for {self.cfg.model_type} (ImageNet-preprocessed)"
                )
                aug_config.pop("brightness")
            aug_layer = DataAugmentation(
                datagen=aug_config,
                image_size=self.cfg.input_shape[:2],
                value_range=(0, 1),
                training_only=True,
                summary=True,
            ).build()

            def apply_tf_aug(img, lbl):
                # training=True is REQUIRED here: inside tf.data there is no
                # Keras learning phase, and Random* layers silently become
                # no-ops without it.
                return aug_layer(img, training=True), lbl

            ds = ds.map(apply_tf_aug, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.cfg.batch_size, drop_remainder=training)

        if training and self.cfg.use_mix:
            augmenter = CutMixMixup(
                img_size=self.cfg.input_shape[:2],
                alpha=1.0,
                mixup_alpha=0.2,
                p_mixup=0.5,
                num_classes=self.cfg.num_classes,
            )
            ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            def apply_one_hot(img, lbl):
                return img, tf.one_hot(tf.cast(lbl, tf.int32), self.cfg.num_classes,
                                       dtype=tf.float32)

            ds = ds.map(apply_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

        return ds.prefetch(tf.data.AUTOTUNE)

    # ======================================================
    # Public Dataset Interfaces
    # ======================================================
    def train_dataset(self) -> tf.data.Dataset:
        """Training dataset: shuffled, augmented, optionally mixed.

        Built once and cached; tf.data datasets are immutable and re-iterable,
        so reuse across epochs / fit() calls is safe. The shuffle uses
        reshuffle_each_iteration=True, so every epoch still sees a new order.
        """
        if self._train_ds is None:
            self._train_ds = self._build_dataset(self.x_train, self.y_train, training=True)
        return self._train_ds

    def val_dataset(self) -> tf.data.Dataset:
        """Validation dataset: deterministic, no augmentation."""
        if self._val_ds is None:
            self._val_ds = self._build_dataset(self.x_val, self.y_val, training=False)
        return self._val_ds

    def test_dataset(self) -> tf.data.Dataset:
        """Test dataset (official CIFAR-10 test split): deterministic."""
        if self._test_ds is None:
            self._test_ds = self._build_dataset(self.x_test, self.y_test, training=False)
        return self._test_ds

    # ======================================================
    # Utility Methods
    # ======================================================
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return detailed information about the datasets."""
        return {
            "dataset": "CIFAR-10",
            "train_samples": len(self.x_train),
            "val_samples": len(self.x_val),
            "test_samples": len(self.x_test),
            "total_samples": len(self.x_train) + len(self.x_val) + len(self.x_test),
            "input_shape": self.cfg.input_shape,
            "num_classes": self.cfg.num_classes,
            "model_type": self.cfg.model_type,
            "preprocessing": "imagenet" if self.need_special_preprocess else "unit_scale",
            "batch_size": self.cfg.batch_size,
            "seed": self.cfg.seed,
            "train_class_distribution": np.bincount(self.y_train).tolist(),
            "val_class_distribution": np.bincount(self.y_val).tolist(),
            "test_class_distribution": np.bincount(self.y_test).tolist(),
        }

    def show_sample_batch(self, num_samples: int = 9, dataset_type: str = "train") -> None:
        """Display a batch of images with class names (for debugging)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("Matplotlib is not installed. Cannot display sample images.")
            return

        try:
            datasets = {"train": self.train_dataset, "val": self.val_dataset,
                        "test": self.test_dataset}
            if dataset_type not in datasets:
                raise ValueError(f"Unknown dataset_type: {dataset_type}")

            class_names = self.cfg.class_names
            for images, labels in datasets[dataset_type]().take(1):
                images = images.numpy()
                labels = tf.argmax(labels, axis=1).numpy()

                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle(f"{dataset_type.capitalize()} Samples - "
                             f"{self.cfg.model_type} ({self.cfg.input_shape})", fontsize=16)

                for i, ax in enumerate(axes.flat):
                    if i >= min(num_samples, len(images)):
                        break
                    img = images[i]
                    if self.need_special_preprocess:
                        # ImageNet-preprocessed values are not displayable; rescale per image
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    ax.imshow(np.clip(img, 0, 1))
                    ax.set_title(f"Label: {class_names[labels[i]]}", fontsize=10)
                    ax.axis("off")

                plt.tight_layout()
                plt.show()
                break
        except Exception as e:
            self.logger.error(f"Error displaying sample batch: {e}")


# ======================================================
# Test Function
# ======================================================
def test_data_loader() -> None:
    """Smoke-test the CIFARDataLoader with the default (cnn) config."""
    from config import parse_args

    cfg = parse_args([])
    cfg.batch_size = 16
    loader = CIFARDataLoader(cfg)

    print("\nDataset Information:")
    for key, value in loader.get_dataset_info().items():
        if "distribution" not in key:
            print(f"  {key}: {value}")

    for x_batch, y_batch in loader.train_dataset().take(1):
        print("\nBatch shape check:")
        print(f"  Images: {x_batch.shape}")
        print(f"  Labels: {y_batch.shape}")

    print("\nCIFAR Data Loader test completed successfully!")


if __name__ == "__main__":
    test_data_loader()
