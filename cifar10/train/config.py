# -*- coding: utf-8 -*-
"""
CIFAR Configuration Module:
1. Environment detection (Colab, Kaggle, local Jupyter, plain script)
2. Logging system setup
3. Command-line arguments and config files (.yaml/.json)
4. .env environment variable support
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml
from dotenv import load_dotenv

# Default input shape per model type. Transfer-learning backbones are
# pretrained at 224x224; the two from-scratch models use native 32x32.
DEFAULT_INPUT_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "cnn": (32, 32, 3),
    "resnet": (32, 32, 3),
    "resnet50": (224, 224, 3),
    "efficientnetb0": (224, 224, 3),
}

MODEL_TYPES = tuple(DEFAULT_INPUT_SHAPES.keys())
LR_MODES = ("cosine", "plateau")


# ============================================================
# Environment Detection
# ============================================================
class ProjectEnv:
    """Detect the runtime environment and determine the project root directory.

    The root anchors all output directories (checkpoints/, artifacts/, logs).
    For local runs it is the ``cifar10/`` directory that contains this package,
    regardless of the current working directory.
    """

    def __init__(self) -> None:
        self.root_dir: Path = self._detect_root_dir()

    @staticmethod
    def _detect_root_dir() -> Path:
        """Return the project root based on the execution environment."""
        try:
            # Google Colab
            if "google.colab" in sys.modules:
                return Path("/content")

            # Kaggle Notebook
            if Path("/kaggle/working").exists():
                return Path("/kaggle/working")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARNING] Environment detection failed: {e}", file=sys.stderr)

        # Local script / Jupyter: anchor to the cifar10/ directory that
        # contains train/ (config.py lives in train/), never to the cwd.
        return Path(__file__).resolve().parent.parent


# ============================================================
# Logging System
# ============================================================
def setup_logger(log_dir: Path, name: str = "train", level: int = logging.INFO) -> logging.Logger:
    """Initialize a logger with console and file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized: {log_path}")
    return logger


# ============================================================
# Global Configuration Class
# ============================================================
@dataclass
class Config:
    """Global configuration parameters for the CIFAR-10 project."""

    # ---------------- Data Parameters ----------------
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility (data split, init, augmentation)"},
    )
    val_split_ratio: float = field(
        default=0.2,
        metadata={"help": "Validation set split ratio", "range": "(0, 1)"},
    )
    class_names: List[str] = field(
        default_factory=lambda: [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        metadata={"help": "CIFAR-10 class names"},
    )

    # ---------------- Model Parameters ----------------
    input_shape: Optional[Tuple[int, int, int]] = field(
        default=None,
        metadata={"help": "Input image shape (H, W, C). None = auto-select by model_type "
                          "(32x32x3 for cnn/resnet, 224x224x3 for pretrained backbones)"},
    )
    num_classes: int = field(
        default=10,
        metadata={"help": "Number of classification classes"},
    )
    model_type: str = field(
        default="cnn",
        metadata={"help": "Model architecture type", "choices": list(MODEL_TYPES)},
    )

    # ---------------- Training Parameters ----------------
    batch_size: int = field(
        default=128,
        metadata={"help": "Global training batch size (split across replicas "
                          "under MirroredStrategy)"},
    )
    epochs: int = field(
        default=30,
        metadata={"help": "Number of training epochs"},
    )
    patience: int = field(
        default=3,
        metadata={"help": "Patience (epochs) for ReduceLROnPlateau; "
                          "EarlyStopping uses patience + 3"},
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "Label smoothing factor (used in BOTH training "
                          "phases so val_loss stays comparable)"},
    )
    lr_mode: str = field(
        default="plateau",
        metadata={"help": "Learning rate scheduling strategy", "choices": list(LR_MODES)},
    )
    init_lr: float = field(
        default=1e-3,
        metadata={"help": "Initial learning rate"},
    )
    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate"},
    )
    factor: float = field(
        default=0.5,
        metadata={"help": "Learning rate reduction factor (plateau mode)"},
    )
    cosine_restart_epochs: int = field(
        default=10,
        metadata={"help": "Epochs per cosine decay cycle before a restart (cosine mode)"},
    )
    mode: str = field(
        default="full",
        metadata={"help": "Training mode", "choices": ["sanity", "full"]},
    )

    # ---------------- Fine-tuning Parameters ----------------
    fine_epochs: int = field(
        default=20,
        metadata={"help": "Fine-tuning epochs (phase 2 for pretrained backbones)"},
    )
    fine_tune_lr: float = field(
        default=1e-5,
        metadata={"help": "Fine-tuning learning rate"},
    )
    fine_tune_at: Optional[int] = field(
        default=None,
        metadata={"help": "Layer index to start unfreezing from"},
    )
    fine_tune_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Unfreeze ratio (e.g. 0.2 = last 20% of backbone layers)",
                  "range": "(0, 1]"},
    )

    # ---------------- Data Augmentation Parameters ----------------
    augmentation: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Augmentation config dict (see data_aug.DataAugmentation)"},
    )
    use_mix: bool = field(
        default=False,
        metadata={"help": "Whether to use MixUp and CutMix batch augmentation"},
    )

    # ---------------- Path Parameters ----------------
    root_dir: Path = field(
        default_factory=lambda: ProjectEnv().root_dir,
        metadata={"help": "Project root directory (anchor for all outputs)"},
    )

    # Auto-generated paths (not serialized)
    ckpt_dir: Path = field(init=False)
    fig_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    # ---------------- Runtime Components ----------------
    logger: Optional[logging.Logger] = field(
        default=None,
        repr=False,
        metadata={"help": "Logger instance"},
    )

    # ============================================================
    # Post-initialization logic
    # ============================================================
    def __post_init__(self) -> None:
        """Validate inputs, create directories, set up logger and seed."""
        if self.model_type not in DEFAULT_INPUT_SHAPES:
            raise ValueError(
                f"Unsupported model_type: {self.model_type!r}. Choices: {list(MODEL_TYPES)}"
            )
        if self.lr_mode not in LR_MODES:
            raise ValueError(f"Unsupported lr_mode: {self.lr_mode!r}. Choices: {list(LR_MODES)}")
        if not 0.0 < self.val_split_ratio < 1.0:
            raise ValueError(f"val_split_ratio must be in (0, 1), got {self.val_split_ratio}")

        # Resolve input shape from model type when not set explicitly.
        if self.input_shape is None:
            self.input_shape = DEFAULT_INPUT_SHAPES[self.model_type]
        else:
            self.input_shape = tuple(int(v) for v in self.input_shape)  # type: ignore[assignment]

        self.ckpt_dir = self.root_dir / "checkpoints" / self.model_type
        self.fig_dir = self.ckpt_dir / "figures"
        self.log_dir = self.ckpt_dir / "logs"
        self.artifacts_dir = self.root_dir / "artifacts"

        for directory in [self.ckpt_dir, self.fig_dir, self.log_dir, self.artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        if self.logger is None:
            self.logger = setup_logger(self.log_dir, name="train")
        self.logger.info(f"Project root directory: {self.root_dir}")

        if self.seed is not None:
            self._set_seed()

        self._validate_fine_tune_params()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility (data split, init, shuffling)."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # Full op determinism trades some GPU throughput for bitwise-reproducible
        # runs. Comment out for faster (but non-deterministic) training.
        tf.config.experimental.enable_op_determinism()

    def _validate_fine_tune_params(self) -> None:
        """Validate fine-tuning parameters (ratio wins over index if both set)."""
        if self.fine_tune_at is not None and self.fine_tune_ratio is not None:
            self.logger.warning(
                "Both fine_tune_at and fine_tune_ratio are set. "
                "fine_tune_ratio will be used and fine_tune_at ignored."
            )
            self.fine_tune_at = None

    @property
    def use_augmentation(self) -> bool:
        """Return True if data augmentation is enabled."""
        return bool(self.augmentation)

    # ============================================================
    # Utility Methods
    # ============================================================
    def to_dict(self) -> Dict[str, Any]:
        """Return a clean, serializable configuration dictionary."""
        config_dict = asdict(self)

        # Remove non-serializable / auto-generated fields
        for field_name in ("logger", "ckpt_dir", "fig_dir", "log_dir", "artifacts_dir"):
            config_dict.pop(field_name, None)

        # Convert Path objects to strings
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        return config_dict

    def to_json(self, indent: int = 2) -> str:
        """Return the configuration as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path = "config.json") -> None:
        """Save the configuration to a JSON file (kept next to exported models)."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        if self.logger:
            self.logger.info(f"Configuration saved to: {path.resolve()}")

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. Only .yaml/.yml/.json supported."
                )

        # Convert string paths back to Path objects
        if isinstance(data.get("root_dir"), str):
            data["root_dir"] = Path(data["root_dir"])

        # Convert list shapes back to tuples
        if isinstance(data.get("input_shape"), list):
            data["input_shape"] = tuple(data["input_shape"])

        # Remove auto-generated fields (recomputed in __post_init__)
        for k in ("ckpt_dir", "fig_dir", "log_dir", "artifacts_dir"):
            data.pop(k, None)

        return cls(**data)

    def __repr__(self) -> str:
        return f"Config(\n{self.to_json(indent=2)}\n)"


# ============================================================
# Argument Parsing
# ============================================================
def parse_shape(value: str) -> Tuple[int, int, int]:
    """Convert a string like '224,224,3' into the tuple (224, 224, 3)."""
    try:
        parts = value.replace("(", "").replace(")", "").replace(" ", "").split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected 3 values, got {len(parts)}")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid input shape: {value!r}, error: {e}")


def parse_args(argv: List[str] | None = None) -> Config:
    """Parse command-line arguments and return a Config instance."""

    class EnglishHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, width=100, max_help_position=40)

    parser = argparse.ArgumentParser(
        description="CIFAR-10 Training Configuration",
        formatter_class=EnglishHelpFormatter,
    )

    # Configuration source
    config_group = parser.add_argument_group("Configuration Source")
    config_group.add_argument("--config", type=str,
                              help="Load configuration from a YAML/JSON file")

    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--seed", type=int, default=42, help="Random seed")
    data_group.add_argument("--val-split", type=float, default=0.2,
                            dest="val_split_ratio", help="Validation split ratio")
    data_group.add_argument("--input-shape", type=parse_shape, default=None,
                            help="Input shape (height,width,channels); default: auto by model type")

    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--batch-size", type=int, default=128,
                                help="Global batch size (split across GPUs under MirroredStrategy)")
    training_group.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    training_group.add_argument("--patience", type=int, default=3, help="Patience for LR reduction")
    training_group.add_argument("--label-smoothing", type=float, default=0.1,
                                help="Label smoothing")
    training_group.add_argument("--lr-mode", type=str, choices=list(LR_MODES),
                                default="plateau", help="Learning rate mode")
    training_group.add_argument("--init-lr", type=float, default=1e-3, help="Initial learning rate")
    training_group.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    training_group.add_argument("--factor", type=float, default=0.5, help="LR reduction factor")
    training_group.add_argument("--cosine-restart-epochs", type=int, default=10,
                                help="Epochs per cosine cycle before restart")
    training_group.add_argument("--mode", type=str, choices=["sanity", "full"],
                                default="full", help="Training mode")

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--model-type", type=str, choices=list(MODEL_TYPES),
                             default="cnn", help="Model architecture")

    # Fine-tuning parameters
    fine_tune_group = parser.add_argument_group("Fine-tuning Parameters")
    fine_tune_group.add_argument("--fine-epochs", type=int, default=20, help="Fine-tuning epochs")
    fine_tune_group.add_argument("--fine-tune-lr", type=float, default=1e-5,
                                 help="Fine-tuning learning rate")
    fine_tune_group.add_argument("--fine-tune-at", type=int, default=None,
                                 help="Unfreeze backbone from this layer index")
    fine_tune_group.add_argument("--fine-tune-ratio", type=float, default=None,
                                 help="Unfreeze ratio (0.2 = last 20%% of backbone layers)")

    # Augmentation parameters
    aug_group = parser.add_argument_group("Data Augmentation Parameters")
    aug_group.add_argument("--width-shift", type=float, default=None, help="Width shift range")
    aug_group.add_argument("--height-shift", type=float, default=None, help="Height shift range")
    aug_group.add_argument("--horizontal-flip", action="store_true", help="Enable horizontal flip")
    aug_group.add_argument("--vertical-flip", action="store_true", help="Enable vertical flip")
    aug_group.add_argument("--rotation", type=float, default=None, help="Rotation range (degrees)")
    aug_group.add_argument("--zoom", type=float, default=None, help="Zoom range")
    aug_group.add_argument("--brightness", type=float, nargs=2, default=None,
                           help="Brightness range")
    aug_group.add_argument("--contrast", type=float, nargs=2, default=None, help="Contrast range")
    aug_group.add_argument("--use-mix", action="store_true", help="Enable MixUp and CutMix")

    args = parser.parse_args(argv)

    load_dotenv()

    if args.config:
        config = Config.from_file(args.config)
        config.logger.info(f"Loaded configuration from: {args.config}")
        return config

    # Build the augmentation config from individual flags.
    # Skip values that are None/False/zero so that e.g. "--rotation 0"
    # does not silently enable an empty augmentation pipeline.
    aug_mapping = {
        "width_shift": args.width_shift,
        "height_shift": args.height_shift,
        "horizontal_flip": args.horizontal_flip,
        "vertical_flip": args.vertical_flip,
        "rotation": args.rotation,
        "zoom": args.zoom,
        "brightness": args.brightness,
        "contrast": args.contrast,
    }
    augmentation_config = {
        key: value for key, value in aug_mapping.items()
        if value is not None and value is not False and value != 0
    }

    config = Config(
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        input_shape=args.input_shape,  # None -> auto-resolved from model_type
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        lr_mode=args.lr_mode,
        init_lr=args.init_lr,
        min_lr=args.min_lr,
        factor=args.factor,
        cosine_restart_epochs=args.cosine_restart_epochs,
        mode=args.mode,
        model_type=args.model_type,
        fine_epochs=args.fine_epochs,
        fine_tune_lr=args.fine_tune_lr,
        fine_tune_at=args.fine_tune_at,
        fine_tune_ratio=args.fine_tune_ratio,
        augmentation=augmentation_config or None,
        use_mix=args.use_mix,
    )

    config.logger.info(config)
    return config


if __name__ == "__main__":
    cfg = parse_args()
    cfg.save("config_debug.json")
    print("Configuration test completed.")
