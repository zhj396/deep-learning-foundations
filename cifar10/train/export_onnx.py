# -*- coding: utf-8 -*-
"""
Standalone ONNX exporter: TF SavedModel -> ONNX (via tf2onnx).

Produces the model.onnx deployment artifact, consumable by ONNX Runtime and
any other ONNX-compatible runtime.
Train.py calls export_saved_model_to_onnx() automatically after training;
this script can also convert an existing SavedModel without retraining
(useful for models trained on Kaggle and downloaded locally).

I/O contract (identical to the SavedModel):
- Input:  NHWC float32, dynamic batch: ("batch", H, W, 3)
- Output: softmax probabilities, shape ("batch", num_classes)

Usage:
    python export_onnx.py --model-type cnn
        # converts <cifar10>/artifacts/cnn/saved_model -> .../model.onnx
    python export_onnx.py --saved-model /path/to/saved_model --output /path/out.onnx
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("train")

# Opset 17: well supported by tf2onnx and any ONNX Runtime >= 1.14.
DEFAULT_OPSET = 17

# Artifacts root when using --model-type: <repo>/cifar10/artifacts
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def export_saved_model_to_onnx(saved_model_dir: str | Path,
                               onnx_path: str | Path,
                               opset: int = DEFAULT_OPSET) -> Path:
    """Convert a TF SavedModel directory to ONNX using the tf2onnx CLI.

    Runs tf2onnx in a subprocess: the CLI path is the most robust entry point
    across TF/Keras versions (SavedModel is a pure TF graph, so Keras 2 vs 3
    differences do not matter here).

    Args:
        saved_model_dir: Directory containing saved_model.pb.
        onnx_path: Destination .onnx file.
        opset: ONNX opset version.

    Returns:
        Path to the written .onnx file.

    Raises:
        FileNotFoundError: saved_model_dir does not look like a SavedModel.
        RuntimeError: tf2onnx conversion failed.
    """
    saved_model_dir = Path(saved_model_dir)
    onnx_path = Path(onnx_path)

    if not (saved_model_dir / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"Not a SavedModel directory (saved_model.pb missing): {saved_model_dir}"
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model_dir),
        "--output", str(onnx_path),
        "--opset", str(opset),
    ]
    logger.info(f"Exporting ONNX: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"tf2onnx failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"ONNX export failed for {saved_model_dir}. "
            f"Is tf2onnx installed? (pip install tf2onnx)"
        )

    logger.info(f"Saved ONNX model: {onnx_path.resolve()}")
    return onnx_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Export a trained SavedModel to ONNX")
    parser.add_argument("--model-type", type=str,
                        choices=["cnn", "resnet", "resnet50", "efficientnetb0"],
                        default=None,
                        help="Model name under the default artifacts directory")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR),
                        help="Artifacts root (default: <cifar10>/artifacts)")
    parser.add_argument("--saved-model", type=str, default=None,
                        help="Explicit SavedModel directory (overrides --model-type)")
    parser.add_argument("--output", type=str, default=None,
                        help="Explicit output .onnx path")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset")
    args = parser.parse_args()

    if args.saved_model:
        saved_model_dir = Path(args.saved_model)
        onnx_path = Path(args.output or saved_model_dir.parent / "model.onnx")
    elif args.model_type:
        model_dir = Path(args.artifacts_dir) / args.model_type
        saved_model_dir = model_dir / "saved_model"
        onnx_path = Path(args.output or model_dir / "model.onnx")
    else:
        parser.error("Provide either --model-type or --saved-model")

    export_saved_model_to_onnx(saved_model_dir, onnx_path, opset=args.opset)
    print(f"ONNX export completed: {onnx_path}")


if __name__ == "__main__":
    main()
