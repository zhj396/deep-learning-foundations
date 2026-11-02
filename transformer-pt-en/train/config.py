"""Training configuration: dataclass + CLI + JSON/YAML persistence.

Every tunable is a real dataclass field. The CLI is generated from the
fields, so adding a knob in one place exposes it everywhere.
"""

import argparse
import json
import sys
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin

import yaml


# Splits the BLEU control probe may read (Config.bleu_split).
SPLITS = ("valid", "test")


# -------------------------
# Runtime environment
# -------------------------
class ProjectEnv:
    """Detect the execution environment and return a writable root dir."""

    def __init__(self) -> None:
        self.root_dir: Path = self._detect_root_dir()

    @staticmethod
    def _detect_root_dir() -> Path:
        # Google Colab
        if "google.colab" in sys.modules:
            return Path("/content")
        # Kaggle notebook: /kaggle/working is the writable output dir
        if Path("/kaggle/working").exists():
            return Path("/kaggle/working")
        # Plain script / Jupyter: current working directory
        return Path.cwd()


# -------------------------
# Configuration
# -------------------------
@dataclass
class Config:
    # ---- special-token ids (must match the SentencePiece model) ----
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3

    # ---- data ----
    seed: int = 41
    data_vocab: Optional[str] = None  # dir with {train,valid,test}.{en,pt} + spm_en_pt.model
    vocab_size: int = 16_000          # informational; overwritten from the spm model at startup
    max_len: int = 128
    batch_size: int = 32
    num_workers: int = 2

    # ---- model ----
    # Narrower than the paper's base: ~19M parameters, which keeps this model
    # at roughly one parameter per ten training tokens on a corpus of a few
    # hundred thousand pairs -- d_model 512 assumed an order of magnitude more
    # data (~4.5M pairs). head_dim stays 64, as in the paper.
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True
    weight_decay: float = 1e-5

    # ---- training ----
    epochs: int = 30
    # Noam counts optimizer UPDATES, and one update covers
    # batch_size * world_size * accumulation_steps sentences, so the ramp has
    # to be sized against the run's total update count instead of copied as
    # the paper's absolute 4000 (there ~4% of its schedule). Pre-LN tolerates
    # the shorter ramp.
    warmup_steps: int = 1500
    factor: float = 1.0
    lr_min_scale: float = 0.1
    clip_grad: float = 1.0
    accumulation_steps: int = 4
    use_amp: bool = False
    label_smoothing: float = 0.1
    patience: int = 8
    bleu_interval: int = 5
    bleu_batches: int = 20      # sentences probed = bleu_batches * batch_size, split over ranks
    bleu_max_len: int = 64      # probe decode cap; probe cost grows with its square
    # Split the control signal comes from. Selection belongs on validation:
    # the best checkpoint, the label-smoothing/LR decays and early stopping all
    # key off this probe, so scoring it on test would report a chosen-maximum
    # rather than a generalization estimate. "test" is offered for deliberate
    # diagnostics of that bias, not as a default.
    bleu_split: str = "valid"
    # Smallest BLEU gain treated as progress. Corpus BLEU over a few hundred
    # sentences wobbles by a point or so on its own, so a strict > comparison
    # would let sampling noise postpone the plateau the controller is looking
    # for. 0.1 keeps the floor well inside that noise band.
    bleu_min_delta: float = 0.1
    final_test: bool = True     # decode the whole test split once, after training
    print_samples: int = 5
    use_beam: bool = False
    beam_size: int = 4
    length_penalty: float = 0.6

    # ---- resume / distributed ----
    resume: bool = False
    resume_ckpt: Optional[str] = None
    no_cuda: bool = False
    timeout: int = 1800                # collective timeout (s); > one rank's BLEU probe
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    find_unused_parameters: bool = False
    trust_checkpoint: bool = False     # opt-in unsafe (full pickle) checkpoint load

    # ---- paths ----
    root_dir: Path = field(default_factory=lambda: ProjectEnv().root_dir)
    ckpt_dir: Path = field(init=False)

    # set when loaded from a file; not a CLI option
    config_file: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.ckpt_dir = self.root_dir / "checkpoints"
        if self.bleu_split not in SPLITS:
            raise ValueError(
                f"bleu_split must be one of {SPLITS}, got {self.bleu_split!r}"
            )

    # -------------------------
    # Serialization
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    def save_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)

    @staticmethod
    def load(path: str) -> "Config":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)

        # ckpt_dir is derived in __post_init__ and not an __init__ arg
        data.pop("ckpt_dir", None)
        if data.get("root_dir") is not None:
            data["root_dir"] = Path(data["root_dir"])

        cfg = Config(**data)
        cfg.config_file = str(path)
        return cfg

    @classmethod
    def get_field_default(cls, field_name: str) -> Any:
        """Default value of a field (resolving default_factory)."""
        f = cls.__dataclass_fields__.get(field_name)
        if f is None:
            return None
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:  # type: ignore[misc]
            return f.default_factory()  # type: ignore[misc]
        return None

    # -------------------------
    # CLI
    # -------------------------
    @staticmethod
    def from_argparse(argv=None) -> "Config":
        parser = argparse.ArgumentParser(
            description="Train a pt->en Transformer (Attention Is All You Need)."
        )

        skip = {"config_file", "ckpt_dir"}
        for f in fields(Config):
            if f.name in skip:
                continue

            default = Config.get_field_default(f.name)

            # Resolve Optional[X] / Union annotations to X for argparse
            t = f.type
            if get_origin(t) is not None:
                concrete = [a for a in get_args(t) if a is not type(None)]
                t = concrete[0] if concrete else str

            if t == bool:
                # BooleanOptionalAction: --flag / --no-flag (Python >= 3.9)
                parser.add_argument(
                    f"--{f.name}", action=argparse.BooleanOptionalAction, default=default
                )
            elif t == Path:
                parser.add_argument(f"--{f.name}", type=Path, default=default)
            else:
                parser.add_argument(f"--{f.name}", type=t, default=default)

        parser.add_argument(
            "--resume-cfg", type=str, default=None,
            help="path to a saved config.json/yaml to resume settings from",
        )

        args = parser.parse_args(argv)

        if args.resume_cfg is not None:
            cfg = Config.load(args.resume_cfg)
            print(f"[Config] Resumed from {args.resume_cfg}")
        else:
            cfg = Config()

        # CLI values override the (possibly loaded) config only when they
        # differ from the dataclass default. Limitation: a True value from a
        # resumed config cannot be flipped off via a plain --flag (pass
        # --no-<flag>, which sets False != default only for True defaults).
        for k, v in vars(args).items():
            if k == "resume_cfg" or not hasattr(cfg, k):
                continue
            if v != Config.get_field_default(k):
                setattr(cfg, k, v)
                print(f"[Config] Override {k} = {v}")

        return cfg
