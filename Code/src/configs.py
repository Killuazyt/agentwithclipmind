import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ExperimentConfig:
    data_root: str = "c:/Users/Killua/Desktop/clip+/LEVIR-MCI-dataset/images"
    label_path: str = "c:/Users/Killua/Desktop/clip+/LEVIR-MCI-dataset/label.json"
    output_dir: str = "c:/Users/Killua/Desktop/clip+/Code/outputs"
    checkpoint_dir: str = ""

    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 224
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine | step
    step_size: int = 10
    step_gamma: float = 0.1

    device: str = "auto"  # auto | cpu | cuda
    seed: int = 42
    pretrained_backbone: bool = True

    lambda_obj: float = 1.0
    lambda_act: float = 1.0
    lambda_loc: float = 0.5

    threshold: float = 0.5
    dropout: float = 0.2

    change_loss_pos_weight: Optional[float] = None
    object_loss_weight: Optional[List[float]] = None
    action_loss_weight: Optional[List[float]] = None
    location_loss_weight: Optional[List[float]] = None

    debug_print_samples: int = 3
    save_metric: str = "change_f1"
    resume_checkpoint: str = ""

    test_checkpoint: str = ""
    test_split: str = "test"
    pred_csv_name: str = "test_predictions.csv"

    def finalize(self) -> "ExperimentConfig":
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        return self

    def to_dict(self):
        return asdict(self)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _parse_float_list(v: str) -> Optional[List[float]]:
    if v is None or v == "" or v.lower() == "none":
        return None
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--label-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["cosine", "step"])
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--step-gamma", type=float, default=None)

    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained-backbone", type=str2bool, default=None)

    parser.add_argument("--lambda-obj", type=float, default=None)
    parser.add_argument("--lambda-act", type=float, default=None)
    parser.add_argument("--lambda-loc", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--change-loss-pos-weight", type=float, default=None)
    parser.add_argument("--object-loss-weight", type=str, default=None)
    parser.add_argument("--action-loss-weight", type=str, default=None)
    parser.add_argument("--location-loss-weight", type=str, default=None)

    parser.add_argument("--debug-print-samples", type=int, default=None)
    parser.add_argument("--save-metric", type=str, default=None)

    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LEVIR-MCI baseline")
    add_common_args(parser)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    return parser


def build_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test LEVIR-MCI baseline")
    add_common_args(parser)
    parser.add_argument("--test-checkpoint", type=str, required=True)
    parser.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--pred-csv-name", type=str, default=None)
    return parser


def apply_args_to_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is None:
            continue
        attr = key.replace("-", "_")
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)

    cfg.object_loss_weight = _parse_float_list(cfg.object_loss_weight) if isinstance(cfg.object_loss_weight, str) else cfg.object_loss_weight
    cfg.action_loss_weight = _parse_float_list(cfg.action_loss_weight) if isinstance(cfg.action_loss_weight, str) else cfg.action_loss_weight
    cfg.location_loss_weight = _parse_float_list(cfg.location_loss_weight) if isinstance(cfg.location_loss_weight, str) else cfg.location_loss_weight

    return cfg.finalize()


def get_train_config_from_args() -> ExperimentConfig:
    parser = build_train_parser()
    args = parser.parse_args()
    cfg = ExperimentConfig().finalize()
    return apply_args_to_config(cfg, args)


def get_test_config_from_args() -> ExperimentConfig:
    parser = build_test_parser()
    args = parser.parse_args()
    cfg = ExperimentConfig().finalize()
    return apply_args_to_config(cfg, args)
