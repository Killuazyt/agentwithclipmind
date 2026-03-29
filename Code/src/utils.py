import json
import logging
import os
import random
from typing import Dict

import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_cfg)


def setup_logger(output_dir: str, name: str = "train") -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{name}.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(state: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def save_json(data: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = sorted(metrics.keys())
    return " | ".join([f"{k}: {metrics[k]:.4f}" for k in keys])
