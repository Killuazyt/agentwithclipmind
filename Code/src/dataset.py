import json
import os
import random
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.configs import ExperimentConfig
from src.label_encoder import LabelEncoder


class PairedTrainTransform:
    """对双时相图像执行一致随机增强。"""

    def __init__(self, image_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, img_t1: Image.Image, img_t2: Image.Image):
        img_t1 = TF.resize(img_t1, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        img_t2 = TF.resize(img_t2, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        if random.random() < 0.5:
            img_t1 = TF.hflip(img_t1)
            img_t2 = TF.hflip(img_t2)
        if random.random() < 0.5:
            img_t1 = TF.vflip(img_t1)
            img_t2 = TF.vflip(img_t2)

        img_t1 = TF.normalize(TF.to_tensor(img_t1), self.mean, self.std)
        img_t2 = TF.normalize(TF.to_tensor(img_t2), self.mean, self.std)
        return img_t1, img_t2


class PairedEvalTransform:
    """验证/测试阶段的确定性双时相变换。"""

    def __init__(self, image_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __call__(self, img_t1: Image.Image, img_t2: Image.Image):
        img_t1 = TF.resize(img_t1, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        img_t2 = TF.resize(img_t2, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        img_t1 = TF.normalize(TF.to_tensor(img_t1), self.mean, self.std)
        img_t2 = TF.normalize(TF.to_tensor(img_t2), self.mean, self.std)
        return img_t1, img_t2


class LEVIRMCIDataset(Dataset):
    """LEVIR-MCI 双时相数据集。"""

    def __init__(
        self,
        label_path: str,
        data_root: str,
        split: str,
        image_size: int,
        label_encoder: LabelEncoder,
        is_train: bool,
        debug_print_samples: int = 0,
    ):
        self.label_path = label_path
        self.data_root = data_root
        self.split = split
        self.label_encoder = label_encoder
        self.debug_print_samples = max(0, debug_print_samples)
        self._printed = 0

        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        with open(self.label_path, "r", encoding="utf-8") as f:
            all_samples: List[Dict] = json.load(f)

        self.samples = [x for x in all_samples if x.get("split") == split]
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split='{split}' in {label_path}")

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = (
            PairedTrainTransform(image_size=image_size, mean=mean, std=std)
            if is_train
            else PairedEvalTransform(image_size=image_size, mean=mean, std=std)
        )

    def __len__(self):
        return len(self.samples)

    def _get_pair_paths(self, filename: str) -> Tuple[str, str]:
        path_t1 = os.path.join(self.data_root, self.split, "A", filename)
        path_t2 = os.path.join(self.data_root, self.split, "B", filename)
        if not os.path.exists(path_t1):
            raise FileNotFoundError(
                f"Missing T1 image for split='{self.split}', filename='{filename}': {os.path.abspath(path_t1)}"
            )
        if not os.path.exists(path_t2):
            raise FileNotFoundError(
                f"Missing T2 image for split='{self.split}', filename='{filename}': {os.path.abspath(path_t2)}"
            )
        return path_t1, path_t2

    def __getitem__(self, index: int):
        sample = self.samples[index]
        filename = sample["filename"]
        path_t1, path_t2 = self._get_pair_paths(filename)

        with Image.open(path_t1) as im1:
            img_t1 = im1.convert("RGB")
        with Image.open(path_t2) as im2:
            img_t2 = im2.convert("RGB")

        img_t1, img_t2 = self.transform(img_t1, img_t2)
        encoded = self.label_encoder.encode_sample(sample)

        if self._printed < self.debug_print_samples:
            print(
                f"[DEBUG][{self.split}] {filename} | change={encoded['change_label']} "
                f"| object={encoded['object_labels'].tolist()} "
                f"| action={encoded['action_labels'].tolist()} "
                f"| location={encoded['location_labels'].tolist()}"
            )
            self._printed += 1

        return {
            "image_t1": img_t1,
            "image_t2": img_t2,
            "change_label": torch.tensor(encoded["change_label"], dtype=torch.float32),
            "object_labels": torch.tensor(encoded["object_labels"], dtype=torch.float32),
            "action_labels": torch.tensor(encoded["action_labels"], dtype=torch.float32),
            "location_labels": torch.tensor(encoded["location_labels"], dtype=torch.float32),
            "filename": filename,
        }


def build_dataloaders(cfg: ExperimentConfig, label_encoder: LabelEncoder):
    train_dataset = LEVIRMCIDataset(
        label_path=cfg.label_path,
        data_root=cfg.data_root,
        split="train",
        image_size=cfg.image_size,
        label_encoder=label_encoder,
        is_train=True,
        debug_print_samples=cfg.debug_print_samples,
    )
    val_dataset = LEVIRMCIDataset(
        label_path=cfg.label_path,
        data_root=cfg.data_root,
        split="val",
        image_size=cfg.image_size,
        label_encoder=label_encoder,
        is_train=False,
        debug_print_samples=0,
    )
    test_dataset = LEVIRMCIDataset(
        label_path=cfg.label_path,
        data_root=cfg.data_root,
        split="test",
        image_size=cfg.image_size,
        label_encoder=label_encoder,
        is_train=False,
        debug_print_samples=0,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
