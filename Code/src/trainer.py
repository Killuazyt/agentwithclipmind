import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.configs import ExperimentConfig
from src.losses import compute_loss
from src.metrics import compute_all_metrics, sigmoid_np
from src.utils import format_metrics, save_checkpoint


class Trainer:
    """封装训练/验证流程。"""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_functions: Dict,
        device: torch.device,
        logger,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_functions = loss_functions
        self.device = device
        self.logger = logger

    def train(self, train_loader, val_loader):
        best_metric = -1.0
        best_epoch = -1
        best_metrics = {}

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_loss_dict = self._train_one_epoch(train_loader)
            val_loss, val_metrics, _ = evaluate_model(
                model=self.model,
                data_loader=val_loader,
                loss_functions=self.loss_functions,
                cfg=self.cfg,
                device=self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            metric_key = self.cfg.save_metric
            current_metric = val_metrics.get(metric_key, 0.0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                best_metrics = val_metrics.copy()

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "cfg": asdict(self.cfg),
                "val_metrics": val_metrics,
            }

            last_ckpt = os.path.join(self.cfg.checkpoint_dir, "last.pt")
            best_ckpt = os.path.join(self.cfg.checkpoint_dir, "best.pt")
            save_checkpoint(state, last_ckpt)
            if is_best:
                save_checkpoint(state, best_ckpt)

            self.logger.info(
                f"Epoch [{epoch}/{self.cfg.epochs}] "
                f"train_total_loss={train_loss:.4f} "
                f"val_total_loss={val_loss:.4f}"
            )
            self.logger.info(f"Train loss details: {format_metrics(train_loss_dict)}")
            self.logger.info(f"Val metrics: {format_metrics(val_metrics)}")

        self.logger.info(f"Training finished. Best epoch={best_epoch}, best_{self.cfg.save_metric}={best_metric:.4f}")
        self.logger.info(f"Best metrics: {format_metrics(best_metrics)}")
        return best_epoch, best_metrics

    def _train_one_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        loss_sums = {
            "total_loss": 0.0,
            "change_loss": 0.0,
            "object_loss": 0.0,
            "action_loss": 0.0,
            "location_loss": 0.0,
        }

        for batch in train_loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch["image_t1"], batch["image_t2"])
            loss, loss_items = compute_loss(outputs, batch, self.loss_functions, self.cfg)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            for k in loss_sums:
                loss_sums[k] += float(loss_items[k].item())

        avg_loss = total_loss / max(1, n_batches)
        avg_loss_dict = {k: v / max(1, n_batches) for k, v in loss_sums.items()}
        return avg_loss, avg_loss_dict


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    loss_functions: Dict,
    cfg: ExperimentConfig,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_change_probs: List[np.ndarray] = []
    all_change_targets: List[np.ndarray] = []
    all_object_probs: List[np.ndarray] = []
    all_object_targets: List[np.ndarray] = []
    all_action_probs: List[np.ndarray] = []
    all_action_targets: List[np.ndarray] = []
    all_location_probs: List[np.ndarray] = []
    all_location_targets: List[np.ndarray] = []

    rows = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image_t1"], batch["image_t2"])
        loss, _ = compute_loss(outputs, batch, loss_functions, cfg)

        total_loss += float(loss.item())
        n_batches += 1

        change_probs = sigmoid_np(outputs["change_logits"].detach().cpu().numpy())
        object_probs = sigmoid_np(outputs["object_logits"].detach().cpu().numpy())
        action_probs = sigmoid_np(outputs["action_logits"].detach().cpu().numpy())
        location_probs = sigmoid_np(outputs["location_logits"].detach().cpu().numpy())

        all_change_probs.append(change_probs)
        all_change_targets.append(batch["change_label"].detach().cpu().numpy())

        all_object_probs.append(object_probs)
        all_object_targets.append(batch["object_labels"].detach().cpu().numpy())

        all_action_probs.append(action_probs)
        all_action_targets.append(batch["action_labels"].detach().cpu().numpy())

        all_location_probs.append(location_probs)
        all_location_targets.append(batch["location_labels"].detach().cpu().numpy())

        for i, fname in enumerate(batch["filename"]):
            rows.append(
                {
                    "filename": fname,
                    "change_prob": float(change_probs[i]),
                    "change_pred": int(change_probs[i] >= cfg.threshold),
                    "object_probs": object_probs[i],
                    "action_probs": action_probs[i],
                    "location_probs": location_probs[i],
                }
            )

    change_probs = np.concatenate(all_change_probs, axis=0)
    change_targets = np.concatenate(all_change_targets, axis=0)

    object_probs = np.concatenate(all_object_probs, axis=0)
    object_targets = np.concatenate(all_object_targets, axis=0)

    action_probs = np.concatenate(all_action_probs, axis=0)
    action_targets = np.concatenate(all_action_targets, axis=0)

    location_probs = np.concatenate(all_location_probs, axis=0)
    location_targets = np.concatenate(all_location_targets, axis=0)

    metrics = compute_all_metrics(
        change_probs=change_probs,
        change_targets=change_targets,
        object_probs=object_probs,
        object_targets=object_targets,
        action_probs=action_probs,
        action_targets=action_targets,
        location_probs=location_probs,
        location_targets=location_targets,
        threshold=cfg.threshold,
    )
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, metrics, rows
