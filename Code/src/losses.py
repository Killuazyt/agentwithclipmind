from typing import Dict, Optional

import torch
import torch.nn as nn

from src.configs import ExperimentConfig


def _to_tensor_or_none(v: Optional[float], device: torch.device):
    if v is None:
        return None
    return torch.tensor([v], dtype=torch.float32, device=device)


def _to_vec_or_none(v, device: torch.device):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return torch.tensor(v, dtype=torch.float32, device=device)
    return None


def build_loss_functions(cfg: ExperimentConfig, device: torch.device) -> Dict[str, nn.Module]:
    """构建损失函数（预留不平衡权重接口）。"""
    change_pos_weight = _to_tensor_or_none(cfg.change_loss_pos_weight, device)
    object_weight = _to_vec_or_none(cfg.object_loss_weight, device)
    action_weight = _to_vec_or_none(cfg.action_loss_weight, device)
    location_weight = _to_vec_or_none(cfg.location_loss_weight, device)

    return {
        "change": nn.BCEWithLogitsLoss(pos_weight=change_pos_weight),
        "object": nn.BCEWithLogitsLoss(weight=object_weight),
        "action": nn.BCEWithLogitsLoss(weight=action_weight),
        "location": nn.BCEWithLogitsLoss(weight=location_weight),
    }


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_functions: Dict[str, nn.Module],
    cfg: ExperimentConfig,
):
    """联合损失: change + λ_obj*object + λ_act*action + λ_loc*location。"""
    change_loss = loss_functions["change"](outputs["change_logits"], batch["change_label"].float())
    object_loss = loss_functions["object"](outputs["object_logits"], batch["object_labels"].float())
    action_loss = loss_functions["action"](outputs["action_logits"], batch["action_labels"].float())
    location_loss = loss_functions["location"](outputs["location_logits"], batch["location_labels"].float())

    total_loss = (
        change_loss
        + cfg.lambda_obj * object_loss
        + cfg.lambda_act * action_loss
        + cfg.lambda_loc * location_loss
    )

    loss_items = {
        "total_loss": total_loss.detach(),
        "change_loss": change_loss.detach(),
        "object_loss": object_loss.detach(),
        "action_loss": action_loss.detach(),
        "location_loss": location_loss.detach(),
    }
    return total_loss, loss_items
