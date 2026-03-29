import os

import torch

from src.configs import get_train_config_from_args
from src.dataset import build_dataloaders
from src.label_encoder import LabelEncoder
from src.losses import build_loss_functions
from src.model import ChangeUnderstandingBaseline
from src.trainer import Trainer
from src.utils import ensure_dir, get_device, load_checkpoint, save_json, set_seed, setup_logger


def main():
    cfg = get_train_config_from_args()
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.checkpoint_dir)

    logger = setup_logger(cfg.output_dir, name="train")
    logger.info("Config loaded.")
    logger.info(str(cfg.to_dict()))

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    label_encoder = LabelEncoder()
    loaders = build_dataloaders(cfg, label_encoder)

    model = ChangeUnderstandingBaseline(pretrained=cfg.pretrained_backbone, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)

    loss_functions = build_loss_functions(cfg, device)

    if cfg.resume_checkpoint:
        ckpt = load_checkpoint(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info(f"Resumed from checkpoint: {cfg.resume_checkpoint}")

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_functions=loss_functions,
        device=device,
        logger=logger,
    )

    best_epoch, best_metrics = trainer.train(loaders["train"], loaders["val"])

    best_json_path = os.path.join(cfg.output_dir, "best_metrics.json")
    save_json({"best_epoch": best_epoch, "best_metrics": best_metrics}, best_json_path)
    logger.info(f"Best metrics saved to: {best_json_path}")


if __name__ == "__main__":
    main()
