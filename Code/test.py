import csv
import os

from src.configs import get_test_config_from_args
from src.dataset import LEVIRMCIDataset
from src.label_encoder import LabelEncoder
from src.losses import build_loss_functions
from src.model import ChangeUnderstandingBaseline
from src.trainer import evaluate_model
from src.utils import ensure_dir, format_metrics, get_device, load_checkpoint, setup_logger


def _labels_to_string(prob_vec, class_names, threshold):
    labels = [name for name, prob in zip(class_names, prob_vec) if float(prob) >= threshold]
    return ",".join(labels)


def main():
    cfg = get_test_config_from_args()
    ensure_dir(cfg.output_dir)
    logger = setup_logger(cfg.output_dir, name="test")

    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    label_encoder = LabelEncoder()

    test_dataset = LEVIRMCIDataset(
        label_path=cfg.label_path,
        data_root=cfg.data_root,
        split=cfg.test_split,
        image_size=cfg.image_size,
        label_encoder=label_encoder,
        is_train=False,
        debug_print_samples=0,
    )

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ChangeUnderstandingBaseline(pretrained=False, dropout=cfg.dropout).to(device)
    ckpt = load_checkpoint(cfg.test_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {cfg.test_checkpoint}")

    loss_functions = build_loss_functions(cfg, device)
    test_loss, metrics, rows = evaluate_model(
        model=model,
        data_loader=test_loader,
        loss_functions=loss_functions,
        cfg=cfg,
        device=device,
    )

    logger.info(f"Test total loss: {test_loss:.4f}")
    logger.info(f"Test metrics: {format_metrics(metrics)}")

    csv_path = os.path.join(cfg.output_dir, cfg.pred_csv_name)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "change_prob",
                "change_pred",
                "object_pred",
                "action_pred",
                "location_pred",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "change_prob": f"{row['change_prob']:.6f}",
                    "change_pred": row["change_pred"],
                    "object_pred": _labels_to_string(row["object_probs"], label_encoder.object_classes, cfg.threshold),
                    "action_pred": _labels_to_string(row["action_probs"], label_encoder.action_classes, cfg.threshold),
                    "location_pred": _labels_to_string(row["location_probs"], label_encoder.location_classes, cfg.threshold),
                }
            )

    logger.info(f"Predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()
