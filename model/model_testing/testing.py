import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from anomalib.models import EfficientAd
from sklearn.metrics import roc_curve, auc
from anomalib.visualization import Visualizer
from anomalib.engine import Engine

from model.datasets.dataset_training.training import TSensorDataModule
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def patched_generate_output_filename(*, input_path: Path, output_path: Path, filename_suffix: str = "", **kwargs) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = Path(input_path).stem
    return output_path / f"{image_name}_{filename_suffix}.png"


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'Pixel AUROC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-level ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_metric_barchart(metrics: dict):
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.grid(axis='y')
    plt.show()


def main():
    # Monkey patch anomalib filename generator for saving outputs
    from anomalib.utils import path as path_utils
    from anomalib.visualization.image import visualizer

    path_utils.generate_output_filename = patched_generate_output_filename
    visualizer.generate_output_filename = patched_generate_output_filename

    # Initialize visualizer and datamodule
    vis = Visualizer()
    datamodule = TSensorDataModule(
        root="./datasets/TempSensor",
        category="sensor",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,  # Important for Windows
    )

    engine = Engine(max_epochs=10)
    model  = EfficientAd(visualizer=vis)  # Keep model reference for direct inference if needed

    # Run test (evaluates the model on test dataset)
    results = engine.test(
        datamodule=datamodule,
        model=model,
        ckpt_path="results/EfficientAd/TSensorDataModule/sensor/v30/weights/lightning/model.ckpt"
    )

    # Print evaluation metrics summary
    print("TempSensor Evaluation Results:")
    if results and isinstance(results, list):
        for k, v in results[0].items():
            print(f"{k}: {v:.4f}")

    # Collect pixel-level scores and masks for ROC plotting
    all_scores = []
    all_labels = []

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images = batch.image.to(device)
            # gt_mask may be None if task is classification
            gt_masks = getattr(batch, "gt_mask", None)
            if gt_masks is None:
                continue  # Skip if no pixel-level masks (cannot plot pixel-level ROC)
            gt_masks = gt_masks.to(device)
            outputs = model(images)
            anomaly_maps = getattr(outputs, "anomaly_map", None)

            if anomaly_maps is None:
                print("No anomaly maps found; skipping pixel ROC curve.")
                continue

            for i in range(images.size(0)):
                scores = anomaly_maps[i].flatten().cpu().numpy()
                labels = gt_masks[i].flatten().cpu().numpy()
                all_scores.extend(scores)
                all_labels.extend(labels)

    if all_scores and all_labels:
        print("Plotting pixel-level ROC curve...")
        plot_roc_curve(all_labels, all_scores)
    else:
        print("No pixel-level ground truth masks found; skipping pixel ROC curve.")

    # Optional: plot bar chart for the key metrics from results
    if results and isinstance(results, list):
        metrics_to_plot = {k: float(v) for k, v in results[0].items() if isinstance(v, (float, int))}
        plot_metric_barchart(metrics_to_plot)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Safe for PyInstaller/frozen apps
    main()
