import torch
from albumentations import PadIfNeeded
from anomalib.metrics import F1Score, AUROC, Evaluator
from anomalib.visualization import ImageVisualizer
from torchvision.transforms.v2 import Compose, Resize, Pad

torch.set_float32_matmul_precision('high')

import logging
from pathlib import Path
import torch
from torchvision.transforms.v2 import Transform
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from dataset import TSensorDataset  # adjust path if needed

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


class TSensorDataModule(AnomalibDataModule):
    """TempSensor AD Datamodule for MVTec-style dataset structure."""

    def __init__(
        self,
        root: Path | str = "./datasets/TempSensor",
        category: str = "sensor",  # Default to match your structure
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root)
        self.category = category
        self.augmentations = augmentations

    def _setup(self, _stage: str | None = None) -> None:
        category_path = self.root / self.category
        self.train_data = TSensorDataset(
            root=self.root,
            category=self.category,
            augmentations=self.train_augmentations or self.augmentations,
            split=Split.TRAIN,
        )
        self.test_data = TSensorDataset(
            root=self.root,
            category=self.category,
            augmentations=self.test_augmentations or self.augmentations,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Check if the expected directory structure exists."""
        expected_dir = self.root / self.category / "train" / "good"
        if not expected_dir.exists():
            raise FileNotFoundError(f"Expected directory not found: {expected_dir}")
        logger.info(f"Dataset directory verified: {expected_dir}")


from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.utils import path as path_utils
from pathlib import Path


def main():


    datamodule = TSensorDataModule(
        root="../images",
        category="sensor",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    # Create visualizer with default settings
    visualizer = ImageVisualizer()



    f1_score = F1Score(fields=["pred_label", "gt_label"])
    auroc = AUROC(fields=["pred_score", "gt_label"])

    # Create evaluator with test metrics (for validation, use val_metrics arg)
    evaluator = Evaluator(test_metrics=[f1_score, auroc])

    model = EfficientAd()
    engine = Engine(max_epochs=10)
    engine.fit(datamodule=datamodule, model=model)

    print("\nLogged Metrics:")
    for metric, value in engine.trainer.callback_metrics.items():
        print(f"{metric}: {value:.4f}")




if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional, but safe for PyInstaller/frozen apps
    main()
