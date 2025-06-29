from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = ".jpg"

class TSensorDataset(AnomalibDataset):
    """TempSensor AD dataset class (MVTec-style).

    Supports train/test splits with ground truth masks for anomaly segmentation.

    Args:
        root (Path | str): Path to root directory containing the dataset.
        category (str): Category name (used to sub-select from the root).
        augmentations (Transform, optional): Input image transforms.
        split (str | Split | None, optional): Split to use: 'train' or 'test'.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/TempSensor",
        category: str = "default",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_custom_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )




def make_custom_ad_dataset(
        root: str | Path,
        split: str | Split | None = None,
        extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Parse custom dataset folder structure into DataFrame samples."""
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Initialize and assign labels
    samples["label_index"] = pd.NA
    samples.loc[samples.label == "good", "label_index"] = LabelName.NORMAL
    samples.loc[samples.label == "anomaly", "label_index"] = LabelName.ABNORMAL
    samples["label_index"] = samples["label_index"].astype("Int64")

    # Process mask paths
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    mask_dict = {Path(p).stem: p for p in mask_samples["image_path"].tolist()}
    samples["mask_path"] = ""
    for idx, row in samples.iterrows():
        if row["split"] == "test" and row["label_index"] == LabelName.ABNORMAL:
            stem = Path(row["image_path"]).stem
            if stem in mask_dict:
                samples.at[idx, "mask_path"] = mask_dict[stem]

    # Validate mask-image correspondence
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if len(abnormal_samples) and not abnormal_samples.apply(
            lambda x: Path(x.image_path).stem in Path(x.mask_path).stem if x.mask_path else True,
            axis=1,
    ).all():
        raise MisMatchError(
            "Mismatch between anomalous images and ground truth masks. "
            "Ensure mask files in 'ground_truth/anomaly' match test anomalies."
        )

    # Determine task type safely
    if "mask_path" in samples.columns:
        has_masks = samples["mask_path"].notna() & (samples["mask_path"] != "")
        samples.attrs["task"] = "segmentation" if has_masks.any() else "classification"
    else:
        samples.attrs["task"] = "classification"

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples