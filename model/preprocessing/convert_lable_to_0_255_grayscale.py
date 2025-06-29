import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === Set your 'chips' folder path ===
chips_dir = Path(r"C:\Users\isaac\PycharmProjects\EngineeringProject\chips")
output_dir = chips_dir / "binary_masks"
output_dir.mkdir(exist_ok=True)

# Loop through all *_json folders
for json_folder in tqdm(chips_dir.glob("*_json"), desc="Converting label.png to binary"):
    label_path = json_folder / "label.png"
    if not label_path.exists():
        print(f"❌ Skipping {json_folder.name} (no label.png found)")
        continue

    # Load label.png as grayscale
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    # Convert to binary mask (0 for background, 255 for anything else)
    binary = np.where(label > 0, 255, 0).astype(np.uint8)

    # Save with the same base name (e.g., 000.png)
    out_name = json_folder.stem.replace("_json", "") + ".png"
    cv2.imwrite(str(output_dir / out_name), binary)

print(f"✅ All masks saved to: {output_dir}")
