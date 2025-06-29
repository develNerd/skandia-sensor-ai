import cv2
import os
from pathlib import Path
from tqdm import tqdm

def pad_and_resize_to_square(image_path: Path, output_path: Path, final_size=256):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Skipped: {image_path} (invalid image)")
        return

    h, w = image.shape[:2]
    # Make square by padding with black
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize to target size
    resized = cv2.resize(image, (final_size, final_size), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)


def preprocess_dataset(root_dir="datasets/TempSensor/sensor", final_size=256):
    image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    root = Path(root_dir)

    print(f"🔍 Starting preprocessing in: {root.resolve()}")
    for path in tqdm(list(root.rglob("*"))):
        if path.suffix not in image_extensions or not path.is_file():
            continue

        pad_and_resize_to_square(path, path, final_size=final_size)  # Overwrite in place

    print("✅ All images padded and resized.")




# Run it
if __name__ == "__main__":
    preprocess_dataset("datasets/TempSensor/sensor", final_size=256)
