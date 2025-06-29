import cv2
from pathlib import Path
from flask import Flask, request, jsonify
from anomalib.models import EfficientAd
from anomalib.data import PredictDataset
from anomalib.engine import Engine
import torch
import os

app = Flask(__name__)

torch.set_float32_matmul_precision("high")
engine = Engine()
model = EfficientAd()
ckpt_path = "../anomalib/results/EfficientAd/TSensorDataModule/sensor/v30/weights/lightning/model.ckpt"

import cv2
import numpy as np


def crop_green_area(img_np):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # Green range (same as your batch script)
    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)

    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological opening to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find external contours on cleaned mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop image to bounding rectangle of largest green contour
        cropped = img_np[y:y + h, x:x + w]
        return cropped
    else:
        # No green contours found, return original image
        return img_np


def pad_and_resize_to_square(image_np, final_size=256):
    h, w = image_np.shape[:2]

    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        image_np = cv2.copyMakeBorder(image_np, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        image_np = cv2.copyMakeBorder(image_np, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    resized = cv2.resize(image_np, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return resized

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    temp_dir = Path("temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    original_path = temp_dir / img_file.filename
    img_file.save(original_path)

    # Load original image as numpy array (BGR)
    img_np = cv2.imread(str(original_path))
    if img_np is None:
        return jsonify({"error": "Invalid image"}), 400

    # Crop green area
    cropped_np = crop_green_area(img_np)

    # Rotate to portrait if width > height
    if cropped_np.shape[1] > cropped_np.shape[0]:
        cropped_np = cv2.rotate(cropped_np, cv2.ROTATE_90_CLOCKWISE)

    if np.array_equal(cropped_np, img_np):
        # No green detected
        return jsonify({'anomaly': True, 'filename': "", 'score': 1.0})


    # Pad and resize cropped image to square 256x256
    processed_np = pad_and_resize_to_square(cropped_np, final_size=256)

    # Save processed image (overwriting original, or use new filename)
    processed_path = temp_dir / f"processed_{img_file.filename}"
    cv2.imwrite(str(processed_path), processed_np)

    # Pass the processed image path to PredictDataset
    dataset = PredictDataset(path=processed_path, image_size=(256, 256))
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)

    if not predictions:
        return jsonify({"error": "Prediction failed"}), 500

    pred = predictions[0]
    result = {
        "filename": str(pred.image_path),
        "anomaly": bool(pred.pred_label),
        "score": float(pred.pred_score),
    }

    print(result)

    return jsonify(result)

if __name__ == "__main__":
    print("Starting API...")
    app.run(debug=True, port=5000)
