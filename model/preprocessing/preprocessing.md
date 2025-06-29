# 🛠️ Preprocessing Scripts for Sensor Image Data

This folder contains preprocessing scripts used to prepare sensor images for training and inference in the Skandia AI-based vision system. Each script performs a dedicated task to clean, crop, convert, or format the dataset as required by the model pipeline.

---

## Scripts Overview

### `check_arrows.py`
Checks for the direction of the arrowhead on the sensor chip.  
This can be used for visual validation or to conditionally process only images with a clearly visible orientation indicator.

---

### `convert_lable_to_0_255_grayscale.py`
Converts labeled images from LabelMe format (typically with red bounding boxes) into grayscale masks using white bounding boxes.

- ✅ **Input**: LabelMe-style RGB annotation images  
- ✅ **Output**: Binary grayscale images (0 for background, 255 for anomaly)

This step ensures the anomaly detection model correctly interprets regions of interest.

---

### `crop_sonsor_area.py`
Crops the sensor region from the raw images.  
This focuses the training and inference pipeline on relevant image sections and removes background noise.

- ✅ Useful for eliminating unnecessary pixels  
- ✅ Improves model performance and inference speed

---

### `pad_and_resize_data.py`
Pads the image to ensure it is square, then resizes it (e.g., to 256×256 or 512×512).

- ✅ Maintains aspect ratio  
- ✅ Required for uniform model input size  
- ✅ Useful for Anomalib and CNNs expecting fixed image dimensions

---

## Result



##  Usage

Run scripts sequentially or independently as needed:

```bash
python check_arrows.py
python convert_lable_to_0_255_grayscale.py
python crop_sonsor_area.py
python pad_and_resize_data.py
