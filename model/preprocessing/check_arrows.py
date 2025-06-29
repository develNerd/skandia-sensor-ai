import cv2
import numpy as np
import json
from pathlib import Path

# === STEP 1: Extract arrow template from labeled image ===

def extract_arrow_template(json_path, image_path, save_path="arrow_template.jpg"):
    with open(json_path) as f:
        data = json.load(f)

    img = cv2.imread(image_path)
    for shape in data["shapes"]:
        if shape["label"].lower() == "arrow":
            points = np.array(shape["points"], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(points)
            arrow_crop = img[y:y + h, x:x + w]
            cv2.imwrite(save_path, arrow_crop)
            print(f"✅ Arrow template saved to {save_path}")
            return save_path
    print("❌ Arrow label not found in JSON.")
    return None

# === STEP 2: Generate rotated templates ===

def generate_rotated_templates(template_path):
    base = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    rotations = {
        "Up": base,
        "Right": cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
        "Down": cv2.rotate(base, cv2.ROTATE_180),
        "Left": cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    return rotations

# === STEP 3: Match template to new image ===

def get_arrow_direction(test_image_path, templates, min_score_threshold=0.4):
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    best_score = -1
    best_dir = None

    for direction, tmpl in templates.items():
        result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"{direction} score: {max_val:.3f}")

        if max_val > best_score:
            best_score = max_val
            best_dir = direction
            best_location = max_loc
            best_template = tmpl

    if best_score >= min_score_threshold:
        # Visualize result
        h, w = best_template.shape
        img_color = cv2.imread(test_image_path)
        top_left = best_location
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img_color, f"{best_dir} ({best_score:.2f})", (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Arrow Detection", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_dir, best_score


# === MAIN WORKFLOW ===

# Paths (update these if needed)
arrow_image_path = "000.jpg"
arrow_json_path = "000.json"
template_path = "arrow_template.jpg"
test_image_path = "042.jpg"  # Image you want to test


# Step 2: Rotate templates
templates = generate_rotated_templates(template_path)

# Step 3: Run direction detection
direction, score = get_arrow_direction(test_image_path, templates)

# Output
if direction:
    print(f"🧭 Arrow points: {direction} (Score: {score:.3f})")
else:
    print("⚠️ No arrow detected with high confidence.")
