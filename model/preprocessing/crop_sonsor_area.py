import cv2


def crop_green_areas(img_np):
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