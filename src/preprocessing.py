import cv2
import numpy as np

def remove_background(image, return_full_mask=False, padding=0):
    h0, w0 = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white = np.count_nonzero(mask == 255)
    if white > mask.size * 0.5:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask

    c = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros((h0, w0), dtype=np.uint8)
    cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(c)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w0, x + w + padding)
    y2 = min(h0, y + h + padding)

    cropped_img = image[y1:y2, x1:x2]

    if return_full_mask:
        return cropped_img, clean_mask
    else:
        return cropped_img, clean_mask[y1:y2, x1:x2]
