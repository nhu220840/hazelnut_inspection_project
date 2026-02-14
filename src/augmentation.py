import cv2
import numpy as np

def augment_image(image):
    """
    Hàm nhân bản dữ liệu ảnh thủ công bằng OpenCV.
    Input: 1 ảnh gốc (OpenCV BGR).
    Output: List gồm 6 biến thể (Gốc + 5 ảnh biến đổi).
    """
    augmented_images = []
    
    # 1. Giữ nguyên ảnh gốc
    augmented_images.append(image)
    
    # 2. Xoay 90 độ (Chiều kim đồng hồ)
    img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(img_90)
    
    # 3. Xoay 180 độ
    img_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_images.append(img_180)
    
    # 4. Xoay 270 độ (Ngược chiều kim đồng hồ)
    img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(img_270)
    
    # 5. Lật ngang (Horizontal Flip) - Giống soi gương
    img_flip_h = cv2.flip(image, 1)
    augmented_images.append(img_flip_h)

    # 6. Lật dọc (Vertical Flip)
    img_flip_v = cv2.flip(image, 0)
    augmented_images.append(img_flip_v)
    
    return augmented_images