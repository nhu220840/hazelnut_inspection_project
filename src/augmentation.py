import cv2
import numpy as np

def augment_image(image):
    """
    Function to manually augment image data using OpenCV.
    Input: 1 original image (OpenCV BGR).
    Output: List of 6 variants (Original + 5 transformed images).
    """
    augmented_images = []
    
    # 1. Keep original image
    augmented_images.append(image)
    
    # 2. Rotate 90 degrees (Clockwise)
    img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(img_90)
    
    # 3. Rotate 180 degrees
    img_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_images.append(img_180)
    
    # 4. Rotate 270 degrees (Counter-clockwise)
    img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(img_270)
    
    # 5. Horizontal Flip - Like a mirror
    img_flip_h = cv2.flip(image, 1)
    augmented_images.append(img_flip_h)

    # 6. Vertical Flip
    img_flip_v = cv2.flip(image, 0)
    augmented_images.append(img_flip_v)
    
    return augmented_images