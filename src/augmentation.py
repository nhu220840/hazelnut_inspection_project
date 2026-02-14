import cv2
import numpy as np

def augment_image(image):
    """Augment image data using OpenCV. Returns list of 6 variants (Original + 5 transformed images)."""
    augmented_images = []
    
    augmented_images.append(image)
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented_images.append(cv2.flip(image, 1))
    augmented_images.append(cv2.flip(image, 0))
    
    return augmented_images