import cv2
import numpy as np
from skimage.feature import hog
from src import config


def extract_hog_features(image):
    """Compute HOG descriptor on resized grayscale image."""
    resized = cv2.resize(image, config.RESIZE_DIM)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    hog_feats = hog(gray, 
                    orientations=config.HOG_ORIENTATIONS, 
                    pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                    cells_per_block=config.HOG_CELLS_PER_BLOCK, 
                    visualize=False)
    return hog_feats


def extract_color_features(image, mask=None):
    """Compute normalized per-channel histograms (BGR), optionally masked."""
    hist_feats = []
    
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], mask, [config.HIST_BINS], [0, 256])
        cv2.normalize(hist, hist)
        hist_feats.extend(hist.flatten())
        
    return np.array(hist_feats)


def extract_features(image, mask=None):
    """Concatenate HOG and color histogram features into one vector."""
    hog_vector = extract_hog_features(image)
    color_vector = extract_color_features(image, mask)
    final_vector = np.hstack([hog_vector, color_vector])
    return final_vector