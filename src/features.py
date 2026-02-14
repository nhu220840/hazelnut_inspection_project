import cv2
import numpy as np
from skimage.feature import hog
from src import config # Import parameters from config

def extract_hog_features(image):
    """
    Extract shape features (HOG)
    """
    # 1. Resize to fixed size (required for HOG)
    resized = cv2.resize(image, config.RESIZE_DIM)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Calculate HOG
    # visualize=False: Only get data vector, don't get image for visualization
    hog_feats = hog(gray, 
                    orientations=config.HOG_ORIENTATIONS, 
                    pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                    cells_per_block=config.HOG_CELLS_PER_BLOCK, 
                    visualize=False)
    return hog_feats

def extract_color_features(image, mask=None):
    """
    Extract color features (Color Histogram)
    """
    # If mask exists, resize mask to match image (if image was resized earlier)
    # But here we calculate on the original cropped image so no need to resize image
    
    hist_feats = []
    
    # Calculate histogram for 3 color channels: Blue, Green, Red
    for channel in range(3):
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        # IMPORTANT: Pass 'mask' to NOT calculate black background color
        hist = cv2.calcHist([image], [channel], mask, [config.HIST_BINS], [0, 256])
        
        # Normalize so that large or small images have equivalent total values
        cv2.normalize(hist, hist)
        
        hist_feats.extend(hist.flatten())
        
    return np.array(hist_feats)

def extract_features(image, mask=None):
    """
    Summary function: Call both functions above and concatenate into 1 single vector
    """
    # 1. Calculate HOG
    hog_vector = extract_hog_features(image)
    
    # 2. Calculate Color (need to pass mask to remove black background)
    color_vector = extract_color_features(image, mask)
    
    # 3. Concatenate
    # Example: HOG has 1000 numbers + Color has 96 numbers = 1096-dimensional vector
    final_vector = np.hstack([hog_vector, color_vector])
    
    return final_vector