import os

# Data path
DATA_PATH = os.path.join("data", "raw", "hazelnut")
MODEL_PATH = "saved_models"

# --- IMPORTANT PARAMETERS ---

# 1. Image resize size before calculating HOG
# Hazelnut is quite round, resizing to 128x128 is sharp enough to catch defects while running fast
RESIZE_DIM = (128, 128) 

# 2. Color Histogram parameters
# Number of bins for each color channel (higher = more color detail but longer vector)
HIST_BINS = 32 

# 3. HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)  