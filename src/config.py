import os

DATA_PATH = os.path.join("data", "raw", "hazelnut")
MODEL_PATH = "saved_models"

RESIZE_DIM = (128, 128)
HIST_BINS = 32
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)  