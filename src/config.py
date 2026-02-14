import os

# Đường dẫn dữ liệu
DATA_PATH = os.path.join("data", "raw", "hazelnut")
MODEL_PATH = "saved_models"

# --- THAM SỐ QUAN TRỌNG ---

# 1. Kích thước ảnh resize trước khi tính HOG
# Hạt Hazelnut khá tròn, resize về 128x128 là đủ nét để bắt lỗi mà chạy nhanh
RESIZE_DIM = (128, 128) 

# 2. Tham số Color Histogram
# Số lượng bin cho mỗi kênh màu (càng cao càng chi tiết màu nhưng vector càng dài)
HIST_BINS = 32 

# 3. Tham số HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)  