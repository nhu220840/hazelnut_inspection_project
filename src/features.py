import cv2
import numpy as np
from skimage.feature import hog
from src import config # Import tham số từ config

def extract_hog_features(image):
    """
    Trích xuất đặc trưng hình dáng (HOG)
    """
    # 1. Resize về kích thước cố định (bắt buộc với HOG)
    resized = cv2.resize(image, config.RESIZE_DIM)
    
    # 2. Chuyển sang ảnh xám
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Tính HOG
    # visualize=False: Chỉ lấy vector dữ liệu, không lấy ảnh để vẽ
    hog_feats = hog(gray, 
                    orientations=config.HOG_ORIENTATIONS, 
                    pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                    cells_per_block=config.HOG_CELLS_PER_BLOCK, 
                    visualize=False)
    return hog_feats

def extract_color_features(image, mask=None):
    """
    Trích xuất đặc trưng màu sắc (Color Histogram)
    """
    # Nếu có mask, resize mask cho khớp với ảnh (nếu ảnh bị resize trước đó)
    # Nhưng ở đây ta tính trên ảnh gốc đã crop nên không cần resize ảnh
    
    hist_feats = []
    
    # Tính histogram cho 3 kênh màu: Blue, Green, Red
    for channel in range(3):
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        # QUAN TRỌNG: Truyền 'mask' vào để KHÔNG tính màu đen của nền
        hist = cv2.calcHist([image], [channel], mask, [config.HIST_BINS], [0, 256])
        
        # Chuẩn hóa (Normalize) để ảnh to hay nhỏ thì tổng giá trị vẫn tương đương nhau
        cv2.normalize(hist, hist)
        
        hist_feats.extend(hist.flatten())
        
    return np.array(hist_feats)

def extract_features(image, mask=None):
    """
    Hàm tổng hợp: Gọi cả 2 hàm trên và nối lại thành 1 vector duy nhất
    """
    # 1. Tính HOG
    hog_vector = extract_hog_features(image)
    
    # 2. Tính Color (cần truyền mask để loại bỏ nền đen)
    color_vector = extract_color_features(image, mask)
    
    # 3. Nối lại (Concatenate)
    # Ví dụ: HOG có 1000 số + Color có 96 số = Vector 1096 chiều
    final_vector = np.hstack([hog_vector, color_vector])
    
    return final_vector