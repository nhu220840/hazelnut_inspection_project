import os
import cv2
import glob
import numpy as np
import pandas as pd  # Dùng để hiển thị bảng thống kê dữ liệu cho đẹp
from src.preprocessing import remove_background
from src.features import extract_features
from src.models import AnomalyDetector, DefectClassifier
from src.augmentation import augment_image # <--- [NEW] Import module mới
from src import config

# ... (Giữ nguyên hàm load_images_and_extract_features cũ nếu muốn, hoặc dùng logic mới dưới đây)

def process_single_image(img):
    """Hàm phụ trợ: Xử lý 1 ảnh -> Trả về feature vector"""
    processed_img, mask = remove_background(img)
    try:
        # extract_features cần mask để loại bỏ nền đen
        return extract_features(processed_img, mask)
    except:
        return None

def main():
    # ==========================================
    # GIAI ĐOẠN 1: HUẤN LUYỆN ONE-CLASS SVM (GIỮ NGUYÊN)
    # ==========================================
    print("\n=== STAGE 1: TRAINING ANOMALY DETECTOR ===")
    train_good_path = os.path.join(config.DATA_PATH, "train", "good")
    image_paths = glob.glob(os.path.join(train_good_path, "*.png"))
    
    X_good = []
    print(f"Loading {len(image_paths)} good images for SVM...")
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        feat = process_single_image(img)
        if feat is not None:
            X_good.append(feat)
            
    if X_good:
        # Lưu ý: Nhớ chỉnh nu=0.2 trong src/models.py như đã bàn ở bước trước
        svm_model = AnomalyDetector()
        svm_model.train(X_good) 
        svm_model.save("anomaly_detector.pkl")
    else:
        print("❌ Lỗi: Không tìm thấy dữ liệu train/good")

    # ==========================================
    # GIAI ĐOẠN 2: HUẤN LUYỆN RANDOM FOREST (CÓ AUGMENTATION)
    # ==========================================
    print("\n=== STAGE 2: TRAINING DEFECT CLASSIFIER WITH AUGMENTATION ===")
    
    defect_types = ['crack', 'cut', 'hole', 'print']
    X_defects = []
    y_defects = []
    
    test_root_path = os.path.join(config.DATA_PATH, "test")
    
    # Bảng thống kê để đưa vào báo cáo
    stats = []

    for idx, defect_name in enumerate(defect_types):
        folder_path = os.path.join(test_root_path, defect_name)
        if not os.path.exists(folder_path): continue
            
        img_paths = glob.glob(os.path.join(folder_path, "*.png"))
        original_count = len(img_paths)
        
        print(f"Processing '{defect_name}': {original_count} images found...")
        
        for path in img_paths:
            img = cv2.imread(path)
            if img is None: continue
            
            # --- [NEW] BƯỚC NHÂN BẢN DỮ LIỆU ---
            # Tạo ra 6 biến thể từ 1 ảnh gốc
            aug_imgs = augment_image(img)
            
            # Trích xuất đặc trưng cho cả 6 ảnh này
            for aug_img in aug_imgs:
                feat = process_single_image(aug_img)
                if feat is not None:
                    X_defects.append(feat)
                    y_defects.append(idx)
        
        # Ghi lại thống kê
        final_count = original_count * 6
        stats.append({
            "Loại lỗi": defect_name, 
            "Số lượng gốc": original_count, 
            "Sau Augmentation": final_count
        })

    # In bảng thống kê ra màn hình (Copy bảng này vào báo cáo rất đẹp)
    print("\n📊 BẢNG THỐNG KÊ DỮ LIỆU SAU KHI NHÂN BẢN:")
    df_stats = pd.DataFrame(stats)
    print(df_stats)
    print(f"\nTổng cộng mẫu để train Random Forest: {len(X_defects)}")

    if X_defects:
        rf_model = DefectClassifier()
        rf_model.train(X_defects, y_defects)
        rf_model.save("defect_classifier.pkl")
    else:
        print("❌ Lỗi: Không tìm thấy dữ liệu lỗi (defect)!")

if __name__ == "__main__":
    main()