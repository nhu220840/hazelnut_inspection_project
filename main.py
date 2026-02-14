import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing import remove_background
from src.features import extract_features
from src import config

# --- CẤU HÌNH ---
# Định nghĩa lại mapping nhãn cho khớp với lúc train
LABEL_MAP = {0: 'crack', 1: 'cut', 2: 'hole', 3: 'print'}
# Thêm nhãn 'good' cho việc đánh giá tổng thể
FULL_LABEL_MAP = {0: 'crack', 1: 'cut', 2: 'hole', 3: 'print', 4: 'good'}

def load_models():
    """Load 2 model đã train từ file .pkl"""
    print("⏳ Loading models...")
    try:
        # Lưu ý: Lúc save ta save cái pipeline, nên lúc load ta được object pipeline
        svm_path = os.path.join(config.MODEL_PATH, "anomaly_detector.pkl")
        rf_path = os.path.join(config.MODEL_PATH, "defect_classifier.pkl")
        
        svm_model = joblib.load(svm_path)
        rf_model = joblib.load(rf_path)
        print("✅ Models loaded successfully!")
        return svm_model, rf_model
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file model. Hãy chạy train.py trước.")
        exit()

def predict_single_image(img, svm_model, rf_model):
    """
    Hàm dự đoán cho 1 bức ảnh duy nhất (Pipeline 2 giai đoạn)
    """
    # 1. Tiền xử lý
    processed_img, mask = remove_background(img)
    
    # 2. Trích xuất đặc trưng
    try:
        feats = extract_features(processed_img, mask)
        feats = feats.reshape(1, -1) # Reshape thành (1, n_features) để predict
    except Exception as e:
        return "error", "error"

    # 3. Giai đoạn 1: Anomaly Detection
    # SVM output: 1 (Inlier/Good), -1 (Outlier/Defect)
    anomaly_score = svm_model.predict(feats)[0]
    
    if anomaly_score == 1:
        return "good", "good" # Kết luận là Good, không cần qua bước 2
    else:
        # 4. Giai đoạn 2: Defect Classification
        defect_code = rf_model.predict(feats)[0]
        defect_name = LABEL_MAP.get(defect_code, "unknown")
        return "defect", defect_name

def evaluate_system():
    svm_model, rf_model = load_models()
    
    test_root = os.path.join(config.DATA_PATH, "test")
    categories = ['good', 'crack', 'cut', 'hole', 'print']
    
    y_true = [] # Nhãn thực tế
    y_pred = [] # Nhãn máy đoán
    
    print("\n🚀 Bắt đầu chạy kiểm thử trên tập Test...\n")
    
    for category in categories:
        folder_path = os.path.join(test_root, category)
        if not os.path.exists(folder_path): continue
            
        print(f"📂 Testing category: {category}...")
        for img_name in os.listdir(folder_path):
            if not img_name.endswith(".png"): continue
            
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            
            # --- CHẠY DỰ ĐOÁN ---
            is_good, final_label = predict_single_image(img, svm_model, rf_model)
            
            # Ghi nhận kết quả
            y_true.append(category)
            y_pred.append(final_label)

    # --- TÍNH TOÁN & HIỂN THỊ KẾT QUẢ ---
    print("\n" + "="*40)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*40)
    
    # 1. Classification Report (Text)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 2. Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label (Máy đoán)')
    plt.ylabel('True Label (Thực tế)')
    plt.title('Confusion Matrix - Hazelnut Inspection System')
    
    # Lưu hình ảnh để cho vào báo cáo
    if not os.path.exists("outputs"): os.makedirs("outputs")
    save_path = "outputs/confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Đã lưu biểu đồ Ma trận nhầm lẫn tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_system()