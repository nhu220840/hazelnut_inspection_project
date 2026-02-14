import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src import config

class AnomalyDetector:
    def __init__(self):
        # CẤU HÌNH TỐI ƯU (Best Config: Accuracy ~75%)
        # 1. Tắt PCA (để giữ chi tiết vết nứt/cắt)
        # 2. nu=0.25 (Mức cân bằng, không quá lỏng như 0.05, không quá gắt như 0.3)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.99)), # <--- ĐÃ TẮT PCA
            ('svm', OneClassSVM(kernel='rbf', nu=0.25, gamma='scale')) 
        ])

    def train(self, X_train):
        print(f"Training Anomaly Detector với {len(X_train)} mẫu...")
        self.model.fit(X_train)
        print("Done.")

    def save(self, filename="anomaly_detector.pkl"):
        # --- FIX LỖI: Tự động tạo thư mục nếu chưa có ---
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        # -----------------------------------------------
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")

class DefectClassifier:
    """
    Giai đoạn 2: Phân loại lỗi (Random Forest)
    Nhiệm vụ: Nếu đã biết là lỗi, thì đó là lỗi gì? (Print, Cut, Crack...)
    """
    def __init__(self):
        # Thêm class_weight='balanced' để chú ý hơn đến các lỗi ít gặp (Hole, Cut)
        self.model = RandomForestClassifier(n_estimators=100, 
                                            class_weight='balanced',
                                            random_state=42)

    def train(self, X_train, y_train):
        print(f"Training Defect Classifier với {len(X_train)} mẫu...")
        self.model.fit(X_train, y_train)
        print("Done.")

    def save(self, filename="defect_classifier.pkl"):
        # --- FIX LỖI: Tự động tạo thư mục nếu chưa có ---
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        # -----------------------------------------------
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")