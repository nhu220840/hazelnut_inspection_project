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
        # OPTIMAL CONFIGURATION (Best Config: Accuracy ~75%)
        # 1. Disable PCA (to preserve crack/cut details)
        # 2. nu=0.25 (Balanced level, not too loose like 0.05, not too strict like 0.3)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=0.99)), # <--- PCA DISABLED
            ('svm', OneClassSVM(kernel='rbf', nu=0.25, gamma='scale')) 
        ])

    def train(self, X_train):
        print(f"Training Anomaly Detector with {len(X_train)} samples...")
        self.model.fit(X_train)
        print("Done.")

    def save(self, filename="anomaly_detector.pkl"):
        # --- FIX: Automatically create directory if it doesn't exist ---
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        # -----------------------------------------------
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")

class DefectClassifier:
    """
    Stage 2: Defect Classification (Random Forest)
    Task: If it's known to be a defect, what type is it? (Print, Cut, Crack...)
    """
    def __init__(self):
        # Add class_weight='balanced' to pay more attention to rare defects (Hole, Cut)
        self.model = RandomForestClassifier(n_estimators=100, 
                                            class_weight='balanced',
                                            random_state=42)

    def train(self, X_train, y_train):
        print(f"Training Defect Classifier with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("Done.")

    def save(self, filename="defect_classifier.pkl"):
        # --- FIX: Automatically create directory if it doesn't exist ---
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        # -----------------------------------------------
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")