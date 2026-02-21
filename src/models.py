import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src import config

class AnomalyDetector:
    """One-class SVM (with StandardScaler) to detect good vs defective hazelnuts."""

    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', OneClassSVM(kernel='rbf', nu=0.25, gamma='scale')) 
        ])

    def train(self, X_train):
        """Fit on feature matrix of good samples only."""
        print(f"Training Anomaly Detector with {len(X_train)} samples...")
        self.model.fit(X_train)

    def save(self, filename="anomaly_detector.pkl"):
        """Serialize pipeline to config.MODEL_PATH."""
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")


class DefectClassifier:
    """Random Forest classifier for defect types (crack, cut, hole, print)."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, 
                                            class_weight='balanced',
                                            random_state=42)

    def train(self, X_train, y_train):
        """Fit on defect features and integer labels."""
        print(f"Training Defect Classifier with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def save(self, filename="defect_classifier.pkl"):
        """Serialize model to config.MODEL_PATH."""
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
            
        save_path = os.path.join(config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")