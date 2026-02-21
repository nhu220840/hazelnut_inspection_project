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
from src.evaluation import run_defect_kfold_cv

LABEL_MAP = {i: name for i, name in enumerate(config.DEFECT_TYPES)}


def load_models():
    """Load saved SVM (anomaly) and RF (defect) models from disk; exit if not found."""
    print("⏳ Loading models...")
    try:
        svm_path = os.path.join(config.MODEL_PATH, "anomaly_detector.pkl")
        rf_path = os.path.join(config.MODEL_PATH, "defect_classifier.pkl")
        
        svm_model = joblib.load(svm_path)
        rf_model = joblib.load(rf_path)
        print("✅ Models loaded successfully!")
        return svm_model, rf_model
    except FileNotFoundError:
        print("❌ Error: Model file not found. Please run train.py first.")
        exit()


def predict_single_image(img, svm_model, rf_model):
    """Two-stage prediction: SVM for good vs defect, then RF for defect type. Returns (status, label)."""
    processed_img, mask = remove_background(img)
    
    try:
        feats = extract_features(processed_img, mask)
        feats = feats.reshape(1, -1)
    except Exception:
        return "error", "error"

    anomaly_score = svm_model.predict(feats)[0]
    
    if anomaly_score == 1:
        return "good", "good"
    else:
        defect_code = rf_model.predict(feats)[0]
        defect_name = LABEL_MAP.get(defect_code, "unknown")
        return "defect", defect_name


def evaluate_system():
    """Evaluate on test set: good with saved model, defects with K-fold CV; print report and save confusion matrix."""
    svm_model, rf_model = load_models()
    test_root = os.path.join(config.DATA_PATH, "test")
    categories = ["good"] + config.DEFECT_TYPES
    y_true = []
    y_pred = []

    print("\n🚀 Evaluating (Good: saved model | Defect: K-fold CV)\n")

    folder_good = os.path.join(test_root, "good")
    if os.path.exists(folder_good):
        print("📂 Good: predicting with saved model...")
        for img_name in os.listdir(folder_good):
            if not img_name.endswith(".png"):
                continue
            img_path = os.path.join(folder_good, img_name)
            img = cv2.imread(img_path)
            _, final_label = predict_single_image(img, svm_model, rf_model)
            y_true.append("good")
            y_pred.append(final_label)

    print("📂 Defect: K-fold CV...")
    cv_true, cv_pred = run_defect_kfold_cv(test_root)
    y_true.extend(cv_true)
    y_pred.extend(cv_pred)

    print("\n" + "=" * 50)
    print("📊 EVALUATION REPORT (Defect = K-fold CV)")
    print("=" * 50)
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=categories)
    annot = [[str(int(v)) for v in row] for row in cm]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=categories, yticklabels=categories,
                annot_kws={'size': 11})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Defect: K-fold CV)')
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    save_path = "outputs/confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Confusion matrix saved at: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_system()