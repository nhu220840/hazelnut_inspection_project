import os
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing import remove_background
from src.features import extract_features
from src import config

LABEL_MAP = {0: 'crack', 1: 'cut', 2: 'hole', 3: 'print'}

def load_models():
    """Load 2 trained models from .pkl files"""
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
    """Prediction function for a single image (2-stage pipeline)"""
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
    svm_model, rf_model = load_models()
    
    test_root = os.path.join(config.DATA_PATH, "test")
    categories = ['good', 'crack', 'cut', 'hole', 'print']
    
    y_true = []
    y_pred = []
    
    print("\n🚀 Starting test on Test set...\n")
    
    for category in categories:
        folder_path = os.path.join(test_root, category)
        if not os.path.exists(folder_path): continue
            
        print(f"📂 Testing category: {category}...")
        for img_name in os.listdir(folder_path):
            if not img_name.endswith(".png"): continue
            
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            
            _, final_label = predict_single_image(img, svm_model, rf_model)
            
            y_true.append(category)
            y_pred.append(final_label)

    print("\n" + "="*40)
    print("📊 EVALUATION REPORT")
    print("="*40)
    
    print(classification_report(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Hazelnut Inspection System')
    
    if not os.path.exists("outputs"): os.makedirs("outputs")
    save_path = "outputs/confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Confusion matrix saved at: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_system()