import os
import cv2
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.models import AnomalyDetector, DefectClassifier
from src.augmentation import augment_image
from src import config
from src.evaluation import get_defect_paths_labels, run_defect_kfold_cv, image_to_feature


def main():
    """Train anomaly detector (SVM on good images), then defect classifier (RF with K-fold CV and 80% train split)."""
    print("\n=== STAGE 1: TRAINING ANOMALY DETECTOR ===")
    train_good_path = os.path.join(config.DATA_PATH, "train", "good")
    image_paths = glob.glob(os.path.join(train_good_path, "*.png"))
    
    X_good = []
    print(f"Loading {len(image_paths)} good images for SVM...")
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
        feat = image_to_feature(img)
        if feat is not None:
            X_good.append(feat)
            
    if X_good:
        svm_model = AnomalyDetector()
        svm_model.train(X_good) 
        svm_model.save("anomaly_detector.pkl")
    else:
        print("❌ Error: No train/good data found")

    print("\n=== STAGE 2: TRAINING DEFECT CLASSIFIER (NO LEAKAGE) ===")
    defect_types = config.DEFECT_TYPES
    test_root_path = os.path.join(config.DATA_PATH, "test")

    all_paths, all_labels = get_defect_paths_labels(test_root_path)
    if len(all_paths) == 0:
        print("❌ Error: No defect data found!")
        return

    for idx, defect_name in enumerate(defect_types):
        n = int((all_labels == idx).sum())
        if n > 0:
            print(f"  '{defect_name}': {n} images")

    n_splits = min(5, int(np.bincount(all_labels).min()))
    n_splits = max(2, n_splits)
    print(f"\n📐 K-FOLD CROSS-VALIDATION (k={n_splits})")
    cv_y_true, cv_y_pred = run_defect_kfold_cv(test_root_path, random_state=42)

    print("\n" + "=" * 50)
    print("📊 K-FOLD CROSS-VALIDATION EVALUATION (Defect Classifier)")
    print("=" * 50)
    print(classification_report(cv_y_true, cv_y_pred, zero_division=0))
    cm = confusion_matrix(cv_y_true, cv_y_pred, labels=defect_types)
    print("Confusion matrix (CV pooled):")
    print(pd.DataFrame(cm, index=defect_types, columns=defect_types))

    train_paths, test_paths, y_train_split, y_test_split = train_test_split(
        all_paths.tolist(), all_labels.tolist(), test_size=0.2, stratify=all_labels, random_state=42
    )
    print(f"\n  Train: {len(train_paths)} images | Hold-out: {len(test_paths)} images")

    X_defects = []
    y_defects = []
    stats = []
    for idx, defect_name in enumerate(defect_types):
        count_train = sum(1 for i, lb in enumerate(y_train_split) if lb == idx)
        count_test = sum(1 for i, lb in enumerate(y_test_split) if lb == idx)
        stats.append({
            "Defect Type": defect_name,
            "Train (original)": count_train,
            "After Augmentation": count_train * 6,
            "Test (hold-out)": count_test,
        })

    for path, label in zip(train_paths, y_train_split):
        img = cv2.imread(path)
        if img is None:
            continue
        for aug_img in augment_image(img):
            feat = image_to_feature(aug_img)
            if feat is not None:
                X_defects.append(feat)
                y_defects.append(label)

    print("\n📊 DATA STATISTICS (train augmented, test hold-out):")
    print(pd.DataFrame(stats))
    print(f"\nTotal samples for Random Forest training: {len(X_defects)} (augmented from {len(train_paths)} images)")

    rf_model = DefectClassifier()
    rf_model.train(X_defects, y_defects)
    rf_model.save("defect_classifier.pkl")

    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
    split_path = os.path.join(config.MODEL_PATH, "defect_test_split.pkl")
    joblib.dump({
        "test_paths": test_paths,
        "y_test": y_test_split,
        "defect_types": defect_types,
    }, split_path)
    print(f"  Hold-out test split saved to {split_path}")

if __name__ == "__main__":
    main()