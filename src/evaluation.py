"""Defect path loading, image-to-feature, and K-fold CV for train.py and main.py."""
import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src import config
from src.preprocessing import remove_background
from src.features import extract_features
from src.models import DefectClassifier
from src.augmentation import augment_image


def image_to_feature(img):
    """Extract feature vector from image (background removal + HOG + color). Returns None on failure."""
    processed_img, mask = remove_background(img)
    try:
        return extract_features(processed_img, mask)
    except Exception:
        return None


def get_defect_paths_labels(test_root):
    """Collect all defect image paths and integer labels from test_root subfolders. Returns (paths, labels) arrays."""
    all_paths = []
    all_labels = []
    for idx, defect_name in enumerate(config.DEFECT_TYPES):
        folder_path = os.path.join(test_root, defect_name)
        if not os.path.exists(folder_path):
            continue
        for path in glob.glob(os.path.join(folder_path, "*.png")):
            all_paths.append(path)
            all_labels.append(idx)
    if not all_paths:
        return np.array([]), np.array([])
    return np.array(all_paths), np.array(all_labels)


def run_defect_kfold_cv(test_root, random_state=42):
    """Run stratified K-fold CV for defect classifier (augment train fold only). Returns (y_true, y_pred) label lists."""
    all_paths, all_labels = get_defect_paths_labels(test_root)
    if len(all_paths) == 0:
        return [], []

    n_splits = min(5, int(np.bincount(all_labels).min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_y_true = []
    cv_y_pred = []

    for train_idx, val_idx in skf.split(all_paths, all_labels):
        train_paths = all_paths[train_idx]
        train_labels = all_labels[train_idx]
        val_paths = all_paths[val_idx]
        val_labels = all_labels[val_idx]

        X_train = []
        y_train = []
        for path, label in zip(train_paths, train_labels):
            img = cv2.imread(path)
            if img is None:
                continue
            for aug_img in augment_image(img):
                feat = image_to_feature(aug_img)
                if feat is not None:
                    X_train.append(feat)
                    y_train.append(label)

        model_fold = DefectClassifier()
        model_fold.train(X_train, y_train)

        for path, true_label in zip(val_paths, val_labels):
            img = cv2.imread(path)
            if img is None:
                continue
            feat = image_to_feature(img)
            if feat is None:
                continue
            feat = np.array(feat).reshape(1, -1)
            pred = model_fold.model.predict(feat)[0]
            cv_y_true.append(config.DEFECT_TYPES[true_label])
            cv_y_pred.append(config.DEFECT_TYPES[pred])

    return cv_y_true, cv_y_pred
