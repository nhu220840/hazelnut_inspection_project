import os
import cv2
import glob
import pandas as pd
from src.preprocessing import remove_background
from src.features import extract_features
from src.models import AnomalyDetector, DefectClassifier
from src.augmentation import augment_image
from src import config

def process_single_image(img):
    """Helper function: Process 1 image -> Return feature vector"""
    processed_img, mask = remove_background(img)
    try:
        return extract_features(processed_img, mask)
    except:
        return None

def main():
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
        svm_model = AnomalyDetector()
        svm_model.train(X_good) 
        svm_model.save("anomaly_detector.pkl")
    else:
        print("❌ Error: No train/good data found")

    print("\n=== STAGE 2: TRAINING DEFECT CLASSIFIER WITH AUGMENTATION ===")
    
    defect_types = ['crack', 'cut', 'hole', 'print']
    X_defects = []
    y_defects = []
    
    test_root_path = os.path.join(config.DATA_PATH, "test")
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
            
            aug_imgs = augment_image(img)
            
            for aug_img in aug_imgs:
                feat = process_single_image(aug_img)
                if feat is not None:
                    X_defects.append(feat)
                    y_defects.append(idx)
        
        final_count = original_count * 6
        stats.append({
            "Defect Type": defect_name, 
            "Original Count": original_count, 
            "After Augmentation": final_count
        })

    print("\n📊 DATA STATISTICS AFTER AUGMENTATION:")
    df_stats = pd.DataFrame(stats)
    print(df_stats)
    print(f"\nTotal samples for Random Forest training: {len(X_defects)}")

    if X_defects:
        rf_model = DefectClassifier()
        rf_model.train(X_defects, y_defects)
        rf_model.save("defect_classifier.pkl")
    else:
        print("❌ Error: No defect data found!")

if __name__ == "__main__":
    main()