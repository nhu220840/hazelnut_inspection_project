import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os
from src.preprocessing import remove_background
from src.features import extract_features
from src import config

LABEL_MAP = {0: 'crack', 1: 'cut', 2: 'hole', 3: 'print'}

@st.cache_resource
def load_models():
    """Load trained models from .pkl files"""
    try:
        svm_path = os.path.join(config.MODEL_PATH, "anomaly_detector.pkl")
        rf_path = os.path.join(config.MODEL_PATH, "defect_classifier.pkl")
        
        svm_model = joblib.load(svm_path)
        rf_model = joblib.load(rf_path)
        return svm_model, rf_model
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train.py first to train the models.")
        st.stop()

def predict_image(img, svm_model, rf_model):
    """Predict defect type for a single image"""
    processed_img, mask = remove_background(img)
    
    try:
        feats = extract_features(processed_img, mask)
        feats = feats.reshape(1, -1)
    except Exception:
        return "error", "error", None, None

    anomaly_score = svm_model.predict(feats)[0]
    
    if anomaly_score == 1:
        return "good", "good", processed_img, mask
    else:
        defect_code = rf_model.predict(feats)[0]
        defect_name = LABEL_MAP.get(defect_code, "unknown")
        return "defect", defect_name, processed_img, mask

def main():
    st.set_page_config(
        page_title="Hazelnut Inspection System",
        page_icon="🌰",
        layout="wide"
    )
    
    st.title("🌰 Hazelnut Inspection System")
    st.markdown("### Automated Defect Detection using Machine Learning")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        svm_model, rf_model = load_models()
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a hazelnut image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a hazelnut to inspect for defects"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Defect Types")
    st.sidebar.markdown("""
    - **Good**: No defects detected
    - **Crack**: Surface cracks
    - **Cut**: Physical cuts
    - **Hole**: Holes in the nut
    - **Print**: Ink marks/stains
    """)
    
    # Main content
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert PIL to OpenCV format (RGB to BGR)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Predict
        with st.spinner("Analyzing image..."):
            status, label, processed_img, mask = predict_image(img_cv, svm_model, rf_model)
        
        with col2:
            st.subheader("🔍 Processed Image")
            if processed_img is not None:
                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, use_container_width=True)
        
        # Results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            if status == "good":
                st.success(f"**Status**: ✅ {status.upper()}")
                st.balloons()
            elif status == "defect":
                st.error(f"**Status**: ⚠️ {status.upper()}")
            else:
                st.warning(f"**Status**: ❌ {status.upper()}")
        
        with col4:
            if label == "good":
                st.info(f"**Label**: 🌰 {label.upper()}")
            elif label in LABEL_MAP.values():
                st.warning(f"**Label**: 🔴 {label.upper()}")
            else:
                st.error(f"**Label**: ❓ {label.upper()}")
        
        with col5:
            if mask is not None:
                st.image(mask, use_container_width=True, caption="Binary Mask")
        
        # Additional info
        if status == "good":
            st.success("🎉 This hazelnut appears to be in good condition with no defects detected!")
        elif status == "defect":
            st.warning(f"⚠️ Defect detected: **{label.upper()}**")
            st.info("💡 This hazelnut may need further inspection or should be rejected.")
        else:
            st.error("❌ Error processing image. Please try again with a different image.")
    
    else:
        st.info("👈 Please upload an image from the sidebar to get started.")
        st.markdown("""
        ### How to use:
        1. Click on **"Browse files"** in the sidebar
        2. Select a hazelnut image (PNG, JPG, or JPEG)
        3. Wait for the analysis to complete
        4. View the results and processed image
        
        ### Supported formats:
        - PNG
        - JPG/JPEG
        """)

if __name__ == "__main__":
    main()

