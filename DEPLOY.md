# 🚀 Streamlit Deployment Guide

This guide will help you deploy the Hazelnut Inspection System using Streamlit.

## 📋 Prerequisites

1. Python 3.8 or higher
2. Trained models (`anomaly_detector.pkl` and `defect_classifier.pkl` in `saved_models/` folder)
3. All dependencies installed

## 🔧 Local Deployment

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train Models (if not already done)

```bash
python train.py
```

This will create the required model files in the `saved_models/` directory.

### Step 3: Run Streamlit App

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new repository on GitHub
2. Push your project to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Hazelnut Inspection System"
git branch -M main
git remote add origin https://github.com/nhu220840/hazelnut-inspection.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `app.py`
6. Click "Deploy"

### Step 3: Configure App Settings

In your Streamlit Cloud dashboard, you may need to configure:

- **Python version**: 3.8 or higher
- **Dependencies**: Streamlit will automatically detect `requirements.txt`

## 📁 Project Structure

```
hazelnut_inspection_project/
├── app.py                 # Streamlit application
├── main.py                # Evaluation script
├── train.py               # Training script
├── run_pipeline.py         # Full pipeline runner
├── requirements.txt       # Python dependencies
├── saved_models/          # Trained models (must exist)
│   ├── anomaly_detector.pkl
│   └── defect_classifier.pkl
├── src/                   # Source code
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── augmentation.py
└── data/                  # Data directory (optional for deployment)
```

## ⚠️ Important Notes

1. **Model Files**: Make sure `saved_models/anomaly_detector.pkl` and `saved_models/defect_classifier.pkl` exist before deploying
2. **File Size**: Model files are large. Consider using Git LFS for version control:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   ```
3. **Data Directory**: The `data/` folder is not required for deployment, only the trained models are needed

## 🔍 Troubleshooting

### Models Not Found Error

If you see "Model files not found":
1. Run `python train.py` to generate models
2. Ensure models are in `saved_models/` directory
3. Check file paths in `src/config.py`

### Import Errors

If you encounter import errors:
1. Verify all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version: `python --version` (should be 3.8+)
3. Ensure you're running from the project root directory

### Streamlit Not Starting

If Streamlit doesn't start:
1. Check if port 8501 is available
2. Try a different port: `streamlit run app.py --server.port 8502`
3. Verify Streamlit is installed: `pip install streamlit`

## 📝 Usage

1. Open the Streamlit app
2. Click "Browse files" in the sidebar
3. Upload a hazelnut image (PNG, JPG, or JPEG)
4. View the prediction results:
   - **Status**: Good or Defect
   - **Label**: Specific defect type (crack, cut, hole, print)
   - **Processed Image**: Background-removed image
   - **Binary Mask**: Detection mask

## 🎯 Features

- ✅ Real-time image upload and processing
- ✅ Visual display of original and processed images
- ✅ Defect classification (crack, cut, hole, print)
- ✅ Binary mask visualization
- ✅ User-friendly interface

## 📞 Support

For issues or questions, please check:
- Project documentation
- Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)

