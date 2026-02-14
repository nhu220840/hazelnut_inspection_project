# 🌰 Hazelnut Inspection System

Automated defect detection system for hazelnuts using machine learning. This project uses a two-stage approach: anomaly detection (One-Class SVM) followed by defect classification (Random Forest).

## 🎯 Features

- **Anomaly Detection**: Identifies good vs defective hazelnuts
- **Defect Classification**: Classifies defects into 4 types:
  - Crack
  - Cut
  - Hole
  - Print (ink marks)
- **Web Interface**: Streamlit app for easy image upload and prediction
- **Data Augmentation**: Automatic augmentation for better model training

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train.py
```

This will:
- Train the anomaly detector on good hazelnuts
- Train the defect classifier on augmented defect images
- Save models to `saved_models/` directory

### 3. Run Evaluation

```bash
python main.py
```

This evaluates the system on the test set and generates a confusion matrix.

### 4. Run Full Pipeline

```bash
python run_pipeline.py
```

This runs both training and evaluation in sequence.

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
hazelnut_inspection_project/
├── app.py                 # Streamlit web application
├── main.py                # Evaluation script
├── train.py               # Training script
├── run_pipeline.py        # Full pipeline runner
├── requirements.txt       # Python dependencies
├── DEPLOY.md              # Deployment guide
├── saved_models/          # Trained models (generated after training)
│   ├── anomaly_detector.pkl
│   └── defect_classifier.pkl
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── preprocessing.py   # Background removal
│   ├── features.py        # Feature extraction (HOG + Color Histogram)
│   ├── models.py          # ML model classes
│   └── augmentation.py   # Data augmentation
├── data/                  # Dataset
│   └── raw/
│       └── hazelnut/
│           ├── train/
│           │   └── good/
│           └── test/
│               ├── good/
│               ├── crack/
│               ├── cut/
│               ├── hole/
│               └── print/
└── notebooks/             # Jupyter notebooks for analysis
    ├── 01_data_analysis.ipynb
    └── 02_feature_visualization.ipynb
```

## 🔧 Configuration

Edit `src/config.py` to adjust:
- Image resize dimensions
- Color histogram bins
- HOG parameters

## 📊 Model Architecture

### Stage 1: Anomaly Detection
- **Model**: One-Class SVM
- **Purpose**: Distinguish good hazelnuts from defective ones
- **Input**: Features extracted from good hazelnuts only

### Stage 2: Defect Classification
- **Model**: Random Forest Classifier
- **Purpose**: Classify defect types (crack, cut, hole, print)
- **Input**: Features from defective hazelnuts with augmentation

### Feature Extraction
- **HOG (Histogram of Oriented Gradients)**: Shape features
- **Color Histogram**: Color distribution features

## 🌐 Deployment

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions including:
- Local deployment
- Streamlit Cloud deployment
- Troubleshooting guide

## 📝 Usage

### Command Line

```bash
# Train models
python train.py

# Evaluate on test set
python main.py

# Run full pipeline
python run_pipeline.py
```

### Web Interface

1. Start Streamlit app: `streamlit run app.py`
2. Upload a hazelnut image
3. View prediction results and processed image

## 🧪 Testing

The system is evaluated on a test set with the following categories:
- Good hazelnuts
- Crack defects
- Cut defects
- Hole defects
- Print defects

## 📈 Performance

The system achieves approximately 75% accuracy on anomaly detection. Performance may vary based on:
- Image quality
- Lighting conditions
- Background complexity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License. See [LICENSE](LICENSE) for details. -->

## Contact

For questions or contributions, please open an issue or contact:

- 📧 Email: [gianhuw.work@gmail.com](mailto:gianhuw.work@gmail.com)
- 💻 GitHub: [nhu220840](https://github.com/nhu220840)
