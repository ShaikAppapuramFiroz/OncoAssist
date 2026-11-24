# Bladder Cancer Type Prediction System (AI-Powered) 🩺

A production-ready, advanced diagnostic support system built with **Python**, **Streamlit**, and **XGBoost**. This system predicts bladder cancer types and related conditions based on clinical laboratory parameters.

---

## 🚀 Features
- **Machine Learning Pipeline:** High accuracy model trained with XGBoost and SMOTE for class balancing.
- **Advanced UI:** Glassmorphism design, animated gradient backgrounds, and Lottie animations.
- **Real-time Prediction:** Instant results with confidence scores and risk assessments.
- **Interactive Visualizations:** Plotly probability charts and feature importance graphs.
- **Dynamic Forms:** Intelligent input handling with autofill capabilities for demonstration.

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Run the following command:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model
Run the data generation and training scripts to build the machine learning artifacts:
```bash
python data/generate_data.py
python model/train.py
```

### 3. Launch Streamlit Application
Start the interactive dashboard:
```bash
streamlit run app.py
```

---

## 🧪 Demo Input Values (For Simulation)

| Target Type | Age | Gender | BloodInUrine | FrequentUrination | Creatinine | Urea | Hemoglobin |
|-------------|-----|--------|--------------|-------------------|------------|------|------------|
| **UrinaryBladder** | 65 | Male | Yes | Yes | 1.10 | 18.00 | 11.50 |
| **Kidney** | 58 | Female | No | No | 2.80 | 45.00 | 10.50 |
| **Prostate** | 72 | Male | Yes | Yes | 1.00 | 15.00 | 14.20 |
| **Cystitis** | 35 | Female | No | Yes | 0.80 | 12.00 | 13.50 |
| **Uterus** | 42 | Female | Yes | No | 0.90 | 14.00 | 12.00 |

---

## 🏗️ Project Structure
```text
bladder_cancer_app/
│
├── app.py                 # Main Streamlit dashboard
├── model/
│   ├── train.py           # ML training pipeline
│   ├── predict.py         # Prediction logic class
│   ├── model.pkl          # Trained XGBoost model
│   ├── scaler.pkl         # Scaler artifact
│   ├── features.pkl       # Selected features list
│   └── target_encoder.pkl # Target label encoder
│
├── utils/
│   ├── preprocessing.py   # Data cleaning & transformation
│   └── visualization.py   # Plotly/Matplotlib charts
│
├── data/
│   └── dataset.csv        # Clinical lab dataset
│
├── assets/
│   └── styles.css         # Advanced Glassmorphism styling
│
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

## ⚙️ Tech Stack
- **Backend:** Python 3.11
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Frontend:** Streamlit, Custom CSS (Glassmorphism)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Animation:** Lottie (streamlit-lottie)

---

## ⚠️ Important Disclaimer
**FOR ACADEMIC DEMONSTRATION ONLY.** This system is a proof-of-concept developed using synthetic clinical data. It is NOT intended for actual medical use, diagnosis, or treatment. Always consult with a qualified medical professional for health-related concerns.
