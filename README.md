<div align="center">

# 🎓 Student Grade Prediction System
### *End-to-End Machine Learning Pipeline, FastAPI Backend & Interactive Streamlit Dashboard*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  Predict academic performance and final letter grades in real time using demographic attributes, study habits, and engagement indicators[cite: 1, 3].
</p>

</div>

---

## 📸 Dashboard Preview

<div align="center">
  <h3>Interactive Input Form & Real-Time Grade Prediction</h3>
  <img src="dashboard_overview.png" alt="image.png" width="900" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
  <br><br>
</div>

---

## 🌟 Overview

The **Student Grade Prediction System** is an end-to-end machine learning platform that evaluates student performance before final exams[cite: 1, 3]. It features:

1. **Supervised Machine Learning Pipeline**: Trained using `RandomForestClassifier` with embedded missing value imputation, standard scaling, and one-hot encoding.
2. **High-Performance FastAPI Service**: Serves real-time predictions via REST endpoints and provides interactive Swagger documentation.
3. **Interactive Streamlit Web Interface**: Features quick-fill student presets, live grade output badges, confidence meters, and full probability distribution charts[cite: 1, 3, 4].
4. **Containerized Deployment**: Packaged with Docker (`python:3.12-slim`) for reproducible deployment across any platform.

---

## ✨ Key Features

- **⚡ Instant Profile Presets:** Pre-populate input fields using archetypes such as **High Achiever**, **Average Student**, or **At-Risk Student**[cite: 1, 3].
- **🎯 Dynamic Probability Visualizer:** View the predicted letter grade alongside model confidence and a multi-class probability bar chart (`AA`, `BA`, `BB`, `CB`, `CC`, `DC`, `DD`, `Fail`)[cite: 4].
- **🔍 Raw API Inspector:** Inspect the direct JSON response payload from the FastAPI `/predict` route directly inside the UI[cite: 4].
- **🛡️ Leak-Free Preprocessing:** Integrated `ColumnTransformer` handles missing data (`SimpleImputer`), numeric feature normalization (`StandardScaler`), and categorical encoding (`OneHotEncoder`).
- **📈 Validated Accuracy:** Evaluated using **5-Fold Stratified Cross-Validation** and holdout test sets with classification reports and weighted F1 scoring.

---

## 🏗️ Architecture & Data Flow

```text
  [ User / Browser ]
          │
          ▼
┌──────────────────┐       HTTP POST        ┌─────────────────────┐
│  Streamlit UI    │ ────────────────────▶  │   FastAPI Server    │
│  (Port 8501)     │ ◀────────────────────  │   (Port 8000)       │
└──────────────────┘     Prediction JSON    └──────────┬──────────┘
                                                       │
                                            Loads Fitted Pipeline
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │ Random Forest Model │
                                            │ (Scikit-Learn .pkl) │
                                            └─────────────────────┘
