"""
Student Grade Classification Model Training Script.
Inspects 'Students Performance .csv', preprocesses features and target,
trains a Scikit-Learn RandomForest classification pipeline, evaluates performance,
and serializes the fitted pipeline to 'student_classifier_api.pkl'.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Students Performance .csv")
OUTPUT_MODEL_FILE = os.path.join(BASE_DIR, "student_classifier_api.pkl")
BACKUP_MODEL_FILE = os.path.join(BASE_DIR, "student_grade_classifier.pkl")


def detect_and_prepare_data(csv_path: str):
    """Loads CSV, detects target and features, and applies necessary preprocessing."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"[1/4] Inspecting dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"      Column list: {list(df.columns)}")

    # Detect target column
    candidate_targets = ["Grade", "grade", "FinalGrade", "Final_Grade", "Score", "G3"]
    target_col = None
    for col in candidate_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"Could not automatically detect target column from: {list(df.columns)}")

    print(f"      Detected target column: '{target_col}'")

    # Clean non-predictive ID column if present
    feature_df = df.copy()
    if "Student_ID" in feature_df.columns:
        feature_df = feature_df.drop(columns=["Student_ID"])

    # Clean known anomalies in input features
    if "Attendance" in feature_df.columns:
        feature_df["Attendance"] = feature_df["Attendance"].replace({"3": "Never"})
    if "Notes" in feature_df.columns:
        feature_df["Notes"] = feature_df["Notes"].replace({"6": "Yes"})
    if "Listening_in_Class" in feature_df.columns:
        feature_df["Listening_in_Class"] = feature_df["Listening_in_Class"].replace({"6": "Yes"})
    if "Scholarship" in feature_df.columns:
        mode_val = feature_df["Scholarship"].mode(dropna=True)[0]
        feature_df["Scholarship"] = feature_df["Scholarship"].fillna(mode_val)

    y = feature_df[target_col].copy()
    X = feature_df.drop(columns=[target_col]).copy()

    # If target is continuous numeric, bin it into letter grades (A, B, C, D, Fail)
    if pd.api.types.is_numeric_dtype(y):
        print("      Target is continuous numeric. Binning into letter grades (A, B, C, D, Fail)...")
        y = pd.cut(
            y,
            bins=[-float("inf"), 50, 65, 75, 85, float("inf")],
            labels=["Fail", "D", "C", "B", "A"],
            right=False
        ).astype(str)
    else:
        print(f"      Target is categorical with {len(y.unique())} classes: {sorted(y.unique().tolist())}")

    print(f"      Feature columns ({len(X.columns)}): {list(X.columns)}")
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Builds a scikit-learn Pipeline with numeric scaling and categorical one-hot encoding."""
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"[2/4] Constructing pipeline:")
    print(f"      Numerical features ({len(num_cols)}): {num_cols}")
    print(f"      Categorical features ({len(cat_cols)}): {cat_cols}")

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ],
        remainder="drop"
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline


def train_and_evaluate():
    X, y = detect_and_prepare_data(DATA_FILE)

    # 5-fold cross-validation
    print("\n[3/4] Performing 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipe = build_pipeline(X)
    cv_acc = cross_val_score(cv_pipe, X, y, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(cv_pipe, X, y, cv=skf, scoring="f1_weighted")
    print(f"      CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"      CV Weighted F1: {cv_f1.mean():.4f}")

    # Holdout evaluation (80/20 train/test split)
    print("\n      Evaluating on 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    eval_pipe = build_pipeline(X)
    eval_pipe.fit(X_train, y_train)
    y_pred = eval_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"      Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("\n      --- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Fit final pipeline on all data
    print("[4/4] Fitting final pipeline on 100% of data and saving...")
    final_pipeline = build_pipeline(X)
    final_pipeline.fit(X, y)

    # Store metadata on pipeline object
    final_pipeline.feature_columns_ = list(X.columns)
    final_pipeline.classes_list_ = list(final_pipeline.classes_)

    # Save to primary required path 'student_classifier_api.pkl'
    joblib.dump(final_pipeline, OUTPUT_MODEL_FILE)
    print(f"      Saved fitted pipeline to: {OUTPUT_MODEL_FILE}")

    # Save backup for compatibility
    joblib.dump(final_pipeline, BACKUP_MODEL_FILE)

    # Verification test
    loaded = joblib.load(OUTPUT_MODEL_FILE)
    sample_row = X.iloc[[0]]
    pred = loaded.predict(sample_row)[0]
    proba = loaded.predict_proba(sample_row)[0]
    conf = round(float(max(proba)) * 100, 2)
    print(f"\n[Verification] Model re-loaded successfully!")
    print(f"               Sample prediction: {pred} (Confidence: {conf}%)")


if __name__ == "__main__":
    train_and_evaluate()
