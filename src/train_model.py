# src/train_model.py
import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.app_db import ensure_schema, register_model


# ------------------------
# Config
# ------------------------
MODEL_NAME = "student_grade_predictor"
VERSION = "1.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Try to discover the dataset automatically to avoid path typos
CANDIDATE_DATASETS = [
    "Students_Performance_Dataset.csv"]
DATA_PATH = None
for _name in CANDIDATE_DATASETS:
    p = DATA_DIR / _name
    if p.exists():
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError(
        f"Dataset not found in {DATA_DIR}. Looked for: {CANDIDATE_DATASETS}"
    )


# ------------------------
# Helpers
# ------------------------
def fmt(x: float) -> str:
    """Format a metric as 3 decimals; 'NA' for None/NaN."""
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def derive_pass_column(df: pd.DataFrame) -> pd.Series:
    """
    Create a binary 'Pass' label from either a 'grade' column (A/B/C = 1, D/F = 0)
    or a 'total' numeric column (>= 60 -> 1, else 0). Returns a Series of 0/1/NaN.
    """
    grade_col = next((c for c in df.columns if c.lower() == "grade"), None)
    total_col = next((c for c in df.columns if "total" in c.lower()), None)

    def derive(row):
        if grade_col and pd.notna(row.get(grade_col)):
            g = str(row[grade_col]).upper()
            if g in {"A", "B", "C"}:
                return 1
            if g in {"D", "F"}:
                return 0
        if total_col and pd.notna(row.get(total_col)):
            try:
                return int(float(row[total_col]) >= 60.0)
            except Exception:
                return np.nan
        return np.nan

    return df.apply(derive, axis=1)


# ------------------------
# Main training routine
# ------------------------
def main():
    # Ensure DB and tables exist
    ensure_schema()

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Derive target
    y_series = derive_pass_column(df)
    df = df.assign(Pass=y_series).dropna(subset=["Pass"])
    y = df["Pass"].astype(int)

    # Avoid leakage: drop 'Pass' + any 'grade'/'total' columns from features
    leakage_cols = ["Pass"]
    grade_col = next((c for c in df.columns if c.lower() == "grade"), None)
    total_col = next((c for c in df.columns if "total" in c.lower()), None)
    if grade_col:
        leakage_cols.append(grade_col)
    if total_col:
        leakage_cols.append(total_col)

    X = df.drop(columns=leakage_cols, errors="ignore")

    # Separate numeric vs categorical
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    # Save model + feature schema
    model_path = MODELS_DIR / "model.pkl"
    dump(pipe, model_path)

    feature_schema = []
    for c in X.columns:
        if c in num_cols:
            feature_schema.append({"name": c, "type": "number"})
        else:
            # small list of choices to help downstream UI
            choices = sorted([str(v) for v in df[c].dropna().unique().tolist()])[:50]
            feature_schema.append({"name": c, "type": "dropdown", "choices": choices})

    cols_path = MODELS_DIR / "feature_schema.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2, ensure_ascii=False)

    # Register model in DB
    register_model(
        name=MODEL_NAME,
        version=VERSION,
        path=str(model_path),
        columns_path=str(cols_path),
        accuracy=None if acc is None else float(acc),
        f1=None if f1 is None else float(f1),
        roc_auc=None if roc is None else float(roc),
    )

    # Logs
    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved schema to {cols_path}")
    print(f"Metrics -> acc={fmt(acc)}, f1={fmt(f1)}, roc_auc={fmt(roc)}")


if __name__ == "__main__":
    main()
