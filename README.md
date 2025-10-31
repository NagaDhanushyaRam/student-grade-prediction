STUDENT GRADE PREDICTION SYSTEM

This project implements a Student Grade Prediction System designed for academic performance tracking and personalized recommendations.
It forms Sprint 3 – Task 5 (Implementation) of the Software Engineering course project.

OVERVIEW

The system enables:

Recording Student Performance
Teachers can enter metrics such as attendance, study hours, stress level, and exam scores for each term.

Predicting Student Outcomes
A trained Machine Learning model (Random Forest Classifier) predicts whether a student is likely to pass or fail.

Providing Recommendations
Based on prediction and performance data, the system offers actionable suggestions (e.g., increase study time, improve attendance).

TECHNOLOGIES USED

Frontend: Streamlit
Backend / ML: Python 3.13, scikit-learn, pandas, numpy, joblib
Database: SQLite (via SQLAlchemy ORM)
Authentication & Utilities: passlib/bcrypt, pydantic
Version Control: Git & GitHub (feature branching workflow)

PROJECT STRUCTURE

student-grade-prediction/
│
├─ data/
│ └─ Students_Performance_Dataset.csv
│
├─ db/
│ ├─ schema.sql (Database design - tables, relations)
│ └─ seed.sql (Optional starter data)
│
├─ models/
│ ├─ model.pkl (Trained ML model - generated automatically)
│ └─ feature_schema.json (Input schema - auto-generated)
│
├─ src/
│ ├─ init.py
│ ├─ db.py (DB helper: schema creation, CRUD, model registry)
│ ├─ train_model.py (Trains RandomForest model and registers it)
│ └─ app.py (Streamlit web app for record & prediction)
│
├─ .gitignore
├─ environment.yml
├─ requirements.txt
└─ README.md

IMPLEMENTATION SUMMARY

Task 5 focused on Implementation, which includes:

Building database schema (schema.sql)

Integrating backend (db.py for CRUD & model registry)

Training the machine learning model (train_model.py)

Generating schema for dynamic UI fields (feature_schema.json)

Creating Streamlit app (app.py) for end-to-end workflow

Achieving full database → model → UI → prediction integration

MACHINE LEARNING MODEL

Algorithm: Random Forest Classifier
Preprocessing: Imputation + One-Hot Encoding + Standard Scaling
Target: Pass (1) / Fail (0)

Evaluation Results:
Accuracy = 0.880
F1 Score = 0.906
ROC-AUC = 0.980

Artifacts:

models/model.pkl — serialized trained pipeline

models/feature_schema.json — defines numeric vs categorical input fields

DATABASE SCHEMA OVERVIEW

Tables Implemented:

users — Stores all user details (Admin, Teacher, Student)
students — Links student accounts to user records
teachers — Links teacher accounts to user records
student_academic_data — Holds term-wise performance metrics
ml_models — Registry of trained models (path, metrics)
prediction_results — Stores PASS/FAIL predictions and probabilities
recommendations — Stores generated feedback for each prediction

All tables are created using SQLite and automatically initialized through ensure_schema().

ENVIRONMENT SETUP

Step 1: Create Conda Environment

conda create -n gradepred python=3.13 -y
conda activate gradepred
python --version

Step 2: Install Dependencies

conda install -c conda-forge streamlit scikit-learn pandas numpy joblib sqlalchemy pydantic bcrypt passlib python-dotenv

(Optional) Export dependencies:
python -m pip freeze > requirements.txt
conda env export --from-history -n gradepred > environment.yml

Step 3: Initialize Database

sqlite3 edutrack.db < db/schema.sql
sqlite3 edutrack.db ".tables"

(If sqlite3 is not installed, ensure_schema() in src/db.py will auto-create tables.)

Step 4: Train the Model

python -m src.train_model

Expected Output:
✅ Saved model to models/model.pkl
✅ Saved schema to models/feature_schema.json
Metrics -> acc=0.880, f1=0.906, roc_auc=0.980

Step 5: Run the Streamlit Application

streamlit run src/app.py

Then in the app:

Use sidebar → Create demo users (Admin, Teacher, Student)

Record Student Performance → Fill attendance, study hours, exam score, etc.

Save / Update record

Predict Student Performance → Enter email & term → click Predict

View PASS/FAIL + probability

Review personalized recommendations
