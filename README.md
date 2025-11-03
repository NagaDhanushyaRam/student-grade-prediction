**STUDENT GRADE PREDICTION SYSTEM**

This project implements a Student Grade Prediction System designed for academic performance tracking and personalized recommendations.It forms Sprint 3 – Task 5 (Implementation) of the Software Engineering course project.

**OVERVIEW**

The system enables:

1.  Recording Student PerformanceTeachers can enter metrics such as attendance, study hours, stress level, and exam scores for each term.
    
2.  Predicting Student OutcomesA trained Machine Learning model (Random Forest Classifier) predicts whether a student is likely to pass or fail.
    
3.  Providing RecommendationsBased on prediction and performance data, the system offers actionable suggestions (e.g., increase study time, improve attendance).
    

**TECHNOLOGIES USED**

Frontend: StreamlitBackend / ML: Python 3.13, scikit-learn, pandas, numpy, joblibDatabase: SQLite (via SQLAlchemy ORM)Authentication & Utilities: passlib/bcrypt, pydanticVersion Control: Git & GitHub (feature branching workflow)

**PROJECT STRUCTURE**

student-grade-prediction/│├─ data/│ └─ Students\_Performance\_Dataset.csv│├─ db/│ ├─ schema.sql (Database design - tables, relations)│ └─ seed.sql (Optional starter data)│├─ models/│ ├─ model.pkl (Trained ML model - generated automatically)│ └─ feature\_schema.json (Input schema - auto-generated)│├─ src/│ ├─ **init**.py│ ├─ db.py (DB helper: schema creation, CRUD, model registry)│ ├─ train\_model.py (Trains RandomForest model and registers it)│ └─ app.py (Streamlit web app for record & prediction)│├─ .gitignore├─ environment.yml├─ requirements.txt└─ README.md

**IMPLEMENTATION SUMMARY**

Task 5 focused on Implementation, which includes:

*   Building database schema (schema.sql)
    
*   Integrating backend (db.py for CRUD & model registry)
    
*   Training the machine learning model (train\_model.py)
    
*   Generating schema for dynamic UI fields (feature\_schema.json)
    
*   Creating Streamlit app (app.py) for end-to-end workflow
    
*   Achieving full database → model → UI → prediction integration
    

**MACHINE LEARNING MODEL**

Algorithm: Random Forest ClassifierPreprocessing: Imputation + One-Hot Encoding + Standard ScalingTarget: Pass (1) / Fail (0)

Evaluation Results:Accuracy = 0.880F1 Score = 0.906ROC-AUC = 0.980

Artifacts:

*   models/model.pkl — serialized trained pipeline
    
*   models/feature\_schema.json — defines numeric vs categorical input fields
    

**DATABASE SCHEMA OVERVIEW**

Tables Implemented:

users — Stores all user details (Admin, Teacher, Student)students — Links student accounts to user recordsteachers — Links teacher accounts to user recordsstudent\_academic\_data — Holds term-wise performance metricsml\_models — Registry of trained models (path, metrics)prediction\_results — Stores PASS/FAIL predictions and probabilitiesrecommendations — Stores generated feedback for each prediction

All tables are created using SQLite and automatically initialized through ensure\_schema().

**ENVIRONMENT SETUP**

Step 1: Create Conda Environment
--------------------------------

conda create -n gradepred python=3.13 -yconda activate gradepredpython --version

Step 2: Install Dependencies
----------------------------

conda install -c conda-forge streamlit scikit-learn pandas numpy joblib sqlalchemy pydantic bcrypt passlib python-dotenv

(Optional) Export dependencies:python -m pip freeze > requirements.txtconda env export --from-history -n gradepred > environment.yml

Step 3: Initialize Database
---------------------------

sqlite3 edutrack.db < db/schema.sqlsqlite3 edutrack.db ".tables"

(If sqlite3 is not installed, ensure\_schema() in src/db.py will auto-create tables.)

Step 4: Train the Model
-----------------------

python -m src.train\_model

Expected Output:✅ Saved model to models/model.pkl✅ Saved schema to models/feature\_schema.jsonMetrics -> acc=0.880, f1=0.906, roc\_auc=0.980

Step 5: Run the Streamlit Application
-------------------------------------

streamlit run src/app.py

Then in the app:

1.  Use sidebar → Create demo users (Admin, Teacher, Student)
    
2.  Record Student Performance → Fill attendance, study hours, exam score, etc.
    
3.  Save / Update record
    
4.  Predict Student Performance → Enter email & term → click Predict
    
5.  View PASS/FAIL + probability
    
6.  Review personalized recommendations
