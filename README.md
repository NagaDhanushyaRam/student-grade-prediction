📘 EduTrack — Student Grade Prediction & Recommendation System

EduTrack is a full-stack Student Grade Prediction & Academic Advising System built using Python, Streamlit, Scikit-Learn, and MongoDB Atlas.
It was created as part of COSC 612 / AIT 624 – Assignment 5 (Implementation & Testing of the Whole System).

The system supports Admins, Teachers, and Students, providing prediction-driven insights and academic recommendations.

⭐ Key Features

    🔐 Authentication & Roles

        Secure email + password login

        SHA-256 hashing + pepper for password protection

        Three user roles: Admin, Teacher, Student

    👨‍💼 1. Admin Features

        Create Teacher and Student accounts

        View site-level academic performance summaries

        Perform high-level data monitoring

    👩‍🏫 2. Teacher Features

        Create/update academic records

        Record inputs such as:

            Attendance

            Study hours

            Exam score

            Stress level

            Sleep hours

            Class participation

            Run PASS/FAIL predictions for advisees

            View probability scores + feature-based risk explanation

            Send messages to students

            Review department-level performance

    🧑‍🎓 3. Student Features

        View personal and academic profile

        Access latest academic record

        See prediction result + probability

        Receive personalised recommendations from the ML and rule engine

        Read messages from their assigned teacher

🤖 Machine Learning Pipeline

    EduTrack uses an end-to-end ML pipeline built with Scikit-Learn:

    Model: RandomForestClassifier wrapped in a Pipeline

    Training script: src/train_model.py

    Trained model stored as models/model.pkl

    MongoDB maintains a model registry (metadata such as version, timestamp)

    A rule-based advice engine converts weak indicators into human-readable recommendations
    (e.g., low sleep → improve sleep schedule, low study hours → increase planned study time)

    ✅ Important:
        Before running the Streamlit app for the first time (or after you change training code/data),
        you must run:

        python -m src.train_model

🗄 MongoDB Atlas Backend

    EduTrack uses MongoDB Atlas as the primary database.
    Main collections:

        Collection	Purpose
        users	Login credentials + roles
        students	Student profile information
        teachers	Teacher/advisor info
        academic_records	Student grades & behaviour features
        messages	Teacher ↔ Student communication
        models	ML model metadata and registry

    MongoDB access and logic live in:

        src/app_db_mongo.py

        src/mongo_client.py

        (Older SQLite artifacts are kept only for legacy reference.)

🧱 Technologies Used
    Backend & Frontend

    Python 3.10+

    Streamlit

    Database

    MongoDB Atlas

    pymongo + certifi for secure connection

    Machine Learning & Data

    scikit-learn

    pandas, numpy

    joblib

    Configuration & Utilities

    python-dotenv (.env loading)

    Git + GitHub for version control and sprint-based branching

📂 Project Structure
    student-grade-prediction/
    ├─ data/
    │   └─ Students_Performance_Dataset.csv        # Training dataset
    ├─ db/
    │   └─ schema.sql                              # Legacy SQLite schema (unused at runtime)
    ├─ models/
    │   ├─ model.pkl                               # Trained ML pipeline (created by train_model.py)
    │   └─ feature_schema.json                     # Feature columns used by the model
    ├─ src/
    │   ├─ app.py                                  # Main Streamlit application
    │   ├─ app_db_mongo.py                         # MongoDB data access & business logic
    │   ├─ mongo_client.py                         # MongoClient factory using MONGODB_URI
    │   ├─ train_model.py                          # Training script (entrypoint for python -m src.train_model)
    │   ├─ migrate_to_mongo.py                     # One-time SQLite → Mongo migration helper
    │   ├─ mongo_smoke_test.py                     # Simple smoke test for Mongo connection
    │   ├─ test_mongo.py                           # Minimal connectivity test
    │   └─ __init__.py
    ├─ requirements.txt
    ├─ environment.yml                             # Optional Conda environment
    ├─ .gitignore
    └─ README.md

⚙️ Installation & Setup

    1️⃣ Clone the Repository
        git clone https://github.com/your-username/student-grade-prediction.git
        cd student-grade-prediction

    2️⃣ Create and Activate Environment
    
        conda create -n gradepred python=3.10 -y
        conda activate gradepred
        pip install -r requirements.txt

    3️⃣ Configure MongoDB

        Create a .env file in the project root:

        MONGODB_URI="your-mongodb-atlas-connection-url"
        MONGODB_DBNAME="edutrack"

        Example:

        MONGODB_URI="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority"
        MONGODB_DBNAME="edutrack"

        Make sure the MongoDB user has read/write permissions for the edutrack database, and your IP/network is allowed in Network Access.
        Use the .env file in src directory for easy access. 

    4️⃣ Train the Model (First Time Only) ✅

        Before running the app, train and register the ML model:

        python -m src.train_model

        This will:

            Load data/Students_Performance_Dataset.csv

            Train the RandomForest-based pipeline

            Save models/model.pkl and models/feature_schema.json

            Optionally register/update model metadata in the models collection

        You only need to re-run this when:

            You change the dataset

            You update model/training code

            You want to retrain using new data

    5️⃣ Run the Streamlit Application

        streamlit run src/app.py

        Then open in your browser:

        http://localhost:8501

        Log in with an existing user (or create one via Admin functionality if seeded).

🧪 Testing
    Smoke Test MongoDB
    python -m src.mongo_smoke_test

    Basic Connectivity Test
    python -m src.test_mongo


    These help verify that:

        Your .env is correctly configured

        MongoDB Atlas is reachable

        The app can read/write basic documents

🔒 Security Notes

    Passwords are hashed using:

        SHA256(PEPPER + raw_password)

        Comparison uses hmac.compare_digest to resist timing attacks

        Secrets (DB URI, etc.) are never hard-coded; they live in .env

🔮 Future Improvements

    UI polish (charts, better dashboards, more filters)

    Model explainability tools (e.g., feature importances per prediction)

    Full CI/CD pipeline (GitHub Actions → Streamlit Cloud / container deploy)

    Fine-grained role management and audit logging

    Automatic model retraining workflow from the UI

📄 License

    MIT License — you are free to use, modify, and distribute this project.

🙌 Contributors

    EduTrack Development Team
      - Naga Dhanushya Ram Munnanuru
      - Ravinder Maini
      - Muhammad Adam
      - Stephen Aboagye-Ntow
      - Ayandayo Adeleke
    Towson University — COSC 612



