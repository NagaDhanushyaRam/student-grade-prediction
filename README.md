# Student Grade Prediction — Task 5 (Implementation)

End-to-end implementation for two major use cases:

1. **Record Student Performance** (teacher enters metrics)
2. **Predict & Recommend** (ML model predicts PASS/FAIL and suggests actions)

The system uses **SQLite** (DB), **Python / scikit-learn** (ML), and **Streamlit** (UI), with a clean **GitHub branch/PR** workflow.

---

## ✅ Status

- Database schema created ✅
- Training pipeline (RandomForest + preprocessing) implemented ✅
- Model + schema saved (`models/model.pkl`, `models/feature_schema.json`) ✅
- Streamlit app wired to DB + model; prediction + recommendations working ✅
- Branch workflow used (`sprint3task5-implementation`) ✅

---

## 🧭 Tech Stack

- **DBMS:** SQLite (SQL scripts + SQLAlchemy)
- **Backend/ML:** Python 3.13, scikit-learn, joblib, pandas, numpy
- **Web UI:** Streamlit
- **Auth/Utils:** passlib/bcrypt, pydantic
- **Version Control:** Git + GitHub (feature branches, PRs)

---

## 📁 Repository Structure

