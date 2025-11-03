import os, json
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy import text

from app_db import (
    ensure_schema, get_engine, create_user, get_user_by_email,
    create_student, create_teacher, upsert_academic_record,
    latest_model, get_latest_record, save_prediction, add_recommendation
)

st.set_page_config(page_title="EduTrack — Schema-Driven Predictor", layout="centered")
st.title("EduTrack — Student Performance (Schema-Driven)")

ensure_schema()

# Sidebar: demo users
with st.sidebar:
    st.header("Quick demo setup")
    demo_admin_pw = st.text_input("Set Admin password (demo)", type="password")
    if st.button("Create demo users"):
        if not demo_admin_pw:
            st.error("Enter an admin password.")
        else:
            try:
                if not get_user_by_email("admin@example.edu"):
                    create_user("Alice Admin","admin@example.edu", demo_admin_pw, "ADMIN")
                if not get_user_by_email("teacher@example.edu"):
                    tid = create_user("Tom Teacher","teacher@example.edu","teacher123","TEACHER")
                    create_teacher(tid, "CS")
                if not get_user_by_email("sara@example.edu"):
                    sid = create_user("Sara Student","sara@example.edu","student123","STUDENT")
                    create_student(sid, 3.2, "CS", "ACTIVE")
                st.success("Demo users ensured.")
            except Exception as e:
                st.error(f"Setup error: {e}")

# Load latest model + schema
mdl = latest_model("rf-pass-pred")
if not mdl:
    st.warning("No trained model found. Run: python -m src.train_model")
    st.stop()

pipe = load(mdl["path"])
with open(mdl["columns_path"], "r", encoding="utf-8") as f:
    feature_schema = json.load(f)

# ---- Section 1: Record performance (teacher) ----
st.subheader("1) Record Student Performance")

with st.form("record_form"):
    email = st.text_input("Student Email (existing or new)", value="sara@example.edu")
    name  = st.text_input("Student Name (if creating new)", value="Sara Student")
    term  = st.text_input("Term (e.g., 2025-Fall)", value="2025-Fall")

    # Core numeric fields (optional for your dataset)
    attendance    = st.number_input("Attendance %", 0.0, 100.0, 80.0, step=0.1)
    study_hours   = st.number_input("Study hours/week", 0.0, 60.0, 10.0, step=0.5)
    exam_score    = st.number_input("Exam score (0-100)", 0.0, 100.0, 70.0, step=0.5)
    stress_level  = st.slider("Stress level (0-10)", 0, 10, 5)
    sleep_hours   = st.number_input("Sleep hours/day", 0.0, 16.0, 7.0, step=0.5)
    participation = st.number_input("Participation %", 0.0, 100.0, 55.0, step=0.5)

    st.markdown("**Additional features (auto from schema)**")
    extra_inputs = {}
    for feat in feature_schema:
        fname, ftype = feat["name"], feat.get("type","number")
        if fname in {"attendance","study_hours","exam_score","stress_level","sleep_hours","participation"}:
            continue  # already captured above as core numerics
        if ftype == "number":
            extra_inputs[fname] = st.number_input(f"{fname}", value=0.0, step=0.5)
        else:
            choices = feat.get("choices") or []
            default_idx = 0 if choices else None
            extra_inputs[fname] = st.selectbox(f"{fname}", options=choices, index=default_idx if choices else None)

    submitted = st.form_submit_button("Save / Update record")
    if submitted:
        try:
            # find or create student
            stu = get_user_by_email(email)
            if not stu:
                uid = create_user(name or "Student", email, "student123", "STUDENT")
                # create student row
                create_student(uid, None, "CS", "ACTIVE")
            # get student_id
            engine = get_engine()
            with engine.begin() as conn:
                query = text("""
                    SELECT s.student_id FROM students s
                    JOIN users u ON u.user_id = s.user_id
                    WHERE u.email = :e
                """)
                student_id = conn.execute(query, {"e": email}).scalar()
            if not student_id:
                st.error("Could not resolve student_id.")
            else:
                rid = upsert_academic_record(
                    student_id, term,
                    attendance=float(attendance),
                    study_hours=float(study_hours),
                    exam_score=float(exam_score),
                    stress_level=int(stress_level),
                    sleep_hours=float(sleep_hours),
                    participation=float(participation),
                    extra=extra_inputs,
                    created_by=None
                )
                st.success(f"Saved record (ID={rid}) for {email} / {term}.")
        except Exception as e:
            st.error(f"Save error: {e}")

# ---- Section 2: Predict ----
st.subheader("2) Predict Student Performance")

email_pred = st.text_input("Student Email to Predict", value="sara@example.edu")
term_pred  = st.text_input("Term", value="2025-Fall")

if st.button("Predict"):
    try:
        # get student_id
        engine = get_engine()
        with engine.begin() as conn:
            row = conn.execute(text(
                """SELECT u.user_id, s.student_id
                     FROM users u JOIN students s ON s.user_id=u.user_id
                    WHERE u.email=:e"""), {"e": email_pred}
            ).mappings().first()

        if not row:
            st.error("Student not found.")
        else:
            student_id = row["student_id"]
            rec = get_latest_record(student_id)
            if not rec:
                st.error("No academic record for this student.")
            else:
                # Build a single-row DataFrame exactly matching training feature names
                # First, collect numeric core fields
                core = {
                    "attendance": rec.get("attendance"),
                    "study_hours": rec.get("study_hours"),
                    "exam_score": rec.get("exam_score"),
                    "stress_level": rec.get("stress_level"),
                    "sleep_hours": rec.get("sleep_hours"),
                    "participation": rec.get("participation")
                }
                extra = rec.get("extra") or {}
                row_dict = {}

                # Use schema to order/build the row
                for feat in feature_schema:
                    name = feat["name"]
                    if name in core and core[name] is not None:
                        row_dict[name] = core[name]
                    elif name in extra:
                        row_dict[name] = extra[name]
                    else:
                        # default if missing: 0 for numbers, first choice for dropdown
                        if feat.get("type") == "number":
                            row_dict[name] = 0.0
                        else:
                            choices = feat.get("choices") or []
                            row_dict[name] = choices[0] if choices else ""

                X = pd.DataFrame([row_dict], columns=[f["name"] for f in feature_schema])

                prob_pass = float(pipe.predict_proba(X)[0, 1])
                label = "PASS" if prob_pass >= 0.5 else "FAIL"
                pred_id = save_prediction(student_id, mdl["model_id"], term_pred, label, prob_pass, 1.0 - prob_pass)

                st.info(f"Predicted: **{label}**  (Pass prob: {prob_pass:.2f})")

                # Simple rule-based tips (optional)
                tips = []
                if core.get("attendance") is not None and core["attendance"] < 75:
                    tips.append(("ATTENDANCE", "Improve attendance to at least 75%."))
                if core.get("study_hours") is not None and core["study_hours"] < 12:
                    tips.append(("STUDY", "Increase study time by 2–4 hours per week."))
                if core.get("sleep_hours") is not None and core["sleep_hours"] < 7:
                    tips.append(("SLEEP", "Aim for 7–8 hours of sleep nightly."))
                if core.get("participation") is not None and core["participation"] < 50:
                    tips.append(("PARTICIPATION", "Engage more in class activities."))

                if tips:
                    st.subheader("Recommendations")
                    for t, msg in tips:
                        add_recommendation(student_id, pred_id, t, msg)
                        st.write(f"• **{t}** — {msg}")
                else:
                    st.success("No specific risk triggers — keep up the good work!")
    except Exception as e:
        st.error(f"Prediction error: {e}")
