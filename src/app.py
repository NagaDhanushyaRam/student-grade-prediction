# app.py
import os
import json
import re
from typing import Dict, Optional, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy import text

from app_db import (
    ensure_schema, get_engine,
    hash_pw, verify_pw,
    any_admin_exists, get_user_by_email, create_user,
    create_teacher_full, create_student_full,
    upsert_academic_record, get_latest_record,
    latest_model, can_run_prediction, teacher_choices,
    # expose DB path so we can show it in UI
    DB_FILE, DB_URL
)

# =====================
# Helpers
# =====================
def to_float(x):
    try:
        return float(x) if x is not None and str(x).strip() != "" else None
    except Exception:
        return None

def to_int(x):
    try:
        return int(float(x)) if x is not None and str(x).strip() != "" else None
    except Exception:
        return None

def _norm(s: str) -> str:
    return (s or "").strip().lower()

# PII that must never appear as dropdowns
PII_FIELDS = {"student_id", "first_name", "last_name", "email", "name"}

# Canonical core fields already collected in the top section
CORE_FIELDS = {"attendance", "study_hours", "examscores", "stress_level", "sleep_hours", "participation"}

# Treat anything containing these tokens as core and SKIP it from "Additional features"
CORE_LIKE_TOKENS = [
    "attendance",
    "study hour",
    "exam",
    "stress",
    "sleep",
    "participation",
]

def _canon(s: str) -> str:
    """lowercase + strip punctuation so 'Study Hours per Week' -> 'study hours per week'."""
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()

# Defaults for categorical suggestions (kept for future use)
DEFAULT_CHOICES = {
    "gender": ["Female", "Male", "Non-binary", "Prefer not to say"],
    "extracurricular activities": ["Yes", "No"],
    "internet access at home": ["Yes", "No"],
    "parent education level": ["High", "Medium", "Low"],
    "family income level": ["Low", "Mid", "High"],
}

# ---- GLOBAL: show recommendations as bullet points (usable by both Teacher & Student)
def render_recommendations_bullets(student_id: int, prediction_id: Optional[int] = None):
    """
    Show recommendations for a student as bullet points.
    If prediction_id is given, filter to that prediction; otherwise show latest/all.
    """
    q = """
        SELECT recommendationType, message
          FROM recommendations
         WHERE student_id = :sid
           AND (:pid IS NULL OR prediction_id = :pid)
      ORDER BY recommendation_id DESC
    """
    with get_engine().begin() as c2:
        recs = c2.execute(text(q), {"sid": student_id, "pid": prediction_id}).mappings().all()

    if not recs:
        st.caption("No recommendations for these inputs.")
        return

    st.markdown("**Recommendations**")
    for r in recs:
        typ = (r["recommendationType"] or "").replace("_", " ").title()
        st.markdown(f"- **{typ}** — {r['message']}")

# =============================
# Prediction save (UPSERT) + id
# =============================
def save_prediction(student_id, model_id, term, label, pass_p, fail_p, conn=None) -> int:
    """
    Upsert a prediction for (student_id, model_id, term) and return prediction_id.
    Pass an open `conn` when you want atomic writes together with recommendations.
    """
    params = {
        "sid": int(student_id),
        "mid": int(model_id),
        "term": term,
        "status": label,
        "pp": float(pass_p),
        "fp": float(fail_p),
    }

    if conn is None:
        with get_engine().begin() as c:
            return _save_prediction_upsert(c, params)
    else:
        return _save_prediction_upsert(conn, params)

def _save_prediction_upsert(conn, params) -> int:
    # Insert-or-update (SQLite ON CONFLICT) using the unique index on (student_id, model_id, term)
    conn.execute(text("""
        INSERT INTO prediction_results (student_id, model_id, term, predictedStatus, passPercentage, failPercentage)
        VALUES (:sid, :mid, :term, :status, :pp, :fp)
        ON CONFLICT(student_id, model_id, term)
        DO UPDATE SET
          predictedStatus = excluded.predictedStatus,
          passPercentage  = excluded.passPercentage,
          failPercentage  = excluded.failPercentage
    """), params)

    pid = conn.execute(text("""
        SELECT prediction_id
          FROM prediction_results
         WHERE student_id=:sid AND model_id=:mid AND term=:term
    """), params).scalar()
    return int(pid)

# =========================================
# Replace recs (single connection/transaction)
# =========================================
def _replace_recommendations_with_conn(conn, student_id: int, prediction_id: int,
                                       recs: List[Tuple[str, str]]):
    # de-dupe
    seen = set()
    recs = [(t, m) for (t, m) in recs if (t, m) not in seen and not seen.add((t, m))]

    conn.execute(text("""
        DELETE FROM recommendations
         WHERE student_id = :sid AND prediction_id = :pid
    """), {"sid": int(student_id), "pid": int(prediction_id)})

    if recs:
        conn.execute(text("""
            INSERT INTO recommendations (student_id, prediction_id, recommendationType, message)
            VALUES (:sid, :pid, :typ, :msg)
        """), [{"sid": int(student_id), "pid": int(prediction_id), "typ": t, "msg": m} for (t, m) in recs])

# =====================
# App init
# =====================
st.set_page_config(page_title="EduTrack — Student Performance", layout="centered")
st.title("EduTrack — Student Performance")

ensure_schema()

# Make the active DB path obvious while debugging
st.caption(f"Database file: `{DB_FILE}`")

# =====================
# Admin bootstrap (first run)
# =====================
if not any_admin_exists():
    st.subheader("Initial Admin Setup")
    with st.form("bootstrap_admin", clear_on_submit=True):
        a_name  = st.text_input("Admin Name", placeholder="Admin")
        a_email = st.text_input("Admin Email", placeholder="admin@university.edu")
        a_pass  = st.text_input("Admin Password", type="password", placeholder="••••••••")
        ok = st.form_submit_button("Create Admin")
    if ok:
        if not (a_name and a_email and a_pass):
            st.error("All fields are required.")
        else:
            try:
                create_user(a_name, a_email, a_pass, "ADMIN")
                st.success(f"Admin created: {a_email}. Please sign in.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create admin: {e}")
    st.stop()

# =====================
# Auth (very light)
# =====================
if "auth" not in st.session_state:
    st.session_state["auth"] = None

def show_login():
    with st.form("login", clear_on_submit=False):
        st.subheader("Sign in")
        login_email = st.text_input("Email", placeholder="you@university.edu")
        login_role  = st.selectbox("Role", ["ADMIN", "TEACHER", "STUDENT"])
        login_pw    = st.text_input("Password", type="password", placeholder="••••••••")
        submit_login = st.form_submit_button("Sign in")

    if not submit_login:
        st.stop()

    u = get_user_by_email(login_email)
    if not u:
        st.error("User not found.")
        st.stop()

    if not verify_pw(login_pw, u["password_hash"]):
        st.error("Incorrect password.")
        st.stop()

    if u.get("role") != login_role:
        st.error("Role mismatch for this account.")
        st.stop()

    st.session_state["auth"] = {
        "user_id": u["user_id"], "email": u["email"],
        "role": u["role"], "name": u.get("name", "")
    }
    st.rerun()

if st.session_state["auth"] is None:
    show_login()

auth = st.session_state["auth"]

with st.sidebar:
    st.caption(f"Signed in as **{auth['email']}** ({auth['role']})")
    role = auth["role"]
    st.sidebar.title("EduTrack")
    if role == "ADMIN":
        choice = st.sidebar.radio("Admin", ["Create Accounts","Manage Accounts","Analytics"])
    elif role == "TEACHER":
        choice = st.sidebar.radio("Teacher", ["My Profile","My Students","Predict"])
    elif role == "STUDENT":
        choice = st.sidebar.radio("Student", ["My Profile","My Performance","Predict"])
    else:
        st.stop()
    if st.button("Log out"):
        st.session_state["auth"] = None
        st.rerun()

# =====================
# Model & feature schema
# =====================
mdl = latest_model("student_grade_predictor")
if not mdl:
    st.warning("No trained model found. Run: python -m src.train_model")
    st.stop()

try:
    pipe = load(mdl["path"])
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    with open(mdl["columns_path"], "r", encoding="utf-8") as f:
        feature_schema = json.load(f)
except Exception as e:
    st.error(f"Failed to load feature schema: {e}")
    st.stop()

def render_extra_features(feature_schema):
    """
    Build 'Additional features' section.
    - Skip PII and anything that looks like a core field (even if named differently)
    - Known categorical fields -> dropdowns with fixed options
    - Others: numeric -> text input, else free text
    """
    st.markdown("**Additional features**")
    values: Dict[str, object] = {}

    CHOICE_MATCHERS = [
        (lambda nf: "extracurricular" in nf and "activit" in nf, ["Yes", "No"]),
        (lambda nf: "internet" in nf and "home" in nf, ["Yes", "No"]),
        (lambda nf: ("parent" in nf or "parental" in nf) and "education" in nf, ["High", "Medium", "Low"]),
        (lambda nf: "family" in nf and "income" in nf, ["Low", "Mid", "High"]),
        (lambda nf: nf == "gender", ["Female", "Male", "Non-binary", "Prefer not to say"]),
    ]

    def looks_core_like(nf: str) -> bool:
        return any(tok in nf for tok in CORE_LIKE_TOKENS)

    for feat in feature_schema:
        fname = feat["name"]
        ftype = feat.get("type", "number")

        nf_raw = (fname or "")
        nf = _canon(nf_raw)
        if nf in (_canon(x) for x in PII_FIELDS):
            continue
        if nf in CORE_FIELDS or looks_core_like(nf):
            continue

        label = nf_raw.replace("_", " ")

        matched = False
        for pred, choices in CHOICE_MATCHERS:
            if pred(nf):
                values[fname] = st.selectbox(label, options=[""] + choices, index=0, key=f"rf_extra_{fname}")
                matched = True
                break
        if matched:
            continue

        if ftype == "number":
            values[fname] = st.text_input(label, placeholder="numeric", key=f"rf_extra_{fname}")
        else:
            values[fname] = st.text_input(label, key=f"rf_extra_{fname}")

    return values

# ==============================================================
# ADMIN DASHBOARD
# ==============================================================
if role == "ADMIN":
    if choice == "Create Accounts":
        tab1, tab2 = st.tabs(["Create Student","Create Teacher"])

        # --- Create Student
        with tab1:
            eng = get_engine()
            with eng.begin() as conn:
                tmap = teacher_choices(conn)

            with st.form("create_student", clear_on_submit=True):
                name = st.text_input("Name")
                email = st.text_input("Email")
                pwd = st.text_input("Password", type="password")
                academic_year = st.text_input("Academic Year", placeholder="2025-26")
                age = st.number_input("Age", min_value=10, max_value=120, step=1)
                gender = st.selectbox("Gender", ["Male","Female","Non-binary","Prefer not to say"])
                ext_id = st.text_input("External Student ID")
                dept = st.text_input("Department Name (normalized)")
                gpa = st.number_input("GPA (optional)", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
                assigned_label = st.selectbox("Assigned Teacher", [""] + list(tmap.keys()))
                submitted = st.form_submit_button("Create Student")

            if submitted:
                try:
                    tid = tmap.get(assigned_label) if assigned_label else None
                    create_student_full(
                        name, email, pwd,
                        academic_year, int(age), gender,
                        ext_id, dept, (float(gpa) if gpa else None),
                        tid
                    )
                    st.success("Student created ✅")
                except Exception as e:
                    st.error(str(e))

        # --- Create Teacher
        with tab2:
            with st.form("create_teacher", clear_on_submit=True):
                t_name = st.text_input("Name")
                t_email = st.text_input("Email")
                t_pwd = st.text_input("Password", type="password")
                t_dept = st.text_input("Department Name (normalized)")
                submitted = st.form_submit_button("Create Teacher")
            if submitted:
                try:
                    create_teacher_full(t_name, t_email, t_pwd, t_dept)
                    st.success("Teacher created ✅")
                except Exception as e:
                    st.error(str(e))

    elif choice == "Manage Accounts":
        st.subheader("Users")
        try:
            eng = get_engine()
            with eng.begin() as conn:
                rows = conn.execute(
                    text("SELECT user_id, name, email, role, is_active, created_at FROM users ORDER BY user_id DESC LIMIT 100")
                ).mappings().all()
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Tip: use SQL console for bulk edits; keep performance edits out of Admin.")
        except Exception as e:
            st.error(f"List error: {e}")

    elif choice == "Analytics":
        st.subheader("Site Analytics")
        try:
            eng = get_engine()
            with eng.begin() as conn:
                total_students = conn.execute(text("SELECT COUNT(*) FROM students")).scalar()
                total_teachers = conn.execute(text("SELECT COUNT(*) FROM teachers")).scalar()
                by_dept = conn.execute(text("""
                    SELECT COALESCE(d.name,'(None)') AS dept, COUNT(*)
                      FROM students s
                      LEFT JOIN departments d ON d.department_id = s.department_id
                     GROUP BY 1
                     ORDER BY 2 DESC
                """)).fetchall()
            c1, c2 = st.columns(2)
            c1.metric("Students", total_students or 0)
            c2.metric("Teachers", total_teachers or 0)
            st.bar_chart({d: c for d, c in by_dept})
        except Exception as e:
            st.error(f"Analytics error: {e}")

# ==============================================================
# TEACHER DASHBOARD
# ==============================================================
if role == "TEACHER":
    if choice == "My Profile":
        st.subheader("My Profile")
        st.write(f"**Name:** {auth.get('name','')}")
        st.write(f"**Email:** {auth['email']}")
        st.write("Use IT admin to reset password if needed.")

    # ---- helper to load students assigned to this teacher
    def _assigned_students_for_teacher(user_id: int):
        with get_engine().begin() as conn:
            rows = conn.execute(text("""
                SELECT s.student_id,
                       u.name AS student_name,
                       s.external_student_id,
                       COALESCE(d.name,'') AS department,
                       s.academic_year,
                       s.gpa
                  FROM students s
                  JOIN teachers t  ON t.teacher_id = s.assigned_teacher_id
                  JOIN users tu     ON tu.user_id   = t.user_id
                  JOIN users u      ON u.user_id    = s.user_id
             LEFT JOIN departments d ON d.department_id = s.department_id
                 WHERE tu.user_id = :uid
              ORDER BY u.name
            """), {"uid": auth["user_id"]}).mappings().all()
        opts = {
            f"{r['student_name']}  (#{r['external_student_id'] or r['student_id']} — {r['department']})": int(r['student_id'])
            for r in rows
        }
        return rows, opts

    # ---------------------------
    # Shared: selector on both tabs
    # ---------------------------
    if choice in ("My Students", "Predict"):
        rows, options = _assigned_students_for_teacher(auth["user_id"])
        if not options:
            st.info("No assigned students yet.")
            st.stop()

        if choice == "My Students":
            st.subheader("Assigned Students")
            st.dataframe(
                pd.DataFrame(rows)[["student_name","external_student_id","department","academic_year","gpa"]],
                use_container_width=True
            )

        default_label = None
        if "open_student_id" in st.session_state:
            for lbl, sid0 in options.items():
                if sid0 == st.session_state["open_student_id"]:
                    default_label = lbl
                    break

        sel_label = st.selectbox(
            "Select student",
            list(options.keys()),
            index=(list(options.keys()).index(default_label) if default_label in options else 0)
        )
        sid = options[sel_label]
        if st.session_state.get("open_student_id") != sid:
            st.session_state["open_student_id"] = sid
            st.rerun()

    # ---------------------------
    # My Students: read-only view
    # ---------------------------
    if choice == "My Students":
        if sid:
            st.subheader("Student Profile & Performance")
            rec = get_latest_record(sid)
            if not rec:
                st.info("No academic record for this student yet.")
            else:
                core_view = {
                    "term": rec.get("term"),
                    "attendance": rec.get("attendance"),
                    "study_hours": rec.get("study_hours"),
                    "examScores": rec.get("examScores"),
                    "stress_level": rec.get("stress_level"),
                    "sleep_hours": rec.get("sleep_hours"),
                    "participation": rec.get("participation"),
                    "created_at": rec.get("created_at"),
                }
                st.table(pd.DataFrame([core_view]))
                with st.expander("Additional saved features"):
                    st.json(rec.get("extra") or {})

                with get_engine().begin() as conn:
                    preds = conn.execute(text("""
                        SELECT prediction_id, model_id, term, predictedStatus, passPercentage, failPercentage, created_at
                          FROM prediction_results
                         WHERE student_id = :sid
                      ORDER BY prediction_id DESC LIMIT 5
                    """), {"sid": sid}).mappings().all()
                if preds:
                    st.markdown("**Recent Predictions**")
                    st.dataframe(pd.DataFrame(preds), use_container_width=True)
                else:
                    st.caption("No predictions yet for this student.")
        st.stop()

    # ---------------------------
    # Predict tab: edit + predict
    # ---------------------------
    if choice == "Predict":
        if sid:
            st.subheader("Student Profile & Performance")
            with st.form("edit_perf"):
                term = st.text_input("Term (e.g., 2025-Fall)", value="")
                attendance = st.number_input("Attendance %", 0, 100, 85)
                study_hours = st.number_input("Study Hours / week", 0, 100, 10)
                exams = st.number_input("Exam Score (0–100)", 0, 100, 70)
                stress = st.slider("Stress (0-10)", 0, 10, 5)
                sleep = st.number_input("Sleep hours / day", 0.0, 24.0, 7.0, step=0.5)
                participation = st.number_input("Participation %", 0, 100, 70)

                # Additional non-PII features from model schema
                extra_vals = render_extra_features(feature_schema)

                submitted = st.form_submit_button("Save Performance")

            if submitted:
                try:
                    if not term:
                        st.error("Term is required.")
                    else:
                        rid = upsert_academic_record(
                            actor_role="TEACHER",
                            actor_user_id=auth["user_id"],
                            student_id=sid,
                            term=term,
                            attendance=float(attendance),
                            study_hours=float(study_hours),
                            examScores=float(exams),
                            stress_level=int(stress),
                            sleep_hours=float(sleep),
                            participation=float(participation),
                            extra=extra_vals,
                        )
                        st.success(f"Saved record (ID={rid}).")
                except Exception as e:
                    st.error(str(e))

            if st.button("Predict"):
                try:
                    with get_engine().begin() as conn:
                        if not can_run_prediction("TEACHER", auth["user_id"], sid, conn):
                            st.error("Not permitted.")
                        else:
                            rec = get_latest_record(sid)
                            if not rec:
                                st.error("No academic record for this student.")
                            else:
                                core = {
                                    "attendance": rec.get("attendance"),
                                    "study_hours": rec.get("study_hours"),
                                    "examScores": rec.get("examScores"),
                                    "stress_level": rec.get("stress_level"),
                                    "sleep_hours": rec.get("sleep_hours"),
                                    "participation": rec.get("participation"),
                                }
                                extra = rec.get("extra") or {}
                                row_dict = {}
                                for feat in feature_schema:
                                    name = feat["name"]
                                    if _norm(name) in {_norm(k) for k in core.keys()} and core.get(name) is not None:
                                        row_dict[name] = core[name]
                                    elif name in core and core[name] is not None:
                                        row_dict[name] = core[name]
                                    elif name in extra and extra[name] not in (None, ""):
                                        row_dict[name] = extra[name]
                                    else:
                                        row_dict[name] = 0.0 if feat.get("type") == "number" else (feat.get("choices") or [""])[0]

                                X = pd.DataFrame([row_dict], columns=[f["name"] for f in feature_schema])
                                prob_pass = float(pipe.predict_proba(X)[0, 1])
                                label = "PASS" if prob_pass >= 0.5 else "FAIL"

                                # ONE TRANSACTION: upsert prediction + replace recs
                                pred_id = save_prediction(
                                    sid, mdl["model_id"], rec.get("term","N/A"),
                                    label, prob_pass, 1.0 - prob_pass,
                                    conn=conn
                                )

                                recs_to_set = []
                                if core.get("sleep_hours") is not None and core["sleep_hours"] < 7:
                                    recs_to_set.append(("SLEEP", "Aim for 7–8 hours of sleep nightly."))
                                if core.get("study_hours") is not None and core["study_hours"] < 12:
                                    recs_to_set.append(("STUDY", "Increase study time by 2–4 hours/week."))
                                if core.get("attendance") is not None and core["attendance"] < 75:
                                    recs_to_set.append(("ATTENDANCE", "Improve attendance to at least 75%."))
                                if core.get("participation") is not None and core["participation"] < 50:
                                    recs_to_set.append(("PARTICIPATION", "Engage more in class."))

                                _replace_recommendations_with_conn(conn, sid, pred_id, recs_to_set)

                                st.info(f"Prediction: **{label}**  (Pass prob: {prob_pass:.2f})")
                                # Show recommendations as bullet points
                                render_recommendations_bullets(sid, pred_id)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ==============================================================
# STUDENT DASHBOARD
# ==============================================================
if role == "STUDENT":
    if choice == "My Profile":
        st.subheader("My Profile")
        try:
            eng = get_engine()
            with eng.begin() as conn:
                row = conn.execute(text("""
                    SELECT u.name, u.email,
                           s.external_student_id, s.academic_year, s.age, s.gender,
                           COALESCE(d.name,'') AS department,
                           s.gpa,
                           (SELECT u2.name
                              FROM teachers t2
                              JOIN users u2 ON u2.user_id = t2.user_id
                             WHERE t2.teacher_id = s.assigned_teacher_id) AS assigned_teacher
                      FROM users u
                INNER JOIN students s ON s.user_id = u.user_id
                 LEFT JOIN departments d ON d.department_id = s.department_id
                     WHERE u.user_id = :uid
                """), {"uid": auth["user_id"]}).mappings().first()
            if row:
                st.json(dict(row))
            else:
                st.info("No student profile found.")
        except Exception as e:
            st.error(f"Profile load error: {e}")

    elif choice == "My Performance":
        st.subheader("Performance (read-only)")
        try:
            eng = get_engine()
            with eng.begin() as conn:
                rows = conn.execute(text("""
                    SELECT term, attendance, study_hours, examScores, stress_level,
                           sleep_hours, participation, extra_json, created_at
                      FROM student_academic_data sad
                      JOIN students s ON s.student_id = sad.student_id
                     WHERE s.user_id = :uid
                  ORDER BY data_id DESC
                """), {"uid": auth["user_id"]}).mappings().all()

                if rows:
                    df = pd.DataFrame([{
                        "term": r["term"],
                        "attendance": r["attendance"],
                        "study_hours": r["study_hours"],
                        "examScores": r["examScores"],
                        "stress_level": r["stress_level"],
                        "sleep_hours": r["sleep_hours"],
                        "participation": r["participation"],
                        "created_at": r["created_at"],
                    } for r in rows])
                    st.dataframe(df, use_container_width=True)

                    with st.expander("See additional saved features"):
                        for r in rows:
                            extra = {}
                            try:
                                extra = json.loads(r.get("extra_json") or "{}")
                            except Exception:
                                pass
                            st.markdown(f"**{r['term']}**")
                            st.json(extra)
                else:
                    st.info("No records yet.")
        except Exception as e:
            st.error(f"Load error: {e}")

    elif choice == "Predict":
        st.subheader("Predict My Result")
        try:
            eng = get_engine()
            with eng.begin() as conn:
                sid = conn.execute(text("SELECT student_id FROM students WHERE user_id=:u"), {"u": auth["user_id"]}).scalar()
                if not sid:
                    st.error("Student profile not found.")
                elif st.button("Predict My Result"):
                    if not can_run_prediction("STUDENT", auth["user_id"], int(sid), conn):
                        st.error("You can only run predictions for your own record.")
                    else:
                        rec = get_latest_record(int(sid))
                        if not rec:
                            st.error("No academic record found.")
                        else:
                            core = {
                                "attendance": rec.get("attendance"),
                                "study_hours": rec.get("study_hours"),
                                "examScores": rec.get("examScores"),
                                "stress_level": rec.get("stress_level"),
                                "sleep_hours": rec.get("sleep_hours"),
                                "participation": rec.get("participation"),
                            }
                            extra = rec.get("extra") or {}
                            row_dict = {}
                            for feat in feature_schema:
                                name = feat["name"]
                                if _norm(name) in {_norm(k) for k in core.keys()} and core.get(name) is not None:
                                    row_dict[name] = core[name]
                                elif name in core and core[name] is not None:
                                    row_dict[name] = core[name]
                                elif name in extra and extra[name] not in (None, ""):
                                    row_dict[name] = extra[name]
                                else:
                                    row_dict[name] = 0.0 if feat.get("type") == "number" else (feat.get("choices") or [""])[0]

                            X = pd.DataFrame([row_dict], columns=[f["name"] for f in feature_schema])
                            prob_pass = float(pipe.predict_proba(X)[0, 1])
                            label = "PASS" if prob_pass >= 0.5 else "FAIL"

                            # ONE TRANSACTION for student flow as well
                            with get_engine().begin() as c2:
                                pred_id = save_prediction(
                                    int(sid), mdl["model_id"], rec.get("term","N/A"),
                                    label, prob_pass, 1.0 - prob_pass,
                                    conn=c2
                                )

                                recs_to_set = []
                                if core.get("sleep_hours") is not None and core["sleep_hours"] < 7:
                                    recs_to_set.append(("SLEEP", "Aim for 7–8 hours of sleep nightly."))
                                if core.get("study_hours") is not None and core["study_hours"] < 12:
                                    recs_to_set.append(("STUDY", "Increase study time by 2–4 hours/week."))
                                if core.get("attendance") is not None and core["attendance"] < 75:
                                    recs_to_set.append(("ATTENDANCE", "Improve attendance to at least 75%."))
                                if core.get("participation") is not None and core["participation"] < 50:
                                    recs_to_set.append(("PARTICIPATION", "Engage more in class."))

                                _replace_recommendations_with_conn(c2, int(sid), pred_id, recs_to_set)

                            st.success(f"{label} ({prob_pass:.1%})")
                            render_recommendations_bullets(int(sid), pred_id)
        except Exception as e:
            st.error(f"Prediction error: {e}")
