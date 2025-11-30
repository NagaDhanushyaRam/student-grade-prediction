import os
import json
import re
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

from mongo_client import get_db
from app_db_mongo import (
    hash_pw, verify_pw,
    any_admin_exists, get_user_by_email, create_user,
    create_teacher_full, create_student_full,
    upsert_academic_record, get_latest_record,
    latest_model, can_run_prediction, teacher_choices,
    send_message, get_inbox, get_sent_messages, mark_message_read,
    get_teacher_students, get_student_advisor,
    get_site_performance_summary, get_department_performance,
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

def _mongo_next_id(coll_name: str, id_field: str) -> int:
    db = get_db()
    last = db[coll_name].find_one(sort=[(id_field, -1)])
    return int(last[id_field]) + 1 if last and id_field in last else 1

# PII that must never appear as dropdowns
PII_FIELDS = {"student_id", "first_name", "last_name", "email", "name"}

# Canonical core fields already collected in the top section
CORE_FIELDS = {
    "attendance",
    "study_hours",
    "examscores",
    "stress_level",
    "sleep_hours",
    "participation",
}

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
    MongoDB version.
    """
    db = get_db()
    query = {"student_id": int(student_id)}
    if prediction_id is not None:
        query["prediction_id"] = int(prediction_id)

    recs = list(
        db["recommendations"]
        .find(query, {"_id": 0})
        .sort("recommendation_id", -1)
    )

    if not recs:
        st.caption("No recommendations for these inputs.")
        return

    st.markdown("**Recommendations**")
    for r in recs:
        typ = (r.get("recommendationType") or "").replace("_", " ").title()
        st.markdown(f"- **{typ}** — {r.get('message','')}")

def build_recommendations(core: dict) -> List[Tuple[str, str]]:
    recs: List[Tuple[str, str]] = []

    TARGETS = {"attendance": 85, "participation": 60, "study_hours": 12, "examScores": 65}

    sl = core.get("sleep_hours")
    if sl is not None:
        if sl < 7:
            recs.append(("SLEEP", "Aim for 7–8 hours of sleep nightly."))
        elif sl > 9:
            recs.append(("SLEEP", "Try to keep sleep in the 7–9 hour range for better alertness."))

    stlvl = core.get("stress_level")
    if stlvl is not None and stlvl >= 7:
        recs.append(("STRESS", "Use short breaks, deep breathing, or a 25-minute Pomodoro cycle."))

    att = core.get("attendance")
    if att is not None and att < TARGETS["attendance"]:
        recs.append(("ATTENDANCE", f"Raise attendance to at least {TARGETS['attendance']}%."))

    part = core.get("participation")
    if part is not None and part < TARGETS["participation"]:
        recs.append(("PARTICIPATION", f"Ask one question per class to lift participation above {TARGETS['participation']}%."))

    sh = core.get("study_hours")
    if sh is not None and sh < TARGETS["study_hours"]:
        recs.append(("STUDY", "Increase study time by 2–4 hours/week and split it across 3–4 sessions."))

    ex = core.get("examScores")
    if ex is not None and ex < TARGETS["examScores"]:
        recs.append(("EXAMS", "Do 2 extra practice sets this week and review misses with a TA or peer."))

    if not recs:
        if sh is not None and sh < (TARGETS["study_hours"] + 2):
            recs.append(("STUDY", "Add one extra 45-minute study block this week to stay ahead."))
        elif att is not None and att < 95:
            recs.append(("ATTENDANCE", "Aim for 95% attendance to keep momentum strong."))
        else:
            recs.append(("GENERAL", "Maintain your routine and plan next week’s study slots on Sunday evening."))

    seen = set()
    return [(t, m) for (t, m) in recs if (t, m) not in seen and not seen.add((t, m))]

# =============================
# Prediction save (UPSERT) + id
# =============================
def save_prediction(student_id, model_id, term, label, pass_p, fail_p, conn=None) -> int:
    """
    Upsert a prediction for (student_id, model_id, term) and return prediction_id.
    MongoDB version (conn is ignored, kept only for API compatibility).
    """
    db = get_db()
    coll = db["prediction_results"]

    sid = int(student_id)
    mid = int(model_id)
    term = term

    existing = coll.find_one({"student_id": sid, "model_id": mid, "term": term})
    if existing:
        prediction_id = int(existing["prediction_id"])
    else:
        prediction_id = _mongo_next_id("prediction_results", "prediction_id")

    coll.update_one(
        {"student_id": sid, "model_id": mid, "term": term},
        {
            "$set": {
                "prediction_id": prediction_id,
                "predictedStatus": label,
                "passPercentage": float(pass_p),
                "failPercentage": float(fail_p),
            },
            "$setOnInsert": {
                "created_at": datetime.utcnow().isoformat(),
            },
        },
        upsert=True,
    )

    return int(prediction_id)

# =========================================
# Replace recs (single logical operation)
# =========================================
def _replace_recommendations_with_conn(conn, student_id: int, prediction_id: int,
                                       recs: List[Tuple[str, str]]):
    """
    Mongo version; `conn` is ignored (kept for API compatibility).
    """
    db = get_db()
    coll = db["recommendations"]

    # de-dupe
    seen = set()
    recs = [(t, m) for (t, m) in recs if (t, m) not in seen and not seen.add((t, m))]

    coll.delete_many({"student_id": int(student_id), "prediction_id": int(prediction_id)})

    if recs:
        base_id = _mongo_next_id("recommendations", "recommendation_id")
        docs = []
        for offset, (typ, msg) in enumerate(recs):
            docs.append(
                {
                    "recommendation_id": base_id + offset,
                    "student_id": int(student_id),
                    "prediction_id": int(prediction_id),
                    "recommendationType": typ,
                    "message": msg,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
        coll.insert_many(docs)

# NEW: ADMIN PERFORMANCE REPORT PAGE
# ============================================
def admin_performance_report_page(auth: dict) -> None:
    st.subheader("Performance Report")

    summary = get_site_performance_summary()
    dept_rows = get_department_performance()

    if not summary or summary.get("total_students") is None:
        st.info("No performance data available yet.")
        return

    # --- Top KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Students", int(summary.get("total_students") or 0))
    with col2:
        st.metric("Teachers", int(summary.get("total_teachers") or 0))
    with col3:
        st.metric("Avg GPA", round(summary.get("avg_gpa") or 0.0, 2))
    with col4:
        avg_att = summary.get("avg_attendance") or 0.0
        st.metric("Avg Attendance", f"{round(avg_att, 1)}%")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Predicted Pass", int(summary.get("pass_count") or 0))
    with col6:
        st.metric("Predicted Fail", int(summary.get("fail_count") or 0))

    st.markdown("---")
    st.subheader("Department-level Performance")

    if not dept_rows:
        st.info("No department data available.")
        return

    df_dept = pd.DataFrame(dept_rows)
    st.dataframe(df_dept)

    # Simple charts (if columns exist)
    if {"department", "avg_gpa"}.issubset(df_dept.columns):
        st.markdown("#### Average GPA by Department")
        st.bar_chart(df_dept.set_index("department")["avg_gpa"])

    if {"department", "avg_attendance"}.issubset(df_dept.columns):
        st.markdown("#### Average Attendance by Department")
        st.bar_chart(df_dept.set_index("department")["avg_attendance"])

    # Export CSV
    csv = df_dept.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Department Performance (CSV)",
        data=csv,
        file_name="department_performance.csv",
        mime="text/csv",
    )

# ============================================
# MESSAGING PAGES (TOP-LEVEL HELPERS)
# ============================================
def messaging_page_teacher(auth: dict) -> None:
    st.subheader("Teacher–Student Messages")

    # --- Send message form ---
    st.markdown("### Send a New Message")
    students = get_teacher_students(auth["user_id"])

    if not students:
        st.info("No students are currently assigned to you.")
    else:
        # Map label -> student_user_id
        options = {
            f"{s['student_name']} ({s['department']})": s["student_user_id"]
            for s in students
        }
        selected_label = st.selectbox("Select student", list(options.keys()))
        subject = st.text_input("Subject")
        body = st.text_area("Message")

        if st.button("Send Message"):
            if body.strip():
                recv_id = options[selected_label]
                send_message(auth["user_id"], recv_id, subject, body)
                st.success("Message sent.")
            else:
                st.warning("Message body cannot be empty.")

    st.markdown("---")
    st.markdown("### Inbox")
    inbox = get_inbox(auth["user_id"])
    if not inbox:
        st.info("No messages in your inbox.")
    else:
        for msg in inbox:
            key_prefix = f"t_in_{msg['message_id']}"
            with st.expander(
                f"{msg['created_at']} – {msg['sender_name']} "
                f"({msg['readStatus']}) – {msg['subject']}"
            ):
                st.write(f"**From:** {msg['sender_name']} <{msg['sender_email']}>")
                st.write("---")
                st.write(msg["content"])
                if msg["readStatus"] == "UNREAD":
                    if st.button("Mark as read", key=f"{key_prefix}_read"):
                        mark_message_read(msg["message_id"], auth["user_id"])
                        st.rerun()

    st.markdown("### Sent")
    sent = get_sent_messages(auth["user_id"])
    if not sent:
        st.info("No sent messages.")
    else:
        for msg in sent:
            key_prefix = f"t_out_{msg['message_id']}"
            with st.expander(
                f"{msg['created_at']} – To {msg['receiver_name']} – {msg['subject']}"
            ):
                st.write(f"**To:** {msg['receiver_name']} <{msg['receiver_email']}>")
                st.write("---")
                st.write(msg["content"])


def messaging_page_student(auth: dict) -> None:
    st.subheader("Messages")

    advisor = get_student_advisor(auth["user_id"])

    # --- Send message to advisor (if one exists) ---
    if advisor:
        st.markdown("### Message Your Advisor")
        st.write(
            f"Advisor: **{advisor['teacher_name']}** "
            f"(<{advisor['teacher_email']}>)"
        )
        subject = st.text_input("Subject", key="stu_msg_subject")
        body = st.text_area("Message", key="stu_msg_body")
        if st.button("Send to Advisor"):
            if body.strip():
                send_message(
                    auth["user_id"],
                    advisor["teacher_user_id"],
                    subject,
                    body,
                )
                st.success("Message sent to your advisor.")
            else:
                st.warning("Message body cannot be empty.")

    st.markdown("---")
    st.markdown("### Inbox")
    inbox = get_inbox(auth["user_id"])
    if not inbox:
        st.info("No messages yet.")
    else:
        for msg in inbox:
            key_prefix = f"s_in_{msg['message_id']}"
            with st.expander(
                f"{msg['created_at']} – {msg['sender_name']} "
                f"({msg['readStatus']}) – {msg['subject']}"
            ):
                st.write(f"**From:** {msg['sender_name']} <{msg['sender_email']}>")
                st.write("---")
                st.write(msg["content"])
                if msg["readStatus"] == "UNREAD":
                    if st.button("Mark as read", key=f"{key_prefix}_read"):
                        mark_message_read(msg["message_id"], auth["user_id"])
                        st.rerun()

    st.markdown("### Sent")
    sent = get_sent_messages(auth["user_id"])
    if not sent:
        st.info("No sent messages.")
    else:
        for msg in sent:
            key_prefix = f"s_out_{msg['message_id']}"
            with st.expander(
                f"{msg['created_at']} – To {msg['receiver_name']} – {msg['subject']}"
            ):
                st.write(f"**To:** {msg['receiver_name']} <{msg['receiver_email']}>")
                st.write("---")
                st.write(msg["content"])


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
                values[fname] = st.selectbox(
                    label,
                    options=choices,
                    index=None,
                    placeholder="Select…",
                    key=f"rf_extra_{fname}",
                )
                matched = True
                break
        if matched:
            continue

        if ftype == "number":
            values[fname] = st.text_input(label, placeholder="numeric", key=f"rf_extra_{fname}")
        else:
            values[fname] = st.text_input(label, key=f"rf_extra_{fname}")

    return values


# =====================
# App init
# =====================
st.set_page_config(page_title="EduTrack — Student Performance", layout="centered")
st.title("EduTrack — Student Performance")

st.caption("Database: MongoDB Atlas (hosted)")


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
                create_user(a_name, a_email.strip().lower(), a_pass, "ADMIN")
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
    st.subheader("Sign in")

    login_email = st.text_input("Email", placeholder="you@university.edu", key="login_email")
    login_role  = st.selectbox("Role", ["ADMIN", "TEACHER", "STUDENT"], key="login_role")
    login_pw    = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")

    all_filled = bool(login_email.strip() and login_pw.strip())
    clicked = st.button("Sign in", disabled=not all_filled)

    if not clicked:
        return  # just show the form

    if not login_email.strip():
        st.error("Email is required.")
        return
    if not login_pw.strip():
        st.error("Password is required.")
        return

    login_email_norm = login_email.strip().lower()

    u = get_user_by_email(login_email_norm)
    if not u:
        st.error("User not found.")
        return

    if not verify_pw(login_pw, u["password_hash"]):
        st.error("Incorrect password.")
        return

    if u.get("role") != login_role:
        st.error("Role mismatch for this account.")
        return

    st.session_state["auth"] = {
        "user_id": u["user_id"],
        "email":  u["email"],
        "role":   u["role"],
        "name":   u.get("name", "")
    }
    st.rerun()


# ---- enforce login before showing rest of app ----
if st.session_state["auth"] is None:
    show_login()
    st.stop()        # ⬅️ important: don’t run the rest of the app until logged in

auth = st.session_state["auth"]


with st.sidebar:
    st.caption(f"Signed in as **{auth['email']}** ({auth['role']})")
    role = auth["role"]
    st.sidebar.title("EduTrack")
    if role == "ADMIN":
        choice = st.sidebar.radio("Admin", ["Create Accounts","Manage Accounts","Analytics","Performance Report"])
    elif role == "TEACHER":
        choice = st.sidebar.radio("Teacher", ["My Profile","My Students","Predict","Messages"])
    elif role == "STUDENT":
        choice = st.sidebar.radio("Student", ["My Profile","My Performance","Predict","Messages"])
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


# ==============================================================
# ADMIN DASHBOARD
# ==============================================================
if role == "ADMIN":
    if choice == "Create Accounts":
        tab1, tab2 = st.tabs(["Create Student","Create Teacher"])

        # --- Create Student
        with tab1:
            tmap = teacher_choices()

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
                teacher_labels = list(tmap.keys())
                assigned_label = st.selectbox(
                    "Assigned Teacher",
                    teacher_labels,
                    index=None,
                    placeholder="Select a teacher (optional)",
                )
                submitted = st.form_submit_button("Create Student")

                if submitted:
                    try:
                        # assigned_label will be None if nothing is chosen
                        tid = tmap.get(assigned_label) if assigned_label else None
                        create_student_full(
                            name,
                            email.strip().lower(),
                            pwd,
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
                    create_teacher_full(t_name, t_email.strip().lower(), t_pwd, t_dept)
                    st.success("Teacher created ✅")
                except Exception as e:
                    st.error(str(e))

    elif choice == "Manage Accounts":
        st.subheader("Users")
        try:
            db = get_db()
            rows = list(
                db["users"]
                .find({}, {"_id": 0, "password_hash": 0})
                .sort("user_id", -1)
                .limit(100)
            )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Tip: for bulk edits use a separate admin script; keep Streamlit UI simple.")
        except Exception as e:
            st.error(f"List error: {e}")

    elif choice == "Analytics":
        st.subheader("Site Analytics")
        try:
            summary = get_site_performance_summary()
            db = get_db()

            total_students = summary.get("total_students") or 0
            total_teachers = summary.get("total_teachers") or 0

            # build dept counts for bar chart
            students = list(db["students"].find({}))
            departments = {d["department_id"]: d["name"] for d in db["departments"].find({})}
            counts = {}
            for s in students:
                dep_id = s.get("department_id")
                dep_name = departments.get(dep_id, "(None)")
                counts[dep_name] = counts.get(dep_name, 0) + 1

            c1, c2 = st.columns(2)
            c1.metric("Students", total_students)
            c2.metric("Teachers", total_teachers)
            if counts:
                st.bar_chart(counts)
            else:
                st.caption("No students yet.")
        except Exception as e:
            st.error(f"Analytics error: {e}")
            
    elif choice == "Performance Report":  # NEW
        admin_performance_report_page(auth)


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
        db = get_db()
        teacher = db["teachers"].find_one({"user_id": int(user_id)})
        if not teacher:
            return [], {}

        tid = int(teacher["teacher_id"])
        students = list(db["students"].find({"assigned_teacher_id": tid}))
        rows = []
        options = {}

        departments = {d["department_id"]: d["name"] for d in db["departments"].find({})}

        for s in students:
            stu_user = db["users"].find_one({"user_id": int(s["user_id"])})
            if not stu_user:
                continue
            dept_name = departments.get(s.get("department_id"), "")
            row = {
                "student_id": int(s["student_id"]),
                "student_name": stu_user["name"],
                "external_student_id": s.get("external_student_id"),
                "department": dept_name,
                "academic_year": s.get("academic_year"),
                "gpa": s.get("gpa"),
            }
            rows.append(row)
            label = f"{row['student_name']}  (#{row['external_student_id'] or row['student_id']} — {row['department']})"
            options[label] = row["student_id"]

        rows.sort(key=lambda r: r["student_name"])
        return rows, options

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

        labels = list(options.keys())
        sel_label = st.selectbox(
            "Select student",
            labels,
            index=(labels.index(default_label) if default_label in labels else 0)
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

                db = get_db()
                preds = list(
                    db["prediction_results"]
                    .find({"student_id": int(sid)}, {"_id": 0})
                    .sort("prediction_id", -1)
                    .limit(5)
                )
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
                    if not can_run_prediction("TEACHER", auth["user_id"], sid):
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

                            pred_id = save_prediction(
                                sid, mdl["model_id"], rec.get("term","N/A"),
                                label, prob_pass, 1.0 - prob_pass,
                                conn=None
                            )
                            recs_to_set = build_recommendations(core)
                            _replace_recommendations_with_conn(None, sid, pred_id, recs_to_set)

                    st.info(f"Prediction: **{label}**  (Pass prob: {prob_pass:.2f})")
                    render_recommendations_bullets(sid, pred_id)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    if choice == "Messages":
        messaging_page_teacher(auth)


# ==============================================================
# STUDENT DASHBOARD
# ==============================================================
if role == "STUDENT":
    if choice == "My Profile":
        st.subheader("My Profile")
        try:
            db = get_db()
            user = db["users"].find_one({"user_id": int(auth["user_id"])})
            student = db["students"].find_one({"user_id": int(auth["user_id"])})
            dept = None
            if student and student.get("department_id") is not None:
                dept = db["departments"].find_one(
                    {"department_id": int(student["department_id"])}
                )
            assigned_name = None
            if student and student.get("assigned_teacher_id"):
                t = db["teachers"].find_one(
                    {"teacher_id": int(student["assigned_teacher_id"])}
                )
                if t:
                    tu = db["users"].find_one({"user_id": int(t["user_id"])})
                    if tu:
                        assigned_name = tu.get("name")

            if user and student:
                row = {
                    "name": user.get("name"),
                    "email": user.get("email"),
                    "external_student_id": student.get("external_student_id"),
                    "academic_year": student.get("academic_year"),
                    "age": student.get("age"),
                    "gender": student.get("gender"),
                    "department": dept.get("name") if dept else "",
                    "gpa": student.get("gpa"),
                    "assigned_teacher": assigned_name,
                }
                st.json(row)
            else:
                st.info("No student profile found.")
        except Exception as e:
            st.error(f"Profile load error: {e}")

    elif choice == "My Performance":
        st.subheader("Performance (read-only)")
        try:
            db = get_db()
            student = db["students"].find_one({"user_id": int(auth["user_id"])})
            if not student:
                st.info("No student record found.")
            else:
                sid = int(student["student_id"])
                rows = list(
                    db["student_academic_data"]
                    .find({"student_id": sid})
                    .sort("data_id", -1)
                )

                if rows:
                    df = pd.DataFrame([{
                        "term": r.get("term"),
                        "attendance": r.get("attendance"),
                        "study_hours": r.get("study_hours"),
                        "examScores": r.get("examScores"),
                        "stress_level": r.get("stress_level"),
                        "sleep_hours": r.get("sleep_hours"),
                        "participation": r.get("participation"),
                        "created_at": r.get("created_at"),
                    } for r in rows])
                    st.dataframe(df, use_container_width=True)

                    with st.expander("See additional saved features"):
                        for r in rows:
                            try:
                                extra = json.loads(r.get("extra_json") or "{}")
                            except Exception:
                                extra = {}
                            st.markdown(f"**{r.get('term')}**")
                            st.json(extra)
                else:
                    st.info("No records yet.")
        except Exception as e:
            st.error(f"Load error: {e}")

    elif choice == "Predict":
        st.subheader("Predict My Result")
        try:
            db = get_db()
            stu_doc = db["students"].find_one({"user_id": int(auth["user_id"])})
            sid = int(stu_doc["student_id"]) if stu_doc else None

            if not sid:
                st.error("Student profile not found.")
            elif st.button("Predict My Result"):
                if not can_run_prediction("STUDENT", auth["user_id"], int(sid)):
                    st.error("You can only run predictions for your own record.")
                else:
                    rec = get_latest_record(int(sid))
                    if not rec:
                        st.error("No academic record found.")
                    else:
                        core = {
                            "attendance":    rec.get("attendance"),
                            "study_hours":   rec.get("study_hours"),
                            "examScores":    rec.get("examScores"),
                            "stress_level":  rec.get("stress_level"),
                            "sleep_hours":   rec.get("sleep_hours"),
                            "participation": rec.get("participation"),
                        }
                        extra = rec.get("extra") or {}

                        # Build model row with exact training columns
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

                        pred_id = save_prediction(
                            int(sid), mdl["model_id"], rec.get("term", "N/A"),
                            label, prob_pass, 1.0 - prob_pass,
                            conn=None
                        )
                        # Always-on recs (same helper as Teacher)
                        recs_to_set = build_recommendations(core)
                        _replace_recommendations_with_conn(None, int(sid), pred_id, recs_to_set)

                if sid:
                    st.success(f"{label} ({prob_pass:.1%})")
                    render_recommendations_bullets(int(sid), pred_id)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    if choice == "Messages":
        messaging_page_student(auth)
