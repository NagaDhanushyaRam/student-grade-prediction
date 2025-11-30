# app_db.py
import os
import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

# ---------------------------------------------------------
# Paths / Engine
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
DB_DIR = ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = os.path.abspath(os.getenv("EDUTRACK_DB", str(DB_DIR / "app.db")))
DB_URL = f"sqlite:///{DB_FILE}"

# A small pepper so hashes aren't plain SHA256(password)
_PWD_PEPPER = "edutrack_pepper_2025"


def get_engine() -> Engine:
    eng = create_engine(
        DB_URL,
        future=True,
        connect_args={
            "timeout": 15,
            "check_same_thread": False,  # allow threads (Streamlit)
        },
        pool_pre_ping=True,
    )

    # Ensure FK enforcement on every connection
    @event.listens_for(eng, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    return eng


# ---------------------------------------------------------
# Password helpers (simple salted/peppered SHA256)
# ---------------------------------------------------------
def hash_pw(raw: str) -> str:
    raw = (raw or "").encode("utf-8")
    return hashlib.sha256(_PWD_PEPPER.encode("utf-8") + raw).hexdigest()


def verify_pw(raw: str, digest: str) -> bool:
    calc = hash_pw(raw)
    return hmac.compare_digest(calc, digest or "")




def ensure_schema() -> None:
    eng = get_engine()
    schema_path = ROOT / "db" / "schema.sql"
    with eng.begin() as conn:
        if eng.dialect.name == "sqlite":
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.connection.executescript(f.read())
        else:
            # For non-SQLite engines (future-proof)
            stmts = [s.strip() for s in open(schema_path, "r", encoding="utf-8").read().split(";") if s.strip()]
            for s in stmts:
                conn.exec_driver_sql(s)


# ---------------------------------------------------------
# Users
# ---------------------------------------------------------
def any_admin_exists() -> bool:
    with get_engine().begin() as conn:
        v = conn.execute(text("SELECT 1 FROM users WHERE role='ADMIN' LIMIT 1")).scalar()
        return v is not None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE LOWER(email)=:e LIMIT 1"),
            {"e": email},
        ).mappings().first()
        return dict(row) if row else None


def create_user(name: str, email: str, password: str, role: str) -> int:
    digest = hash_pw(password)
    with get_engine().begin() as conn:
        r = conn.execute(
            text("""
                INSERT INTO users(name, email, password_hash, role)
                VALUES (:n, :e, :p, :r)
            """),
            {"n": name, "e": email, "p": digest, "r": role},
        )
        return int(r.lastrowid)


# ---------------------------------------------------------
# Departments helper
# ---------------------------------------------------------
def _ensure_department(conn, dept_name: Optional[str]) -> Optional[int]:
    if not dept_name:
        return None
    dept_name = dept_name.strip()
    if not dept_name:
        return None
    # upsert department by name
    conn.execute(
        text("INSERT INTO departments(name) VALUES (:n) ON CONFLICT(name) DO NOTHING"),
        {"n": dept_name},
    )
    did = conn.execute(
        text("SELECT department_id FROM departments WHERE name=:n"),
        {"n": dept_name},
    ).scalar()
    return int(did) if did else None


# ---------------------------------------------------------
# Create Teacher / Student (full)
# ---------------------------------------------------------
def create_teacher_full(name: str, email: str, password: str, dept_name: Optional[str]) -> int:
    with get_engine().begin() as conn:
        # user
        uid = conn.execute(
            text("""INSERT INTO users(name, email, password_hash, role)
                    VALUES (:n,:e,:p,'TEACHER')"""),
            {"n": name, "e": email, "p": hash_pw(password)},
        ).lastrowid

        did = _ensure_department(conn, dept_name)

        tid = conn.execute(
            text("""INSERT INTO teachers(user_id, department_id)
                    VALUES (:u, :d)"""),
            {"u": uid, "d": did},
        ).lastrowid
        return int(tid)


def create_student_full(
    name: str, email: str, password: str,
    academic_year: Optional[str], age: Optional[int], gender: Optional[str],
    external_student_id: Optional[str], dept_name: Optional[str],
    gpa: Optional[float], assigned_teacher_id: Optional[int]
) -> int:
    with get_engine().begin() as conn:
        uid = conn.execute(
            text("""INSERT INTO users(name, email, password_hash, role)
                    VALUES (:n,:e,:p,'STUDENT')"""),
            {"n": name, "e": email, "p": hash_pw(password)},
        ).lastrowid

        did = _ensure_department(conn, dept_name)

        sid = conn.execute(
            text("""
                INSERT INTO students(
                    user_id, academic_year, age, gender, external_student_id,
                    department_id, assigned_teacher_id, gpa
                ) VALUES (:u,:ay,:age,:g,:ext,:d,:t,:gpa)
            """),
            {
                "u": uid, "ay": academic_year, "age": age, "g": gender,
                "ext": external_student_id, "d": did,
                "t": assigned_teacher_id, "gpa": gpa,
            },
        ).lastrowid
        return int(sid)


# ---------------------------------------------------------
# Academic records
# ---------------------------------------------------------
def upsert_academic_record(
    actor_role: str,
    actor_user_id: int,
    student_id: int,
    term: str,
    attendance: Optional[float],
    study_hours: Optional[float],
    examScores: Optional[float],
    stress_level: Optional[int],
    sleep_hours: Optional[float],
    participation: Optional[float],
    extra: Optional[Dict[str, Any]],
) -> int:
    """
    Upsert by (student_id, term) and return data_id.
    """
    payload = json.dumps(extra or {}, ensure_ascii=False)

    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO student_academic_data (
                    student_id, term, attendance, study_hours, examScores,
                    stress_level, sleep_hours, participation, extra_json
                ) VALUES (
                    :sid, :term, :att, :sh, :ex, :st, :sl, :part, :extra
                )
                ON CONFLICT(student_id, term) DO UPDATE SET
                    attendance   = excluded.attendance,
                    study_hours  = excluded.study_hours,
                    examScores   = excluded.examScores,
                    stress_level = excluded.stress_level,
                    sleep_hours  = excluded.sleep_hours,
                    participation= excluded.participation,
                    extra_json   = excluded.extra_json
            """),
            {
                "sid": int(student_id),
                "term": term,
                "att": attendance,
                "sh": study_hours,
                "ex": examScores,
                "st": stress_level,
                "sl": sleep_hours,
                "part": participation,
                "extra": payload,
            },
        )

        did = conn.execute(
            text("""
                SELECT data_id
                  FROM student_academic_data
                 WHERE student_id=:sid AND term=:term
            """),
            {"sid": int(student_id), "term": term},
        ).scalar()

        return int(did)


def get_latest_record(student_id: int) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        r = conn.execute(
            text("""
                SELECT data_id, student_id, term, attendance, study_hours, examScores,
                       stress_level, sleep_hours, participation, extra_json, created_at
                  FROM student_academic_data
                 WHERE student_id = :sid
              ORDER BY data_id DESC
                 LIMIT 1
            """),
            {"sid": int(student_id)},
        ).mappings().first()

        if not r:
            return None

        extra = {}
        try:
            extra = json.loads(r.get("extra_json") or "{}")
        except Exception:
            pass

        return {
            "data_id": r["data_id"],
            "student_id": r["student_id"],
            "term": r["term"],
            "attendance": r["attendance"],
            "study_hours": r["study_hours"],
            "examScores": r["examScores"],
            "stress_level": r["stress_level"],
            "sleep_hours": r["sleep_hours"],
            "participation": r["participation"],
            "extra": extra,
            "created_at": r["created_at"],
        }


# ---------------------------------------------------------
# Model registry
# ---------------------------------------------------------
def latest_model(name: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        r = conn.execute(
            text("""
                SELECT model_id, model_name, version, path, columns_path, created_at
                  FROM ml_models
                 WHERE model_name = :n
              ORDER BY model_id DESC
                 LIMIT 1
            """),
            {"n": name},
        ).mappings().first()
        return dict(r) if r else None

def register_model(
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: str = "",
    columns_path: str = "",
    version: Optional[str] = None,   # <— NEW
    **metrics,
) -> int:
    final_name = (name or model_name or "").strip()
    if not final_name:
        raise ValueError("register_model: `name` or `model_name` is required")
    if not model_path or not columns_path:
        raise ValueError("register_model: `model_path` and `columns_path` are required")

    # Default version if caller doesn’t pass one (timestamp so it’s always unique)
    ver = (version or datetime.now().strftime("%Y%m%d%H%M%S")).strip()

    with get_engine().begin() as conn:
        r = conn.execute(
            text("""
                INSERT INTO ml_models (model_name, version, path, columns_path)
                VALUES (:n, :v, :p, :c)
                ON CONFLICT(model_name, version) DO UPDATE SET
                    path = excluded.path,
                    columns_path = excluded.columns_path,
                    created_at = datetime('now')
            """),
            {"n": final_name, "v": ver, "p": model_path, "c": columns_path},
        )
        return int(r.lastrowid or conn.execute(
            text("SELECT model_id FROM ml_models WHERE model_name=:n AND version=:v"),
            {"n": final_name, "v": ver},
        ).scalar())
    
# ---------------------------------------------------------
# Permissions helpers
# ---------------------------------------------------------
def can_run_prediction(role: str, user_id: int, student_id: int, conn) -> bool:
    """
    Use the provided open connection to keep things consistent with callers.
    """
    role = (role or "").upper()
    if role == "ADMIN":
        return True
    if role == "STUDENT":
        # student can run for themselves
        v = conn.execute(
            text("""
                SELECT 1
                  FROM students s
                  JOIN users u ON u.user_id = s.user_id
                 WHERE s.student_id = :sid AND u.user_id = :uid
            """),
            {"sid": int(student_id), "uid": int(user_id)},
        ).scalar()
        return v is not None
    if role == "TEACHER":
        # teacher must be assigned to that student
        v = conn.execute(
            text("""
                SELECT 1
                  FROM students s
                  JOIN teachers t ON t.teacher_id = s.assigned_teacher_id
                  JOIN users tu    ON tu.user_id = t.user_id
                 WHERE s.student_id = :sid AND tu.user_id = :uid
            """),
            {"sid": int(student_id), "uid": int(user_id)},
        ).scalar()
        return v is not None
    return False


def teacher_choices(conn) -> Dict[str, int]:
    """
    Build label -> teacher_id for the teacher picker.
    Uses the provided open connection.
    """
    rows = conn.execute(
        text("""
            SELECT t.teacher_id, u.name AS tname, COALESCE(d.name,'') AS dname
              FROM teachers t
              JOIN users u ON u.user_id = t.user_id
         LEFT JOIN departments d ON d.department_id = t.department_id
          ORDER BY u.name
        """)
    ).mappings().all()

    mapping: Dict[str, int] = {}
    for r in rows:
        label = f"{r['tname']} ({r['dname']})" if r["dname"] else r["tname"]
        mapping[label] = int(r["teacher_id"])
    return mapping

# ============================================
# NEW: PERFORMANCE REPORT HELPERS (ADMIN)
# ============================================

def get_site_performance_summary() -> dict:
    """
    High-level performance metrics across the whole site.
    Returns a dict with keys:
      total_students, total_teachers, avg_gpa,
      avg_attendance, pass_count, fail_count
    """
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT
              COUNT(DISTINCT s.student_id)                      AS total_students,
              COUNT(DISTINCT t.teacher_id)                      AS total_teachers,
              AVG(s.gpa)                                        AS avg_gpa,
              AVG(a.attendance)                                 AS avg_attendance,
              SUM(CASE WHEN pr.predictedStatus = 'PASS' THEN 1 ELSE 0 END) AS pass_count,
              SUM(CASE WHEN pr.predictedStatus = 'FAIL' THEN 1 ELSE 0 END) AS fail_count
            FROM students s
            LEFT JOIN student_academic_data a
                   ON a.student_id = s.student_id
            LEFT JOIN prediction_results pr
                   ON pr.student_id = s.student_id
            LEFT JOIN teachers t
                   ON t.teacher_id = s.assigned_teacher_id
        """)).mappings().first()

    return dict(row) if row else {}


def get_department_performance() -> list[dict]:
    """
    Department-level performance metrics.
    Each dict has keys:
      department, total_students, avg_gpa,
      avg_attendance, pass_count, fail_count
    """
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            SELECT
              d.name                                           AS department,
              COUNT(DISTINCT s.student_id)                     AS total_students,
              AVG(s.gpa)                                       AS avg_gpa,
              AVG(a.attendance)                                AS avg_attendance,
              SUM(CASE WHEN pr.predictedStatus = 'PASS' THEN 1 ELSE 0 END) AS pass_count,
              SUM(CASE WHEN pr.predictedStatus = 'FAIL' THEN 1 ELSE 0 END) AS fail_count
            FROM departments d
            LEFT JOIN students s
                   ON s.department_id = d.department_id
            LEFT JOIN student_academic_data a
                   ON a.student_id = s.student_id
            LEFT JOIN prediction_results pr
                   ON pr.student_id = s.student_id
            GROUP BY d.department_id, d.name
            ORDER BY d.name
        """)).mappings().all()

    return [dict(r) for r in rows]

# ============================================
# MESSAGING HELPERS (TEACHER–STUDENT)
# ============================================

def send_message(sender_id: int, receiver_id: int,
                 subject: str | None, content: str) -> None:
    """
    Insert a new message.
    """
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (sender_id, receiver_id, subject, content)
                VALUES (:sid, :rid, :subj, :body)
            """),
            {"sid": sender_id, "rid": receiver_id,
             "subj": subject or "", "body": content},
        )


def get_inbox(user_id: int) -> list[dict]:
    """
    All messages received by this user.
    """
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                  m.message_id,
                  m.subject,
                  m.content,
                  m.readStatus,
                  m.created_at,
                  u.name  AS sender_name,
                  u.email AS sender_email
                FROM messages m
                JOIN users u ON u.user_id = m.sender_id
                WHERE m.receiver_id = :uid
                ORDER BY m.created_at DESC
            """),
            {"uid": user_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_sent_messages(user_id: int) -> list[dict]:
    """
    All messages sent by this user.
    """
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                  m.message_id,
                  m.subject,
                  m.content,
                  m.readStatus,
                  m.created_at,
                  u.name  AS receiver_name,
                  u.email AS receiver_email
                FROM messages m
                JOIN users u ON u.user_id = m.receiver_id
                WHERE m.sender_id = :uid
                ORDER BY m.created_at DESC
            """),
            {"uid": user_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def mark_message_read(message_id: int, user_id: int) -> None:
    """
    Mark a message as READ (only by the receiver).
    """
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                UPDATE messages
                   SET readStatus = 'READ'
                 WHERE message_id = :mid
                   AND receiver_id = :uid
            """),
            {"mid": message_id, "uid": user_id},
        )


def get_teacher_students(teacher_user_id: int) -> list[dict]:
    """
    Students assigned to a given teacher (identified by the teacher's user_id).
    Returns: student_user_id, student_name, department.
    """
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                  s.student_id,
                  u.user_id AS student_user_id,
                  u.name    AS student_name,
                  COALESCE(d.name, '') AS department
                FROM teachers t
                JOIN students s ON s.assigned_teacher_id = t.teacher_id
                JOIN users u    ON u.user_id = s.user_id
                LEFT JOIN departments d ON d.department_id = s.department_id
                WHERE t.user_id = :uid
                ORDER BY u.name
            """),
            {"uid": teacher_user_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_student_advisor(student_user_id: int) -> dict | None:
    """
    Advisor (teacher) for a given student (by student's user_id).
    Returns: {teacher_user_id, teacher_name, teacher_email} or None.
    """
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("""
                SELECT
                  t.teacher_id,
                  u.user_id AS teacher_user_id,
                  u.name    AS teacher_name,
                  u.email   AS teacher_email
                FROM students s
                JOIN teachers t ON t.teacher_id = s.assigned_teacher_id
                JOIN users u    ON u.user_id = t.user_id
                WHERE s.user_id = :uid
            """),
            {"uid": student_user_id},
        ).mappings().first()
    return dict(row) if row else None
