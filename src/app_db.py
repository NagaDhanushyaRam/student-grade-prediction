# app_db.py
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from passlib.hash import bcrypt_sha256
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

# ==========================
# Paths & DB configuration
# ==========================
ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# One canonical SQLite file. You can override via env if needed.
DB_FILE = os.path.abspath(os.getenv("EDUTRACK_DB", str(DB_DIR / "app.db")))
DB_URL = f"sqlite:///{DB_FILE}"

SCHEMA_PATH = os.path.abspath(str(DB_DIR / "schema.sql"))

# Single engine instance
_ENGINE: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a singleton SQLAlchemy engine with SQLite FK pragma enabled."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DB_URL, future=True)

        # Ensure FK constraints for SQLite
        @event.listens_for(_ENGINE, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):
            try:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
            except Exception:
                # Non-fatal; best-effort
                pass

    return _ENGINE


# ==========================
# Schema bootstrap
# ==========================
def ensure_schema() -> None:
    """
    Apply db/schema.sql exactly once (unless EDUTRACK_RESET=1).
    Prevents wiping data on every Streamlit rerun.
    """
    eng = get_engine()

    # If not resetting, skip when 'users' table exists
    do_reset = os.getenv("EDUTRACK_RESET", "0") == "1"
    with eng.begin() as conn:
        users_exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'")
        ).scalar() is not None

    if users_exists and not do_reset:
        return  # schema already applied

    # Apply schema.sql once (or on explicit reset)
    raw = eng.raw_connection()  # sqlite3 connection
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
        raw.executescript(sql)
        raw.commit()
    finally:
        raw.close()


# ==========================
# Auth helpers
# ==========================
def hash_pw(plain: str) -> str:
    return bcrypt_sha256.hash(plain)


def verify_pw(plain: str, pw_hash: str) -> bool:
    return bcrypt_sha256.verify(plain, pw_hash)


# ==========================
# Departments (normalized)
# ==========================
def get_or_create_department_id(conn, name: Optional[str]) -> Optional[int]:
    """Return department_id for a given name; create if missing. None if name is blank."""
    if not name or not str(name).strip():
        return None
    name = str(name).strip()
    row = conn.execute(
        text("SELECT department_id FROM departments WHERE name=:n"),
        {"n": name},
    ).fetchone()
    if row:
        return int(row[0])
    res = conn.execute(
        text("INSERT INTO departments(name) VALUES(:n)"),
        {"n": name},
    )
    return int(res.lastrowid)


# ==========================
# Users & Profiles
# ==========================
def _create_user(conn, *, name: str, email: str, password_plain: str, role: str) -> int:
    res = conn.execute(
        text("""
            INSERT INTO users (name, email, password_hash, role)
            VALUES (:n, :e, :p, :r)
        """),
        {"n": name, "e": email, "p": hash_pw(password_plain), "r": role.upper()},
    )
    return int(res.lastrowid)


def create_user(name: str, email: str, password_plain: str, role: str) -> int:
    with get_engine().begin() as conn:
        return _create_user(
            conn, name=name, email=email, password_plain=password_plain, role=role
        )


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email=:e"),
            {"e": email},
        ).mappings().first()
        return dict(row) if row else None


def create_teacher(user_id: int, department_name: Optional[str] = None) -> int:
    with get_engine().begin() as conn:
        dep_id = get_or_create_department_id(conn, department_name)
        res = conn.execute(
            text("""
                INSERT INTO teachers (user_id, department_id)
                VALUES (:u, :d)
            """),
            {"u": user_id, "d": dep_id},
        )
        return int(res.lastrowid)


def create_student(
    user_id: int,
    gpa: float = None,
    academic_status: str = "ACTIVE",
    department_name: Optional[str] = None,
    assigned_teacher_id: Optional[int] = None,
    external_student_id: Optional[str] = None,
    academic_year: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> int:
    with get_engine().begin() as conn:
        dep_id = get_or_create_department_id(conn, department_name)
        res = conn.execute(
            text("""
                INSERT INTO students (
                    user_id, external_student_id, academic_year, age, gender,
                    department_id, assigned_teacher_id, gpa, academicStatus
                )
                VALUES (:u, :ext, :ay, :age, :gender, :dep, :tid, :gpa, :status)
            """),
            {
                "u": user_id,
                "ext": external_student_id,
                "ay": academic_year,
                "age": age,
                "gender": gender,
                "dep": dep_id,
                "tid": assigned_teacher_id,
                "gpa": gpa,
                "status": academic_status,
            },
        )
        return int(res.lastrowid)


def create_student_full(
    name: str,
    email: str,
    password_plain: str,
    academic_year: Optional[str],
    age: Optional[int],
    gender: Optional[str],
    external_student_id: Optional[str],
    department: Optional[str],
    gpa: Optional[float],
    assigned_teacher_id: Optional[int],
) -> int:
    with get_engine().begin() as conn:
        uid = _create_user(
            conn, name=name, email=email, password_plain=password_plain, role="STUDENT"
        )
        dep_id = get_or_create_department_id(conn, department)
        conn.execute(
            text("""
                INSERT INTO students (
                    user_id, external_student_id, academic_year, age, gender,
                    department_id, assigned_teacher_id, gpa, academicStatus
                )
                VALUES (:uid, :extid, :ay, :age, :gender, :dep, :tid, :gpa, 'ACTIVE')
            """),
            {
                "uid": uid,
                "extid": external_student_id,
                "ay": academic_year,
                "age": age,
                "gender": gender,
                "dep": dep_id,
                "tid": assigned_teacher_id,
                "gpa": gpa,
            },
        )
        return uid


def create_teacher_full(
    name: str, email: str, password_plain: str, department: Optional[str]
) -> int:
    with get_engine().begin() as conn:
        uid = _create_user(
            conn, name=name, email=email, password_plain=password_plain, role="TEACHER"
        )
        dep_id = get_or_create_department_id(conn, department)
        conn.execute(
            text("INSERT INTO teachers (user_id, department_id) VALUES (:uid, :dep)"),
            {"uid": uid, "dep": dep_id},
        )
        return uid


# ==========================
# Permissions & Ownership
# ==========================
def can_teacher_edit_student(conn, teacher_user_id: int, target_student_id: int) -> bool:
    """True if the teacher (by users.user_id) is assigned to the target student."""
    row = conn.execute(
        text("""
            SELECT t.teacher_id
              FROM teachers t
              JOIN users u ON u.user_id = t.user_id
             WHERE u.user_id = :uid
        """),
        {"uid": teacher_user_id},
    ).fetchone()
    if not row:
        return False
    teacher_id = int(row[0])
    assigned = conn.execute(
        text("""
            SELECT 1
              FROM students
             WHERE student_id = :sid AND assigned_teacher_id = :tid
        """),
        {"sid": target_student_id, "tid": teacher_id},
    ).fetchone()
    return assigned is not None


def can_run_prediction(actor_role: str, actor_user_id: int, student_id: int, conn) -> bool:
    """
    Admins can predict for anyone;
    Teachers only for their assigned students;
    Students only for themselves.
    """
    role = (actor_role or "").upper()
    if role in ("ADMIN", "TEACHER"):
        if role == "TEACHER":
            return can_teacher_edit_student(conn, actor_user_id, student_id)
        return True
    if role == "STUDENT":
        owner = conn.execute(
            text("""
                SELECT 1
                  FROM students s
                  JOIN users u ON u.user_id = s.user_id
                 WHERE s.student_id = :sid AND u.user_id = :uid
            """),
            {"sid": student_id, "uid": actor_user_id},
        ).fetchone()
        return owner is not None
    return False


# ==========================
# Academic Data
# ==========================
def upsert_academic_record(
    actor_role: str,
    actor_user_id: int,
    student_id: int,
    *,
    term: str,
    attendance: float = None,
    study_hours: float = None,
    examScores: float = None,
    stress_level: int = None,
    sleep_hours: float = None,
    participation: float = None,
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Upsert student_academic_data row with permission checks.
    Returns data_id.
    """
    role = (actor_role or "").upper()
    extra_json = json.dumps(extra or {}, ensure_ascii=False)

    eng = get_engine()
    with eng.begin() as conn:
        if role == "TEACHER":
            if not can_teacher_edit_student(conn, actor_user_id, student_id):
                raise PermissionError("You can only edit performance for your assigned students.")
        elif role == "ADMIN":
            raise PermissionError("Admins cannot record or edit student performance.")
        elif role == "STUDENT":
            raise PermissionError("Students cannot edit performance data.")

        # Try update first
        upd = conn.execute(
            text("""
                UPDATE student_academic_data
                   SET attendance=:a, study_hours=:sh, examScores=:es, stress_level=:sl,
                       sleep_hours=:slp, participation=:p, extra_json=:x
                 WHERE student_id=:sid AND term=:t
            """),
            {
                "a": attendance,
                "sh": study_hours,
                "es": examScores,
                "sl": stress_level,
                "slp": sleep_hours,
                "p": participation,
                "x": extra_json,
                "sid": student_id,
                "t": term,
            },
        )

        if upd.rowcount == 0:
            res = conn.execute(
                text("""
                    INSERT INTO student_academic_data (
                        student_id, term, attendance, study_hours, examScores, stress_level,
                        sleep_hours, participation, extra_json, created_by
                    )
                    VALUES (:sid, :t, :a, :sh, :es, :sl, :slp, :p, :x, :cb)
                """),
                {
                    "sid": student_id,
                    "t": term,
                    "a": attendance,
                    "sh": study_hours,
                    "es": examScores,
                    "sl": stress_level,
                    "slp": sleep_hours,
                    "p": participation,
                    "x": extra_json,
                    "cb": actor_user_id,
                },
            )
            return int(res.lastrowid)

        rid = conn.execute(
            text("""
                SELECT data_id
                  FROM student_academic_data
                 WHERE student_id=:sid AND term=:t
            """),
            {"sid": student_id, "t": term},
        ).scalar_one()
        return int(rid)


def get_latest_record(student_id: int) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(
            text("""
                SELECT *
                  FROM student_academic_data
                 WHERE student_id=:sid
              ORDER BY data_id DESC
                 LIMIT 1
            """),
            {"sid": student_id},
        ).mappings().first()
        if not row:
            return None
        d = dict(row)
        try:
            d["extra"] = json.loads(d.get("extra_json") or "{}")
        except Exception:
            d["extra"] = {}
        return d


# ==========================
# Models & Predictions
# ==========================
def register_model(
    model_name: str,
    version: str,
    path: str,
    columns_path: str,
    accuracy: float = None,
    f1: float = None,
    roc_auc: float = None,
) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(
            text("""
                INSERT INTO ml_models (model_name, version, path, columns_path, accuracy, f1_score, roc_auc)
                VALUES (:n, :v, :p, :c, :a, :f, :r)
            """),
            {
                "n": model_name,
                "v": version,
                "p": path,
                "c": columns_path,
                "a": accuracy,
                "f": f1,
                "r": roc_auc,
            },
        )
        return int(res.lastrowid)


def latest_model(model_name: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(
            text("""
                SELECT *
                  FROM ml_models
                 WHERE model_name=:n
              ORDER BY model_id DESC
                 LIMIT 1
            """),
            {"n": model_name},
        ).mappings().first()
        return dict(row) if row else None


def save_prediction(
    student_id: int,
    model_id: int,
    term: str,
    label: str,
    pass_prob: float,
    fail_prob: float,
) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(
            text("""
                INSERT INTO prediction_results (
                    student_id, model_id, term, predictedStatus, passPercentage, failPercentage
                )
                VALUES (:sid, :mid, :t, :lbl, :pp, :fp)
                ON CONFLICT(student_id, model_id, term) DO UPDATE SET
                    predictedStatus = excluded.predictedStatus,
                    passPercentage  = excluded.passPercentage,
                    failPercentage  = excluded.failPercentage
            """),
            {
                "sid": student_id,
                "mid": model_id,
                "t": term,
                "lbl": label,
                "pp": pass_prob,
                "fp": fail_prob,
            },
        )
        if res.lastrowid:
            return int(res.lastrowid)

        pid = conn.execute(
            text("""
                SELECT prediction_id
                  FROM prediction_results
                 WHERE student_id=:sid AND model_id=:mid AND term=:t
            """),
            {"sid": student_id, "mid": model_id, "t": term},
        ).scalar_one()
        return int(pid)


def add_recommendation(student_id: int, prediction_id: Optional[int], typ: str, message: str) -> int:
    """
    Insert a recommendation. If the given prediction_id violates a FK (e.g., predicate row
    not visible or schema mismatch), gracefully retry with NULL prediction_id so the rec
    is not lost. Still always ties to the student_id.
    """
    with get_engine().begin() as conn:
        try:
            res = conn.execute(
                text("""
                    INSERT INTO recommendations(student_id, prediction_id, recommendationType, message)
                    VALUES (:sid, :pid, :ty, :msg)
                """),
                {"sid": student_id, "pid": prediction_id, "ty": typ, "msg": message},
            )
            return int(res.lastrowid)
        except IntegrityError:
            # retry without prediction_id
            res = conn.execute(
                text("""
                    INSERT INTO recommendations(student_id, prediction_id, recommendationType, message)
                    VALUES (:sid, NULL, :ty, :msg)
                """),
                {"sid": student_id, "ty": typ, "msg": message},
            )
            return int(res.lastrowid)


# ==========================
# Convenience queries
# ==========================
def any_admin_exists() -> bool:
    with get_engine().begin() as conn:
        r = conn.execute(
            text("SELECT 1 FROM users WHERE role='ADMIN' LIMIT 1")
        ).fetchone()
        return r is not None


def teacher_choices(conn) -> Dict[str, int]:
    rows = conn.execute(
        text("""
            SELECT t.teacher_id, u.name
              FROM teachers t
              JOIN users u ON u.user_id = t.user_id
          ORDER BY u.name
        """)
    ).fetchall()
    return {f"{r[1]} (#{r[0]})": int(r[0]) for r in rows}


# ==========================
# Debug helper
# ==========================
if __name__ == "__main__":
    ensure_schema()
    with get_engine().begin() as c:
        tables = c.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).scalars().all()
    print("DB_FILE:", DB_FILE)
    print("Tables:", tables)
