import os, json
from typing import Optional, Dict, Any
from pathlib import Path

from passlib.hash import bcrypt_sha256
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# --- Paths: use the actual app.db inside the db/ folder
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("EDUTRACK_DB", str(ROOT / "db" / "app.db"))
SCHEMA_PATH = str(ROOT / "db" / "schema.sql")

def get_engine() -> Engine:
    # sqlite URL must be absolute for reliability
    return create_engine(f"sqlite:///{os.path.abspath(DB_PATH)}", future=True)

def ensure_schema() -> None:
    """
    Apply schema.sql using the raw sqlite3 connection so we can call executescript().
    """
    engine = get_engine()
    with engine.begin() as _:
        pass  # just ensures file/dir exists

    raw = engine.raw_connection()  # sqlite3 connection
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            raw.executescript(f.read())
        raw.commit()
    finally:
        raw.close()

def hash_pw(plain: str) -> str:
    return bcrypt_sha256.hash(plain)

def verify_pw(pw: str, pw_hash: str) -> bool:
    return bcrypt_sha256.verify(pw, pw_hash)

def create_user(name: str, email: str, password: str, role: str) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(text("""
            INSERT INTO users(name, email, password_hash, role)
            VALUES (:n, :e, :p, :r)
        """), {"n": name, "e": email, "p": hash_pw(password), "r": role.upper()})
        return res.lastrowid

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(text("SELECT * FROM users WHERE email=:e"),
                           {"e": email}).mappings().first()
        return dict(row) if row else None

def create_student(user_id: int, gpa: float=None, department: str=None, status: str="ACTIVE") -> int:
    with get_engine().begin() as conn:
        res = conn.execute(text("""
            INSERT INTO students(user_id, gpa, department, status)
            VALUES(:u, :g, :d, :s)
        """), {"u": user_id, "g": gpa, "d": department, "s": status})
        return res.lastrowid

def create_teacher(user_id: int, department: str=None) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(text(
            "INSERT INTO teachers(user_id, department) VALUES(:u, :d)"
        ), {"u": user_id, "d": department})
        return res.lastrowid

def upsert_academic_record(student_id: int, term: str,
                           attendance: float=None, study_hours: float=None,
                           exam_score: float=None, stress_level: int=None,
                           sleep_hours: float=None, participation: float=None,
                           extra: Optional[Dict[str, Any]]=None,
                           created_by: Optional[int]=None) -> int:
    extra_json = json.dumps(extra or {}, ensure_ascii=False)
    with get_engine().begin() as conn:
        upd = conn.execute(text("""
            UPDATE student_academic_data
               SET attendance=:a, study_hours=:sh, exam_score=:es, stress_level=:sl,
                   sleep_hours=:slp, participation=:p, extra_json=:x
             WHERE student_id=:sid AND term=:t
        """), {"a": attendance, "sh": study_hours, "es": exam_score, "sl": stress_level,
               "slp": sleep_hours, "p": participation, "x": extra_json,
               "sid": student_id, "t": term})
        if upd.rowcount == 0:
            res = conn.execute(text("""
                INSERT INTO student_academic_data
                  (student_id, term, attendance, study_hours, exam_score, stress_level,
                   sleep_hours, participation, extra_json, created_by)
                VALUES (:sid, :t, :a, :sh, :es, :sl, :slp, :p, :x, :cb)
            """), {"sid": student_id, "t": term, "a": attendance, "sh": study_hours,
                   "es": exam_score, "sl": stress_level, "slp": sleep_hours,
                   "p": participation, "x": extra_json, "cb": created_by})
            return res.lastrowid
        rid = conn.execute(text("""
            SELECT data_id FROM student_academic_data
            WHERE student_id=:sid AND term=:t
        """), {"sid": student_id, "t": term}).scalar_one()
        return rid

def get_latest_record(student_id: int) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(text("""
            SELECT * FROM student_academic_data
            WHERE student_id=:sid ORDER BY data_id DESC LIMIT 1
        """), {"sid": student_id}).mappings().first()
        if not row:
            return None
        d = dict(row)
        try:
            d["extra"] = json.loads(d.get("extra_json") or "{}")
        except Exception:
            d["extra"] = {}
        return d

def register_model(name: str, version: str, path: str, columns_path: str,
                   accuracy: float=None, f1: float=None, roc_auc: float=None) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(text("""
            INSERT INTO ml_models(name, version, path, columns_path, accuracy, f1_score, roc_auc)
            VALUES (:n,:v,:p,:c,:a,:f,:r)
        """), {"n": name, "v": version, "p": path, "c": columns_path,
               "a": accuracy, "f": f1, "r": roc_auc})
        return res.lastrowid

def latest_model(name: str) -> Optional[Dict[str, Any]]:
    with get_engine().begin() as conn:
        row = conn.execute(text("""
            SELECT * FROM ml_models WHERE name=:n
            ORDER BY model_id DESC LIMIT 1
        """), {"n": name}).mappings().first()
        return dict(row) if row else None

def save_prediction(student_id: int, model_id: int, term: str,
                    label: str, pass_prob: float, fail_prob: float) -> int:
    with get_engine().begin() as conn:
        res = conn.execute(text("""
            INSERT INTO prediction_results(student_id, model_id, term, predicted_label, pass_prob, fail_prob)
            VALUES (:sid,:mid,:t,:lbl,:pp,:fp)
            ON CONFLICT(student_id, model_id, term) DO UPDATE SET
               predicted_label=excluded.predicted_label,
               pass_prob=excluded.pass_prob,
               fail_prob=excluded.fail_prob
        """), {"sid": student_id, "mid": model_id, "t": term,
               "lbl": label, "pp": pass_prob, "fp": fail_prob})
        if res.lastrowid:
            return res.lastrowid
        pid = conn.execute(text("""
            SELECT prediction_id FROM prediction_results
            WHERE student_id=:sid AND model_id=:mid AND term=:t
        """), {"sid": student_id, "mid": model_id, "t": term}).scalar_one()
        return pid

def add_recommendation(student_id: int, prediction_id: int, typ: str, message: str) -> int:
    """
    Insert a recommendation record into the recommendations table.
    """
    with get_engine().begin() as conn:
        res = conn.execute(text("""
            INSERT INTO recommendations(student_id, prediction_id, type, message)
            VALUES (:sid, :pid, :ty, :msg)
        """), {"sid": student_id, "pid": prediction_id, "ty": typ, "msg": message})
        return res.lastrowid

if __name__ == "__main__":
    ensure_schema()
    with get_engine().begin() as c:
        tables = c.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )).scalars().all()
    print("Tables:", tables)
