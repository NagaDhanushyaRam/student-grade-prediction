# app_db_mongo.py
import json
import hashlib
import hmac
from datetime import datetime
from typing import Optional, Dict, Any, List

from mongo_client import get_db

# same pepper as before
_PWD_PEPPER = "edutrack_pepper_2025"


# ----------------------------
# Helpers
# ----------------------------
def hash_pw(raw: str) -> str:
    raw = (raw or "").encode("utf-8")
    return hashlib.sha256(_PWD_PEPPER.encode("utf-8") + raw).hexdigest()


def verify_pw(raw: str, digest: str) -> bool:
    calc = hash_pw(raw)
    return hmac.compare_digest(calc, digest or "")


def _next_id(coll_name: str, id_field: str) -> int:
    db = get_db()
    last = db[coll_name].find_one(sort=[(id_field, -1)])
    return int(last[id_field]) + 1 if last and id_field in last else 1


# ----------------------------
# Users
# ----------------------------
def any_admin_exists() -> bool:
    db = get_db()
    return db["users"].count_documents({"role": "ADMIN"}, limit=1) > 0


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    doc = db["users"].find_one(
        {"email": {"$regex": f"^{email}$", "$options": "i"}},
        {"_id": 0}
    )
    return doc


def create_user(name: str, email: str, password: str, role: str) -> int:
    db = get_db()
    user_id = _next_id("users", "user_id")
    db["users"].insert_one(
        {
            "user_id": user_id,
            "name": name,
            "email": email,
            "password_hash": hash_pw(password),
            "role": role,
            "is_active": 1,
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    return user_id


# ----------------------------
# Departments / Teachers / Students
# ----------------------------
def _ensure_department(dept_name: Optional[str]) -> Optional[int]:
    if not dept_name:
        return None
    dept_name = dept_name.strip()
    if not dept_name:
        return None

    db = get_db()
    coll = db["departments"]
    existing = coll.find_one({"name": dept_name})
    if existing:
        return int(existing["department_id"])

    dep_id = _next_id("departments", "department_id")
    coll.insert_one({"department_id": dep_id, "name": dept_name})
    return dep_id


def create_teacher_full(
    name: str,
    email: str,
    password: str,
    dept_name: Optional[str]
) -> int:
    db = get_db()
    user_id = create_user(name, email, password, "TEACHER")
    dep_id = _ensure_department(dept_name)

    teacher_id = _next_id("teachers", "teacher_id")
    db["teachers"].insert_one(
        {
            "teacher_id": teacher_id,
            "user_id": user_id,
            "department_id": dep_id,
        }
    )
    return teacher_id


def create_student_full(
    name: str,
    email: str,
    password: str,
    academic_year: Optional[str],
    age: Optional[int],
    gender: Optional[str],
    external_student_id: Optional[str],
    dept_name: Optional[str],
    gpa: Optional[float],
    assigned_teacher_id: Optional[int],
) -> int:
    db = get_db()
    user_id = create_user(name, email, password, "STUDENT")
    dep_id = _ensure_department(dept_name)

    student_id = _next_id("students", "student_id")
    db["students"].insert_one(
        {
            "student_id": student_id,
            "user_id": user_id,
            "academic_year": academic_year,
            "age": age,
            "gender": gender,
            "external_student_id": external_student_id,
            "department_id": dep_id,
            "assigned_teacher_id": assigned_teacher_id,
            "gpa": gpa,
        }
    )
    return student_id


# ----------------------------
# Academic records
# ----------------------------
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
    We keep the same fields as SQLite (including extra_json).
    """
    db = get_db()
    coll = db["student_academic_data"]

    sid = int(student_id)
    payload = json.dumps(extra or {}, ensure_ascii=False)

    existing = coll.find_one({"student_id": sid, "term": term})
    if existing:
        data_id = int(existing["data_id"])
    else:
        data_id = _next_id("student_academic_data", "data_id")

    coll.update_one(
        {"student_id": sid, "term": term},
        {
            "$set": {
                "data_id": data_id,
                "attendance": attendance,
                "study_hours": study_hours,
                "examScores": examScores,
                "stress_level": stress_level,
                "sleep_hours": sleep_hours,
                "participation": participation,
                "extra_json": payload,
            },
            "$setOnInsert": {
                "created_at": datetime.utcnow().isoformat(),
            },
        },
        upsert=True,
    )

    # audit_log is optional; skipping for brevity
    return data_id


def get_latest_record(student_id: int) -> Optional[Dict[str, Any]]:
    db = get_db()
    coll = db["student_academic_data"]
    doc = coll.find_one(
        {"student_id": int(student_id)},
        sort=[("data_id", -1)]
    )
    if not doc:
        return None

    extra = {}
    try:
        extra = json.loads(doc.get("extra_json") or "{}")
    except Exception:
        extra = {}

    return {
        "data_id": doc.get("data_id"),
        "student_id": doc.get("student_id"),
        "term": doc.get("term"),
        "attendance": doc.get("attendance"),
        "study_hours": doc.get("study_hours"),
        "examScores": doc.get("examScores"),
        "stress_level": doc.get("stress_level"),
        "sleep_hours": doc.get("sleep_hours"),
        "participation": doc.get("participation"),
        "extra": extra,
        "created_at": doc.get("created_at"),
    }


# ----------------------------
# Model registry
# ----------------------------
def latest_model(name: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    coll = db["ml_models"]
    doc = coll.find_one(
        {"model_name": name},
        sort=[("model_id", -1)]
    )
    if not doc:
        return None
    # keep keys same as old SQL version
    return {
        "model_id": doc.get("model_id"),
        "model_name": doc.get("model_name"),
        "version": doc.get("version"),
        "path": doc.get("path"),
        "columns_path": doc.get("columns_path"),
        "created_at": doc.get("created_at"),
    }


def register_model(
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: str = "",
    columns_path: str = "",
    version: Optional[str] = None,
    **metrics,
) -> int:
    db = get_db()
    coll = db["ml_models"]

    final_name = (name or model_name or "").strip()
    if not final_name:
        raise ValueError("register_model: `name` or `model_name` is required")
    if not model_path or not columns_path:
        raise ValueError("register_model: `model_path` and `columns_path` are required")

    ver = (version or datetime.utcnow().strftime("%Y%m%d%H%M%S")).strip()

    existing = coll.find_one({"model_name": final_name, "version": ver})
    if existing:
        model_id = int(existing["model_id"])
    else:
        model_id = _next_id("ml_models", "model_id")

    coll.update_one(
        {"model_name": final_name, "version": ver},
        {
            "$set": {
                "model_id": model_id,
                "model_name": final_name,
                "version": ver,
                "path": model_path,
                "columns_path": columns_path,
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
        },
        upsert=True,
    )

    return model_id


# ----------------------------
# Permissions helpers
# ----------------------------
def can_run_prediction(role: str, user_id: int, student_id: int) -> bool:
    db = get_db()
    role = (role or "").upper()
    sid = int(student_id)
    uid = int(user_id)

    if role == "ADMIN":
        return True

    if role == "STUDENT":
        student = db["students"].find_one({"student_id": sid, "user_id": uid})
        return student is not None

    if role == "TEACHER":
        student = db["students"].find_one({"student_id": sid})
        if not student:
            return False
        teacher = db["teachers"].find_one({"teacher_id": student.get("assigned_teacher_id")})
        return bool(teacher and int(teacher.get("user_id")) == uid)

    return False


def teacher_choices() -> Dict[str, int]:
    """
    Build label -> teacher_id for the teacher picker.
    """
    db = get_db()
    teachers = list(db["teachers"].find({}))
    mapping: Dict[str, int] = {}

    for t in teachers:
        teacher_id = int(t["teacher_id"])
        u = db["users"].find_one({"user_id": int(t["user_id"])})
        d = None
        if t.get("department_id") is not None:
            d = db["departments"].find_one({"department_id": int(t["department_id"])})
        tname = u["name"] if u else f"Teacher {teacher_id}"
        dname = d["name"] if d else ""
        label = f"{tname} ({dname})" if dname else tname
        mapping[label] = teacher_id

    return mapping


# ----------------------------
# Performance report helpers
# ----------------------------
def _avg(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def get_site_performance_summary() -> dict:
    db = get_db()
    students = list(db["students"].find({}))
    teachers = list(db["teachers"].find({}))
    acad = list(db["student_academic_data"].find({}))
    preds = list(db["prediction_results"].find({}))

    total_students = len(students)
    total_teachers = len(teachers)
    avg_gpa = _avg([s.get("gpa") for s in students])
    avg_att = _avg([a.get("attendance") for a in acad])

    pass_count = sum(1 for p in preds if p.get("predictedStatus") == "PASS")
    fail_count = sum(1 for p in preds if p.get("predictedStatus") == "FAIL")

    return {
        "total_students": total_students,
        "total_teachers": total_teachers,
        "avg_gpa": avg_gpa,
        "avg_attendance": avg_att,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }


def get_department_performance() -> List[dict]:
    db = get_db()
    departments = {d["department_id"]: d["name"] for d in db["departments"].find({})}
    students = list(db["students"].find({}))
    acad_by_student = {}
    for a in db["student_academic_data"].find({}):
        sid = int(a["student_id"])
        acad_by_student.setdefault(sid, []).append(a)

    preds_by_student = {}
    for p in db["prediction_results"].find({}):
        sid = int(p["student_id"])
        preds_by_student.setdefault(sid, []).append(p)

    result: List[dict] = []

    for dep_id, dep_name in departments.items():
        stu = [s for s in students if s.get("department_id") == dep_id]
        stu_ids = [int(s["student_id"]) for s in stu]
        gpas = [s.get("gpa") for s in stu]

        att_vals: List[Optional[float]] = []
        pass_cnt = 0
        fail_cnt = 0

        for sid in stu_ids:
            for a in acad_by_student.get(sid, []):
                att_vals.append(a.get("attendance"))
            for p in preds_by_student.get(sid, []):
                if p.get("predictedStatus") == "PASS":
                    pass_cnt += 1
                elif p.get("predictedStatus") == "FAIL":
                    fail_cnt += 1

        result.append(
            {
                "department": dep_name,
                "total_students": len(stu),
                "avg_gpa": _avg(gpas),
                "avg_attendance": _avg(att_vals),
                "pass_count": pass_cnt,
                "fail_count": fail_cnt,
            }
        )

    # sort by name
    result.sort(key=lambda r: r["department"])
    return result


# ----------------------------
# Messaging helpers
# ----------------------------
def send_message(sender_id: int, receiver_id: int,
                 subject: str | None, content: str) -> None:
    db = get_db()
    coll = db["messages"]
    mid = _next_id("messages", "message_id")
    coll.insert_one(
        {
            "message_id": mid,
            "sender_id": int(sender_id),
            "receiver_id": int(receiver_id),
            "subject": subject or "",
            "content": content,
            "readStatus": "UNREAD",
            "created_at": datetime.utcnow().isoformat(),
        }
    )


def get_inbox(user_id: int) -> List[dict]:
    db = get_db()
    msgs = list(
        db["messages"].find({"receiver_id": int(user_id)}).sort("created_at", -1)
    )
    out: List[dict] = []
    for m in msgs:
        sender = db["users"].find_one({"user_id": int(m["sender_id"])})
        out.append(
            {
                "message_id": m["message_id"],
                "subject": m.get("subject", ""),
                "content": m.get("content", ""),
                "readStatus": m.get("readStatus", "UNREAD"),
                "created_at": m.get("created_at"),
                "sender_name": sender["name"] if sender else "Unknown",
                "sender_email": sender["email"] if sender else "",
            }
        )
    return out


def get_sent_messages(user_id: int) -> List[dict]:
    db = get_db()
    msgs = list(
        db["messages"].find({"sender_id": int(user_id)}).sort("created_at", -1)
    )
    out: List[dict] = []
    for m in msgs:
        receiver = db["users"].find_one({"user_id": int(m["receiver_id"])})
        out.append(
            {
                "message_id": m["message_id"],
                "subject": m.get("subject", ""),
                "content": m.get("content", ""),
                "readStatus": m.get("readStatus", "UNREAD"),
                "created_at": m.get("created_at"),
                "receiver_name": receiver["name"] if receiver else "Unknown",
                "receiver_email": receiver["email"] if receiver else "",
            }
        )
    return out


def mark_message_read(message_id: int, user_id: int) -> None:
    db = get_db()
    db["messages"].update_one(
        {"message_id": int(message_id), "receiver_id": int(user_id)},
        {"$set": {"readStatus": "READ"}},
    )


def get_teacher_students(teacher_user_id: int) -> List[dict]:
    db = get_db()
    teacher = db["teachers"].find_one({"user_id": int(teacher_user_id)})
    if not teacher:
        return []
    tid = int(teacher["teacher_id"])

    students = list(db["students"].find({"assigned_teacher_id": tid}))
    result: List[dict] = []
    for s in students:
        stu_user = db["users"].find_one({"user_id": int(s["user_id"])})
        dept = None
        if s.get("department_id") is not None:
            dept = db["departments"].find_one({"department_id": int(s["department_id"])})
        result.append(
            {
                "student_id": int(s["student_id"]),
                "student_user_id": int(s["user_id"]),
                "student_name": stu_user["name"] if stu_user else "Unknown",
                "department": dept["name"] if dept else "",
            }
        )
    # sort by name
    result.sort(key=lambda r: r["student_name"])
    return result


def get_student_advisor(student_user_id: int) -> Optional[dict]:
    db = get_db()
    student = db["students"].find_one({"user_id": int(student_user_id)})
    if not student or not student.get("assigned_teacher_id"):
        return None

    teacher = db["teachers"].find_one(
        {"teacher_id": int(student["assigned_teacher_id"])}
    )
    if not teacher:
        return None

    user = db["users"].find_one({"user_id": int(teacher["user_id"])})
    if not user:
        return None

    return {
        "teacher_user_id": int(user["user_id"]),
        "teacher_name": user["name"],
        "teacher_email": user["email"],
    }
