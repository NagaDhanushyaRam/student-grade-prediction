-- =========================================
-- EduTrack Schema (SQLite) — Fresh Install (drop+create)
-- =========================================
PRAGMA foreign_keys = ON;
BEGIN TRANSACTION;



-- =========
-- Reference: Departments
-- =========
CREATE TABLE IF NOT EXISTS departments (
  department_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name            TEXT NOT NULL UNIQUE
);

-- =========
-- Core Users & Roles
-- =========
CREATE TABLE IF NOT EXISTS users (
  user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL,
  email         TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('ADMIN','TEACHER','STUDENT')),
  is_active     INTEGER NOT NULL DEFAULT 1,   -- soft delete/disable
  created_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_users_role   ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- =========
-- Admins (1:1 with users where role='ADMIN')
-- =========
CREATE TABLE IF NOT EXISTS administrators (
  admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id  INTEGER NOT NULL UNIQUE,
  name     TEXT,
  email    TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_admins_user ON administrators(user_id);

-- =========
-- Teachers
-- =========
CREATE TABLE IF NOT EXISTS teachers (
  teacher_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id        INTEGER NOT NULL UNIQUE,
  department_id  INTEGER,
  FOREIGN KEY (user_id)       REFERENCES users(user_id)             ON DELETE CASCADE,
  FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_teachers_user ON teachers(user_id);
CREATE INDEX IF NOT EXISTS idx_teachers_dept ON teachers(department_id);

-- =========
-- Students
-- =========
CREATE TABLE IF NOT EXISTS students (
  student_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id             INTEGER NOT NULL UNIQUE,
  external_student_id TEXT UNIQUE,                        -- visible Student ID
  academic_year       TEXT,                               -- e.g., "2025-26"
  age                 INTEGER CHECK (age BETWEEN 10 AND 120),
  gender              TEXT CHECK (gender IN ('Male','Female','Non-binary','Prefer not to say')),
  department_id       INTEGER,
  assigned_teacher_id INTEGER,                            -- who may edit performance
  gpa                 REAL,
  academicStatus      TEXT,
  FOREIGN KEY (user_id)             REFERENCES users(user_id)             ON DELETE CASCADE,
  FOREIGN KEY (department_id)       REFERENCES departments(department_id) ON DELETE SET NULL,
  FOREIGN KEY (assigned_teacher_id) REFERENCES teachers(teacher_id)       ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_students_user    ON students(user_id);
CREATE INDEX IF NOT EXISTS idx_students_dept    ON students(department_id);
CREATE INDEX IF NOT EXISTS idx_students_teacher ON students(assigned_teacher_id);

-- =========
-- Academic Data (Student → AcademicData = 1:N)
-- =========
CREATE TABLE IF NOT EXISTS student_academic_data (
  data_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id     INTEGER NOT NULL,
  term           TEXT NOT NULL,
  attendance     REAL,
  study_hours    REAL,
  examScores     INTEGER,
  stress_level   INTEGER,
  sleep_hours    REAL,
  participation  REAL,
  extra_json     TEXT,                   -- flexible JSON payload
  created_by     INTEGER,                -- user_id who created the row
  created_at     TEXT DEFAULT (datetime('now')),
  UNIQUE (student_id, term),
  FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES users(user_id)       ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_academic_student_term
  ON student_academic_data(student_id, term);

-- =========
-- ML Models
-- =========
CREATE TABLE IF NOT EXISTS ml_models (
  model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name   TEXT NOT NULL,
  model_type   TEXT,
  version      TEXT NOT NULL,
  path         TEXT NOT NULL,
  columns_path TEXT NOT NULL,
  accuracy     REAL,
  f1_score     REAL,
  roc_auc      REAL,
  created_at   TEXT DEFAULT (datetime('now')),
  UNIQUE (model_name, version)
);

-- =========
-- Predictions
-- =========
CREATE TABLE IF NOT EXISTS prediction_results (
  prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id        INTEGER NOT NULL,
  model_id          INTEGER NOT NULL,
  term              TEXT NOT NULL,
  predictedStatus   TEXT NOT NULL CHECK (predictedStatus IN ('PASS','FAIL')),
  passPercentage    REAL NOT NULL,
  failPercentage    REAL NOT NULL,
  created_at        TEXT DEFAULT (datetime('now')),
  UNIQUE (student_id, model_id, term),
  FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (model_id)   REFERENCES ml_models(model_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pred_student_term
  ON prediction_results(student_id, term);

-- =========
-- Recommendations
-- =========
CREATE TABLE IF NOT EXISTS recommendations (
  recommendation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id       INTEGER,       -- nullable to allow manual notes
  student_id          INTEGER NOT NULL,
  recommendationType  TEXT NOT NULL,
  message             TEXT NOT NULL,
  created_at          TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (prediction_id) REFERENCES prediction_results(prediction_id) ON DELETE CASCADE,
  FOREIGN KEY (student_id)    REFERENCES students(student_id)              ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_recs_student ON recommendations(student_id);

-- =========
-- Messages (Users M:N)
-- =========
CREATE TABLE IF NOT EXISTS messages (
  message_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  sender_id    INTEGER NOT NULL,      -- users.user_id
  receiver_id  INTEGER NOT NULL,      -- users.user_id
  subject      TEXT,
  content      TEXT,
  readStatus   TEXT NOT NULL DEFAULT 'UNREAD' CHECK (readStatus IN ('UNREAD','READ')),
  created_at   TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (sender_id)   REFERENCES users(user_id) ON DELETE CASCADE,
  FOREIGN KEY (receiver_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_sender
  ON messages(sender_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_receiver
  ON messages(receiver_id, readStatus, created_at);

-- =========
-- Dashboards (Users 1:1)
-- =========
CREATE TABLE IF NOT EXISTS dashboards (
  dashboard_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id         INTEGER NOT NULL UNIQUE,
  dashboardType   TEXT,          -- 'ADMIN' | 'TEACHER' | 'STUDENT'
  customSettings  TEXT,          -- JSON blob (cards, filters, etc.)
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dashboards_user ON dashboards(user_id);

-- =========
-- Audit Log
-- =========
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_user_id  INTEGER NOT NULL,
  entity_type    TEXT NOT NULL,     -- 'STUDENT','TEACHER','ACADEMIC_DATA','USER','PREDICTION', etc.
  entity_id      INTEGER NOT NULL,
  action         TEXT NOT NULL,     -- 'CREATE','UPDATE','DELETE','PREDICT'
  details_json   TEXT,
  created_at     TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (actor_user_id) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_entity
  ON audit_log(entity_type, entity_id, created_at);
