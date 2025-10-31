PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL,
  email         TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('ADMIN','TEACHER','STUDENT')),
  created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS students (
  student_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER NOT NULL UNIQUE,
  gpa        REAL,
  department TEXT,
  status     TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS teachers (
  teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER NOT NULL UNIQUE,
  department TEXT,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Numeric “core” fields + extra_json for categorical/optional fields
CREATE TABLE IF NOT EXISTS student_academic_data (
  data_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id    INTEGER NOT NULL,
  term          TEXT NOT NULL,
  attendance    REAL,
  study_hours   REAL,
  exam_score    REAL,
  stress_level  INTEGER,
  sleep_hours   REAL,
  participation REAL,
  extra_json    TEXT,                -- JSON string of additional features (categoricals etc.)
  created_by    INTEGER,
  created_at    TEXT DEFAULT (datetime('now')),
  UNIQUE (student_id, term),
  FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ml_models (
  model_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT NOT NULL,
  version      TEXT NOT NULL,
  path         TEXT NOT NULL,
  columns_path TEXT NOT NULL,
  accuracy     REAL,
  f1_score     REAL,
  roc_auc      REAL,
  created_at   TEXT DEFAULT (datetime('now')),
  UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS prediction_results (
  prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id      INTEGER NOT NULL,
  model_id        INTEGER NOT NULL,
  term            TEXT NOT NULL,
  predicted_label TEXT NOT NULL CHECK (predicted_label IN ('PASS','FAIL')),
  pass_prob       REAL NOT NULL,
  fail_prob       REAL NOT NULL,
  created_at      TEXT DEFAULT (datetime('now')),
  UNIQUE (student_id, model_id, term),
  FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (model_id)   REFERENCES ml_models(model_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendations (
  recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id     INTEGER,
  student_id        INTEGER NOT NULL,
  type              TEXT NOT NULL,
  message           TEXT NOT NULL,
  created_at        TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (prediction_id) REFERENCES prediction_results(prediction_id) ON DELETE CASCADE,
  FOREIGN KEY (student_id)    REFERENCES students(student_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_academic_student_term ON student_academic_data(student_id, term);
CREATE INDEX IF NOT EXISTS idx_pred_student_term     ON prediction_results(student_id, term);
