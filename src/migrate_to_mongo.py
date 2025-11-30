# migrate_to_mongo.py
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# 1. Load environment (MONGODB_URI, MONGODB_DBNAME)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME", "edutrack")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set in .env")

# 2. Paths
ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "db"
SQLITE_PATH = DB_DIR / "app.db"   # adjust if different

print("Using SQLite file:", SQLITE_PATH)

if not SQLITE_PATH.exists():
    raise FileNotFoundError(f"SQLite DB not found at {SQLITE_PATH}")

# 3. Connect to SQLite
sqlite_conn = sqlite3.connect(SQLITE_PATH)
sqlite_conn.row_factory = sqlite3.Row  # access by column name
cur = sqlite_conn.cursor()

# 4. Connect to Mongo
client = MongoClient(MONGODB_URI)
mongo_db = client[MONGODB_DBNAME]

def copy_table(table_name: str):
    print(f"➡️  Copying table: {table_name}")

    # Read all rows
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()

    docs = []
    for row in rows:
        d = dict(row)
        # Optionally drop SQLite-specific internal fields if any (none in your schema)
        docs.append(d)

    coll = mongo_db[table_name]

    # Optional: clear collection before inserting
    coll.delete_many({})

    if docs:
        coll.insert_many(docs)
        print(f"   Inserted {len(docs)} documents into '{table_name}'")
    else:
        print(f"   No rows found in '{table_name}'")

def main():
    tables = [
    "administrators",
    "audit_log",
    "dashboards",
    "departments",
    "messages",
    "ml_models",
    "models",
    "prediction_results",
    "recommendations",
    "sqlite_sequence",
    "student_academic_data",
    "students",
    "teachers",
    "users",
    ]

    for t in tables:
        copy_table(t)

    print("✅ Migration complete.")

if __name__ == "__main__":
    main()
