import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
import certifi  # <-- new

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME", "edutrack")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set in environment")

@lru_cache(maxsize=1)
def get_db():
    # Use certifi’s CA bundle so TLS trust is clean on Windows
    client = MongoClient(
        MONGODB_URI,
        tlsCAFile=certifi.where(),           # <– important
        serverSelectionTimeoutMS=30000,      # 30s (same as default, explicit)
    )
    return client[MONGODB_DBNAME]
