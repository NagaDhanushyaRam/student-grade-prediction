# src/mongo_client.py
import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME", "edutrack")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set in environment")

@lru_cache(maxsize=1)
def get_db():
    client = MongoClient(MONGODB_URI)
    return client[MONGODB_DBNAME]
