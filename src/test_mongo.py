from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")
print("Using URI:", uri)

client = MongoClient(uri)
print(client.list_database_names())
