# src/mongo_smoke_test.py
from src.mongo_client import get_db

def main():
    db = get_db()

    print("Collections:", db.list_collection_names())

    # Show a few users
    print("\nUsers:")
    for doc in db["users"].find({}, {"_id": 0}).limit(5):
        print(doc)

    # Show a few students
    print("\nStudents:")
    for doc in db["students"].find({}, {"_id": 0}).limit(5):
        print(doc)

if __name__ == "__main__":
    main()
