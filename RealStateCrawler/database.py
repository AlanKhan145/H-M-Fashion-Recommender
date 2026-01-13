from pymongo import MongoClient
from config import MONGO_URL, DATABASE_NAME

def get_database():
    try:
        client = MongoClient(MONGO_URL)
        db = client[DATABASE_NAME]
        print(f"Kết nối thành công đến database '{DATABASE_NAME}'")
        return db
    except Exception as e:
        print(f"Lỗi kết nối MongoDB: {e}")
        return None
