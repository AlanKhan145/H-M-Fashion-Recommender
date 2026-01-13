from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

MONGODB_URL = "mongodb+srv://CgvHub:o@cluster0.ktuns.mongodb.net/"
DATABASE_NAME = "todosapp"

# Kết nối tới MongoDB
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Nếu sử dụng async
async_client = AsyncIOMotorClient(MONGODB_URL)
async_db = async_client[DATABASE_NAME]

# Ví dụ thao tác với MongoDB
collection = db["todos"]

def get_database():
    return db

def get_collection(name):
    return db[name]

# Nếu muốn dùng async
async def get_async_database():
    return async_db

async def get_async_collection(name):
    return async_db[name]
