from pymongo import MongoClient
from django.contrib.auth.hashers import make_password
import django
import os

# Point vers ton settings.py
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend_with_mongodb.settings")
django.setup()

client = MongoClient("mongodb://localhost:27017/")
db = client["csv_anonymizer_db"]
users = db["users"]

admin_user = {
    "name": "Admin",
    "email": "admin@example.com",
    "password": make_password("admin123"),
    "role": "admin"
}

if not users.find_one({"email": admin_user["email"]}):
    users.insert_one(admin_user)
    print("✅ Admin account created!")
else:
    print("ℹ️ Admin already exists.")

