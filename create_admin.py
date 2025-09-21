from pymongo import MongoClient  
from django.contrib.auth.hashers import make_password  
import django  
import os  
import time  
import sys  
  
# Point vers ton settings.py  
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend_with_mongodb.settings")  
django.setup()  
  
def wait_for_mongodb(max_retries=30, delay=2):  
    """Attendre que MongoDB soit disponible"""  
    for attempt in range(max_retries):  
        try:  
            client = MongoClient("mongodb://mongodb:27017/", serverSelectionTimeoutMS=5000)  
            client.admin.command('ping')  
            print(f"✅ MongoDB connecté après {attempt + 1} tentatives")  
            return client  
        except Exception as e:  
            print(f"Tentative {attempt + 1}/{max_retries}: MongoDB non disponible - {e}")  
            if attempt < max_retries - 1:  
                time.sleep(delay)  
            else:  
                print("❌ Impossible de se connecter à MongoDB")  
                sys.exit(1)  
  
# Attendre MongoDB  
client = wait_for_mongodb()  
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