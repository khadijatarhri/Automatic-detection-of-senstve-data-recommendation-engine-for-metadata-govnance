from django.shortcuts import render
from rest_framework.views import APIView 
from rest_framework.response import Response
from rest_framework import status
from db_connections import db
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import check_password
from db_connections import db
import os
from django.views import View 
from bson import ObjectId  


users = db["users"]

# --- Register Logic ---
def register_form(request):
    return render(request, "authapp/register.html")

class RegisterView(View):  # CHANGER APIView → View
    def post(self, request):
        data = request.POST  # CHANGER request.data → request.POST
        if users.find_one({"email": data["email"]}):
            return render(request, "authapp/register.html", {"error": "Email already exists"})
        
        new_user = {
            "name": data["name"],
            "email": data["email"],
            "password": make_password(data["password"])
        }
        users.insert_one(new_user)
        return redirect("authapp:login_form")

# --- Login Logic ---

def login_form(request):
    if request.method == 'POST':
        print("Login POST received")  # DEBUG
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = users.find_one({'email': email})

        if not user:
            print("No user found with this email")  # DEBUG
            messages.error(request, "Invalid email or password.")
            return redirect('authapp:login_form')

        if 'password' not in user:
            print("User document missing 'password' field")  # DEBUG
            messages.error(request, "Account error: password not set.")
            return redirect('authapp:login_form')
        if check_password(password, user['password']):
            request.session['user_email'] = email
            if user.get("role") == "admin":
                return redirect("csv_anonymizer:upload")  # à créer si besoin
            return redirect('csv_anonymizer:upload')

        if check_password(password, user['password']):
            print("Login success")  # DEBUG
            request.session['user_email'] = email
            return redirect('csv_anonymizer:upload')
        else:
            print("Invalid password")  # DEBUG
            messages.error(request, "Invalid email or password.")
            return redirect('authapp:login_form')

    print("Login GET request")  # DEBUG
    return render(request, 'authapp/login.html')


# --- Home Page ---
def home_view(request):  
    if not request.session.get("user_email"):  
        return redirect("/login/")  
      
    user_email = request.session.get("user_email")  
    current_user = users.find_one({'email': user_email})  
      
    # Si c'est un admin, rediriger vers l'interface complète  
    if current_user and current_user.get('role') == 'admin':  
        return render(request, "authapp/home.html")  
    else:  
        # Pour les utilisateurs normaux, afficher seulement le tableau  
        return render(request, "authapp/user_home.html")

# --- Upload API ---
class UploadFileView(APIView):  
    def post(self, request):  
        if not request.session.get("user_email"):  
            return Response({"error": "User not authenticated"}, status=401)  
              
        file = request.FILES.get("file")  
        if file:  
            save_path = f"media/uploads/{file.name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            return Response({"message": "File uploaded successfully"}, status=201)
        return Response({"error": "No file provided"}, status=400)


class AdminView(View):  
    def get(self, request):  
        if not request.session.get("user_email"):  
            return redirect('authapp:login_form')  
          
        user_email = request.session.get("user_email")  
        current_user = users.find_one({'email': user_email})  
          
        if not current_user or current_user.get('role') != 'admin':  
            return redirect('authapp:home')  
          
        # Récupérer tous les utilisateurs et convertir _id en string  
        all_users = list(users.find({}, {'password': 0}))  
          
        # Convertir les ObjectId en strings pour le template  
        for user in all_users:  
            user['id'] = str(user['_id'])  # Créer un nouveau champ 'id'  
          
        return render(request, 'authapp/admin.html', {'users': all_users})  
      
    def post(self, request):  
        if not request.session.get("user_email"):  
            return redirect('authapp:login_form')  
          
        action = request.POST.get('action')  
          
        if action == 'create':  
            return self.create_user(request)  
        elif action == 'update':  
            return self.update_user(request)  
        elif action == 'delete':  
            return self.delete_user(request)  
          
        return redirect('admin')  
      
    def create_user(self, request):  
        data = request.POST  
        if users.find_one({"email": data["email"]}):  
            messages.error(request, "Email already exists")  
            return redirect('admin')  
          
        new_user = {  
            "name": data["name"],  
            "email": data["email"],  
            "password": make_password(data["password"]),  
            "role": data.get("role", "user")  # Par défaut 'user'  
        }  
        users.insert_one(new_user)  
        messages.success(request, "User created successfully")  
        return redirect('admin')  
      
    def update_user(self, request):  
        user_id = request.POST.get('user_id')  
        update_data = {  
            "name": request.POST.get("name"),  
            "email": request.POST.get("email"),  
            "role": request.POST.get("role", "user")  
        }  
          
        if request.POST.get("password"):  
            update_data["password"] = make_password(request.POST.get("password"))  
          
        users.update_one(  
            {"_id": ObjectId(user_id)},  
            {"$set": update_data}  
        )  
        messages.success(request, "User updated successfully")  
        return redirect('admin')  
      
    def delete_user(self, request):  
        user_id = request.POST.get('user_id')  
        users.delete_one({"_id": ObjectId(user_id)})  
        messages.success(request, "User deleted successfully")  
        return redirect('admin')

def logout_view(request):  
    request.session.flush()  # Supprime toutes les données de session  
    return redirect('authapp:login_form')