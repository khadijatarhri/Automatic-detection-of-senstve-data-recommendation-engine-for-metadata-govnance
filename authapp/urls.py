from django.urls import path
from .views import RegisterView,download_file, UploadFileView,logout_view, AdminView,login_form, register_form, home_view


app_name = 'authapp'

urlpatterns = [
    path("register/", register_form, name="register_form"),
    path("api/register/", RegisterView.as_view(), name="api_register"),
    path("login/", login_form, name="login_form"),
    path("home/", home_view, name="home"),
    path("api/upload/", UploadFileView.as_view()),
    path('admin/', AdminView.as_view(), name='admin'),  
    path('logout/', logout_view, name='logout'),
    path("download/<str:job_id>/", download_file, name="download_file"),  # Nouvelle URL  
  

]