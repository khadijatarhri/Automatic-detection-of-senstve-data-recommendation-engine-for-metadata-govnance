from django.urls import path  
from .views import UploadCSVView, ProcessCSVView ,StatisticsView  # Ajoute cette importation  
  
app_name = 'csv_anonymizer'  
  
urlpatterns = [  
    path('upload/', UploadCSVView.as_view(), name='upload'),  
    path('process/<str:job_id>/', ProcessCSVView.as_view(), name='process'),  
    path('statistics/', StatisticsView.as_view(), name='statistics'),

]
