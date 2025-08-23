from django.urls import path  
from .views import UploadCSVView , DataStewardDashboardView, ProcessCSVView ,StatisticsView , RemoveDuplicatesView,  RemoveMissingValuesView # Ajoute cette importation  
  
app_name = 'csv_anonymizer'  
  
urlpatterns = [  
    path('upload/', UploadCSVView.as_view(), name='upload'),  
    path('process/<str:job_id>/', ProcessCSVView.as_view(), name='process'),  
    path('statistics/', StatisticsView.as_view(), name='statistics'),
    path('remove-duplicates/', RemoveDuplicatesView.as_view(), name='remove_duplicates'),  
    path('remove-missing-values/', RemoveMissingValuesView.as_view(), name='remove_missing_values'), 
    path('datasteward-dashboard/', DataStewardDashboardView.as_view(), name='datasteward_dashboard'),

 
]
