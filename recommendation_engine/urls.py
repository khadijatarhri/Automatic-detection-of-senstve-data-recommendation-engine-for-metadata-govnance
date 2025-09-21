from django.urls import path  
from . import views  
  
app_name = 'recommendation_engine'  
  
urlpatterns = [  
    path('recommendations/<str:job_id>/', views.RecommendationView.as_view(), name='recommendations'),  
    path('api/recommendations/<str:job_id>/', views.RecommendationAPIView.as_view(), name='recommendations_api'),  
    path('metadata/<str:job_id>/', views.MetadataView.as_view(), name='metadata'), 
    path('validation/<str:job_id>/<str:entity_id>/', views.ValidationWorkflowView.as_view(), name='validation'),
    path('column-validation/<str:job_id>/<str:column_name>/', views.ColumnValidationWorkflowView.as_view(), name='column_validation'),
    path('glossary/', views.GlossaryView.as_view(), name='glossary'),  
    path('glossary/sync/', views.GlossarySyncView.as_view(), name='glossary_sync'),
    path('data-quality/<str:job_id>/', views.DataQualityView.as_view(), name='data_quality'),  


 

]