from django.urls import path  
from . import views  
  
app_name = 'recommendation_engine'  
  
urlpatterns = [  
    path('recommendations/<str:job_id>/', views.RecommendationView.as_view(), name='recommendations'),  
    path('api/recommendations/<str:job_id>/', views.RecommendationAPIView.as_view(), name='api_recommendations'),  
]