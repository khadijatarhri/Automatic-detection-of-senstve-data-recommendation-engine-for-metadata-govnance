
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('mongo_auth/', include('mongo_auth.urls')),
    path('csv-anonymizer/', include('csv_anonymizer.urls')),  
    path('authapp/', include('authapp.urls')),  
    path("admin/", admin.site.urls),
    path('recommendations/', include('recommendation_engine.urls')),  
    path("", include("authapp.urls")),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
