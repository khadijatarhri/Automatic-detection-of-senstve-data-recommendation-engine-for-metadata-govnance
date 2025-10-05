
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView

urlpatterns = [
    path('mongo_auth/', include('mongo_auth.urls')),
    path('csv-anonymizer/', include('csv_anonymizer.urls')),  
    path('authapp/', include('authapp.urls')),  
    path("admin/", admin.site.urls),
    path('recommendations/', include('recommendation_engine.urls')),  
    path("", include("authapp.urls")),
    path('favicon.ico', RedirectView.as_view(
        url=staticfiles_storage.url('favicon.ico'),
        permanent=True
    ), name='favicon'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
