FROM python:3.10-bullseye  
  
WORKDIR /app  
  
# Installer les dépendances système  
RUN apt-get update && apt-get install -y \  
    gcc \  
    g++ \  
    && rm -rf /var/lib/apt/lists/*  
  
# Copier et installer les dépendances Python  
COPY requirements.txt ./  
RUN pip install --no-cache-dir -r requirements.txt  
  
COPY create_admin.py ./  
  
# Copier le code de l'application  
COPY . .  
  
EXPOSE 8000  
  
# CORRECTION: UN SEUL CMD par défaut (pour web)
CMD ["sh", "-c", "python create_admin.py && python manage.py runserver 0.0.0.0:8000"]