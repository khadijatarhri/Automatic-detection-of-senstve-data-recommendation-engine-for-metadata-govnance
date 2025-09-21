# =============================================================================
# CONSUMER KAFKA POUR DJANGO
# Écoute les messages Kafka et les traite comme des données CSV
# =============================================================================

import json        # Pour décoder les messages JSON
import logging     # Pour les logs
from kafka import KafkaConsumer    # Client Kafka Python
from pymongo import MongoClient    # Client MongoDB
import datetime    # Pour les timestamps
from django.conf import settings  # Configuration Django

# Configuration des logs
logger = logging.getLogger(__name__)

class OdooCustomerDataConsumer:
    """
    Classe qui écoute le topic 'odoo-customer-data' et traite les messages
    comme s'ils venaient d'un fichier CSV uploadé
    """
    
    def __init__(self):
        """Initialise le consumer Kafka et la connexion MongoDB"""
        
        # === CONFIGURATION KAFKA CONSUMER ===
        self.consumer = KafkaConsumer(
            'odoo-customer-data',                    # Topic à écouter
            bootstrap_servers=['kafka-broker:29092'], # Adresse du broker Kafka
            group_id='django-governance-group',      # ID du groupe de consumers
            # Fonction pour décoder les messages JSON
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'               # Commence par les nouveaux messages
        )
          
        # === CONFIGURATION MONGODB ===
        # Connexion à MongoDB (même config que ton app Django)
        self.client = MongoClient('mongodb://mongodb:27017/')
        self.main_db = self.client['main_db']           # Base principale
        self.csv_db = self.client['csv_anonymizer_db']  # Base pour les CSV
          
    def start_consuming(self):
        """Démarre l'écoute des messages Kafka en boucle infinie"""
        logger.info("🚀 Démarrage du consumer Kafka pour odoo-customer-data")
          
        # Boucle infinie d'écoute
        for message in self.consumer:
            try:
                customer_data = message.value  # Données du client depuis Kafka
                self.process_customer_data(customer_data)
            except Exception as e:
                logger.error(f"Erreur traitement message Kafka: {e}")
      
    def process_customer_data(self, customer_data):
        """
        Traite les données client reçues d'Odoo comme un CSV importé
        Simule le processus d'upload CSV de ton app Django
        """
          
        # === CRÉATION D'UN JOB D'ANONYMISATION ===
        # Simule la structure d'un job CSV comme dans UploadCSVView
        job_data = {
            'user_email': 'system@kafka.consumer',  # Utilisateur système
            # Nom de fichier généré automatiquement
            'original_filename': f"odoo_customers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            'upload_date': datetime.datetime.now(),
            'status': 'pending',                     # En attente de traitement
            'source': 'kafka_odoo_vrp',            # Source : Kafka depuis Odoo VRP
            'shared_with_data_stewards': True,      # Partagé avec les data stewards
        }
          
        # === INSERTION DU JOB EN BASE ===
        # Insère le job comme dans UploadCSVView.post()
        result = self.main_db.anonymization_jobs.insert_one(job_data)
        job_id = result.inserted_id
          
        # === CONVERSION DES DONNÉES ODOO EN FORMAT CSV ===
        headers = ['name', 'email', 'phone', 'location', 'customer_id']
        csv_data = [{
            'name': customer_data.get('name', ''),          # Nom du client
            'email': customer_data.get('email', ''),        # Email du client
            'phone': customer_data.get('phone', ''),        # Téléphone du client
            'location': customer_data.get('location', ''),  # Localisation du client
            'customer_id': customer_data.get('id', '')      # ID du client dans Odoo
        }]
          
        # === STOCKAGE DES DONNÉES CSV ===
        # Stocke comme dans UploadCSVView
        self.csv_db.csv_data.insert_one({
            'job_id': str(job_id),  # Référence vers le job
            'headers': headers,      # En-têtes du CSV
            'data': csv_data        # Données du CSV
        })
          
        logger.info(f"✅ Client Odoo traité: {customer_data.get('name')} - Job ID: {job_id}")