import json
import logging
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from pymongo import MongoClient
from bson import ObjectId  # AJOUT IMPORTANT
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OdooCustomerDataConsumer:
    """Consumer Kafka pour données Odoo Sale Orders"""
    
    def __init__(self):
        mongo_uri = 'mongodb://mongodb:27017/'
        logger.info(f"Connexion MongoDB: {mongo_uri}")
        

        from db_connections import db as main_db  
        from pymongo import MongoClient  
  
        self.client = MongoClient('mongodb://mongodb:27017/')  
        self.main_db = main_db  # Utiliser la conne xion centralisée 
        self.csv_db = self.client['csv_anonymizer_db']
        
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connecté avec succès")
        except Exception as e:
            logger.error(f"Erreur connexion MongoDB: {e}")
            raise
        
        self.main_db = self.client['main_db']
        self.csv_db = self.client['csv_anonymizer_db']
        
        logger.info(f"Bases MongoDB disponibles: {self.client.list_database_names()}")
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.consumer = KafkaConsumer(
                    'odoo-customer-data',
                    bootstrap_servers=['kafka-broker:29092'],
                    group_id='django-governance-group',
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    max_poll_records=10,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000
                )
                logger.info("Consumer Kafka connecté avec succès")
                break
            except KafkaError as e:
                retry_count += 1
                logger.warning(f"Tentative {retry_count}/{max_retries} - Kafka non disponible: {e}")
                time.sleep(5)
        
        if retry_count == max_retries:
            raise Exception("Impossible de se connecter à Kafka après plusieurs tentatives")
    
    def start_consuming(self):
        logger.info("Démarrage du consumer Kafka pour odoo-customer-data")
        logger.info(f"Topics disponibles: {self.consumer.topics()}")
        
        message_count = 0
        
        try:
            for message in self.consumer:
                try:
                    message_count += 1
                    customer_data = message.value
                    
                    logger.info(f"Message #{message_count} reçu - Topic: {message.topic}, Partition: {message.partition}, Offset: {message.offset}")
                    
                    job_id = self.process_customer_data(customer_data)
                    
                    if job_id:
                        logger.info(f"Message traité avec succès - Job ID: {job_id}")
                    else:
                        logger.error("Échec du traitement du message")
                    
                except Exception as e:
                    logger.error(f"Erreur traitement message: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except KeyboardInterrupt:
            logger.info("Arrêt du consumer par l'utilisateur")
        finally:
            self.consumer.close()
            logger.info(f"Total messages traités: {message_count}")
    
    def process_customer_data(self, customer_data):
        """Traite les données client Odoo et les stocke en chunks"""
        
        try:
            job_data = {
                'user_email': 'system@kafka.consumer',
                'original_filename': f"odoo_sale_order_{customer_data.get('order_reference', 'unknown')}.csv",
                'upload_date': datetime.now(),
                'status': 'pending',
                'source': 'kafka_odoo_vrp',
                'shared_with_data_stewards': True,
                'odoo_metadata': {
                    'order_reference': customer_data.get('order_reference'),
                    'order_amount': customer_data.get('order_amount'),
                    'order_date': customer_data.get('order_date'),
                    'customer_id': customer_data.get('customer_id'),
                    'created_at': customer_data.get('created_at')
                }
            }
            
            result = self.main_db.anonymization_jobs.insert_one(job_data)
            job_id = str(result.inserted_id)
            
            logger.info(f"Job créé: {job_id}")
            
            # Stocker en chunks
            headers = ['customer_id', 'name', 'email', 'phone', 'location']
            
            row_data = {
                'customer_id': customer_data.get('customer_id', ''),
                'name': customer_data.get('name', ''),
                'email': customer_data.get('email', ''),
                'phone': customer_data.get('phone', ''),
                'location': customer_data.get('location', '')
            }
            
            chunk_doc = {
                'job_id': job_id,
                'chunk_number': 0,
                'headers': headers,
                'data': [row_data],
                'created_at': datetime.now()
            }
            
            self.csv_db.csv_chunks.insert_one(chunk_doc)
            
            logger.info(f"Données stockées - Job: {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Erreur process_customer_data: {e}")
            import traceback
            traceback.print_exc()
            return None