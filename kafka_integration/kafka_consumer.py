# kafka_consumer.py - VERSION BATCH

import json
import logging
from kafka import KafkaConsumer
from pymongo import MongoClient
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class OdooCustomerDataConsumer:
    def __init__(self):
        self.client = MongoClient('mongodb://mongodb:27017/')
        self.csv_db = self.client['csv_anonymizer_db']
        self.main_db = self.client['main_db']
        
        # Buffer pour agréger les messages
        self.message_buffer = []
        self.buffer_size = 50  # Taille du batch
        self.last_flush = time.time()
        self.flush_interval = 300  # 5 minutes
        
        self.consumer = KafkaConsumer(
            'odoo-customer-data',
            bootstrap_servers=['kafka-broker:29092'],
            group_id='django-governance-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        
    def start_consuming(self):
        logger.info("Démarrage consumer Kafka - MODE BATCH")
        
        try:
            for message in self.consumer:
                customer_data = message.value
                self.message_buffer.append(customer_data)
                
                # Flush si buffer plein OU timeout atteint
                if (len(self.message_buffer) >= self.buffer_size or 
                    time.time() - self.last_flush > self.flush_interval):
                    self.flush_batch()
                    
        except KeyboardInterrupt:
            # Flush final avant arrêt
            if self.message_buffer:
                self.flush_batch()
            logger.info("Consumer arrêté")
        finally:
            self.consumer.close()
    
    def flush_batch(self):
        """Traite le batch accumulé comme un seul job"""
        if not self.message_buffer:
            return
        
        try:
            # Créer UN SEUL job pour tout le batch
            job_data = {
                'user_email': 'system@kafka.consumer',
                'original_filename': f'odoo_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                'upload_date': datetime.now(),
                'status': 'pending',
                'source': 'kafka_odoo_vrp',
                'shared_with_data_stewards': True,
                'batch_size': len(self.message_buffer),
                'odoo_metadata': {
                    'batch_period': f'{self.last_flush} - {datetime.now()}',
                    'record_count': len(self.message_buffer)
                }
            }
            
            result = self.main_db.anonymization_jobs.insert_one(job_data)
            job_id = str(result.inserted_id)
            
            # Headers uniques pour tous les records Odoo
            headers = ['customer_id', 'name', 'email', 'phone', 'location', 
                      'order_reference', 'order_amount', 'order_date']
            
            # Convertir le buffer en format chunk
            chunk_data = []
            for customer_data in self.message_buffer:
                row = {
                    'customer_id': customer_data.get('customer_id', ''),
                    'name': customer_data.get('name', ''),
                    'email': customer_data.get('email', ''),
                    'phone': customer_data.get('phone', ''),
                    'location': customer_data.get('location', ''),
                    'order_reference': customer_data.get('order_reference', ''),
                    'order_amount': customer_data.get('order_amount', 0),
                    'order_date': customer_data.get('order_date', '')
                }
                chunk_data.append(row)
            
            # Sauvegarder en chunks (1000 lignes max par chunk)
            chunk_size = 1000
            for i in range(0, len(chunk_data), chunk_size):
                chunk = chunk_data[i:i + chunk_size]
                chunk_doc = {
                    'job_id': job_id,
                    'chunk_number': i // chunk_size,
                    'headers': headers,
                    'data': chunk,
                    'created_at': datetime.now()
                }
                self.csv_db.csv_chunks.insert_one(chunk_doc)
            
            logger.info(f"✅ Batch traité: {len(self.message_buffer)} records → Job {job_id}")
            
            # Reset buffer
            self.message_buffer = []
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Erreur flush batch: {e}")
            import traceback
            traceback.print_exc()