from django.core.management.base import BaseCommand  
from kafka_integration.kafka_consumer import OdooCustomerDataConsumer  
  
class Command(BaseCommand):  
    help = 'Démarre le consumer Kafka pour les données Odoo'  
      
    def handle(self, *args, **options):  
        consumer = OdooCustomerDataConsumer()  
        consumer.start_consuming()