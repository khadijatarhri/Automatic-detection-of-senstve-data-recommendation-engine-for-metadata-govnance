import logging
from django.core.management.base import BaseCommand
from kafka_integration.kafka_consumer import OdooCustomerDataConsumer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Démarre le consumer Kafka pour les données Odoo'
    
    def handle(self, *args, **options):
        """Point d'entrée de la commande Django"""
        self.stdout.write(self.style.SUCCESS('Démarrage du Kafka Consumer...'))
        
        try:
            consumer = OdooCustomerDataConsumer()
            consumer.start_consuming()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('Consumer arrêté par l\'utilisateur'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Erreur fatale: {e}'))
            logger.exception("Erreur dans le consumer Kafka")
            raise