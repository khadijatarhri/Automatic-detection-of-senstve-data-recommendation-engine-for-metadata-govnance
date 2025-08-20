from django.core.management.base import BaseCommand  
from celery import shared_task  
  
@shared_task  
def sync_metadata_to_sandbox():  
    """Tâche Celery pour synchronisation automatique"""  
    from hive_sandbox_api import HiveSandboxAPI  
    from pymongo import MongoClient  
      
    # Récupérer les nouvelles annotations validées  
    client = MongoClient('mongodb://mongodb:27017/')  
    metadata_db = client['metadata_validation_db']  
      
    # Synchroniser vers HDP Sandbox  
    hive_api = HiveSandboxAPI()  
    new_records = metadata_db['column_annotations'].find({  
        'validation_status': 'validated',  
        'synced_to_hive': {'$ne': True}  
    })  
      
    hive_api.sync_metadata_to_hive(list(new_records))  
      
    # Marquer comme synchronisé  
    metadata_db['column_annotations'].update_many(  
        {'validation_status': 'validated'},  
        {'$set': {'synced_to_hive': True}}  
    )