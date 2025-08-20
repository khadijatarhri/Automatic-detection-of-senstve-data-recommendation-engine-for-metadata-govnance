import requests  
import json  
  
class HiveSandboxAPI:  
    def __init__(self, sandbox_url='http://localhost:4200'):  
        self.sandbox_url = sandbox_url  
        self.session = requests.Session()  
      
    def execute_hive_query(self, query):  
        """Exécute une requête Hive via l'API REST du sandbox"""  
        payload = {  
            'query': query,  
            'database': 'default'  
        }  
        response = self.session.post(  
            f"{self.sandbox_url}/api/hive/execute",  
            json=payload  
        )  
        return response.json()  
      
    def sync_metadata_to_hive(self, metadata_records):  
        """Synchronise les métadonnées vers Hive"""  
        # Créer la table si elle n'existe pas  
        create_table_query = """  
            CREATE TABLE IF NOT EXISTS metadata_quality (  
                job_id STRING,  
                column_name STRING,  
                entity_type STRING,  
                rgpd_category STRING,  
                validation_status STRING  
            ) STORED AS PARQUET  
        """  
        self.execute_hive_query(create_table_query)  
          
        # Insérer les données  
        for record in metadata_records:  
            insert_query = f"""  
                INSERT INTO metadata_quality VALUES   
                ('{record['job_id']}', '{record['column_name']}',   
                 '{record['entity_type']}', '{record['rgpd_category']}',   
                 '{record['validation_status']}')  
            """  
            self.execute_hive_query(insert_query)