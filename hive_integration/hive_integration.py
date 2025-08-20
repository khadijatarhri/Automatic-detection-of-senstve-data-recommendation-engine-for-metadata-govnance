from pyhive import hive  
from pymongo import MongoClient  
import pandas as pd  
  
class HiveMetadataSync:  
    def __init__(self, hive_host='localhost', hive_port=10000):  
        self.hive_conn = hive.Connection(host=hive_host, port=hive_port)  
        self.mongo_client = MongoClient('mongodb://mongodb:27017/')  
          
    def create_metadata_tables(self):  
        """Crée les tables de métadonnées dans Hive"""  
        cursor = self.hive_conn.cursor()  
          
        # Table pour les annotations de colonnes  
        cursor.execute("""  
            CREATE TABLE IF NOT EXISTS metadata_quality (  
                job_id STRING,  
                column_name STRING,  
                entity_type STRING,  
                rgpd_category STRING,  
                sensitivity_level STRING,  
                anonymization_method STRING,  
                validation_status STRING,  
                confidence_score DOUBLE,  
                validated_by STRING,  
                validation_date TIMESTAMP  
            )  
            STORED AS PARQUET  
        """)  
          
    def sync_column_annotations(self, job_id=None):  
        """Synchronise les annotations de colonnes vers Hive"""  
        metadata_db = self.mongo_client['metadata_validation_db']  
        annotations = metadata_db['column_annotations']  
          
        query = {'validation_status': 'validated'}  
        if job_id:  
            query['job_id'] = job_id  
              
        validated_annotations = list(annotations.find(query))  
          
        if validated_annotations:  
            df = pd.DataFrame(validated_annotations)  
            # Convertir en format Hive et insérer  
            self._insert_to_hive_table('metadata_quality', df)


    def _insert_to_hive_table(self, table_name, df):  
        """Insère les données pandas dans une table Hive"""  
        cursor = self.hive_conn.cursor()  
          
        # Nettoyer les données et convertir les types  
        df = df.fillna('')  # Remplacer les NaN  
          
        for _, row in df.iterrows():  
            values = tuple(row.values)  
            placeholders = ', '.join(['%s'] * len(values))  
              
            cursor.execute(f"""  
                INSERT INTO {table_name} VALUES ({placeholders})  
            """, values)


    def __enter__(self):  
        return self  
      
    def __exit__(self, exc_type, exc_val, exc_tb):  
        if self.hive_conn:  
            self.hive_conn.close()  
        if self.mongo_client:  
            self.mongo_client.close()