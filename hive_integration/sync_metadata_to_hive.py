#!/usr/bin/env python3  
from hive_integration import HiveMetadataSync  
  
def main():  
    print("🚀 Démarrage de la synchronisation des métadonnées vers Hive...")  
      
    try:  
        # Étape 1: Initialiser la connexion  
        with HiveMetadataSync() as hive_sync:  
              
            # Étape 2: Créer les tables  
            print("📊 Création des tables de métadonnées...")  
            hive_sync.create_metadata_tables()  
            print("✅ Tables créées avec succès")  
              
            # Étape 3: Synchroniser les données  
            print("🔄 Synchronisation des annotations validées...")  
            hive_sync.sync_column_annotations()  
            print("✅ Synchronisation terminée")  
              
    except Exception as e:  
        print(f"❌ Erreur lors de la synchronisation: {e}")  
        raise  
  
if __name__ == "__main__":  
    main()