import requests  
import json  
from pymongo import MongoClient  
from datetime import datetime  
  
# Configuration Atlas  
ATLAS_URL = "http://172.17.0.1:21000"  
ATLAS_USER = "admin"  
ATLAS_PASS = "allahyarani123"  
  
def create_glossary():  
    """Créer le glossaire principal"""  
    glossary_data = {  
        "name": "RGPD Metadata_Glossary",  
        "shortDescription": "Glossaire complet des métadonnées détectées",  
        "longDescription": "Synchronisation de toutes les métadonnées avant et après validation"  
    }  
      
    response = requests.post(  
        f"{ATLAS_URL}/api/atlas/v2/glossary",  
        json=glossary_data,  
        auth=(ATLAS_USER, ATLAS_PASS)  
    )  
      
    if response.status_code == 200:  
        return response.json()['guid']  
    else:  
        print(f"Erreur création glossaire: {response.text}")  
        return None  
  
def create_categories(glossary_guid):  
    """Créer les catégories RGPD"""  
    categories = [  
        {"name": "Données d'identification", "description": "Informations d'identité personnelle"},  
        {"name": "Données financières", "description": "Informations bancaires et financières"},  
        {"name": "Données de contact", "description": "Informations de contact"},  
        {"name": "Données de localisation", "description": "Informations géographiques"},  
        {"name": "Données temporelles", "description": "Informations de date et heure"}  
    ]  
      
    category_guids = {}  
    for category in categories:  
        cat_data = {  
            "name": category["name"],  
            "shortDescription": category["description"],  
            "anchor": {"glossaryGuid": glossary_guid}  
        }  
          
        response = requests.post(  
            f"{ATLAS_URL}/api/atlas/v2/glossary/category",  
            json=cat_data,  
            auth=(ATLAS_USER, ATLAS_PASS)  
        )  
          
        if response.status_code == 200:  
            category_guids[category["name"]] = response.json()['guid']  
            print(f"Catégorie créée: {category['name']}")  
      
    return category_guids  


def extract_sensitivity_levels_from_db():  
    """Extraire les niveaux de sensibilité uniques depuis MongoDB"""  
    client = MongoClient('mongodb://mongodb:27017/')  
    metadata_db = client['metadata_validation_db']  
    annotations = metadata_db['column_annotations']  
      
    # Extraire tous les niveaux de sensibilité uniques  
    sensitivity_levels = annotations.distinct('sensitivity_level')  
      
    # Filtrer les valeurs nulles/vides  
    sensitivity_levels = [level for level in sensitivity_levels if level]  
      
    print(f"Niveaux de sensibilité trouvés: {sensitivity_levels}")  
    return sensitivity_levels  
  
def create_dynamic_atlas_classifications():  
    """Créer les classifications Atlas basées sur les données réelles"""  
    sensitivity_levels = extract_sensitivity_levels_from_db()  
      
    classification_defs = []  
      
    for level in sensitivity_levels:  
        classification_def = {  
            "name": level,  
            "description": f"Classification de sensibilité: {level}",  
            "attributeDefs": [  
                {"name": "level", "typeName": "string", "isOptional": True},  
                {"name": "source", "typeName": "string", "isOptional": True},  
                {"name": "ranger_policy", "typeName": "string", "isOptional": True},  
                {"name": "rgpd_category", "typeName": "string", "isOptional": True}  
            ]  
        }  
        classification_defs.append(classification_def)  
      
    # Créer les classifications dans Atlas  
    if classification_defs:  
        classification_batch = {"classificationDefs": classification_defs}  
          
        response = requests.post(  
            f"{ATLAS_URL}/api/atlas/v2/types/typedefs",  
            json=classification_batch,  
            auth=(ATLAS_USER, ATLAS_PASS)  
        )  
          
        if response.status_code == 200:  
            print(f"✓ {len(classification_defs)} classifications créées dans Atlas")  
            return True  
        else:  
            print(f"✗ Erreur création classifications: {response.text}")  
            return False  
      
    return False



  
def sync_all_metadata():  
    """Synchroniser toutes les métadonnées vers Atlas"""  
      

    create_dynamic_atlas_classifications()  

    # 1. Créer le glossaire  
    glossary_guid = create_glossary()  
    if not glossary_guid:  
        return  
      
    # 2. Créer les catégories  
    category_guids = create_categories(glossary_guid)  
      
    # 3. Connexion MongoDB pour extraire TOUTES les métadonnées  
    client = MongoClient('mongodb://mongodb:27017/')  
      
    # Base de données des annotations (validées ET non-validées)  
    metadata_db = client['metadata_validation_db']  
    annotations = metadata_db['column_annotations']  
      
    # Base de données CSV pour les métadonnées enrichies  
    csv_db = client['csv_anonymizer_db']  
    csv_collection = csv_db['csv_data']  
      
    # 4. Extraire toutes les annotations (tous statuts)  
    all_annotations = list(annotations.find({}))  
    print(f"Trouvé {len(all_annotations)} annotations à synchroniser")  
      
    # 5. Synchroniser les termes depuis les annotations  
    synced_terms = set()  
    for annotation in all_annotations:  
        entity_type = annotation['entity_type']  
          
        if entity_type not in synced_terms:  
            # Mapper la catégorie RGPD  
            rgpd_category = annotation.get('rgpd_category', 'Non classifié')  
            category_guid = category_guids.get(rgpd_category)  
              
            # Déterminer le statut de validation  
            validation_status = annotation.get('validation_status', 'pending')  
              
            term_data = {  
                "name": entity_type,  
                "qualifiedName": f"complete_metadata.{entity_type}@cluster1",  
                "shortDescription": f"Entité {entity_type} - Statut: {validation_status}",  
                "longDescription": f"Type d'entité: {entity_type}. Catégorie RGPD: {rgpd_category}. Méthode d'anonymisation: {annotation.get('anonymization_method', 'masquage')}. Statut de validation: {validation_status}",  
                "anchor": {"glossaryGuid": glossary_guid},  
                "attributes": {  
                    "entity_type": entity_type,  
                    "validation_status": validation_status,  
                    "rgpd_category": rgpd_category,  
                    "anonymization_method": annotation.get('anonymization_method', 'masquage'),  
                    "validated_by": annotation.get('validated_by', 'Non validé'),  
                    "sync_date": datetime.now().isoformat()  
                },  
                "classifications": [  
                        {  
                           "typeName": annotation.get('anonymization_method', 'masquage'),   
                           "attributes": {  
                               "level": validation_status,  
                               "source": "automatic_detection",  
                               "ranger_policy": annotation.get('anonymization_method', 'masquage')  
                            }  
                        }  
                ]  
            }  
              
            # Ajouter la catégorie si elle existe  
            if category_guid:  
                term_data["categories"] = [{"categoryGuid": category_guid}]  
              
            # Créer le terme dans Atlas  
            response = requests.post(  
                f"{ATLAS_URL}/api/atlas/v2/glossary/term",  
                json=term_data,  
                auth=(ATLAS_USER, ATLAS_PASS)  
            )  
              
            if response.status_code == 200:  
                print(f"✓ Terme synchronisé: {entity_type} (statut: {validation_status})")  
                synced_terms.add(entity_type)  
            else:  
                print(f"✗ Erreur terme {entity_type}: {response.text}")  
      
    # 6. Synchroniser les métadonnées enrichies depuis semantic_engine  
    print("\n=== Synchronisation des métadonnées enrichies ===")  
      
    # Récupérer tous les jobs CSV  
    csv_jobs = list(csv_collection.find({}))  
      
    for job in csv_jobs:  
        job_id = job.get('job_id')  
        headers = job.get('headers', [])  
          
        print(f"Traitement job: {job_id}")  
          
        # Pour chaque colonne, créer un terme avec les métadonnées enrichies  
        for header in headers:  
            column_term_name = f"COLUMN_{header}_{job_id}"  
              
            if column_term_name not in synced_terms:  
                column_term_data = {  
                    "name": column_term_name,  
                    "qualifiedName": f"complete_metadata.{column_term_name}@cluster1",  
                    "shortDescription": f"Colonne {header} du dataset {job_id}",  
                    "longDescription": f"Métadonnées enrichies pour la colonne {header}",  
                    "anchor": {"glossaryGuid": glossary_guid},  
                    "attributes": {  
                        "column_name": header,  
                        "job_id": job_id,  
                        "source": "semantic_engine",  
                        "sync_date": datetime.now().isoformat()  
                    }  
                }  
                  
                response = requests.post(  
                    f"{ATLAS_URL}/api/atlas/v2/glossary/term",  
                    json=column_term_data,  
                    auth=(ATLAS_USER, ATLAS_PASS)  
                )  
                  
                if response.status_code == 200:  
                    print(f"✓ Colonne synchronisée: {header}")  
                    synced_terms.add(column_term_name)  
      
    print(f"\n=== Synchronisation terminée ===")  
    print(f"Total termes synchronisés: {len(synced_terms)}")  
    print(f"Glossaire GUID: {glossary_guid}")  
      
    return {  
        "success": True,  
        "glossary_guid": glossary_guid,  
        "terms_synced": len(synced_terms),  
        "categories_created": len(category_guids)  
    }  
  
if __name__ == "__main__":  
    result = sync_all_metadata()  
    print(f"\nRésultat final: {result}")
