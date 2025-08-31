import requests
import json
from pymongo import MongoClient
from datetime import datetime
import logging

ATLAS_URL = "http://172.26.0.2:21000"  # Vérifiez que votre Atlas est accessible
ATLAS_USER = "admin"                    # Vérifiez les credentials
ATLAS_PASS = "ensias123"          # Vérifiez les credentials

# Configuration MongoDB - À AJUSTER selon votre environnement
MONGO_URI = 'mongodb://mongodb:27017/' 

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AtlasMetadataGovernance:
    def __init__(self):
        self.atlas_url = ATLAS_URL
        self.auth = (ATLAS_USER, ATLAS_PASS)
        self.mongo_client = MongoClient(MONGO_URI)
        self.metadata_db = self.mongo_client['metadata_validation_db']
        
        # Test de connectivité initial
        self._test_connections()
        
    def _test_connections(self):
        """Tester les connexions Atlas et MongoDB"""
        logger.info("🔍 Test des connexions...")
        
        # Test MongoDB
        try:
            self.mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connecté")
        except Exception as e:
            logger.error(f"❌ MongoDB non accessible: {e}")
            raise
        
        # Test Atlas
        


    def get_hive_table_entity(self, table_name):  
     """Récupérer l'entité table Hive et ses colonnes"""  
     search_url = f"{self.atlas_url}/api/atlas/v2/search/dsl"  
     dsl_query = f"hive_table where name='{table_name}'"  
      
     response = requests.get(  
        search_url,  
        auth=self.auth,  
        params={'query': dsl_query}  
     )  
     logger.info(f"Recherche table {table_name}: {response.status_code}")  

     if response.status_code == 200:  
        entities = response.json().get('entities', [])  
        logger.info(f"Entités trouvées: {len(entities)}")  

        if entities:  
            return entities[0]['guid']  
     return None  
  
    def get_table_columns(self, table_guid):  
     """Récupérer toutes les colonnes de la table"""  
     entity_url = f"{self.atlas_url}/api/atlas/v2/entity/guid/{table_guid}"  
      
     response = requests.get(  
        entity_url,  
        auth=self.auth,  
        params={'ignoreRelationships': 'false'}  
     )  
      
     if response.status_code == 200:  
        entity = response.json()['entity']  
        columns = entity.get('relationshipAttributes', {}).get('columns', [])  
          
        column_info = []  
        for col in columns:  
            column_info.append({  
                'guid': col['guid'],  
                'name': col['displayText'],  
                'type': col['typeName']  
            })  
        return column_info  
     return []
    

    def assign_glossary_term_to_column(self, column_guid, term_guid):  
     """Assigner un terme de glossaire à une colonne Hive"""  
     assign_url = f"{self.atlas_url}/api/atlas/v2/entity/guid/{column_guid}/meanings"  
      
     payload = [{  
        "termGuid": term_guid,  
        "relationGuid": None  
     }]  
      
     response = requests.post(  
        assign_url,  
        auth=self.auth,  
        headers={'Content-Type': 'application/json'},  
        json=payload  
     )  
      

     if response.status_code != 200:  
        logger.error(f"Erreur assignation: {response.status_code} - {response.text}")  
      
     return response.status_code == 200  
  
    def get_glossary_term_guid(self, term_name):  
     """Récupérer le GUID d'un terme du glossaire"""  
     search_url = f"{self.atlas_url}/api/atlas/v2/search/basic"  
      
     response = requests.get(  
        search_url,  
        auth=self.auth,  
        params={  
            'query': term_name,  
            'typeName': 'AtlasGlossaryTerm',  
            'excludeDeletedEntities': 'true'  
        }  
     )  
     logger.info(f"Recherche terme {term_name}: {response.status_code}")  

     if response.status_code == 200:  
        entities = response.json().get('entities', [])  
        logger.info(f"Termes trouvés: {len(entities)}")  
        for entity in entities:  
            if entity['displayText'] == term_name:  
                return entity['guid']  
     return None
    

    def create_column_term_mapping(self):  
     """Créer mapping entre colonnes Hive et termes du glossaire"""  
     # Utiliser vos métadonnées validées existantes  
     enriched_metadata = self.metadata_db['enriched_metadata']  
     validated_metadata = list(enriched_metadata.find({"validation_status": "validated"}))  
      
     mapping = {}  
      
     for metadata in validated_metadata:  
        column_name = metadata['column_name']  
        entity_types = metadata.get('entity_types', [])  
        rgpd_category = metadata.get('recommended_rgpd_category')  
          
        # Créer le nom du terme comme dans votre méthode create_validated_metadata_terms  
        term_name = f"{column_name.upper()}_TERM"  
          
        mapping[column_name] = term_name  
          
        logger.info(f"Mapping créé: {column_name} → {term_name}")  
      
     return mapping


    def get_glossary_term_guid(self, term_name):  
     """Récupérer le GUID d'un terme du glossaire avec sélection précise"""  
     search_url = f"{self.atlas_url}/api/atlas/v2/search/basic"  
      
     response = requests.get(  
        search_url,  
        auth=self.auth,  
        params={  
            'query': term_name,  
            'typeName': 'AtlasGlossaryTerm',  
            'excludeDeletedEntities': 'true'  
        }  
     )  
      
     if response.status_code == 200:  
        entities = response.json().get('entities', [])  
          
        # Filtrer par glossaire exact et nom exact  
        for entity in entities:  
            # Vérifier que le terme appartient au bon glossaire  
            if (entity['displayText'] == term_name and   
                entity.get('attributes', {}).get('anchor', {}).get('glossaryGuid') == self.current_glossary_guid):  
                return entity['guid']  
          
        # Fallback : sélection par nom exact seulement  
        for entity in entities:  
            if entity['displayText'] == term_name:  
                return entity['guid']  
      
     return None
    
    def preview_sync_data(self):
        """Prévisualiser les données qui seront synchronisées"""
        enriched_metadata = self.metadata_db['enriched_metadata']
        
        # Statistiques
        total_metadata = enriched_metadata.count_documents({})
        validated_metadata = enriched_metadata.count_documents({"validation_status": "validated"})
        pending_metadata = enriched_metadata.count_documents({"validation_status": "pending"})
        
        # Catégories et niveaux uniques
        categories = enriched_metadata.distinct('recommended_rgpd_category')
        sensitivity_levels = enriched_metadata.distinct('recommended_sensitivity_level')
        
        preview = {
            "total_metadata": total_metadata,
            "validated_metadata": validated_metadata,
            "pending_metadata": pending_metadata,
            "rgpd_categories": [cat for cat in categories if cat],
            "sensitivity_levels": [level for level in sensitivity_levels if level],
            "will_sync": validated_metadata > 0
        }
        
        logger.info("📊 PRÉVISUALISATION DE LA SYNCHRONISATION")
        logger.info(f"📝 Total métadonnées: {total_metadata}")
        logger.info(f"✅ Métadonnées validées (à synchroniser): {validated_metadata}")
        logger.info(f"⏳ Métadonnées en attente: {pending_metadata}")
        logger.info(f"📂 Catégories RGPD: {preview['rgpd_categories']}")
        logger.info(f"🔒 Niveaux de sensibilité: {preview['sensitivity_levels']}")
        
        if validated_metadata == 0:
            logger.warning("⚠️  AUCUNE MÉTADONNÉE VALIDÉE - Rien ne sera synchronisé!")
            
        return preview
        
        # Test de connectivité initial
        self._test_connections()
        
    def create_business_glossary(self):
        """Créer le glossaire métier principal"""
        glossary_data = {
            "name": "Data_Governance_Glossary11",
            "shortDescription": "Glossaire métier pour la gouvernance des données",
            "longDescription": "Glossaire centralisé contenant toutes les métadonnées validées par les data stewards, enrichies avec les recommandations IA et conformes aux exigences RGPD"
        }
        
        response = requests.post(
            f"{self.atlas_url}/api/atlas/v2/glossary",
            json=glossary_data,
            auth=self.auth,
            timeout=(30, 60)
        )
        
        if response.status_code == 200:  
              glossary_guid = response.json()['guid']  
              self.current_glossary_guid = glossary_guid  # Stocker pour la sélection des termes  
              logger.info("✓ Glossaire métier créé avec succès")  
              return glossary_guid
        else:
            logger.error(f"✗ Erreur création glossaire: {response.text}")
            return {"success": False, "error": "Échec "}

    def extract_rgpd_categories_from_db(self):
        """Extraire les catégories RGPD réelles depuis la base"""
        enriched_metadata = self.metadata_db['enriched_metadata']
        categories = enriched_metadata.distinct('recommended_rgpd_category')
        categories = [cat for cat in categories if cat and cat.strip()]
        logger.info(f"Catégories RGPD détectées: {categories}")
        return categories

    def create_rgpd_categories(self, glossary_guid):
        """Créer les catégories RGPD basées sur les données réelles"""
        real_categories = self.extract_rgpd_categories_from_db()
        category_guids = {}
        
        # Mapping des descriptions métier
        category_descriptions = {
            "Données d'identification": "Informations permettant d'identifier directement ou indirectement une personne physique",
            "Données financières": "Informations bancaires, financières et de paiement",
            "Données de contact": "Informations de contact et de communication",
            "Données de localisation": "Informations géographiques et d'adresse",
            "Données temporelles": "Informations de date, heure et temporelles",
            "Données de santé": "Informations médicales et de santé",
            "Données biométriques": "Données biométriques d'identification",
            "Données de comportement": "Données de navigation et comportementales"
        }
        
        for category in real_categories:
            cat_data = {
                "name": category,
                "shortDescription": category_descriptions.get(category, f"Catégorie RGPD: {category}"),
                "longDescription": f"Catégorie de données personnelles selon le RGPD: {category}. Gestion automatisée avec validation data steward.",
                "anchor": {"glossaryGuid": glossary_guid}
            }
            
            response = requests.post(
                f"{self.atlas_url}/api/atlas/v2/glossary/category",
                json=cat_data,
                auth=self.auth
            )
            
            if response.status_code == 200:
                category_guids[category] = response.json()['guid']
                logger.info(f"✓ Catégorie RGPD créée: {category}")
            else:
                logger.error(f"✗ Erreur catégorie {category}: {response.text}")
        
        return category_guids

    def create_sensitivity_classifications(self):
        """Créer les classifications de sensibilité basées sur les données réelles"""
        enriched_metadata = self.metadata_db['enriched_metadata']
        sensitivity_levels = enriched_metadata.distinct('recommended_sensitivity_level')
        sensitivity_levels = [level for level in sensitivity_levels if level]
        
        logger.info(f"Niveaux de sensibilité détectés: {sensitivity_levels}")
        
        # Mapping des attributs métier pour chaque niveau
        sensitivity_mapping = {
            "PUBLIC": {"risk_level": "LOW", "retention_period": "UNLIMITED", "access_level": "PUBLIC"},
            "INTERNAL": {"risk_level": "LOW", "retention_period": "7_YEARS", "access_level": "INTERNAL"},
            "CONFIDENTIAL": {"risk_level": "MEDIUM", "retention_period": "5_YEARS", "access_level": "RESTRICTED"},
            "PERSONAL_DATA": {"risk_level": "HIGH", "retention_period": "2_YEARS", "access_level": "CONTROLLED"},
            "RESTRICTED": {"risk_level": "CRITICAL", "retention_period": "1_YEAR", "access_level": "HIGHLY_RESTRICTED"}
        }
        
        classification_defs = []
        
        for level in sensitivity_levels:
            attrs = sensitivity_mapping.get(level, {"risk_level": "MEDIUM", "retention_period": "3_YEARS", "access_level": "RESTRICTED"})
            
            classification_def = {
                "name": f"DataSensitivity_{level}",
                "description": f"Classification de sensibilité des données: {level}",
                "attributeDefs": [
                    {"name": "sensitivity_level", "typeName": "string", "isOptional": False},
                    {"name": "risk_level", "typeName": "string", "isOptional": True},
                    {"name": "retention_period", "typeName": "string", "isOptional": True},
                    {"name": "access_level", "typeName": "string", "isOptional": True},
                    {"name": "rgpd_compliant", "typeName": "boolean", "isOptional": True},
                    {"name": "data_steward", "typeName": "string", "isOptional": True},
                    {"name": "validation_date", "typeName": "date", "isOptional": True}
                ]
            }
            classification_defs.append(classification_def)
        
        if classification_defs:
            classification_batch = {"classificationDefs": classification_defs}
            
            response = requests.post(
                f"{self.atlas_url}/api/atlas/v2/types/typedefs",
                json=classification_batch,
                auth=self.auth
            )
            
            if response.status_code == 200:
                logger.info(f"✓ {len(classification_defs)} classifications de sensibilité créées")
                {"success": True, "nonerror": "non echec"}
            else:
                logger.error(f"✗ Erreur création classifications: {response.text}")
                return {"success": False, "error": "Échec création classifications"}
        
        return {"success": False, "error": "Échec "}


    def automate_hive_glossary_assignment(self, table_name="entites_marocaines"):  
     """Workflow principal d'assignation automatique"""  
     logger.info(f"🚀 Début assignation automatique pour table: {table_name}")  
      
     try:  
        # 1. Récupérer la table Hive  
        table_guid = self.get_hive_table_entity(table_name)  
        if not table_guid:  
            logger.error(f"❌ Table {table_name} non trouvée dans Atlas")  
            return {"success": False, "error": "Table non trouvée"}  
          
        # 2. Récupérer les colonnes  
        columns = self.get_table_columns(table_guid)  
        logger.info(f"📋 Colonnes trouvées: {[col['name'] for col in columns]}")  
          
        # 3. Créer le mapping colonnes → termes  
        column_term_mapping = self.create_column_term_mapping()  
          
        # 4. Assigner les termes aux colonnes  
        assigned_count = 0  
        for column in columns:  
            column_name = column['name']  
              
            if column_name in column_term_mapping:  
                term_name = column_term_mapping[column_name]  
                  
                # Récupérer le GUID du terme  
                term_guid = self.get_glossary_term_guid(term_name)  
                  
                if term_guid:  
                    success = self.assign_glossary_term_to_column(column['guid'], term_guid)  
                      
                    if success:  
                        logger.info(f"✅ Terme '{term_name}' assigné à colonne '{column_name}'")  
                        assigned_count += 1  
                    else:  
                        logger.error(f"❌ Échec assignation pour '{column_name}'")  
                else:  
                    logger.warning(f"⚠️  Terme '{term_name}' non trouvé dans glossaire")  
            else:  
                logger.info(f"ℹ️  Pas de mapping pour colonne '{column_name}'")  
          
        return {  
            "success": True,  
            "table_guid": table_guid,  
            "columns_processed": len(columns),  
            "terms_assigned": assigned_count  
        }  
          
     except Exception as e:  
        logger.error(f"❌ Erreur assignation: {str(e)}")  
        return {"success": False, "error": str(e)}
    



    def create_validated_metadata_terms(self, glossary_guid, category_guids):
        """Créer les termes du glossaire à partir des métadonnées validées"""
        enriched_metadata = self.metadata_db['enriched_metadata']
        
        # Récupérer uniquement les métadonnées validées
        validated_metadata = list(enriched_metadata.find({"validation_status": "validated"}))
        logger.info(f"Métadonnées validées à synchroniser: {len(validated_metadata)}")
        
        synced_terms = 0
        
        for metadata in validated_metadata:
            column_name = metadata['column_name']
            job_id = metadata['job_id']
            
            # Créer un nom de terme unique et métier
            term_name = f"{column_name.upper()}_TERM"
            qualified_name = f"datagovernance.{column_name}_{job_id}@production"
            
            # Préparer les attributs métier
            attributes = {
                "source_column": column_name,
                "source_dataset": job_id,
                "entity_types": metadata.get('entity_types', []),
                "total_entities": metadata.get('total_entities', 0),
                "sensitivity_level": metadata.get('recommended_sensitivity_level'),
                "rgpd_category": metadata.get('recommended_rgpd_category'),
                "ranger_policy": metadata.get('recommended_ranger_policy'),
                "validation_date": datetime.now().isoformat(),
                "data_quality_score": self._calculate_data_quality_score(metadata),
                "business_owner": "Data Steward",
                "technical_owner": "Data Engineering Team"
            }
            
            # Obtenir la catégorie RGPD
            rgpd_category = metadata.get('recommended_rgpd_category')
            category_guid = category_guids.get(rgpd_category)
            
            # Préparer les classifications
            sensitivity_level = metadata.get('recommended_sensitivity_level')
            classifications = []
            if sensitivity_level:
                classifications.append({
                    "typeName": f"DataSensitivity_{sensitivity_level}",
                    "attributes": {
                        "sensitivity_level": sensitivity_level,
                        "rgpd_compliant": True,
                        "data_steward": "Validated",
                        "validation_date": datetime.now().isoformat()
                    }
                })
            
            term_data = {
                "name": term_name,
                "qualifiedName": qualified_name,
                "shortDescription": f"Attribut métier validé: {column_name}",
                "longDescription": self._generate_business_description(metadata),
                "anchor": {"glossaryGuid": glossary_guid}
               
            }
            
            # Ajouter la catégorie RGPD si disponible
            if category_guid:
                term_data["categories"] = [{"categoryGuid": category_guid}]
            
            # Créer le terme dans Atlas
            response = requests.post(
                f"{self.atlas_url}/api/atlas/v2/glossary/term",
                json=term_data,
                auth=self.auth
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Terme métier synchronisé: {term_name}")
                synced_terms += 1
            else:
                logger.error(f"✗ Erreur terme {term_name}: {response.text}")
        
        return synced_terms

    def _calculate_data_quality_score(self, metadata):
        """Calculer un score de qualité des données"""
        score = 0
        
        # Présence d'entités détectées
        if metadata.get('total_entities', 0) > 0:
            score += 30
        
        # Diversité des types d'entités
        entity_types = metadata.get('entity_types', [])
        if len(entity_types) > 0:
            score += 20
        
        # Validation par data steward
        if metadata.get('validation_status') == 'validated':
            score += 40
        
        # Présence d'échantillons
        if metadata.get('sample_values') and len(metadata.get('sample_values', [])) > 0:
            score += 10
        
        return min(score, 100)

    def _generate_business_description(self, metadata):
        """Générer une description métier riche"""
        column_name = metadata['column_name']
        entity_types = metadata.get('entity_types', [])
        sensitivity = metadata.get('recommended_sensitivity_level', 'INTERNAL')
        rgpd_category = metadata.get('recommended_rgpd_category', 'Non classifié')
        total_entities = metadata.get('total_entities', 0)
        
        description = f"""
ATTRIBUT MÉTIER VALIDÉ: {column_name.upper()}

🔍 ANALYSE AUTOMATIQUE:
• Types d'entités détectées: {', '.join(entity_types) if entity_types else 'Aucune entité spécifique'}
• Nombre total d'entités: {total_entities}
• Niveau de sensibilité: {sensitivity}

📋 CLASSIFICATION RGPD:
• Catégorie: {rgpd_category}
• Politique Ranger recommandée: {metadata.get('recommended_ranger_policy', 'Non définie')}

✅ VALIDATION:
• Statut: Validé par Data Steward
• Date de validation: {datetime.now().strftime('%Y-%m-%d')}

📊 ÉCHANTILLONS:
{self._format_sample_values(metadata.get('sample_values', []))}
        """.strip()
        
        return description

    def _format_sample_values(self, sample_values):
        """Formater les valeurs d'échantillon pour la description"""
        if not sample_values:
            return "• Aucun échantillon disponible"
        
        formatted_samples = []
        for i, sample in enumerate(sample_values[:3], 1):  # Limiter à 3 échantillons
            # Masquer partiellement pour la confidentialité
            if len(sample) > 10:
                masked_sample = sample[:3] + "***" + sample[-2:]
            else:
                masked_sample = sample[:2] + "***"
            formatted_samples.append(f"• Échantillon {i}: {masked_sample}")
        
        return '\n'.join(formatted_samples)

    def create_data_lineage_entities(self):
        """Créer des entités pour la traçabilité des données"""
        # Cette fonction pourrait être étendue pour créer des entités Atlas
        # représentant les datasets, tables, et leurs relations
        logger.info("🔗 Fonction de lignage des données - À implémenter selon vos besoins")
        
    def sync_governance_metadata(self, preview_only=False):
        """Fonction principale de synchronisation pour la gouvernance"""
        logger.info("🚀 Début de la synchronisation métadonnées gouvernance")
        
        try:
            # Prévisualisation obligatoire
            preview = self.preview_sync_data()
            
            if preview_only:
                logger.info("👁️  Mode prévisualisation uniquement - Aucune modification dans Atlas")
                return {"success": True, "preview": preview, "sync_executed": False}
            
            if not preview["will_sync"]:
                logger.warning("🛑 Arrêt: Aucune métadonnée validée à synchroniser")
                return {"success": False, "error": "Aucune métadonnée validée", "preview": preview}
            
            # Demander confirmation
            if not self._confirm_sync(preview):
                logger.info("🛑 Synchronisation annulée par l'utilisateur")
                return {"success": False, "error": "Annulée par l'utilisateur", "preview": preview}
            # Demander confirmation
            if not self._confirm_sync(preview):
                logger.info("🛑 Synchronisation annulée par l'utilisateur")
                return {"success": False, "error": "Annulée par l'utilisateur", "preview": preview}
            
            # Sauvegarder l'état actuel d'Atlas (optionnel)
            self._backup_atlas_state()
            
            logger.info("▶️  EXÉCUTION DE LA SYNCHRONISATION...")
            
            # 1. Créer les classifications de sensibilité
            if not self.create_sensitivity_classifications():
                logger.error("Échec création classifications")
                return False
            
            # 2. Créer le glossaire métier
            glossary_guid = self.create_business_glossary()
            if not glossary_guid:
                logger.error("Échec création glossaire")
                return False
            
            # 3. Créer les catégories RGPD basées sur les données réelles
            category_guids = self.create_rgpd_categories(glossary_guid)
            
            # 4. Synchroniser uniquement les métadonnées validées
            synced_terms = self.create_validated_metadata_terms(glossary_guid, category_guids)
            
            # 5. Créer la lignage (optionnel)
            self.create_data_lineage_entities()
            # 6. Assigner automatiquement les termes aux colonnes Hive  
            assignment_result = self.automate_hive_glossary_assignment("entites_marocaines")  
   
            result = {  
             "success": True,  
             "glossary_guid": glossary_guid,  
             "validated_terms_synced": synced_terms,  
             "categories_created": len(category_guids),  
             "sync_timestamp": datetime.now().isoformat(),  
             "preview": preview  
            }  
  





  
            result.update({  
              "hive_assignment": assignment_result  
            })
            
            # Marquer les métadonnées comme synchronisées
            self._mark_as_synced(synced_terms)
            
            result = {
                "success": True,
                "glossary_guid": glossary_guid,
                "validated_terms_synced": synced_terms,
                "categories_created": len(category_guids),
                "sync_timestamp": datetime.now().isoformat(),
                "preview": preview
            }
            
            logger.info(f"✅ Synchronisation terminée avec succès: {synced_terms} termes validés")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation: {str(e)}")
            return {"success": False, "error": str(e)}
        
        finally:
            self.mongo_client.close()

    def _confirm_sync(self, preview):
        """Demander confirmation avant synchronisation"""
        print("\n" + "="*60)
        print("⚠️  CONFIRMATION REQUISE")
        print("="*60)
        print(f"Vous allez synchroniser {preview['validated_metadata']} métadonnées validées vers Atlas")
        print(f"Catégories RGPD: {', '.join(preview['rgpd_categories'])}")
        print(f"Niveaux de sensibilité: {', '.join(preview['sensitivity_levels'])}")
        print("\n⚠️  Cette opération va créer/modifier des éléments dans Apache Atlas")
        
        response = input("\n🤔 Continuer la synchronisation? (oui/non): ").lower().strip()
        return response in ['oui', 'o', 'yes', 'y']
    
    def _backup_atlas_state(self):
        """Sauvegarder l'état actuel d'Atlas (optionnel)"""
        logger.info("💾 Sauvegarde de l'état Atlas (recommandé)")
        # Implémentation optionnelle pour sauvegarder l'état actuel
        pass
    
    def _mark_as_synced(self, synced_count):
        """Marquer les métadonnées synchronisées"""
        if synced_count > 0:
            enriched_metadata = self.metadata_db['enriched_metadata']
            enriched_metadata.update_many(
                {"validation_status": "validated"},
                {"$set": {"atlas_sync_status": "synced", "atlas_sync_date": datetime.now()}}
            )
            logger.info(f"📝 {synced_count} métadonnées marquées comme synchronisées")

def main():
    """Point d'entrée principal"""
    governance = AtlasMetadataGovernance()
    result = governance.sync_governance_metadata()
    
    print("\n" + "="*60)
    print("RÉSULTAT DE LA SYNCHRONISATION GOUVERNANCE")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("success"):
        print(f"\n✅ Synchronisation réussie!")
        print(f"📚 Termes validés synchronisés: {result.get('validated_terms_synced', 0)}")
        print(f"📂 Catégories RGPD créées: {result.get('categories_created', 0)}")
        print(f"🆔 GUID du glossaire: {result.get('glossary_guid')}")
    else:
        print(f"\n❌ Échec de la synchronisation: {result.get('error')}")

if __name__ == "__main__":
    main()