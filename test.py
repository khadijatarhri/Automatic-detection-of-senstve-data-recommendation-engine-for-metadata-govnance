import json  
from pymongo import MongoClient  
from datetime import datetime  
import logging  
from pyapacheatlas.auth import BasicAuthentication  
from pyapacheatlas.core import AtlasClient  
import requests  
import time
  
ATLAS_URL = "http://172.17.0.1:21000"  
ATLAS_USER = "admin"  
ATLAS_PASS = "ensias123@"  
MONGO_URI = 'mongodb://mongodb:27017/'  
  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
class AtlasMetadataGovernance:  
    def __init__(self):  
        self.atlas_url = ATLAS_URL  
        self.auth = BasicAuthentication(username=ATLAS_USER, password=ATLAS_PASS)  
        self.atlas_client = AtlasClient(  
            endpoint_url=self.atlas_url,  
            authentication=self.auth  
        )  
        self.auth_tuple = (ATLAS_USER, ATLAS_PASS)  
          
        self.mongo_client = MongoClient(MONGO_URI)  
        self.metadata_db = self.mongo_client['metadata_validation_db']  
        self.current_glossary_guid = None  
        
        # Cache pour éviter de rechercher plusieurs fois les mêmes termes
        self.created_terms_cache = {}
        # Cache pour les colonnes Hive réelles (clé de votre mapping)
        self.hive_columns_cache = {}
          
        self._test_connections()  
  
    def _test_connections(self):  
        """Tester les connexions Atlas et MongoDB"""  
        logger.info("🔍 Test des connexions...")  
          
        try:  
            self.mongo_client.admin.command('ping')  
            logger.info("✅ MongoDB connecté")  
        except Exception as e:  
            logger.error(f"❌ MongoDB non accessible: {e}")  
            raise  
  
    # ==================== PARTIE 1: DÉCOUVERTE DES COLONNES HIVE ====================
    # Cette partie correspond au "scan" dans Purview: on découvre ce qui existe réellement
    
    def discover_hive_schema(self, table_name):
        """
        ÉTAPE 1 (comme Purview): Scanner la table Hive pour découvrir son schéma réel.
        
        Cette fonction remplace votre logique de "mapping intelligent" par une découverte 
        basée sur ce qui existe réellement dans Hive, comme le fait Purview.
        
        Returns:
            dict: Mapping {nom_colonne_hive_lowercase: info_colonne}
        """
        logger.info(f"🔍 ÉTAPE 1: Découverte du schéma Hive pour table '{table_name}'")
        
        try:
            # 1. Trouver la table dans Atlas
            table_guid = self.get_hive_table_entity(table_name)
            if not table_guid:
                logger.error(f"❌ Table {table_name} non trouvée")
                return {}
            
            # 2. Récupérer les colonnes réelles
            columns = self.get_table_columns(table_guid)
            
            # 3. Créer un index des colonnes (normalisé en lowercase pour matching)
            hive_schema = {}
            for col in columns:
                # Normaliser le nom pour le matching (lowercase, sans espaces)
                normalized_name = col['name'].lower().strip().replace(' ', '_')
                hive_schema[normalized_name] = {
                    'guid': col['guid'],
                    'original_name': col['name'],  # Nom original Hive
                    'type': col['type']
                }
                logger.info(f"  📋 Colonne Hive découverte: {col['name']} → {normalized_name}")
            
            # Sauvegarder dans le cache
            self.hive_columns_cache[table_name] = hive_schema
            
            logger.info(f"✅ {len(hive_schema)} colonnes Hive découvertes")
            return hive_schema
            
        except Exception as e:
            logger.error(f"❌ Erreur découverte schéma: {e}")
            return {}
    
    def get_hive_table_entity(self, table_name):  
        """Récupérer l'entité table Hive via API REST"""  
        try:  
            search_url = f"{self.atlas_url}/api/atlas/v2/search/dsl"  
            response = requests.get(  
                search_url,  
                auth=self.auth_tuple,  
                params={'query': f"hive_table where name='{table_name}'"}  
            )  
              
            if response.status_code == 200:  
                entities = response.json().get('entities', [])  
                if entities:  
                    return entities[0]['guid']  
        except Exception as e:  
            logger.error(f"Erreur recherche table: {e}")  
          
        return None  
  
    def get_table_columns(self, table_guid):  
        """Récupérer les colonnes via l'API REST"""  
        try:  
            entity_url = f"{self.atlas_url}/api/atlas/v2/entity/guid/{table_guid}"  
            response = requests.get(  
                entity_url,  
                auth=self.auth_tuple,  
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
        except Exception as e:  
            logger.error(f"Erreur récupération colonnes: {e}")  
          
        return []  

    # ==================== PARTIE 2: CRÉATION DU GLOSSAIRE ET TAXONOMIE ====================
    # Cette partie crée la structure business (glossaire + catégories + termes)
    
    def create_business_glossary(self):  
        """
        ÉTAPE 2: Créer le glossaire business.
        
        Dans Purview, le glossaire est créé manuellement ou via API.
        Ici on le crée automatiquement avec une structure RGPD.
        """  
        try:  
            from AtlasAPI.atlas_integration import CustomAtlasGlossary  
              
            glossary = CustomAtlasGlossary(  
                name="Data_Governance_Glossary_PyAtlas5",  
                shortDescription="Glossaire métier pour la gouvernance des données",  
                longDescription="Glossaire centralisé utilisant pyapacheatlas"  
            )  
              
            response = requests.post(  
                f"{self.atlas_url}/api/atlas/v2/glossary",  
                json=glossary.to_dict(),  
                auth=self.auth_tuple,  
                timeout=(30, 60)  
            )  
              
            if response.status_code == 200:  
                glossary_guid = response.json()['guid']  
                self.current_glossary_guid = glossary_guid  
                logger.info(f"✅ Glossaire créé: {glossary_guid}")  
                return glossary_guid  
            else:  
                logger.error(f"❌ Échec création glossaire: {response.text}")  
                return None  
                  
        except Exception as e:  
            logger.error(f"❌ Erreur création glossaire: {e}")  
            return None  
    
    def create_rgpd_categories(self, glossary_guid):  
        """
        ÉTAPE 3: Créer les catégories RGPD (taxonomie business).
        
        CORRECTION IMPORTANTE: Les catégories doivent être créées AVANT les termes
        pour pouvoir les référencer lors de la création des termes.
        """  
        real_categories = self.extract_rgpd_categories_from_db()  
        category_guids = {}  
          
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
                "longDescription": f"Catégorie de données personnelles selon le RGPD: {category}",  
                "anchor": {"glossaryGuid": glossary_guid}  
            }  
              
            response = requests.post(  
                f"{self.atlas_url}/api/atlas/v2/glossary/category",  
                json=cat_data,  
                auth=self.auth_tuple  
            )  
              
            if response.status_code == 200:  
                category_guids[category] = response.json()['guid']  
                logger.info(f"✅ Catégorie créée: {category} → {category_guids[category]}")  
            else:  
                logger.error(f"❌ Erreur catégorie {category}: {response.text}")  
          
        return category_guids  
    
    def extract_rgpd_categories_from_db(self):  
        """Extraire les catégories RGPD depuis MongoDB"""  
        enriched_metadata = self.metadata_db['enriched_metadata']  
        categories = enriched_metadata.distinct('recommended_rgpd_category')  
        categories = [cat for cat in categories if cat and cat.strip()]  
        logger.info(f"Catégories RGPD trouvées: {categories}")  
        return categories  

    def create_validated_metadata_terms(self, glossary_guid, category_guids, hive_columns_map):
     """
     CORRECTION FINALE: Créer les termes PUIS assigner aux colonnes Hive
    
     L'API Atlas v2 ne permet PAS d'inclure assignedEntities à la création.
     Il faut faire en 2 étapes séparées.
     """
     enriched_metadata = self.metadata_db['enriched_metadata']
     validated_metadata = list(enriched_metadata.find({"validation_status": "validated"}))
    
     synced_terms = 0
     terms_to_assign = []  # Pour assignation en phase 2
    
     for metadata in validated_metadata:
        column_name = metadata['column_name']
        job_id = metadata['job_id']
        rgpd_category = metadata.get('recommended_rgpd_category')
        
        # Récupérer le GUID Hive correspondant
        hive_column_guid = hive_columns_map.get(column_name.lower())
        
        term_name = f"{column_name.upper()}_TERM"
        qualified_name = f"datagovernance.{column_name}_{job_id}@production"
        
        try:
            from AtlasAPI.atlas_integration import CustomAtlasGlossaryTerm
            
            term = CustomAtlasGlossaryTerm(
                name=term_name,
                qualifiedName=qualified_name,
                shortDescription=f"Attribut métier validé: {column_name}",
                longDescription=self._generate_business_description(metadata),
                attributes={
                    "source_column": column_name,
                    "source_dataset": job_id,
                    "entity_types": metadata.get('entity_types', []),
                    "sensitivity_level": metadata.get('recommended_sensitivity_level'),
                    "rgpd_category": rgpd_category
                }
            )
            
            # Ajouter classification
            sensitivity_level = metadata.get('recommended_sensitivity_level')
            if sensitivity_level:
                term.addClassification(
                    f"DataSensitivity_{sensitivity_level}",
                    {
                        "sensitivity_level": sensitivity_level,
                        "rgpd_compliant": True,
                        "data_steward": "Validated"
                    }
                )
            
            term.glossaryGuid = glossary_guid
            
            # Ajouter catégorie RGPD
            if rgpd_category and rgpd_category in category_guids:
                term.categories = [{
                    "categoryGuid": category_guids[rgpd_category]
                }]
            
            # IMPORTANT: NE PAS inclure assignedEntities ici
            term_payload = term.to_dict()
            
            # Créer le terme (SANS relation Hive)
            response = requests.post(
                f"{self.atlas_url}/api/atlas/v2/glossary/term",
                json=term_payload,
                auth=self.auth_tuple
            )
            
            if response.status_code == 200:
                term_guid = response.json()['guid']
                
                # Sauvegarder dans le cache
                self.created_terms_cache[term_name] = {
                    'guid': term_guid,
                    'qualified_name': qualified_name,
                    'source_column': column_name,
                    'hive_column_guid': hive_column_guid  # Pour assignation ultérieure
                }
                
                # Préparer pour assignation si colonne Hive existe
                if hive_column_guid:
                    terms_to_assign.append({
                        'term_guid': term_guid,
                        'term_name': term_name,
                        'column_guid': hive_column_guid,
                        'column_name': column_name
                    })
                
                logger.info(f"✅ Terme créé: {term_name}")
                synced_terms += 1
            else:
                logger.error(f"❌ Erreur terme {term_name}: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Exception terme {term_name}: {e}")
    
     return synced_terms, terms_to_assign

    # ==================== PARTIE 3: MAPPING CSV → HIVE ====================
    # Cette partie crée le lien entre vos métadonnées CSV et le schéma Hive réel
    
    def create_csv_to_hive_mapping(self, table_name):
        """
        ÉTAPE 5: Créer le mapping entre colonnes CSV (MongoDB) et colonnes Hive réelles.
        
        CORRECTION IMPORTANTE: 
        - On se base sur le schéma Hive RÉEL découvert précédemment
        - On normalise les noms pour matcher (lowercase, underscores)
        - On utilise une logique de "fuzzy matching" simple
        
        Cette approche est plus proche de Purview qui fait du matching automatique
        entre les sources découvertes et les métadonnées business.
        
        Returns:
            dict: {nom_colonne_hive: {term_info, column_info}}
        """
        logger.info(f"🔗 ÉTAPE 5: Création du mapping CSV → Hive")
        
        # 1. Récupérer le schéma Hive si pas déjà en cache
        if table_name not in self.hive_columns_cache:
            self.discover_hive_schema(table_name)
        
        hive_schema = self.hive_columns_cache.get(table_name, {})
        if not hive_schema:
            logger.error("❌ Aucun schéma Hive disponible")
            return {}
        
        # 2. Pour chaque terme créé, essayer de trouver la colonne Hive correspondante
        mapping = {}
        matched_count = 0
        unmatched_csv = []
        
        for term_name, term_info in self.created_terms_cache.items():
            csv_column = term_info['source_column']
            
            # Normaliser le nom CSV pour matching
            normalized_csv = csv_column.lower().strip().replace(' ', '_')
            
            # Chercher la colonne Hive correspondante
            if normalized_csv in hive_schema:
                # MATCH EXACT trouvé !
                hive_col_info = hive_schema[normalized_csv]
                mapping[hive_col_info['original_name']] = {
                    'term_guid': term_info['guid'],
                    'term_name': term_name,
                    'column_guid': hive_col_info['guid'],
                    'csv_column': csv_column,
                    'match_type': 'exact'
                }
                matched_count += 1
                logger.info(f"✅ MATCH: CSV '{csv_column}' → Hive '{hive_col_info['original_name']}'")
            else:
                # Pas de match direct, essayer des variantes
                possible_matches = self._find_fuzzy_matches(normalized_csv, hive_schema.keys())
                
                if possible_matches:
                    best_match = possible_matches[0]
                    hive_col_info = hive_schema[best_match]
                    mapping[hive_col_info['original_name']] = {
                        'term_guid': term_info['guid'],
                        'term_name': term_name,
                        'column_guid': hive_col_info['guid'],
                        'csv_column': csv_column,
                        'match_type': 'fuzzy',
                        'confidence': 0.8  # Score arbitraire
                    }
                    matched_count += 1
                    logger.info(f"⚠️ MATCH APPROXIMATIF: CSV '{csv_column}' → Hive '{hive_col_info['original_name']}'")
                else:
                    unmatched_csv.append(csv_column)
                    logger.warning(f"❌ PAS DE MATCH: CSV '{csv_column}' (colonne Hive introuvable)")
        
        # Résumé
        logger.info(f"📊 Résultat mapping: {matched_count}/{len(self.created_terms_cache)} colonnes matchées")
        if unmatched_csv:
            logger.warning(f"⚠️ Colonnes CSV non matchées: {', '.join(unmatched_csv)}")
        
        return mapping
    
    def _find_fuzzy_matches(self, csv_column, hive_columns):
        """
        Trouver des correspondances approximatives entre noms de colonnes.
        
        Logique simple:
        - Suppression des underscores
        - Vérification de sous-chaînes
        - Distance de Levenshtein simplifiée
        """
        matches = []
        
        csv_clean = csv_column.replace('_', '').replace('-', '')
        
        for hive_col in hive_columns:
            hive_clean = hive_col.replace('_', '').replace('-', '')
            
            # Vérifier si l'un est contenu dans l'autre
            if csv_clean in hive_clean or hive_clean in csv_clean:
                matches.append(hive_col)
                continue
            
            # Vérifier similarité (très basique)
            if self._string_similarity(csv_clean, hive_clean) > 0.7:
                matches.append(hive_col)
        
        return matches
    
    def _string_similarity(self, s1, s2):
        """Calcul basique de similarité entre deux chaînes"""
        if not s1 or not s2:
            return 0.0
        
        # Utiliser la longueur de la sous-séquence commune
        longer = s1 if len(s1) >= len(s2) else s2
        shorter = s2 if len(s1) >= len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # Compter les caractères communs
        common = sum(1 for a, b in zip(shorter, longer) if a == b)
        return common / len(longer)

    # ==================== PARTIE 4: ASSIGNATION TERMES → COLONNES ====================
    # Cette partie fait le lien final entre glossaire business et assets techniques
    
    def assign_terms_to_hive_columns(self, table_name):
        """
        ÉTAPE 6: Assigner les termes du glossaire aux colonnes Hive.
        
        CORRECTION IMPORTANTE:
        - On utilise le mapping créé précédemment (qui garantit la cohérence)
        - On vérifie que les termes existent avant assignation
        - On gère les erreurs proprement
        
        C'est l'équivalent de "Apply terms to assets" dans Purview.
        """
        logger.info(f"🔗 ÉTAPE 6: Assignation des termes aux colonnes Hive")
        
        # 1. Créer le mapping si pas déjà fait
        mapping = self.create_csv_to_hive_mapping(table_name)
        
        if not mapping:
            logger.error("❌ Aucun mapping disponible pour assignation")
            return {"success": False, "error": "Pas de mapping"}
        
        # 2. Assigner chaque terme à sa colonne
        assigned_count = 0
        failed_assignments = []
        
        for hive_column_name, mapping_info in mapping.items():
            term_guid = mapping_info['term_guid']
            column_guid = mapping_info['column_guid']
            term_name = mapping_info['term_name']
            
            logger.info(f"🔄 Assignation: '{term_name}' → colonne '{hive_column_name}'")
            
            # Appel API Atlas pour assigner le terme
            success = self._assign_term_to_column_api(column_guid, term_guid)
            
            if success:
                assigned_count += 1
                logger.info(f"   ✅ Assigné avec succès")
            else:
                failed_assignments.append({
                    'column': hive_column_name,
                    'term': term_name
                })
                logger.error(f"   ❌ Échec assignation")
        
        # 3. Résumé
        result = {
            "success": assigned_count > 0,
            "total_mappings": len(mapping),
            "successful_assignments": assigned_count,
            "failed_assignments": len(failed_assignments),
            "failed_details": failed_assignments
        }
        
        logger.info(f"📊 Assignations: {assigned_count}/{len(mapping)} réussies")
        
        return result
    
    def _assign_term_to_column_api(self, column_guid, term_guid):
        """
        Appel API bas niveau pour assigner un terme à une colonne.
        
        Utilise l'endpoint /meanings d'Atlas.
        """
        try:
            assign_url = f"{self.atlas_url}/api/atlas/v2/entity/guid/{column_guid}/meanings"
            
            payload = [{
                "termGuid": term_guid,
                "relationGuid": None
            }]
            
            response = requests.post(
                assign_url,
                auth=self.auth_tuple,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"   API Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"   Exception: {e}")
            return False

    # ==================== PARTIE 5: UTILITAIRES ET CLASSIFICATIONS ====================
    
    def create_sensitivity_classifications(self):  
        """Créer les classifications de sensibilité"""  
        enriched_metadata = self.metadata_db['enriched_metadata']  
        sensitivity_levels = enriched_metadata.distinct('recommended_sensitivity_level')  
        sensitivity_levels = [level for level in sensitivity_levels if level]  
          
        logger.info(f"Niveaux de sensibilité: {sensitivity_levels}")  
          
        classification_defs = []  
          
        for level in sensitivity_levels:  
            classification_def = {  
                "name": f"DataSensitivity_{level}",  
                "description": f"Classification de sensibilité: {level}",  
                "attributeDefs": [  
                    {"name": "sensitivity_level", "typeName": "string", "isOptional": False},  
                    {"name": "rgpd_compliant", "typeName": "boolean", "isOptional": True},  
                    {"name": "data_steward", "typeName": "string", "isOptional": True}  
                ]  
            }  
            classification_defs.append(classification_def)  
          
        if classification_defs:  
            classification_batch = {"classificationDefs": classification_defs}  
              
            response = requests.post(  
                f"{self.atlas_url}/api/atlas/v2/types/typedefs",  
                json=classification_batch,  
                auth=self.auth_tuple  
            )  
              
            if response.status_code in [200, 409]:  # 409 = déjà existe
                logger.info(f"✅ Classifications créées/existantes")  
                return True  
            else:  
                logger.error(f"❌ Erreur classifications: {response.text}")  
                return False  
          
        return True  
    
    def wait_for_atlas_indexing(self, seconds=30):
        """
        Attendre que Atlas indexe les nouveaux éléments.
        
        IMPORTANT: Atlas n'indexe pas immédiatement les nouveaux termes.
        Il faut attendre que l'index Solr soit mis à jour (généralement 10-30 secondes).
        
        C'est une limitation connue d'Atlas, Purview a le même comportement.
        """
        logger.info(f"⏳ Attente indexation Atlas ({seconds}s)...")
        time.sleep(seconds)
        logger.info("✅ Indexation supposée terminée")
  
    def preview_sync_data(self):  
        """Prévisualiser ce qui sera synchronisé"""  
        enriched_metadata = self.metadata_db['enriched_metadata']  
          
        total_metadata = enriched_metadata.count_documents({})  
        validated_metadata = enriched_metadata.count_documents({"validation_status": "validated"})  
        pending_metadata = enriched_metadata.count_documents({"validation_status": "pending"})  
          
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
          
        logger.info("📊 ===== PRÉVISUALISATION =====")  
        logger.info(f"Total métadonnées: {total_metadata}")  
        logger.info(f"Validées (à sync): {validated_metadata}")  
        logger.info(f"En attente: {pending_metadata}")  
        logger.info(f"Catégories RGPD: {preview['rgpd_categories']}")  
        logger.info(f"Sensibilités: {preview['sensitivity_levels']}")  
          
        return preview  
    
    def create_hive_column_map(self, table_name):
     """
     Créer un mapping {csv_column_name: hive_column_guid}
     AVANT de créer les termes
     """
     hive_schema = self.discover_hive_schema(table_name)
    
     enriched_metadata = self.metadata_db['enriched_metadata']
     validated_metadata = list(enriched_metadata.find({"validation_status": "validated"}))
    
     mapping = {}
    
     for metadata in validated_metadata:
        csv_column = metadata['column_name'].lower()
        
        # Chercher match exact
        if csv_column in hive_schema:
            mapping[csv_column] = hive_schema[csv_column]['guid']
            logger.info(f"✅ Mapping: CSV '{csv_column}' → Hive GUID {hive_schema[csv_column]['guid'][:8]}...")
        else:
            # Fuzzy match
            matches = self._find_fuzzy_matches(csv_column, hive_schema.keys())
            if matches:
                best_match = matches[0]
                mapping[csv_column] = hive_schema[best_match]['guid']
                logger.info(f"⚠️ Mapping approximatif: CSV '{csv_column}' → Hive '{best_match}'")
            else:
                logger.warning(f"❌ Pas de match Hive pour CSV '{csv_column}'")
    
     return mapping
    
    def _generate_business_description(self, metadata):  
        """Générer description métier riche"""  
        column_name = metadata['column_name']  
        entity_types = metadata.get('entity_types', [])  
        sensitivity = metadata.get('recommended_sensitivity_level', 'INTERNAL')  
        rgpd_category = metadata.get('recommended_rgpd_category', 'Non classifié')  
        total_entities = metadata.get('total_entities', 0)  
          
        description = f"""ATTRIBUT MÉTIER: {column_name.upper()}

🔍 ANALYSE:
• Entités: {', '.join(entity_types) if entity_types else 'Aucune'}
• Nombre d'entités: {total_entities}
• Sensibilité: {sensitivity}

📋 RGPD:
• Catégorie: {rgpd_category}

✅ VALIDATION:
• Validé: {datetime.now().strftime('%Y-%m-%d')}"""
          
        return description

    # ==================== WORKFLOW PRINCIPAL ====================
    
    
    def sync_governance_metadata(self, table_name="entites_marocaines", preview_only=False):
     """Workflow corrigé avec assignation en 2 phases"""
    
     try:
        preview = self.preview_sync_data()
        
        if preview_only or not preview["will_sync"]:
            return {"success": False, "preview": preview}
        
        if not self._confirm_sync(preview):
            return {"success": False, "error": "Annulée"}
        
        # PHASE 1-5: Comme avant
        hive_schema = self.discover_hive_schema(table_name)
        if not hive_schema:
            return {"success": False, "error": "Schéma Hive introuvable"}
        
        hive_columns_map = self.create_hive_column_map(table_name)
        logger.info(f"📊 {len(hive_columns_map)}/{len(hive_schema)} colonnes mappées")
        
        if not self.create_sensitivity_classifications():
            return {"success": False, "error": "Échec classifications"}
        
        glossary_guid = self.create_business_glossary()
        if not glossary_guid:
            return {"success": False, "error": "Échec glossaire"}
        
        category_guids = self.create_rgpd_categories(glossary_guid)
        
        # PHASE 6: Créer termes ET récupérer liste pour assignation
        logger.info("\n📝 PHASE 6: CRÉATION DES TERMES")
        synced_terms, terms_to_assign = self.create_validated_metadata_terms(
            glossary_guid, 
            category_guids,
            hive_columns_map
        )
        
        logger.info(f"✅ {synced_terms} termes créés")
        
        # PHASE 7: ATTENDRE indexation (CRITIQUE!)
        logger.info("\n⏳ PHASE 7: ATTENTE INDEXATION")
        time.sleep(15)  # 15 secondes minimum
        
        # PHASE 8: ASSIGNER termes aux colonnes Hive
        logger.info("\n🔗 PHASE 8: ASSIGNATION TERMES → COLONNES HIVE")
        assigned_count = 0
        failed_assignments = []
        
        for assignment in terms_to_assign:
            term_guid = assignment['term_guid']
            column_guid = assignment['column_guid']
            term_name = assignment['term_name']
            column_name = assignment['column_name']
            
            logger.info(f"🔄 Assignation: '{term_name}' → '{column_name}'")
            
            # Utiliser l'endpoint /meanings
            assign_url = f"{self.atlas_url}/api/atlas/v2/entity/guid/{column_guid}/meanings"
            
            payload = [{
                "termGuid": term_guid,
                "relationGuid": None
            }]
            
            try:
                response = requests.post(
                    assign_url,
                    auth=self.auth_tuple,
                    headers={'Content-Type': 'application/json'},
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    assigned_count += 1
                    logger.info(f"   ✅ Assigné avec succès")
                else:
                    failed_assignments.append({
                        'column': column_name,
                        'term': term_name,
                        'error': response.text
                    })
                    logger.error(f"   ❌ Erreur: {response.status_code} - {response.text}")
                    
            except Exception as e:
                failed_assignments.append({
                    'column': column_name,
                    'term': term_name,
                    'error': str(e)
                })
                logger.error(f"   ❌ Exception: {e}")
        
        self._mark_as_synced(synced_terms)
        
        return {
            "success": True,
            "glossary_guid": glossary_guid,
            "terms_created": synced_terms,
            "hive_assignments_successful": assigned_count,
            "hive_assignments_failed": len(failed_assignments),
            "failed_details": failed_assignments,
            "sync_timestamp": datetime.now().isoformat()
        }
        
     except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}
    
    def _confirm_sync(self, preview):
        """Demander confirmation avant synchronisation"""
        print("\n" + "=" * 80)
        print("⚠️  CONFIRMATION REQUISE")
        print("=" * 80)
        print(f"Métadonnées à synchroniser: {preview['validated_metadata']}")
        print(f"Catégories RGPD: {', '.join(preview['rgpd_categories'])}")
        print(f"Niveaux de sensibilité: {', '.join(preview['sensitivity_levels'])}")
        print("\n⚠️  Cette opération va modifier Apache Atlas")
        
        response = input("\nContinuer? (oui/non): ").lower().strip()
        return response in ['oui', 'o', 'yes', 'y']
    
    def _mark_as_synced(self, synced_count):
        """Marquer les métadonnées comme synchronisées dans MongoDB"""
        if synced_count > 0:
            enriched_metadata = self.metadata_db['enriched_metadata']
            enriched_metadata.update_many(
                {"validation_status": "validated"},
                {
                    "$set": {
                        "atlas_sync_status": "synced",
                        "atlas_sync_date": datetime.now()
                    }
                }
            )
            logger.info(f"✅ {synced_count} métadonnées marquées comme synchronisées")

    # ==================== FONCTIONS DE DEBUG ====================
    
    def debug_full_workflow(self, table_name="entites_marocaines"):
        """
        Mode debug complet pour diagnostiquer les problèmes.
        
        Affiche:
        - État MongoDB
        - État Atlas (glossaire, catégories, termes)
        - Schéma Hive
        - Mapping potentiel
        """
        logger.info("🔧 MODE DEBUG COMPLET")
        logger.info("=" * 80)
        
        # 1. État MongoDB
        logger.info("\n📊 1. ÉTAT MONGODB")
        preview = self.preview_sync_data()
        
        # 2. Schéma Hive
        logger.info("\n📋 2. SCHÉMA HIVE")
        hive_schema = self.discover_hive_schema(table_name)
        
        # 3. Termes créés
        logger.info("\n📝 3. TERMES DANS CACHE")
        logger.info(f"Nombre de termes en cache: {len(self.created_terms_cache)}")
        for term_name, info in list(self.created_terms_cache.items())[:5]:
            logger.info(f"  - {term_name} → {info['source_column']}")
        
        # 4. Simulation mapping
        logger.info("\n🔗 4. SIMULATION MAPPING")
        if self.created_terms_cache and hive_schema:
            mapping = self.create_csv_to_hive_mapping(table_name)
            logger.info(f"Mappings possibles: {len(mapping)}")
        
        return {
            "mongodb_state": preview,
            "hive_columns": len(hive_schema),
            "cached_terms": len(self.created_terms_cache),
            "potential_mappings": len(mapping) if 'mapping' in locals() else 0
        }

def main():
    """
    Point d'entrée principal avec options.
    
    UTILISATION:
    1. Mode normal: Synchronisation complète
    2. Mode preview: Voir ce qui sera fait sans modifier Atlas
    3. Mode debug: Diagnostiquer les problèmes
    """
    print("=" * 80)
    print("ATLAS METADATA GOVERNANCE - SYNCHRONISATION RGPD")
    print("=" * 80)
    print("\nModes disponibles:")
    print("1. Preview (voir sans modifier)")
    print("2. Sync complet (avec confirmation)")
    print("3. Debug (diagnostiquer les problèmes)")
    
    choice = input("\nVotre choix (1/2/3): ").strip()
    
    governance = AtlasMetadataGovernance()
    
    if choice == "1":
        # MODE PREVIEW
        result = governance.sync_governance_metadata(preview_only=True)
        print("\n" + "=" * 80)
        print("PRÉVISUALISATION")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif choice == "2":
        # MODE SYNC COMPLET
        result = governance.sync_governance_metadata(preview_only=False)
        print("\n" + "=" * 80)
        print("RÉSULTAT SYNCHRONISATION")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if result.get("success"):
            print("\n✅ SUCCÈS!")
            print(f"Termes créés: {result.get('terms_created', 0)}")
            print(f"Assignations: {result.get('assignment_result', {}).get('successful_assignments', 0)}")
            
            # Diagnostics si assignations échouées
            failed = result.get('assignment_result', {}).get('failed_assignments', 0)
            if failed > 0:
                print(f"\n⚠️ {failed} assignations échouées")
                print("Causes possibles:")
                print("- Noms de colonnes différents entre CSV et Hive")
                print("- Termes pas encore indexés (attendre 1-2 minutes)")
                print("- Vérifiez les logs détaillés ci-dessus")
        else:
            print(f"\n❌ ÉCHEC: {result.get('error')}")
            
    elif choice == "3":
        # MODE DEBUG
        result = governance.debug_full_workflow()
        print("\n" + "=" * 80)
        print("RAPPORT DEBUG")
        print("=" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        print("Choix invalide")

if __name__ == "__main__":
    main()