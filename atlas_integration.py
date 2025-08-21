import os  
from typing import Dict, List  
from datetime import datetime  
from pyapacheatlas.auth import BasicAuthentication  
from pyapacheatlas.core import AtlasClient, AtlasEntity  
from pymongo import MongoClient  
from glossary_manager import GlossaryManager, GlossaryTermExtractor  
  
class AtlasTermFormatter:  
    """Formate les termes pour Atlas avec les classes personnalisées"""  
      
    @staticmethod  
    def format_term_for_atlas(term: Dict) -> Dict:  
        """Formate un terme avec classifications Atlas ET méthodes Ranger"""  
          
        # Récupérer les méthodes Ranger (comme avant)  
        from semantic_engine import SemanticAnalyzer  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        entity_type = term['name']  
        ranger_method = semantic_analyzer.anonymization_methods.get(entity_type, 'masquage')  
          
        # Mapper les classifications Atlas  
        sensitivity_classifications = {  
            'PUBLIC': 'Public',  
            'INTERNAL': 'Internal',  
            'CONFIDENTIAL': 'Confidential',  
            'RESTRICTED': 'Restricted',  
            'PERSONAL_DATA': 'PersonalData'  
        }  
          
        sensitivity_level = term.get('sensitivity_level', 'INTERNAL')  
        classification = sensitivity_classifications.get(sensitivity_level, 'Internal')  
          
        # Retourner un dictionnaire compatible avec CustomAtlasGlossaryTerm  
        return {  
            "name": term['name'],  
            "shortDescription": term['definition'],  
            "longDescription": f"Terme RGPD validé - Méthode Ranger: {ranger_method}. Catégorie: {term['category']}. Validations: {term.get('validation_count', 0)}",  
            "qualifiedName": f"rgpd_glossary.{term['name']}@cluster1",  
            "classifications": [  
                {  
                    "typeName": classification,  
                    "attributes": {  
                        "level": sensitivity_level,  
                        "source": "automatic_detection",  
                        "ranger_policy": ranger_method  
                    }  
                }  
            ],  
            "attributes": {  
                "rgpd_category": term['category'],  
                "original_anonymization_method": term['anonymization_method'],  
                "ranger_anonymization_method": ranger_method,  
                "sensitivity_level": sensitivity_level,  
                "validation_count": term.get('validation_count', 0),  
                "source": "validated_with_ranger_integration",  
                "last_updated": term.get('updated_at', '').isoformat() if term.get('updated_at') else ''  
            }  
        }

  
class AtlasGlossaryClient:  
    """Client pour interagir avec Apache Atlas utilisant pyapacheatlas 0.16.0"""  
      
    def __init__(self, atlas_url: str, username: str, password: str):  
        self.atlas_url = atlas_url.rstrip('/')  
          
        # Utiliser pyapacheatlas pour l'authentification et le client  
        self.auth = BasicAuthentication(username=username, password=password)  
        self.client = AtlasClient(  
            endpoint_url=self.atlas_url,  
            authentication=self.auth  
        )  
      
    def create_glossary(self, glossary_name: str) -> str:  
        """Crée un glossaire via l'API REST directe"""  
        try:  
            # Utiliser l'API REST directe pour les glossaires  
            glossary_data = CustomAtlasGlossary(  
                name=glossary_name,  
                shortDescription="Glossaire RGPD automatisé",  
                longDescription="Termes générés par le système de détection RGPD"  
            )  
              
            # Utiliser les méthodes HTTP du client pyapacheatlas  
            response = self.client._client.post(  
                f"{self.atlas_url}/api/atlas/v2/glossary",  
                json=glossary_data.to_dict()  
            )  
            response.raise_for_status()  
            return response.json()['guid']  
              
        except Exception as e:  
            print(f"Erreur lors de la création du glossaire: {e}")  
            raise  
      
    def create_term(self, glossary_guid: str, term_data: Dict) -> str:  
        """Crée un terme via l'API REST directe"""  
        try:  
            # Créer un terme personnalisé  
            custom_term = CustomAtlasGlossaryTerm(  
                name=term_data['name'],  
                qualifiedName=term_data.get('qualifiedName', f"rgpd_glossary.{term_data['name']}@cluster1"),  
                shortDescription=term_data.get('shortDescription', ''),  
                longDescription=term_data.get('longDescription', ''),  
                attributes=term_data.get('attributes', {})  
            )  
              
            # Ajouter les classifications  
            for classification in term_data.get('classifications', []):  
                custom_term.addClassification(  
                    classification['typeName'],  
                    classification.get('attributes', {})  
                )  
              
            custom_term.glossaryGuid = glossary_guid  
              
            # Utiliser l'API REST directe  
            response = self.client._client.post(  
                f"{self.atlas_url}/api/atlas/v2/glossary/term",  
                json=custom_term.to_dict()  
            )  
            response.raise_for_status()  
            return response.json()['guid']  
              
        except Exception as e:  
            print(f"Erreur lors de la création du terme: {e}")  
            raise


class GlossarySyncService:  
    """Service de synchronisation complète avec Atlas utilisant pyapacheatlas"""  
      
    def __init__(self, atlas_url: str, atlas_username: str, atlas_password: str):  
        self.extractor = GlossaryTermExtractor(GlossaryManager())  
        self.atlas_client = AtlasGlossaryClient(atlas_url, atlas_username, atlas_password)  
      
    def sync_validated_terms_to_atlas(self) -> Dict:  
        """Synchronise les termes validés vers Atlas"""  
        try:  
            # 1. Extraire les termes validés  
            validated_terms = self.extractor.extract_validated_terms()  
              
            # 2. Propager vers Atlas  
            term_guids = self.atlas_client.propagate_terms(validated_terms)  
              
            # 3. Enregistrer la synchronisation  
            self._log_sync_results(validated_terms, term_guids)  
              
            return {  
                'success': True,  
                'terms_synced': len(validated_terms),  
                'atlas_guids': term_guids  
            }  
              
        except Exception as e:  
            return {  
                'success': False,  
                'error': str(e)  
            }  
      
    def sync_with_categories_and_classifications(self) -> Dict:  
        """Synchronise avec création automatique des catégories et classifications"""  
        try:  
            # 1. Créer ou récupérer le glossaire  
            glossary_guid = self.atlas_client.create_glossary("RGPD_Glossary")  
              
            # 2. Créer les catégories RGPD automatiquement  
            category_guids = self._create_rgpd_categories(glossary_guid)  
              
            # 3. Extraire et synchroniser les termes validés  
            validated_terms = self.extractor.extract_validated_terms()  
              
            # 4. Enrichir les termes avec les GUIDs de catégories  
            for term in validated_terms:  
                category_name = term.get('category', 'Non classifié')  
                if category_name in category_guids:  
                    term['category_guid'] = category_guids[category_name]  
              
            # 5. Propager vers Atlas avec classifications  
            term_guids = self.atlas_client.propagate_terms(validated_terms)  
              
            return {  
                'success': True,  
                'terms_synced': len(validated_terms),  
                'categories_created': len(category_guids),  
                'atlas_guids': term_guids,  
                'category_guids': category_guids  
            }  
              
        except Exception as e:  
            return {  
                'success': False,  
                'error': str(e)  
            }  
      
    def _create_rgpd_categories(self, glossary_guid: str) -> Dict[str, str]:  
        """Crée automatiquement les catégories RGPD dans Atlas"""  
        categories_data = [  
            {  
                "name": "Données d'identification",  
                "shortDescription": "Informations permettant d'identifier une personne"  
            },  
            {  
                "name": "Données de contact",  
                "shortDescription": "Informations de contact personnel"  
            },  
            {  
                "name": "Données financières",  
                "shortDescription": "Informations bancaires et financières"  
            },  
            {  
                "name": "Données de localisation",  
                "shortDescription": "Informations géographiques et d'adresse"  
            },  
            {  
                "name": "Données temporelles",  
                "shortDescription": "Informations de date et heure"  
            }  
        ]  
          
        category_guids = {}  
        for category_data in categories_data:  
            try:  
                # Créer la catégorie avec pyapacheatlas  
                category = AtlasGlossary(  
                    name=category_data['name'],  
                    shortDescription=category_data['shortDescription'],  
                    glossaryGuid=glossary_guid  
                )  
                  
                result = self.atlas_client.client.upload_glossary_category(category)  
                category_guids[category_data['name']] = result['guid']  
                  
            except Exception as e:  
                print(f"Erreur création catégorie {category_data['name']}: {e}")  
          
        return category_guids  
      
    def _log_sync_results(self, terms: List[Dict], guids: Dict[str, str]):  
        """Enregistre les résultats de synchronisation"""  
        print(f"Synchronisation terminée: {len(terms)} termes propagés vers Atlas")  
        for term_name, guid in guids.items():  
            print(f"- {term_name}: {guid}")


class CustomAtlasGlossary:  
    """Wrapper personnalisé pour les glossaires Atlas"""  
      
    def __init__(self, name: str, shortDescription: str = "", longDescription: str = "", language: str = "French"):  
        self.name = name  
        self.shortDescription = shortDescription  
        self.longDescription = longDescription  
        self.language = language  
      
    def to_dict(self):  
        return {  
            "name": self.name,  
            "shortDescription": self.shortDescription,  
            "longDescription": self.longDescription,  
            "language": self.language  
        }  
  
class CustomAtlasGlossaryTerm:  
    """Wrapper personnalisé pour les termes de glossaire Atlas"""  
      
    def __init__(self, name: str, qualifiedName: str, shortDescription: str = "", longDescription: str = "", attributes: Dict = None):  
        self.name = name  
        self.qualifiedName = qualifiedName  
        self.shortDescription = shortDescription  
        self.longDescription = longDescription  
        self.attributes = attributes or {}  
        self.classifications = []  
        self.glossaryGuid = None  
      
    def addClassification(self, typeName: str, attributes: Dict = None):  
        self.classifications.append({  
            "typeName": typeName,  
            "attributes": attributes or {}  
        })  
      
    def to_dict(self):  
        result = {  
            "name": self.name,  
            "qualifiedName": self.qualifiedName,  
            "shortDescription": self.shortDescription,  
            "longDescription": self.longDescription,  
            "attributes": self.attributes  
        }  
          
        if self.classifications:  
            result["classifications"] = self.classifications  
          
        if self.glossaryGuid:  
            result["anchor"] = {"glossaryGuid": self.glossaryGuid}  
          
        return result