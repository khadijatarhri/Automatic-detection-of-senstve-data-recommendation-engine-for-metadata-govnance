import requests  
import json  
from typing import Dict, List  
from datetime import datetime  
from pymongo import MongoClient  
from glossary_manager import GlossaryManager, GlossaryTermExtractor  
import os


class AtlasTermFormatter:  
  """Formate les termes pour Atlas"""  
      
  @staticmethod  
  def format_term_for_atlas(term: Dict) -> Dict:  
    """Formate un terme avec classifications Atlas ET méthodes Ranger"""  
      
    # Récupérer les méthodes Ranger  
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
    """Client pour interagir avec Apache Atlas"""  
      
    def __init__(self, atlas_url: str, username: str, password: str):  
        self.atlas_url = atlas_url.rstrip('/')  
        self.auth = (username, password)  
        self.session = requests.Session()  
        self.session.auth = self.auth  
        self.session.headers.update({'Content-Type': 'application/json'})  
      
    def create_glossary(self, glossary_name: str) -> str:  
     """Crée un glossaire dans Atlas ou récupère l'existant"""  
     url = f"{self.atlas_url}/api/atlas/v2/glossary"  
      
     # Essayer de récupérer le glossaire existant d'abord  
     try:  
        response = self.session.get(url)  
        if response.status_code == 200:  
            glossaries = response.json()  
            for glossary in glossaries:  
                if glossary.get('name') == glossary_name:  
                    return glossary['guid']  
     except:  
        pass  
      
     # Si pas trouvé, créer un nouveau  
     payload = {  
        "name": glossary_name,  
        "shortDescription": "Glossaire RGPD automatisé",  
        "longDescription": "Termes générés par le système de détection RGPD",  
        "language": "French",  
        "usage": "Gouvernance des données RGPD"  
    }  
      
     try:  
        response = self.session.post(url, json=payload)  
        response.raise_for_status()  
        return response.json()['guid']  
     except requests.exceptions.HTTPError as e:  
        if e.response.status_code == 409:  
            # Glossaire existe déjà, essayer de le récupérer à nouveau  
            response = self.session.get(url)  
            glossaries = response.json()  
            for glossary in glossaries:  
                if glossary.get('name') == glossary_name:  
                    return glossary['guid']  
        raise



    def create_term(self, glossary_guid: str, term_data: Dict) -> str:  
     """Crée un terme dans le glossaire Atlas en écrasant l'existant"""  
      
     # 1. Vérifier si le terme existe déjà et le supprimer  
     qualified_name = term_data.get('qualifiedName')  
     if qualified_name:  
        try:  
            # Rechercher le terme existant  
            search_url = f"{self.atlas_url}/api/atlas/v2/search/basic"  
            search_payload = {  
                "query": qualified_name,  
                "typeName": "AtlasGlossaryTerm"  
            }  
            search_response = self.session.post(search_url, json=search_payload)  
              
            if search_response.status_code == 200:  
                search_results = search_response.json()  
                if search_results.get('entities'):  
                    # Terme existe déjà, le supprimer  
                    existing_term = search_results['entities'][0]  
                    existing_guid = existing_term['guid']  
                    print(f"Suppression du terme existant: {qualified_name}")  
                    self._delete_term(existing_guid)  
        except Exception as e:  
            print(f"Erreur lors de la recherche/suppression du terme: {e}")  
      
     # 2. Créer le nouveau terme (avec méthodes Ranger)  
     url = f"{self.atlas_url}/api/atlas/v2/glossary/term"  
      
     # Enrichir avec les méthodes d'anonymisation Ranger  
     enriched_term_data = self._enrich_with_ranger_methods(term_data)  
      
     payload = {  
        **enriched_term_data,  
        "anchor": {"glossaryGuid": glossary_guid}  
     }  
      
     try:  
        response = self.session.post(url, json=payload)  
        response.raise_for_status()  
        print(f"Nouveau terme créé: {term_data.get('name')}")  
        return response.json()['guid']  
     except requests.exceptions.HTTPError as e:  
        if e.response.status_code == 409:  
            # Si conflit persiste, forcer la suppression et recréer  
            print(f"Conflit persistant pour: {term_data.get('name')}")  
            existing_guid = self._find_existing_term_guid(glossary_guid, term_data.get('name'))  
            self._delete_term(existing_guid)  
            # Réessayer la création  
            response = self.session.post(url, json=payload)  
            response.raise_for_status()  
            return response.json()['guid']  
        raise  
  
    def _delete_term(self, term_guid: str):  
     """Supprime un terme existant d'Atlas"""  
     try:  
        url = f"{self.atlas_url}/api/atlas/v2/glossary/term/{term_guid}"  
        response = self.session.delete(url)  
        response.raise_for_status()  
        print(f"Terme supprimé: {term_guid}")  
     except Exception as e:  
        print(f"Erreur lors de la suppression du terme {term_guid}: {e}")  
        raise  
  
    def _enrich_with_ranger_methods(self, term_data: Dict) -> Dict:  
     """Enrichit les données du terme avec les méthodes d'anonymisation Ranger"""  
      
     # Mapping des méthodes d'anonymisation vers les politiques Ranger  
     ranger_anonymization_methods = {  
        'PERSON': 'ranger_masking_policy_person',  
        'ID_MAROC': 'ranger_hashing_policy_id',  
        'PHONE_NUMBER': 'ranger_partial_masking_policy_phone',  
        'EMAIL_ADDRESS': 'ranger_partial_masking_policy_email',  
        'LOCATION': 'ranger_generalization_policy_location',  
        'IBAN_CODE': 'ranger_encryption_policy_financial',  
        'DATE_TIME': 'ranger_date_shifting_policy'  
     }  
      
     entity_type = term_data.get('name')  
     ranger_method = ranger_anonymization_methods.get(entity_type, 'ranger_default_masking_policy')  
      
     # Enrichir les attributs avec les informations Ranger  
     enriched_data = term_data.copy()  
     enriched_data['attributes'] = {  
        **term_data.get('attributes', {}),  
        'ranger_policy': ranger_method,  
        'anonymization_source': 'apache_ranger',  
        'policy_enforcement': 'automatic'  
     }  
      
     # Mettre à jour la description longue  
     enriched_data['longDescription'] = f"{term_data.get('longDescription', '')} - Politique Ranger: {ranger_method}"  
      
     return enriched_data



      
    def propagate_terms(self, terms: List[Dict]) -> Dict[str, str]:  
        """Propage tous les termes vers Atlas"""  
        try:  
            # Créer ou récupérer le glossaire  
            glossary_guid = self.create_glossary("RGPD_Glossary")  
              
            term_guids = {}  
            formatter = AtlasTermFormatter()  
              
            for term in terms:  
                formatted_term = formatter.format_term_for_atlas(term)  
                term_guid = self.create_term(glossary_guid, formatted_term)  
                term_guids[term['name']] = term_guid  
              
            return term_guids  
              
        except requests.exceptions.RequestException as e:  
            raise Exception(f"Erreur lors de la propagation vers Atlas: {e}")  
  
class GlossarySyncService:  
    """Service de synchronisation complète avec Atlas"""  
      
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
        from atlas_category_manager import AtlasCategoryManager  
        category_manager = AtlasCategoryManager(self.atlas_client)  
        category_guids = category_manager.create_rgpd_categories(glossary_guid)  
          
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









    def _log_sync_results(self, terms: List[Dict], guids: Dict[str, str]):  
        """Enregistre les résultats de synchronisation"""  
        print(f"Synchronisation terminée: {len(terms)} termes propagés vers Atlas")  
        for term_name, guid in guids.items():  
            print(f"- {term_name}: {guid}")