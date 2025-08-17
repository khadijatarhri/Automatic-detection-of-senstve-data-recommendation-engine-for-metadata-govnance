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
        """Formate un terme selon le schéma Atlas"""  
        return {  
            "name": term['name'],  
            "shortDescription": term['definition'],  
            "longDescription": f"Terme RGPD validé automatiquement. Catégorie: {term['category']}. Validations: {term.get('validation_count', 0)}",  
            "qualifiedName": f"rgpd_glossary.{term['name']}@cluster1",  
            "attributes": {  
                "rgpd_category": term['category'],  
                "anonymization_method": term['anonymization_method'],  
                "sensitivity_level": term.get('sensitivity_level', 'INTERNAL'),  
                "validation_count": term.get('validation_count', 0),  
                "source": term.get('source', 'system'),  
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
        """Crée un glossaire dans Atlas"""  
        url = f"{self.atlas_url}/api/atlas/v2/glossary"  
        payload = {  
            "name": glossary_name,  
            "shortDescription": "Glossaire RGPD automatisé",  
            "longDescription": "Termes générés par le système de détection RGPD",  
            "language": "French",  
            "usage": "Gouvernance des données RGPD"  
        }  
          
        response = self.session.post(url, json=payload)  
        response.raise_for_status()  
        return response.json()['guid']  
      
    def create_term(self, glossary_guid: str, term_data: Dict) -> str:  
        """Crée un terme dans le glossaire Atlas"""  
        url = f"{self.atlas_url}/api/atlas/v2/glossary/term"  
          
        payload = {  
            **term_data,  
            "anchor": {"glossaryGuid": glossary_guid}  
        }  
          
        response = self.session.post(url, json=payload)  
        response.raise_for_status()  
        return response.json()['guid']  
      
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
      
    def _log_sync_results(self, terms: List[Dict], guids: Dict[str, str]):  
        """Enregistre les résultats de synchronisation"""  
        print(f"Synchronisation terminée: {len(terms)} termes propagés vers Atlas")  
        for term_name, guid in guids.items():  
            print(f"- {term_name}: {guid}")