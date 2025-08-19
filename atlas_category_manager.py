import requests  
from typing import Dict, List  
  
class AtlasCategoryManager:  
    """Gestionnaire des catégories de glossaire Atlas"""  
      
    def __init__(self, atlas_client):  
        self.atlas_client = atlas_client  
          
    def create_rgpd_categories(self, glossary_guid: str) -> Dict[str, str]:  
        """Crée automatiquement les catégories RGPD dans Atlas"""  
        categories = [  
            {  
                "name": "Données d'identification",  
                "shortDescription": "Informations permettant d'identifier une personne",  
                "anchor": {"glossaryGuid": glossary_guid}  
            },  
            {  
                "name": "Données de contact",   
                "shortDescription": "Informations de contact personnel",  
                "anchor": {"glossaryGuid": glossary_guid}  
            },  
            {  
                "name": "Données financières",  
                "shortDescription": "Informations bancaires et financières",  
                "anchor": {"glossaryGuid": glossary_guid}  
            },  
            {  
                "name": "Données de localisation",  
                "shortDescription": "Informations géographiques et d'adresse",  
                "anchor": {"glossaryGuid": glossary_guid}  
            },  
            {  
                "name": "Données temporelles",  
                "shortDescription": "Informations de date et heure",  
                "anchor": {"glossaryGuid": glossary_guid}  
            }  
        ]  
          
        category_guids = {}  
        for category in categories:  
            try:  
                url = f"{self.atlas_client.atlas_url}/api/atlas/v2/glossary/category"  
                response = self.atlas_client.session.post(url, json=category)  
                response.raise_for_status()  
                category_guids[category['name']] = response.json()['guid']  
            except Exception as e:  
                print(f"Erreur création catégorie {category['name']}: {e}")  
                  
        return category_guids