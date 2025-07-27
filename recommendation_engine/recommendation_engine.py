import asyncio  
import aiohttp  
import json  
from typing import Dict, List  
from .models import RecommendationItem, RecommendationType, RecommendationStorage  
from datetime import datetime  
  
class DeepSeekClient:  
    def __init__(self, api_key: str):  
        self.api_key = api_key  
        self.base_url = "https://api.deepseek.com/v1"  
      
    async def generate_recommendations(self, prompt: str) -> str:  
        headers = {  
            "Authorization": f"Bearer {self.api_key}",  
            "Content-Type": "application/json"  
        }  
          
        payload = {  
            "model": "deepseek-chat",  
            "messages": [  
                {  
                    "role": "system",  
                    "content": "Tu es un expert en gouvernance des données et conformité RGPD."  
                },  
                {"role": "user", "content": prompt}  
            ],  
            "max_tokens": 1500,  
            "temperature": 0.7  
        }  
          
        async with aiohttp.ClientSession() as session:  
            async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:  
                if response.status == 200:  
                    result = await response.json()  
                    return result['choices'][0]['message']['content']  
                else:  
                    raise Exception(f"Erreur API DeepSeek: {response.status}")  
  
class IntelligentRecommendationEngine:  
    def __init__(self, deepseek_api_key: str):  
        self.deepseek_client = DeepSeekClient(deepseek_api_key)  
        self.storage = RecommendationStorage()  
      
    def create_dataset_profile_from_presidio(self, job_id: str, detected_entities: set, headers: list, csv_data: list) -> dict:  
        """Crée un profil de dataset à partir des données Presidio"""  
        entity_distribution = {}  
        for entity in detected_entities:  
            entity_distribution[entity] = sum(1 for row in csv_data for value in row.values()   
                                            if isinstance(value, str) and entity in str(value))  
          
        return {  
            'dataset_id': job_id,  
            'name': f'Dataset_{job_id}',  
            'entity_distribution': entity_distribution,  
            'detected_entities': list(detected_entities),  
            'headers': headers,  
            'total_rows': len(csv_data),  
            'quality_score': self._calculate_quality_score(csv_data),  
            'has_personal_data': bool(detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}),  
            'rgpd_compliance_score': self._calculate_rgpd_score(detected_entities)  
        }  
      
    def _calculate_quality_score(self, csv_data: list) -> float:  
        """Calcule un score de qualité basique"""  
        if not csv_data:  
            return 0.0  
          
        total_cells = sum(len(row) for row in csv_data)  
        empty_cells = sum(1 for row in csv_data for value in row.values() if not value or value.strip() == '')  
          
        return max(0.0, (total_cells - empty_cells) / total_cells * 10)  
      
    def _calculate_rgpd_score(self, detected_entities: set) -> float:  
        """Calcule un score RGPD basique"""  
        sensitive_entities = {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD'}  
        found_sensitive = len(detected_entities & sensitive_entities)  
          
        if found_sensitive == 0:  
            return 10.0  
        elif found_sensitive <= 2:  
            return 7.0  
        else:  
            return 4.0  
      
    async def generate_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations basées sur le profil"""  
        recommendations = []  
          
        # Recommandations de qualité  
        if dataset_profile.get('quality_score', 0) < 8.0:  
            rec = RecommendationItem(  
                id=f"quality_{dataset_profile['dataset_id']}",  
                title="Améliorer la qualité des données",  
                description="Des cellules vides ont été détectées. Considérez un nettoyage des données.",  
                category=RecommendationType.QUALITY_IMPROVEMENT.value,  
                priority=7.0,  
                confidence=0.85,  
                metadata={"quality_score": dataset_profile.get('quality_score')},  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Recommandations RGPD  
        if dataset_profile.get('has_personal_data', False):  
            rec = RecommendationItem(  
                id=f"rgpd_{dataset_profile['dataset_id']}",  
                title="Conformité RGPD requise",  
                description="Des données personnelles ont été détectées. Assurez-vous de la conformité RGPD.",  
                category=RecommendationType.COMPLIANCE_RGPD.value,  
                priority=9.0,  
                confidence=0.95,  
                metadata={"detected_entities": dataset_profile.get('detected_entities', [])},  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Sauvegarder les recommandations  
        self.storage.save_recommendations(dataset_profile['dataset_id'], recommendations)  
          
        return recommendations