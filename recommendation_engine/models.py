from django.db import models

# Create your models here.
from pymongo import MongoClient  
from datetime import datetime  
import json  
from typing import Dict, List, Optional, Any  
from dataclasses import dataclass, asdict  
from enum import Enum  
  
# Connexion MongoDB  
client = MongoClient('mongodb://mongodb:27017/')
recommendations_db = client['recommendations_db']  
  
@dataclass  
class RecommendationItem:  
    id: str  
    title: str  
    description: str  
    category: str  
    priority: float  
    confidence: float  
    metadata: Dict[str, Any]  
    created_at: datetime  
  
class RecommendationType(Enum):  
    QUALITY_IMPROVEMENT = "QUALITY_IMPROVEMENT"  
    SECURITY_ENHANCEMENT = "SECURITY_ENHANCEMENT"  
    COMPLIANCE_RGPD = "COMPLIANCE_RGPD"  
    METADATA_ENRICHMENT = "METADATA_ENRICHMENT"  
    CLASSIFICATION_OPTIMIZATION = "CLASSIFICATION_OPTIMIZATION"  
    ANONYMIZATION_STRATEGY = "ANONYMIZATION_STRATEGY"  
  
class RecommendationStorage:  
    def __init__(self):  
        self.collection = recommendations_db.recommendations  
        self.analysis_collection = recommendations_db.dataset_analysis  
      
    def save_recommendations(self, dataset_id: str, recommendations: List[RecommendationItem]):  
        """Sauvegarde les recommandations dans MongoDB"""  
        for rec in recommendations:  
            doc = asdict(rec)  
            doc['dataset_id'] = dataset_id  
            doc['created_at'] = rec.created_at.isoformat()  
            self.collection.insert_one(doc)  
      
    def get_recommendations(self, dataset_id: str) -> List[RecommendationItem]:  
        """Récupère les recommandations depuis MongoDB"""  
        docs = self.collection.find({'dataset_id': dataset_id}).sort('priority', -1)  
        recommendations = []  
        for doc in docs:  
            doc['created_at'] = datetime.fromisoformat(doc['created_at'])  
            recommendations.append(RecommendationItem(**{k: v for k, v in doc.items() if k != '_id' and k != 'dataset_id'}))  
        return recommendations