from django.db import models

# Create your models here.
from pymongo import MongoClient  
from datetime import datetime  
import json  
from typing import Dict, List, Optional, Any  
from dataclasses import dataclass, asdict  
from enum import Enum  
import numpy as np  

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

            if 'metadata' in doc and doc['metadata']:  
                 doc['metadata'] = self._convert_numpy_types(doc['metadata'])   

            self.collection.insert_one(doc)  
      

    def _convert_numpy_types(self, obj):  
     if isinstance(obj, dict):  
        return {k: self._convert_numpy_types(v) for k, v in obj.items()}  
     elif isinstance(obj, list):  
        return [self._convert_numpy_types(item) for item in obj]  
     elif isinstance(obj, np.integer):  
        return int(obj)  
     elif isinstance(obj, np.floating):  
        return float(obj)  
     elif isinstance(obj, np.ndarray):  
        return obj.tolist()  
     else:  
        return obj


      
    def get_recommendations(self, dataset_id: str) -> List[RecommendationItem]:  
        """Récupère les recommandations depuis MongoDB"""  
        docs = self.collection.find({'dataset_id': dataset_id}).sort('priority', -1)  
        recommendations = []  
        for doc in docs:  
            doc['created_at'] = datetime.fromisoformat(doc['created_at'])  
            recommendations.append(RecommendationItem(**{k: v for k, v in doc.items() if k != '_id' and k != 'dataset_id'}))  
        return recommendations