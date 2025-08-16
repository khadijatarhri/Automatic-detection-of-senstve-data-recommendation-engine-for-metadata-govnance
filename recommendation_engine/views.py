from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect  
from django.http import JsonResponse  
from django.views import View  
from .MoteurDeRecommandationAvecDeepSeekML import IntelligentRecommendationEngine
from .models import RecommendationStorage  
import os  
from pymongo import MongoClient  
from bson import ObjectId  
from semantic_engine import SemanticAnalyzer, IntelligentAutoTagger, EntityMetadata  
from presidio_custom import create_enhanced_analyzer_engine  
from db_connections import db as main_db  
import pandas as pd  



class RecommendationView(View):  
    def get(self, request, job_id):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        return render(request, 'recommendation_engine/recommendations.html', {  
            'job_id': job_id,  
            'recommendations': recommendations  
        })  
  
class RecommendationAPIView(View):  
    def get(self, request, job_id):  
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        data = []  
        for rec in recommendations:  
            data.append({  
                'id': rec.id,  
                'title': rec.title,  
                'description': rec.description,  
                'category': rec.category,  
                'priority': rec.priority,  
                'confidence': rec.confidence,  
                'created_at': rec.created_at.isoformat()  
            })  
          
        return JsonResponse({'recommendations': data})
    
class RecommendationView(View):  
    def get(self, request, job_id):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        # Calculer les scores dynamiquement  
        overall_score = self._calculate_overall_score(recommendations)  
        rgpd_score = self._calculate_rgpd_score(recommendations)  
          
        return render(request, 'recommendation_engine/recommendations.html', {  
            'job_id': job_id,  
            'recommendations': recommendations,  
            'overall_score': overall_score,  
            'rgpd_score': rgpd_score  
        })  
      
    def _calculate_overall_score(self, recommendations):  
        if not recommendations:  
            return 10.0  
          
        # Score basé sur la priorité moyenne des recommandations  
        avg_priority = sum(rec.priority for rec in recommendations) / len(recommendations)  
        return max(0.0, 10.0 - avg_priority)  
      
    def _calculate_rgpd_score(self, recommendations):  
        rgpd_recs = [rec for rec in recommendations if rec.category == "COMPLIANCE_RGPD"]  
        if not rgpd_recs:  
            return 9.0  
          
        # Score RGPD basé sur le nombre et la priorité des recommandations RGPD  
        avg_rgpd_priority = sum(rec.priority for rec in rgpd_recs) / len(rgpd_recs)  
        return max(0.0, 10.0 - avg_rgpd_priority)
    



class MetadataView(View):  
    def get(self, request, job_id):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        # Récupérer les métadonnées enrichies depuis semantic_engine  
        metadata = self._get_enriched_metadata(job_id)  
          
        return render(request, 'recommendation_engine/metadata.html', {  
            'job_id': job_id,  
            'metadata': metadata  
        })  
      
    def _get_enriched_metadata(self, job_id):  
        """  
        Récupère les métadonnées enrichies du semantic_engine pour un job donné  
          
        Args:  
            job_id: ID du job d'analyse  
              
        Returns:  
            List[EntityMetadata]: Liste des entités enrichies avec métadonnées  
        """  
        try:  
            # Convertir job_id en ObjectId si nécessaire  
            if len(job_id) == 24:  
                object_id = ObjectId(job_id)  
            else:  
                object_id = job_id  
              
            # Récupérer les données du job depuis MongoDB  
            collection = main_db["csv_analysis_jobs"]  
            job_data = collection.find_one({'job_id': str(object_id)})  
              
            if not job_data:  
                return []  
              
            # Récupérer le nom du fichier original pour le contexte  
            job = main_db.anonymization_jobs.find_one({'_id': object_id})  
            original_filename = job['original_filename'] if job else 'dataset.csv'  
              
            # Initialiser les moteurs d'analyse sémantique  
            analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
            semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
            auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
              
            # Récupérer les données CSV  
            headers = job_data['headers']  
            csv_data = job_data['data']  
            df = pd.DataFrame(csv_data)  
              
            enriched_entities = []  
              
            # Analyser chaque cellule du DataFrame  
            for column in df.columns:  
                for index, value in df[column].items():  
                    if isinstance(value, str) and value.strip():  
                        # Utiliser l'IntelligentAutoTagger pour analyser et enrichir  
                        entities, tags = auto_tagger.analyze_and_tag(  
                            value,   
                            dataset_name=original_filename  
                        )  
                          
                        # Ajouter les entités enrichies à la liste  
                        for entity in entities:  
                            # Créer une copie avec des informations supplémentaires  
                            enriched_entity = EntityMetadata(  
                                entity_type=entity.entity_type,  
                                entity_value=entity.entity_value,  
                                start_pos=entity.start_pos,  
                                end_pos=entity.end_pos,  
                                confidence_score=entity.confidence_score * 100,  # Convertir en pourcentage  
                                sensitivity_level=entity.sensitivity_level,  
                                data_category=entity.data_category,  
                                semantic_context=entity.semantic_context,  
                                rgpd_category=entity.rgpd_category,  
                                anonymization_method=entity.anonymization_method  
                            )  
                            enriched_entities.append(enriched_entity)  
              
            # Supprimer les doublons basés sur entity_type et entity_value  
            unique_entities = []  
            seen = set()  
            for entity in enriched_entities:  
                key = (entity.entity_type, entity.entity_value)  
                if key not in seen:  
                    seen.add(key)  
                    unique_entities.append(entity)  
              
            return unique_entities  
              
        except Exception as e:  
            print(f"Erreur lors de la récupération des métadonnées: {e}")  
            return []