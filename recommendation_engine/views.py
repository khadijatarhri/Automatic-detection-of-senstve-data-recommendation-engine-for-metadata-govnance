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
    """Récupère et génère les métadonnées enrichies pour un job donné"""  
    try:  
        # Connexion à la bonne base de données  
        client = MongoClient('mongodb://mongodb:27017/')  
        csv_db = client['csv_anonymizer_db']  
        collection = csv_db['csv_data']  
          
        # Récupérer les données CSV originales  
        job_data = collection.find_one({'job_id': str(job_id)})  
          
        if not job_data:  
            print(f"Aucune donnée trouvée pour job_id: {job_id}")  
            return []  
          
        # Récupérer le nom du fichier original  
        main_db_client = MongoClient('mongodb://mongodb:27017/')  
        main_db = main_db_client['db']  
        job_info = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})  
        original_filename = job_info['original_filename'] if job_info else 'dataset.csv'  
          
        # Initialiser les moteurs d'analyse sémantique  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
          
        # Analyser les données et générer les métadonnées enrichies  
        enriched_entities = []  
        headers = job_data.get('headers', [])  
        csv_data = job_data.get('data', [])  
          
        print(f"Analyse de {len(csv_data)} lignes de données...")  
          
        # Analyser chaque cellule du DataFrame  
        for row_idx, row in enumerate(csv_data):  
            print(f"Traitement ligne {row_idx + 1}/{len(csv_data)}")  
            for header, value in row.items():  
                if isinstance(value, str) and value.strip():  
                    try:  
                        # Utiliser l'IntelligentAutoTagger pour analyser  
                        entities, tags = auto_tagger.analyze_and_tag(  
                            value,   
                            dataset_name=original_filename  
                        )  
                          
                        # Ajouter les entités enrichies  
                        for entity in entities:  
                            enriched_entity = {  
                                'entity_type': entity.entity_type,  
                                'entity_value': entity.entity_value,  
                                'confidence_score': entity.confidence_score * 100,  
                                'sensitivity_level': entity.sensitivity_level.value if hasattr(entity.sensitivity_level, 'value') else str(entity.sensitivity_level),  
                                'data_category': entity.data_category.value if hasattr(entity.data_category, 'value') else str(entity.data_category),  
                                'rgpd_category': entity.rgpd_category,  
                                'anonymization_method': entity.anonymization_method  
                            }  
                            enriched_entities.append(enriched_entity)  
                    except Exception as e:  
                        print(f"Erreur lors de l'analyse de '{value}': {e}")  
                        continue  
          
        # CORRECTION: Supprimer les doublons APRÈS avoir traité toutes les lignes  
        unique_entities = []  
        seen = set()  
        for entity in enriched_entities:  
            key = (entity['entity_type'], entity['entity_value'])  
            if key not in seen:  
                seen.add(key)  
                unique_entities.append(entity)  
          
        print(f"Métadonnées enrichies générées: {len(unique_entities)} entités uniques sur {len(enriched_entities)} total")  
        return unique_entities  
          
    except Exception as e:  
        print(f"Erreur lors de la récupération des métadonnées: {e}")  
        import traceback  
        traceback.print_exc()  
        return []
