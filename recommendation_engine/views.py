from django.shortcuts import render
import asyncio
# Create your views here.
from django.shortcuts import render, redirect  
from django.http import JsonResponse  
from django.views import View  

from .MoteurDeRecommandationAvecDeepSeekML import IntelligentRecommendationEngine , GeminiClient , DataQualityEngine
from .models import RecommendationStorage  
import os  
from pymongo import MongoClient  
from bson import ObjectId  
from semantic_engine import SemanticAnalyzer, IntelligentAutoTagger, EntityMetadata  
from presidio_custom import create_enhanced_analyzer_engine  
from db_connections import db as main_db  
import pandas as pd  
from datetime import datetime

from atlas_integration import GlossarySyncService  
  
class GlossarySyncView(View):  
    def post(self, request):  
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        sync_service = GlossarySyncService(  
            atlas_url=os.getenv('ATLAS_URL', 'http://127.0.0.1:21000'),  
            atlas_username=os.getenv('ATLAS_USERNAME', 'admin'),  
            atlas_password=os.getenv('ATLAS_PASSWORD', 'allahyarani123')  
        )  
          
        result = sync_service.sync_with_categories_and_classifications()
        return JsonResponse(result)  
  
class GlossaryView(View):  
    def get(self, request):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        from glossary_manager import GlossaryManager  
        glossary_manager = GlossaryManager()  
        terms = glossary_manager.get_all_terms()  
          
        return render(request, 'recommendation_engine/glossary.html', {  
            'terms': terms  
        })






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
        


        # Vérification du rôle data steward  
        user_email = request.session.get("user_email")  
        from pymongo import MongoClient  
        client = MongoClient('mongodb://mongodb:27017/')  
        db = client['csv_anonymizer_db']  
        users = db['users']  
        user = users.find_one({'email': user_email})  
          
        # Autoriser seulement les data stewards (qui ont le rôle 'user' mais s'affichent comme Data Steward)  
        if not user or user.get('role') != 'user':  
            return redirect('authapp:home') 
          
        # Récupérer les métadonnées enrichies depuis semantic_engine  
        metadata = self._get_enriched_metadata(job_id)  
          
        return render(request, 'recommendation_engine/metadata.html', {  
            'job_id': job_id,  
            'metadata': metadata  
        })  
      
  def _get_enriched_metadata(self, job_id):  
    """Récupère et génère les métadonnées enrichies groupées par colonne"""  
    try:  
        # Connexion à la bonne base de données  
        client = MongoClient('mongodb://mongodb:27017/')  
        csv_db = client['csv_anonymizer_db']  
        collection = csv_db['csv_data']  
          
        # Connexion à la base de données des annotations  
        metadata_db = client['metadata_validation_db']  
        annotations_collection = metadata_db['column_annotations']  
          
        # Récupérer les annotations existantes  
        annotations = {}  
        for annotation in annotations_collection.find({'job_id': job_id}):  
            key = f"{annotation['column_name']}_{annotation['entity_type']}"  
            annotations[key] = annotation  
          
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
          
        # Grouper les entités par colonne  
        column_metadata = {}  
        headers = job_data.get('headers', [])  
        csv_data = job_data.get('data', [])  
          
        print(f"Analyse de {len(csv_data)} lignes de données par colonne...")  
          
        # Analyser chaque cellule du DataFrame et grouper par colonne  
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
                          
                        # Grouper par colonne au lieu d'entité individuelle  
                        if header not in column_metadata:  
                            column_metadata[header] = {  
                                'column_name': header,  
                                'entity_types': set(),  
                                'sample_values': [],  
                                'total_entities': 0,  
                                'confidence_scores': [],  
                                'sensitivity_levels': set(),  
                                'rgpd_categories': set(),  
                                'anonymization_methods': set()  
                            }  
                          
                        for entity in entities:  
                            column_metadata[header]['entity_types'].add(entity.entity_type)  
                            column_metadata[header]['sample_values'].append(entity.entity_value)  
                            column_metadata[header]['total_entities'] += 1  
                            column_metadata[header]['confidence_scores'].append(entity.confidence_score * 100)  
                              
                            # Ajouter les métadonnées enrichies  
                            sensitivity_level = entity.sensitivity_level.value if hasattr(entity.sensitivity_level, 'value') else str(entity.sensitivity_level)  
                            column_metadata[header]['sensitivity_levels'].add(sensitivity_level)  
                              
                            if entity.rgpd_category:  
                                column_metadata[header]['rgpd_categories'].add(entity.rgpd_category)  
                              
                            if entity.anonymization_method:  
                                column_metadata[header]['anonymization_methods'].add(entity.anonymization_method)  
                                  
                    except Exception as e:  
                        print(f"Erreur lors de l'analyse de '{value}': {e}")  
                        continue  
          
        # Convertir les sets en listes et calculer les moyennes  
        result = []  
        for column_name, data in column_metadata.items():  
            # Limiter les échantillons pour l'affichage  
            data['sample_values'] = list(set(data['sample_values']))[:5]  
            data['entity_types'] = list(data['entity_types'])  
            data['sensitivity_levels'] = list(data['sensitivity_levels'])  
            data['rgpd_categories'] = list(data['rgpd_categories'])  
            data['anonymization_methods'] = list(data['anonymization_methods'])  
              
            # Calculer la confiance moyenne  
            if data['confidence_scores']:  
                data['avg_confidence'] = sum(data['confidence_scores']) / len(data['confidence_scores'])  
            else:  
                data['avg_confidence'] = 0  
              
            # Vérifier le statut de validation  
            validation_status = "pending"  
            rejected_count = 0  
            validated_count = 0  
  
            for entity_type in data['entity_types']:  
                 key = f"{column_name}_{entity_type}"  
                 if key in annotations:  
                       annotation_status = annotations[key]['validation_status']  
                       if annotation_status == 'validated':  
                           validated_count += 1  
                       elif annotation_status == 'rejected':  
                           rejected_count += 1  
  
           # Déterminer le statut global de la colonne  
            if validated_count > 0 and rejected_count == 0:  
                 validation_status = "validated"  
            elif rejected_count > 0 and validated_count == 0:  
                 validation_status = "rejected"  
            elif validated_count > 0 and rejected_count > 0:  
                 validation_status = "mixed"  # Optionnel: gérer le cas mixte  
            else:  
                 validation_status = "pending"  
  
            data['validation_status'] = validation_status
            result.append(data)  
          
        print(f"Métadonnées enrichies générées: {len(result)} colonnes analysées")  
        return result  
          
    except Exception as e:  
        print(f"Erreur lors de la récupération des métadonnées: {e}")  
        import traceback  
        traceback.print_exc()  
        return []
class ColumnValidationWorkflowView(View):  
    def post(self, request, job_id, column_name):  
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        try:  
            import json  
            data = json.loads(request.body)  
            validation_status = data.get('validation_status')  
            annotation_comments = data.get('annotation_comments', '')  
            entity_type = data.get('entity_type')  
            rgpd_category = data.get('rgpd_category')  
            anonymization_method = data.get('anonymization_method')  
              
            # Connexion MongoDB  
            client = MongoClient('mongodb://mongodb:27017/')  
            metadata_db = client['metadata_validation_db']  
            column_annotations_collection = metadata_db['column_annotations']  
              
            # Créer ou mettre à jour l'annotation de colonne  
            annotation_doc = {  
                'job_id': job_id,  
                'column_name': column_name,  
                'entity_type': entity_type,  
                'validation_status': validation_status,  
                'annotation_comments': annotation_comments,  
                'rgpd_category': rgpd_category,  
                'anonymization_method': anonymization_method,  
                'validated_by': request.session.get("user_email"),  
                'validation_date': datetime.now(),  
                'updated_at': datetime.now()  
            }  
              
            # Upsert (insert ou update)  
            column_annotations_collection.update_one(  
                   {'job_id': job_id, 'column_name': column_name, 'entity_type': entity_type},  
                   {'$set': annotation_doc},  
                   upsert=True  
            )  
  
            sync_status = {'hive_sync': False, 'atlas_sync': False, 'ranger_sync': False}  
  
            
           
            try:  
                    from hive_integration.hive_integration import HiveMetadataSync  
          
                    with HiveMetadataSync(  
                         hive_host='sandbox-hdp.hortonworks.com',   
                         hive_port=2181  
                    ) as hive_sync:  
                         hive_sync.create_metadata_tables()  
                         hive_sync.sync_column_annotations(job_id=job_id)  
              
                    sync_status['hive_sync'] = True  
            except Exception as e:  
                   print(f"Erreur synchronisation Hive Sandbox: {e}")  
                   sync_status['hive_sync'] = False


            # Déclencher la synchronisation automatique vers Atlas  
            if sync_status['hive_sync']:
                   try:  
                      from atlas_integration import GlossarySyncService  
                      sync_service = GlossarySyncService(  
                         atlas_url=os.getenv('ATLAS_URL', 'http://127.0.0.1:21000'),  
                         atlas_username=os.getenv('ATLAS_USERNAME', 'admin'),  
                         atlas_password=os.getenv('ATLAS_PASSWORD', 'allahyarani123')  
                      )  
                      sync_result = sync_service.sync_with_categories_and_classifications()
                      sync_status['atlas_sync'] = sync_result.get('success', False)  
                      print(f"Synchronisation Atlas automatique: {sync_result}")  
                   except Exception as e:  
                      print(f"Erreur synchronisation Atlas: {e}")  
                      sync_status['atlas_error'] = str(e)  



            return JsonResponse({  
                   'success': True,  
                   'message': 'Validation sauvegardée avec succès',  
                   'sync_status': sync_status,  
                   'atlas_synced': sync_status['atlas_sync']  
            })
        
        except Exception as e:  
            return JsonResponse({'error': str(e)}, status=500)
        



class ValidationWorkflowView(View):  
    def post(self, request, job_id, entity_id):  
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        try:  
            import json  
            data = json.loads(request.body)  
            validation_status = data.get('validation_status')  
            annotation_comments = data.get('annotation_comments', '')  
              
            # Connexion MongoDB  
            client = MongoClient('mongodb://mongodb:27017/')  
            metadata_db = client['metadata_validation_db']  
            annotations_collection = metadata_db['entity_annotations']  
              
            # Créer ou mettre à jour l'annotation  
            annotation_doc = {  
                'job_id': job_id,  
                'entity_id': entity_id,  
                'validation_status': validation_status,  
                'annotation_comments': annotation_comments,  
                'validated_by': request.session.get("user_email"),  
                'validation_date': datetime.now(),  
                'updated_at': datetime.now()  
            }  
              
            # Upsert (insert ou update)  
            annotations_collection.update_one(  
                {'job_id': job_id, 'entity_id': entity_id},  
                {'$set': annotation_doc},  
                upsert=True  
            )  
              
            return JsonResponse({'success': True})  
              
        except Exception as e:  
            return JsonResponse({'error': str(e)}, status=500)


class DataQualityView(View):  
    def get(self, request, job_id):    
        if not request.session.get("user_email"):    
            return redirect('login_form')    
          
        # Vérification du rôle data steward  
        user_email = request.session.get("user_email")  
        from pymongo import MongoClient  
        client = MongoClient('mongodb://mongodb:27017/')  
        db = client['csv_anonymizer_db']  
        users = db['users']  
        user = users.find_one({'email': user_email})  
          
        # Autoriser seulement les data stewards  
        #if not user or user.get('role') != 'user':  
        #    return redirect('authapp:home')  


        # Récupérer les données depuis MongoDB  
        csv_db = client['csv_anonymizer_db']  
        collection = csv_db['csv_data']  
          
        job_data = collection.find_one({'job_id': str(job_id)})  
        if not job_data:  
            return JsonResponse({'error': 'Données non trouvées'}, status=404)  
          
        headers = job_data.get('headers', [])  
        csv_data = job_data.get('data', [])  
            
        # Utiliser votre DataQualityEngine existante  
        gemini_api_key = os.getenv('GEMINI_API_KEY')    
          
        async def analyze_quality():    
            async with GeminiClient(gemini_api_key) as gemini_client:    
                quality_engine = DataQualityEngine(gemini_client)  
                return await quality_engine.analyze_data_quality(csv_data, headers)  # Ajout d'await  
            
        quality_analysis = asyncio.run(analyze_quality())    
            
        return render(request, 'recommendation_engine/data_quality.html', {    
            'job_id': job_id,    
            'quality_analysis': quality_analysis    
        })


    def post(self, request, job_id):  
        """Appliquer les corrections de qualité"""  
        action = request.POST.get('action')  
          
        if action == 'remove_duplicates':  
            return self._remove_duplicates(request, job_id)  
        elif action == 'fix_inconsistencies':  
            return self._fix_inconsistencies(request, job_id)  
        elif action == 'fill_missing_values':  
            return self._fill_missing_values(request, job_id)  
          
        return JsonResponse({'error': 'Action non reconnue'}, status=400)  
      
    def _remove_duplicates(self, request, job_id):  
        """Suppression des doublons avec confirmation"""  
        duplicate_strategy = request.POST.get('duplicate_strategy', 'keep_first')  
        columns_to_check = request.POST.getlist('columns_to_check')  
          
        # Récupérer les données  
        client = MongoClient('mongodb://mongodb:27017/')  
        csv_db = client['csv_anonymizer_db']  
        collection = csv_db['csv_data']  
          
        job_data = collection.find_one({'job_id': str(job_id)})  
        if not job_data:  
            return JsonResponse({'error': 'Données non trouvées'}, status=404)  
          
        # Appliquer la suppression des doublons  
        df = pd.DataFrame(job_data['data'])  
        original_count = len(df)  
          
        if columns_to_check:  
            df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=duplicate_strategy)  
        else:  
            df_cleaned = df.drop_duplicates(keep=duplicate_strategy)  
          
        removed_count = original_count - len(df_cleaned)  
          
        # Sauvegarder les données nettoyées  
        job_data['data'] = df_cleaned.to_dict('records')  
        job_data['quality_actions'] = job_data.get('quality_actions', [])  
        job_data['quality_actions'].append({  
            'action': 'remove_duplicates',  
            'timestamp': datetime.now(),  
            'removed_count': removed_count,  
            'strategy': duplicate_strategy  
        })  
          
        collection.replace_one({'job_id': str(job_id)}, job_data)  
          
        return JsonResponse({  
            'success': True,  
            'removed_count': removed_count,  
            'remaining_count': len(df_cleaned)  
        })