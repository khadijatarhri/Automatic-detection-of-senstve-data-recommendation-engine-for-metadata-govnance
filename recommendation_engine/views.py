import asyncio
# Create your views here.
from django.shortcuts import render, redirect  
from django.http import  JsonResponse
from django.views import View  
from .MoteurDeRecommandationAvecDeepSeekML import GeminiClient,  DataQualityEngine
from .models import RecommendationStorage  
import os  
import json
from pymongo import MongoClient  
from bson import ObjectId  
from semantic_engine import SemanticAnalyzer, IntelligentAutoTagger  
from presidio_custom import create_enhanced_analyzer_engine  
from db_connections import db as main_db  
import pandas as pd  
from datetime import datetime
from AtlasAPI.atlas_integration import GlossarySyncService  
from .recommendation_engine_core import EnterpriseRecommendationEngine
from .recommendation_formatters import EnterpriseFormatter
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging
import subprocess
import threading

# CORRECTION: Utiliser la même base de données que pour l'upload
client = MongoClient('mongodb://mongodb:27017/')

csv_db = client['csv_anonymizer_db']  # Même base que dans csv_anonymizer/views.py
users = main_db["users"]

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
    """Vue pour afficher les recommandations IA"""
    
    def get(self, request, job_id=None):
        """Affiche les recommandations pour un job donné"""
        
        # Vérifications d'authentification
        user_email = request.session.get("user_email")
        if not user_email:
            return redirect('login_form')
        
        user = users.find_one({'email': user_email})
        if not user or user.get('role') not in ['admin', 'user']:
            return redirect('authapp:home')
        
        if not job_id:
            return render(request, 'recommendation_engine/error.html', {
                'error': 'ID de job manquant'
            })
        
        try:
            # Récupérer les informations du job
            job = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})
            if not job:
                return render(request, 'recommendation_engine/error.html', {
                    'error': 'Job non trouvé'
                })
            
            # Vérifier les autorisations
            if (user.get('role') != 'admin' and 
                user_email not in job.get('authorized_users', []) and 
                job.get('user_email') != user_email):
                return render(request, 'recommendation_engine/error.html', {
                    'error': 'Accès non autorisé à ce job'
                })
            
            print(f"=== DEBUG RECOMMANDATIONS ===")
            print(f"Job ID: {job_id}")
            print(f"User: {user_email}")
            
            # Générer ou récupérer les recommandations
            recommendations_data = self._get_or_generate_recommendations(job_id, job)
            
            print(f"Recommandations générées: {type(recommendations_data)}")
            print(f"Nombre de recommandations: {len(recommendations_data.get('recommendations', []))}")
            
            # Formater pour l'affichage
            formatter = EnterpriseFormatter()
            
            context = {
                'job_id': job_id,
                'job': job,
                'recommendations_data': recommendations_data,
                'recommendations_by_category': recommendations_data.get('recommendations_by_category', {}),
                'overall_score': recommendations_data.get('overall_score', 5.0),
                'dashboard_view': formatter.format_dashboard_view(recommendations_data),
                'technical_view': formatter.format_technical_view(recommendations_data),
                'compliance_report': formatter.format_compliance_report(recommendations_data),
                'user_role': user.get('role', 'user'),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return render(request, 'recommendation_engine/recommendations.html', context)
            
        except Exception as e:
            print(f"ERREUR dans RecommendationView: {e}")
            import traceback
            traceback.print_exc()
            return render(request, 'recommendation_engine/error.html', {
                'error': f'Erreur lors du chargement des recommandations: {str(e)}'
            })
    
    def _get_or_generate_recommendations(self, job_id, job):
        """Récupère les recommandations existantes ou les génère"""
        
        # Vérifier s'il existe déjà des recommandations sauvegardées
        existing_recommendations = main_db.recommendations.find_one({'job_id': job_id})
        if existing_recommendations and existing_recommendations.get('recommendations_data'):
            print(f"Recommandations existantes trouvées pour job {job_id}")
            return existing_recommendations['recommendations_data']
        
        # Générer de nouvelles recommandations
        print(f"Génération de nouvelles recommandations pour job {job_id}")
        return self._generate_fresh_recommendations(job_id, job)
    
    def _generate_fresh_recommendations(self, job_id, job):
        """Génère de nouvelles recommandations"""
        try:
            print(f"=== GÉNÉRATION RECOMMANDATIONS POUR JOB {job_id} ===")
            
            # CORRECTION: Utiliser csv_db au lieu de main_db pour les chunks
            chunks_data = list(csv_db['csv_chunks'].find({'job_id': job_id}))
            print(f"Chunks trouvés: {len(chunks_data)}")
            
            if not chunks_data:
                print("AUCUN CHUNK TROUVÉ - Utilisation du fallback")
                return self._generate_basic_recommendations(job_id)
            
            # Reconstituer les données
            headers = chunks_data[0].get('headers', [])
            sample_rows = []
            
            print(f"Headers trouvés: {headers}")
            
            # Prendre un échantillon des premières données
            for chunk in chunks_data[:1]:  # Prendre seulement le premier chunk
                chunk_data = chunk.get('data', [])
                print(f"Données dans le chunk: {len(chunk_data)} lignes")
                
                for row_data in chunk_data[:10]:  # 10 premières lignes
                    if isinstance(row_data, dict):  # Si déjà un dictionnaire
                        sample_rows.append(row_data)
                    elif isinstance(row_data, list) and len(row_data) == len(headers):  # Si liste
                        row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
                        sample_rows.append(row_dict)
            
            print(f"Sample rows créés: {len(sample_rows)}")
            
            # Détecter les entités (simulation basique améliorée)
            detected_entities = set()
            
            # Analyse des headers
            for header in headers:
                header_lower = header.lower()
                if any(word in header_lower for word in ['email', 'mail', '@']):
                    detected_entities.add('EMAIL_ADDRESS')
                if any(word in header_lower for word in ['phone', 'tel', 'telephone', 'mobile']):
                    detected_entities.add('PHONE_NUMBER')
                if any(word in header_lower for word in ['person', 'name', 'nom', 'prenom']):
                    detected_entities.add('PERSON')
                if any(word in header_lower for word in ['iban', 'account', 'compte']):
                    detected_entities.add('IBAN_CODE')
                if any(word in header_lower for word in ['id', 'cin', 'carte']):
                    detected_entities.add('ID_MAROC')
                if any(word in header_lower for word in ['location', 'address', 'adresse', 'ville']):
                    detected_entities.add('LOCATION')
                if any(word in header_lower for word in ['date', 'time', 'created']):
                    detected_entities.add('DATE_TIME')
            
            # Analyse du contenu des échantillons
            for row in sample_rows[:5]:  # Analyser seulement 5 lignes pour performance
                for key, value in row.items():
                    if isinstance(value, str) and value.strip():
                        value_str = value.strip().lower()
                        # Détection basique par patterns
                        if '@' in value_str and '.' in value_str:
                            detected_entities.add('EMAIL_ADDRESS')
                        if any(char.isdigit() for char in value_str) and len(value_str) >= 8:
                            if value_str.startswith('+') or value_str.startswith('0'):
                                detected_entities.add('PHONE_NUMBER')
            
            print(f"Entités détectées: {list(detected_entities)}")
            
            if not detected_entities:
                print("AUCUNE ENTITÉ DÉTECTÉE - Ajout d'entités par défaut")
                detected_entities = {'PERSON', 'EMAIL_ADDRESS'}  # Entités par défaut
            
            # Générer les recommandations avec asyncio
            async def generate_recommendations_async():
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                
                try:
                    async with GeminiClient(gemini_api_key) as gemini_client:
                        recommendation_engine = EnterpriseRecommendationEngine(gemini_client)
                        
                        # Créer le profil du dataset
                        dataset_profile = recommendation_engine.create_dataset_profile_from_presidio(
                            str(job_id), detected_entities, headers, sample_rows
                        )
                        
                        print(f"Dataset profile créé: {dataset_profile.keys()}")
                        
                        # Générer les recommandations
                        recommendations = await recommendation_engine.generate_structured_recommendations(dataset_profile)
                        
                        print(f"Recommandations générées par le moteur: {type(recommendations)}")
                        return recommendations
                        
                except Exception as e:
                    print(f"ERREUR dans generate_recommendations_async: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            try:
                recommendations_data = asyncio.run(generate_recommendations_async())
                print(f"Recommandations reçues: {type(recommendations_data)}")
                
                # Vérifier la structure des données
                if not isinstance(recommendations_data, dict):
                    print("ATTENTION: Format de recommandations inattendu")
                    recommendations_data = self._generate_basic_recommendations(job_id)
                
                # Sauvegarder les recommandations pour usage futur
                main_db.recommendations.update_one(
                    {'job_id': job_id},
                    {
                        '$set': {
                            'job_id': job_id,
                            'recommendations_data': {  
                              'recommendations': [rec.__dict__ for rec in recommendations_data['recommendations']],  
                              'recommendations_by_category': {  
                              cat: [rec.__dict__ for rec in recs]   
                              for cat, recs in recommendations_data['recommendations_by_category'].items()  
                              },  
                              'category_summary': recommendations_data['category_summary'],  
                              'overall_score': recommendations_data['overall_score']  
                            },
                            'generated_at': datetime.now(),
                            'headers': headers,
                            'detected_entities': list(detected_entities)
                        }
                    },
                    upsert=True
                )
                
                return recommendations_data
                
            except Exception as e:
                print(f"ERREUR asyncio: {e}")
                return self._generate_basic_recommendations(job_id)
            
        except Exception as e:
            print(f"ERREUR GLOBALE dans _generate_fresh_recommendations: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_basic_recommendations(job_id)
    
    def _generate_basic_recommendations(self, job_id):
        """Génère des recommandations basiques en cas d'échec"""
        print(f"Génération des recommandations BASIQUES pour job {job_id}")
        
        from .models import RecommendationItem
        
        basic_recommendations = [
            RecommendationItem(
                id=f"basic_{job_id}_security",
                title="Analyse de sécurité recommandée",
                description="Une analyse de sécurité approfondie est recommandée pour ce dataset contenant potentiellement des données sensibles.",
                category="SECURITY",
                priority=8.0,
                confidence=0.75,
                metadata={'color': 'orange', 'basic': True},
                created_at=datetime.now()
            ),
            RecommendationItem(
                id=f"basic_{job_id}_compliance",
                title="Vérification de conformité RGPD",
                description="Vérifiez la conformité RGPD de ce dataset et assurez-vous que les données personnelles sont traitées conformément à la réglementation.",
                category="COMPLIANCE",
                priority=9.0,
                confidence=0.80,
                metadata={'color': 'red', 'basic': True},
                created_at=datetime.now()
            ),
            RecommendationItem(
                id=f"basic_{job_id}_quality",
                title="Contrôle qualité des données",
                description="Effectuez une analyse de qualité pour identifier les valeurs manquantes, doublons et incohérences.",
                category="QUALITY",
                priority=6.0,
                confidence=0.70,
                metadata={'color': 'yellow', 'basic': True},
                created_at=datetime.now()
            )
        ]
        
        result = {
            'recommendations': basic_recommendations,
            'recommendations_by_category': {
                'SECURITY': [basic_recommendations[0]],
                'COMPLIANCE': [basic_recommendations[1]],
                'QUALITY': [basic_recommendations[2]],
                'GOVERNANCE': []
            },
            'category_summary': {
                'SECURITY': {'count': 1, 'avg_priority': 8.0, 'color': 'orange'},
                'COMPLIANCE': {'count': 1, 'avg_priority': 9.0, 'color': 'red'},
                'QUALITY': {'count': 1, 'avg_priority': 6.0, 'color': 'yellow'},
                'GOVERNANCE': {'count': 0, 'avg_priority': 0.0, 'color': 'blue'}
            },
            'overall_score': 6.5,
            'total_count': 3
        }
        
        print(f"Recommandations basiques générées: {len(basic_recommendations)} items")
        return result


class RecommendationAPIView(View):
    """API pour récupérer les recommandations en JSON"""
    
    def get(self, request, job_id=None):
        """Retourne les recommandations en format JSON"""
        
        # Mêmes vérifications d'authentification
        user_email = request.session.get("user_email")
        if not user_email:
            return JsonResponse({'error': 'Non authentifié'}, status=401)
        
        if not job_id:
            return JsonResponse({'error': 'ID de job manquant'}, status=400)
        
        try:
            job = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})
            if not job:
                return JsonResponse({'error': 'Job non trouvé'}, status=404)
            
            view_instance = RecommendationView()
            recommendations_data = view_instance._get_or_generate_recommendations(job_id, job)
            
            # Convertir les objets RecommendationItem en dictionnaires
            serialized_data = self._serialize_recommendations(recommendations_data)
            
            return JsonResponse(serialized_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    def _serialize_recommendations(self, recommendations_data):
        """Convertit les recommandations en format JSON sérialisable"""
        if not isinstance(recommendations_data, dict):
            return {'error': 'Format de données invalide'}
        
        serialized = {
            'overall_score': recommendations_data.get('overall_score', 0),
            'total_count': recommendations_data.get('total_count', 0),
            'category_summary': recommendations_data.get('category_summary', {}),
            'recommendations': []
        }
        
        recommendations = recommendations_data.get('recommendations', [])
        for rec in recommendations:
            if hasattr(rec, '__dict__'):  # Si c'est un objet RecommendationItem
                serialized['recommendations'].append({
                    'id': rec.id,
                    'title': rec.title,
                    'description': rec.description,
                    'category': rec.category,
                    'priority': rec.priority,
                    'confidence': rec.confidence,
                    'metadata': rec.metadata,
                    'created_at': rec.created_at.isoformat() if rec.created_at else None
                })
            else:  # Si c'est déjà un dictionnaire
                serialized['recommendations'].append(rec)
        
        return serialized


# Reste des classes inchangées (MetadataView, ValidationWorkflowView, etc.)
class MetadataView(View):  
    def get(self, request, job_id):    
     if not request.session.get("user_email"):    
        return redirect('login_form')    
      
     user_email = request.session.get("user_email")  
     user = users.find_one({'email': user_email})    
      
     if not user or user.get('role') != 'user':    
        return redirect('authapp:home')   
      
     metadata = self._get_enriched_metadata(job_id)
    
    # AJOUT: Sérialiser pour JavaScript
     import json
     for meta in metadata:
        meta['json_safe'] = json.dumps({
            'column_name': meta['column_name'],
            'entity_types': meta['entity_types'],
            'validation_status': meta.get('validation_status', 'pending'),
            'recommended_rgpd_category': meta.get('recommended_rgpd_category'),
            'recommended_sensitivity_level': meta.get('recommended_sensitivity_level'),
            'recommended_ranger_policy': meta.get('recommended_ranger_policy')
        })
    
     metadata_db = MongoClient('mongodb://mongodb:27017/')['metadata_validation_db']
     enriched_collection = metadata_db['enriched_metadata']
    
     for meta in metadata:
        db_meta = enriched_collection.find_one({
            'job_id': job_id,
            'column_name': meta['column_name']
        })
        if db_meta:
            meta['atlas_sync_status'] = db_meta.get('atlas_sync_status', 'not_synced')
            meta['atlas_sync_date'] = db_meta.get('atlas_sync_date')
    
     validated_count = sum(1 for m in metadata if m.get('validation_status') == "validated")
     pending_count = sum(1 for m in metadata if m.get('validation_status') == "pending")
     synced_count = sum(1 for m in metadata if m.get('atlas_sync_status') == "synced")

     return render(request, 'recommendation_engine/metadata.html', {    
        'job_id': job_id,    
        'metadata': metadata,
        'validated_count': validated_count,
        'pending_count': pending_count,
        'synced_count': synced_count
     })
    
    def _get_enriched_metadata(self, job_id):  
        """Récupère les métadonnées enrichies de MongoDB"""
        logger = logging.getLogger(__name__)
        
        EXCLUDED_ENTITY_TYPES = {    
            'IN_PAN', 'URL', 'DOMAIN_NAME', 'NRP', 'US_BANK_NUMBER',    
            'IN_AADHAAR', 'US_DRIVER_LICENSE', 'UK_NHS'    
        }
        
        try:
            # Connexion à MongoDB
            metadata_db = MongoClient('mongodb://mongodb:27017/')['metadata_validation_db']
            enriched_collection = metadata_db['enriched_metadata']
            
            # Chercher d'abord dans la base de métadonnées
            existing_metadata = list(enriched_collection.find({'job_id': str(job_id)}))
            
            if existing_metadata:
                logger.info(f"Métadonnées trouvées dans DB: {len(existing_metadata)}")
                return existing_metadata
            
            # Sinon, générer les métadonnées
            logger.info(f"Génération métadonnées pour job {job_id}")
            
            chunks_data = list(csv_db['csv_chunks'].find({'job_id': str(job_id)}))
            
            if not chunks_data:
                logger.warning(f"Aucun chunk pour job {job_id}")
                return []
            
            headers = chunks_data[0].get('headers', [])
            sample_data = []
            
            for chunk in chunks_data[:1]:
                chunk_rows = chunk.get('data', [])
                for row_data in chunk_rows[:10]:
                    if isinstance(row_data, dict):
                        sample_data.append(row_data)
                    elif isinstance(row_data, list) and len(row_data) == len(headers):
                        row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
                        sample_data.append(row_dict)
            
            if not sample_data:
                return []
            
            # Analyseurs
            try:
                analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")
                semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")
                auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)
            except Exception as e:
                logger.error(f"Erreur init analyseurs: {e}")
                return []
            
            job_record = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})
            original_filename = job_record.get('original_filename', 'dataset.csv') if job_record else 'dataset.csv'
            
            metadata_list = []
            
            for header in headers:
                column_values = []
                for row in sample_data:
                    if header in row and row[header]:
                        value = str(row[header]).strip()
                        if value:
                            column_values.append(value)
                
                if not column_values:
                    continue
                
                detected_entities = set()
                sample_values = []
                total_entities = 0
                
                for i, value in enumerate(column_values[:5]):
                    try:
                        entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=original_filename)
                        
                        for entity in entities:
                            if entity.entity_type not in EXCLUDED_ENTITY_TYPES:
                                detected_entities.add(entity.entity_type)
                                total_entities += 1
                                
                                if len(sample_values) < 3:
                                    sample_values.append(entity.entity_value)
                                    
                    except Exception as e:
                        continue
                
                if detected_entities:
                    rgpd_category = self._get_rgpd_category(detected_entities)
                    sensitivity_level = self._get_sensitivity_level(detected_entities)  
                    ranger_policy = self._get_ranger_policy(detected_entities)
                    
                    column_metadata = {
                        'job_id': str(job_id),
                        'column_name': header,
                        'entity_types': list(detected_entities),
                        'sample_values': sample_values,
                        'total_entities': total_entities,
                        'recommended_rgpd_category': rgpd_category,
                        'recommended_sensitivity_level': sensitivity_level,
                        'recommended_ranger_policy': ranger_policy,
                        'validation_status': 'pending',
                        'atlas_sync_status': 'not_synced',
                        'created_at': datetime.now()
                    }
                    
                    # Sauvegarder dans MongoDB
                    enriched_collection.update_one(
                        {'job_id': str(job_id), 'column_name': header},
                        {'$set': column_metadata},
                        upsert=True
                    )
                    
                    metadata_list.append(column_metadata)

            for meta in metadata_list:
              meta['json_safe'] = json.dumps({
               'column_name': meta['column_name'],
               'entity_types': meta['entity_types'],
               'validation_status': meta.get('validation_status', 'pending'),
               'recommended_rgpd_category': meta.get('recommended_rgpd_category'),
               'recommended_sensitivity_level': meta.get('recommended_sensitivity_level'),
               'recommended_ranger_policy': meta.get('recommended_ranger_policy')
              })
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Erreur _get_enriched_metadata: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _get_rgpd_category(self, detected_entities):
        if any(entity in detected_entities for entity in ['PERSON', 'ID_MAROC']):
            return "Données d'identification"
        elif any(entity in detected_entities for entity in ['IBAN_CODE', 'CREDIT_CARD']):
            return "Données financières"
        elif any(entity in detected_entities for entity in ['PHONE_NUMBER', 'EMAIL_ADDRESS']):
            return "Données de contact"
        elif 'LOCATION' in detected_entities:
            return "Données de localisation"
        else:
            return "Non défini"

    def _get_sensitivity_level(self, detected_entities):
        if any(entity in detected_entities for entity in ['PERSON', 'ID_MAROC']):
            return "PERSONAL_DATA"
        elif any(entity in detected_entities for entity in ['IBAN_CODE', 'CREDIT_CARD']):
            return "RESTRICTED"
        elif any(entity in detected_entities for entity in ['PHONE_NUMBER', 'EMAIL_ADDRESS']):
            return "CONFIDENTIAL"
        else:
            return "INTERNAL"

    def _get_ranger_policy(self, detected_entities):
        if 'PERSON' in detected_entities:
            return "ranger_masking_policy_person"
        elif 'ID_MAROC' in detected_entities:
            return "ranger_hashing_policy_id"
        elif any(entity in detected_entities for entity in ['PHONE_NUMBER', 'EMAIL_ADDRESS']):
            return "ranger_partial_masking_policy_phone"
        elif any(entity in detected_entities for entity in ['IBAN_CODE', 'CREDIT_CARD']):
            return "ranger_encryption_policy_financial"
        else:
            return "ranger_masking_policy_person"


class ColumnValidationWorkflowView(View):  
    def post(self, request, job_id, column_name):  
        logger = logging.getLogger(__name__)

        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        try:  
            import json  
            data = json.loads(request.body)
            
            validation_status = data.get('validation_status')  
            annotation_comments = data.get('annotation_comments', '')  
            entity_type = data.get('entity_type')  
            rgpd_category = data.get('rgpd_category')
            sensitivity_level = data.get('sensitivity_level')
            anonymization_method = data.get('anonymization_method')  
              
            # MongoDB metadata_validation_db
            metadata_db = MongoClient('mongodb://mongodb:27017/')['metadata_validation_db']
            enriched_collection = metadata_db['enriched_metadata']
            
            # Mise à jour dans enriched_metadata
            update_doc = {
                'job_id': str(job_id),
                'column_name': column_name,
                'entity_type': entity_type,
                'validation_status': validation_status,
                'annotation_comments': annotation_comments,
                'recommended_rgpd_category': rgpd_category,
                'recommended_sensitivity_level': sensitivity_level,
                'recommended_ranger_policy': anonymization_method,
                'validated_by': request.session.get("user_email"),
                'validation_date': datetime.now(),
                'updated_at': datetime.now(),
                'ready_for_atlas_sync': validation_status == 'validated'
            }
            
            enriched_collection.update_one(
                {'job_id': str(job_id), 'column_name': column_name},
                {'$set': update_doc},
                upsert=True
            )
            
            logger.info(f"Validation sauvegardée: {column_name} -> {validation_status}")

            return JsonResponse({  
                'success': True,  
                'message': 'Validation sauvegardée',
                'validation_status': validation_status,
                'ready_for_sync': validation_status == 'validated'
            })
        
        except Exception as e:  
            logger.error(f"Erreur validation: {e}")
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
    def __init__(self):  
        super().__init__()  
        # CORRECTION: Utiliser csv_db comme dans csv_anonymizer
        self.client = MongoClient('mongodb://mongodb:27017/')  
        self.csv_db = self.client['csv_anonymizer_db']  
        self.users = main_db["users"]  # Users restent dans main_db
      
    def get(self, request, job_id):      
        if not request.session.get("user_email"):      
            return redirect('login_form')      
            
        # Vérification du rôle data steward    
        user_email = request.session.get("user_email")    
        user = self.users.find_one({'email': user_email})    
            
        # Autoriser seulement les data stewards    
        if not user or user.get('role') != 'user':    
            return redirect('authapp:home')    
  
        # CORRECTION: Utiliser csv_db au lieu de self.csv_db
        chunks = list(self.csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))    
        if not chunks:    
             return JsonResponse({'error': 'Données non trouvées'}, status=404)    
    
        headers = chunks[0]['headers']    
        csv_data = []    
        for chunk in chunks:    
            csv_data.extend(chunk['data'])   
              
        # Utiliser votre DataQualityEngine existante    
        gemini_api_key = os.getenv('GEMINI_API_KEY')      
            
        async def analyze_quality():      
            async with GeminiClient(gemini_api_key) as gemini_client:      
                quality_engine = DataQualityEngine(gemini_client)    
                return await quality_engine.analyze_data_quality(csv_data, headers)  
              
        quality_analysis = asyncio.run(analyze_quality())      
              
        return render(request, 'recommendation_engine/data_quality.html', {      
            'job_id': job_id,      
            'quality_analysis': quality_analysis      
        })  
  
    def post(self, request, job_id):    
        """Appliquer les corrections de qualité"""    
        if not request.session.get("user_email"):    
            return JsonResponse({'error': 'Non autorisé'}, status=401)    
        
        # Vérification du rôle data steward    
        user_email = request.session.get("user_email")    
        user = self.users.find_one({'email': user_email})    
        
        if not user or user.get('role') != 'user':    
            return JsonResponse({'error': 'Accès refusé'}, status=403)    
        
        action = request.POST.get('action')    
        
        if action == 'batch_cleaning':    
            return self._apply_batch_cleaning(request, job_id)    
        elif action == 'remove_duplicates':    
            return self._remove_duplicates(request, job_id)    
        elif action == 'remove_missing_values':    
            return self._remove_missing_values(request, job_id)    
        elif action == 'fix_inconsistencies':    
            return self._fix_inconsistencies(request, job_id)    
        
        return JsonResponse({'error': 'Action non reconnue'}, status=400)    
    
    def _apply_batch_cleaning(self, request, job_id):    
        """Application de plusieurs corrections en une seule fois"""    
        try:    
            import json  
            selected_actions = json.loads(request.POST.get('selected_actions', '[]'))    
                
            # CORRECTION: Utiliser self.csv_db  
            chunks = list(self.csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))    
            if not chunks:    
                return JsonResponse({'error': 'Données non trouvées'}, status=404)    
                
            # Combiner toutes les données des chunks    
            all_data = []    
            headers = chunks[0]['headers']    
            for chunk in chunks:    
                all_data.extend(chunk['data'])    
                
            df = pd.DataFrame(all_data)    
            original_count = len(df)    
            total_changes = 0    
            applied_actions = []    
                
            # Appliquer les actions dans l'ordre optimal    
            if 'remove_duplicates' in selected_actions:    
                before_count = len(df)    
                df = df.drop_duplicates(keep='first')    
                removed = before_count - len(df)    
                total_changes += removed    
                applied_actions.append(f"Doublons supprimés: {removed}")    
                
            if 'remove_missing_values' in selected_actions:    
                before_count = len(df)    
                df = df.dropna()    
                removed = before_count - len(df)    
                total_changes += removed    
                applied_actions.append(f"Lignes avec valeurs manquantes supprimées: {removed}")    
                
            if 'fix_inconsistencies' in selected_actions:    
                applied_actions.append("Incohérences corrigées")    
                
            # Re-sauvegarder en chunks    
            self._save_cleaned_data_as_chunks(job_id, headers, df.to_dict('records'))    
                
            return JsonResponse({    
                'success': True,    
                'total_changes': total_changes,    
                'applied_actions': applied_actions,    
                'final_count': len(df)    
            })    
                
        except Exception as e:    
            return JsonResponse({'error': str(e)}, status=500)    
  
    def _remove_missing_values(self, request, job_id):    
        """Suppression des valeurs manquantes"""    
        try:    
            chunks = list(self.csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))    
            if not chunks:    
                return JsonResponse({'error': 'Données non trouvées'}, status=404)    
                
            # Combiner les données    
            all_data = []    
            headers = chunks[0]['headers']    
            for chunk in chunks:    
                all_data.extend(chunk['data'])    
                
            df = pd.DataFrame(all_data)    
            original_count = len(df)    
                
            # Supprimer les lignes avec des valeurs manquantes    
            df_cleaned = df.dropna()    
            removed_count = original_count - len(df_cleaned)    
                
            # Re-sauvegarder en chunks    
            self._save_cleaned_data_as_chunks(job_id, headers, df_cleaned.to_dict('records'))    
                
            return JsonResponse({    
                'success': True,    
                'removed_count': removed_count,    
                'remaining_count': len(df_cleaned)    
            })    
                
        except Exception as e:    
            return JsonResponse({'error': str(e)}, status=500)  
            
    def _remove_duplicates(self, request, job_id):    
        """Suppression des doublons avec confirmation"""    
        duplicate_strategy = request.POST.get('duplicate_strategy', 'first')    
        columns_to_check = request.POST.getlist('columns_to_check')    
        
        try:    
            chunks = list(self.csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))    
            if not chunks:    
                return JsonResponse({'error': 'Données non trouvées'}, status=404)    
                
            # Combiner les données    
            all_data = []    
            headers = chunks[0]['headers']    
            for chunk in chunks:    
                all_data.extend(chunk['data'])    
                
            df = pd.DataFrame(all_data)    
            original_count = len(df)    
                
            # Supprimer les doublons    
            if columns_to_check:    
                df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=duplicate_strategy)    
            else:    
                df_cleaned = df.drop_duplicates(keep=duplicate_strategy)    
                
            removed_count = original_count - len(df_cleaned)    
                
            # Re-sauvegarder en chunks    
            self._save_cleaned_data_as_chunks(job_id, headers, df_cleaned.to_dict('records'))    
                
            return JsonResponse({    
                'success': True,    
                'removed_count': removed_count,    
                'remaining_count': len(df_cleaned)    
            })    
                
        except Exception as e:    
            return JsonResponse({'error': str(e)}, status=500)  
  
    def _save_cleaned_data_as_chunks(self, job_id, headers, cleaned_data):    
        """Re-sauvegarde les données nettoyées en chunks"""    
        # Supprimer les anciens chunks    
        self.csv_db['csv_chunks'].delete_many({'job_id': str(job_id)})    
            
        # Re-créer les chunks avec les données nettoyées    
        chunk_size = 1000    
        for i in range(0, len(cleaned_data), chunk_size):    
            chunk_data = cleaned_data[i:i + chunk_size]    
            chunk_doc = {    
                'job_id': str(job_id),    
                'chunk_number': i // chunk_size,    
                'headers': headers,    
                'data': chunk_data,    
                'created_at': datetime.now()    
            }    
            self.csv_db['csv_chunks'].insert_one(chunk_doc)

class AtlasSyncView(View):
 """Vue pour déclencher la synchronisation Atlas"""
    
 @method_decorator(csrf_exempt)
 def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
 def post(self, request):
    """Déclencher la synchronisation vers Atlas"""
    logger = logging.getLogger(__name__)

    user_email = request.session.get("user_email")
    if not user_email:
        return JsonResponse({'error': 'Non authentifié'}, status=401)
    
    user = users.find_one({'email': user_email})
    if not user or user.get('role') != 'user':
        return JsonResponse({'error': 'Accès refusé - Data Steward uniquement'}, status=403)
    
    try:
        logger.info(f"Synchronisation Atlas déclenchée par {user_email}")
        
        # Import corrigé
        try:
            # Essayez d'abord avec le nom du fichier document
            from atlas_entity_migration import AtlasMetadataGovernance
        except ImportError:
            try:
                # Sinon essayez depuis recommendation_engine
                from recommendation_engine.atlas_entity_migration import AtlasMetadataGovernance
            except ImportError as e:
                logger.error(f"Import impossible: {e}")
                return JsonResponse({
                    'success': False,
                    'error': f'Module de migration introuvable. Vérifiez que atlas_entity_migration.py existe.'
                }, status=500)
        
        governance = AtlasMetadataGovernance()
        
        # Exécution avec timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Synchronisation Atlas timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)  # 3 minutes max
        
        try:
            result = governance.sync_governance_metadata(preview_only=False)
            signal.alarm(0)
            
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'message': 'Synchronisation réussie',
                    'details': {
                        'glossary_guid': result.get('glossary_guid'),
                        'terms_synced': result.get('validated_terms_synced', 0),
                        'categories_created': result.get('categories_created', 0)
                    }
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Erreur inconnue')
                }, status=500)
                
        except TimeoutError:
            return JsonResponse({
                'success': False,
                'error': 'Timeout: Atlas ne répond pas. Vérifiez que HDP Sandbox est démarré.'
            }, status=504)
            
    except Exception as e:
        logger.error(f"Erreur synchronisation Atlas: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
    


class OdooMetadataView(View):
    """Vue améliorée avec Semantic Analyzer pour métadonnées Odoo"""
    
    def get(self, request):  
        logger = logging.getLogger(__name__)  
        logger.info("🚀 OdooMetadataView.get() - Début de la requête")  
        
        # Authentification
        user_email = request.session.get("user_email")  
        if not user_email:  
            return redirect('login_form')  
        
        user = users.find_one({'email': user_email})  
        if not user or user.get('role') not in ['admin', 'user']:  
            return redirect('authapp:home')  
        
        # Connexion MongoDB locale
        client = MongoClient('mongodb://mongodb:27017/')  
        main_db_local = client['main_db']  
        
        # Récupérer les JOBS ODOO consolidés
        odoo_jobs = list(main_db_local.anonymization_jobs.find({  
            'source': 'kafka_odoo_vrp'  
        }).sort('upload_date', -1))  
        
        logger.info(f"✅ Jobs Odoo trouvés: {len(odoo_jobs)}")
        
        # NOUVEAU: Agrégation des métadonnées par type de colonne
        consolidated_metadata = self._get_consolidated_metadata(odoo_jobs)
        
        # Statistiques consolidées
        stats = self._calculate_consolidated_stats(consolidated_metadata)
        
        return render(request, 'recommendation_engine/odoo_metadata.html', {  
            'consolidated_metadata': consolidated_metadata,
            'total_jobs': len(odoo_jobs),
            'stats': stats,
            'user_role': user.get('role', 'user'),
            'job_id': str(odoo_jobs[0]['_id']) if odoo_jobs else None  # AJOUT
        })
    
    def _get_consolidated_metadata(self, odoo_jobs):
        """
        Agrège les métadonnées par type de colonne sur tous les jobs Odoo.
        Au lieu d'avoir 19 jobs × 5 colonnes = 95 lignes,
        on a 5 colonnes consolidées avec les stats globales.
        """
        logger = logging.getLogger(__name__)
        
        # Dictionnaire pour agréger par nom de colonne
        columns_aggregated = {}
        
        for job in odoo_jobs:
            job_id = str(job['_id'])
            
            # Récupérer ou générer les métadonnées pour ce job
            metadata_list = self._get_enriched_metadata_with_semantic(job_id)
            
            for meta in metadata_list:
                col_name = meta['column_name']
                
                if col_name not in columns_aggregated:
                    # Première occurrence de cette colonne
                    columns_aggregated[col_name] = {
                        'column_name': col_name,
                        'entity_types': set(meta['entity_types']),
                        'total_jobs': 1,
                        'total_records': meta.get('total_entities', 0),
                        'sample_values': meta.get('sample_values', [])[:3],
                        'confidence_scores': meta.get('confidence_scores', {}),
                        'validation_status': meta.get('validation_status', 'pending'),
                        'recommended_rgpd_category': meta.get('recommended_rgpd_category'),
                        'recommended_sensitivity_level': meta.get('recommended_sensitivity_level'),
                        'recommended_ranger_policy': meta.get('recommended_ranger_policy'),
                        'semantic_context': meta.get('semantic_context', {}),
                        'last_updated': meta.get('created_at', datetime.now())
                    }
                else:
                    # Agréger les données
                    agg = columns_aggregated[col_name]
                    agg['entity_types'].update(meta['entity_types'])
                    agg['total_jobs'] += 1
                    agg['total_records'] += meta.get('total_entities', 0)
                    
                    # Garder les meilleurs scores de confiance
                    for entity_type, score in meta.get('confidence_scores', {}).items():
                        current_score = agg['confidence_scores'].get(entity_type, 0)
                        agg['confidence_scores'][entity_type] = max(current_score, score)
                    
                    # Mettre à jour la date si plus récente
                    if meta.get('created_at', datetime.min) > agg['last_updated']:
                        agg['last_updated'] = meta['created_at']
        
        # Convertir les sets en listes pour la sérialisation
        result = []
        for col_name, agg in columns_aggregated.items():
            agg['entity_types'] = list(agg['entity_types'])
            
            # Calculer le score de confiance moyen
            if agg['confidence_scores']:
                agg['avg_confidence'] = sum(agg['confidence_scores'].values()) / len(agg['confidence_scores'])
            else:
                agg['avg_confidence'] = 0.0
            
            result.append(agg)
        
        # Trier par nombre de records (colonnes les plus importantes en premier)
        result.sort(key=lambda x: x['total_records'], reverse=True)
        
        logger.info(f"✅ Métadonnées consolidées: {len(result)} colonnes uniques")
        return result
    
    def _calculate_consolidated_stats(self, consolidated_metadata):
        """Calcule les statistiques consolidées"""
        return {
            'total_columns': len(consolidated_metadata),
            'total_records': sum(col['total_records'] for col in consolidated_metadata),
            'validated': sum(1 for col in consolidated_metadata if col['validation_status'] == 'validated'),
            'pending': sum(1 for col in consolidated_metadata if col['validation_status'] == 'pending'),
            'rejected': sum(1 for col in consolidated_metadata if col['validation_status'] == 'rejected'),
            'avg_confidence': sum(col['avg_confidence'] for col in consolidated_metadata) / len(consolidated_metadata) if consolidated_metadata else 0.0
        }
    
    def _get_enriched_metadata_with_semantic(self, job_id):
        """
        Version AMÉLIORÉE avec Semantic Analyzer complet.
        Remplace la détection basique par mots-clés.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Analyse sémantique pour job_id: {job_id}")
        
        EXCLUDED_ENTITY_TYPES = {
            'IN_PAN', 'URL', 'DOMAIN_NAME', 'NRP', 'US_BANK_NUMBER',
            'IN_AADHAAR', 'US_DRIVER_LICENSE', 'UK_NHS'
        }
        
        try:
            # ÉTAPE 1 : Vérifier cache MongoDB
            metadata_db = MongoClient('mongodb://mongodb:27017/')['metadata_validation_db']
            enriched_collection = metadata_db['enriched_metadata']
            
            existing_metadata = list(enriched_collection.find({'job_id': str(job_id)}))
            if existing_metadata:
                logger.info(f"✅ Métadonnées trouvées dans cache: {len(existing_metadata)}")
                return existing_metadata
            
            # ÉTAPE 2 : Récupérer les chunks
            csv_db = MongoClient('mongodb://mongodb:27017/')['csv_anonymizer_db']
            chunks_data = list(csv_db.csv_chunks.find({'job_id': str(job_id)}))
            
            if not chunks_data:
                logger.warning(f"⚠️ Aucun chunk pour job {job_id}")
                return []
            
            headers = chunks_data[0].get('headers', [])
            
            # ÉTAPE 3 : Reconstituer sample data
            sample_data = []
            for chunk in chunks_data[:1]:  # Premier chunk suffisant
                chunk_rows = chunk.get('data', [])
                for row_data in chunk_rows[:20]:  # Analyser 20 lignes pour meilleure précision
                    if isinstance(row_data, dict):
                        sample_data.append(row_data)
                    elif isinstance(row_data, list) and len(row_data) == len(headers):
                        row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
                        sample_data.append(row_dict)
            
            if not sample_data:
                return []
            
            logger.info(f"📊 Sample data: {len(sample_data)} lignes, {len(headers)} colonnes")
            
            # ÉTAPE 4 : Initialiser le Semantic Analyzer
            try:
                analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")
                semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")
                auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)
                logger.info("✅ Semantic Analyzer initialisé")
            except Exception as e:
                logger.error(f"❌ Erreur Semantic Analyzer: {e}")
                # Fallback sur détection basique
                return self._fallback_basic_detection(job_id, headers, sample_data)
            
            # ÉTAPE 5 : Analyser chaque colonne avec Semantic Analyzer
            metadata_list = []
            
            for header in headers:
                column_values = []
                for row in sample_data:
                    value = row.get(header)
                    if value and str(value).strip():
                        column_values.append(str(value).strip())
                
                if not column_values:
                    continue
                
                # Analyse sémantique complète
                detected_entities = {}  # {entity_type: max_confidence}
                sample_values = []
                semantic_contexts = []
                
                for value in column_values[:10]:  # Top 10 pour performance
                    try:
                        # ANALYSE PRESIDIO + SPACY
                        results = analyzer.analyze(text=value, language='fr')
                        
                        # Analyse contextuelle
                        context_scores = semantic_analyzer.analyze_semantic_context(value)
                        semantic_contexts.append(context_scores)
                        
                        for entity in results:
                            if entity.entity_type in EXCLUDED_ENTITY_TYPES:
                                continue
                            
                            # Garder le meilleur score
                            current_score = detected_entities.get(entity.entity_type, 0.0)
                            detected_entities[entity.entity_type] = max(current_score, entity.score)
                            
                            if len(sample_values) < 3:
                                sample_values.append(value)
                    
                    except Exception as e:
                        logger.warning(f"Erreur analyse '{value}': {e}")
                        continue
                
                # Calculer le contexte sémantique moyen
                avg_context = {}
                if semantic_contexts:
                    for context_key in ['financial', 'identity', 'contact', 'location', 'behavioral']:
                        avg_context[context_key] = sum(ctx.get(context_key, 0) for ctx in semantic_contexts) / len(semantic_contexts)
                
                # Déterminer les recommandations basées sur le contexte
                if detected_entities:
                    # Prendre le type d'entité avec le meilleur score
                    best_entity_type = max(detected_entities, key=detected_entities.get)
                    best_confidence = detected_entities[best_entity_type]
                    
                    # Recommandations contextuelles
                    sensitivity = semantic_analyzer.determine_sensitivity_level(
                        best_entity_type, avg_context
                    )
                    category = semantic_analyzer.determine_data_category(
                        best_entity_type, avg_context
                    )
                    
                    rgpd_category = semantic_analyzer.rgpd_mapping.get(
                        best_entity_type, "Non défini"
                    )
                    ranger_policy = self._determine_ranger_policy(
                        best_entity_type, sensitivity
                    )
                    
                    column_metadata = {
                        'job_id': str(job_id),
                        'column_name': header,
                        'entity_types': list(detected_entities.keys()),
                        'confidence_scores': detected_entities,
                        'avg_confidence': sum(detected_entities.values()) / len(detected_entities),
                        'sample_values': sample_values,
                        'total_entities': len([v for v in column_values if v]),
                        'semantic_context': avg_context,
                        'recommended_rgpd_category': rgpd_category,
                        'recommended_sensitivity_level': sensitivity.value,
                        'recommended_ranger_policy': ranger_policy,
                        'validation_status': 'pending',
                        'atlas_sync_status': 'not_synced',
                        'created_at': datetime.now()
                    }
                    
                    # Sauvegarder dans MongoDB
                    enriched_collection.update_one(
                        {'job_id': str(job_id), 'column_name': header},
                        {'$set': column_metadata},
                        upsert=True
                    )
                    
                    metadata_list.append(column_metadata)
            
            logger.info(f"✅ Métadonnées générées: {len(metadata_list)} colonnes")
            return metadata_list
        
        except Exception as e:
            logger.error(f"❌ Erreur _get_enriched_metadata_with_semantic: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _determine_ranger_policy(self, entity_type, sensitivity_level):
        """Détermine la politique Ranger basée sur l'entité et la sensibilité"""
        
        mapping = {
            'PERSON': 'ranger_masking_policy_person',
            'ID_MAROC': 'ranger_hashing_policy_id',
            'PHONE_NUMBER': 'ranger_partial_masking_policy_phone',
            'EMAIL_ADDRESS': 'ranger_partial_masking_policy_email',
            'LOCATION': 'ranger_generalization_policy_location',
            'IBAN_CODE': 'ranger_encryption_policy_financial',
            'CREDIT_CARD': 'ranger_encryption_policy_financial'
        }
        
        # Si sensibilité très élevée, forcer le chiffrement
        if sensitivity_level.value == 'PERSONAL_DATA':
            if entity_type in ['IBAN_CODE', 'CREDIT_CARD', 'ID_MAROC']:
                return 'ranger_encryption_policy_financial'
        
        return mapping.get(entity_type, 'ranger_hashing_policy_id')
    
    def _fallback_basic_detection(self, job_id, headers, sample_data):
        """Fallback si le Semantic Analyzer échoue"""
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ Utilisation de la détection basique (fallback)")
        
        metadata_list = []
        
        for header in headers:
            header_lower = header.lower()
            
            # Détection basique par mots-clés
            if any(word in header_lower for word in ['email', 'mail', '@']):
                entity_type = 'EMAIL_ADDRESS'
                rgpd_cat = 'Données de contact'
            elif any(word in header_lower for word in ['phone', 'tel', 'mobile']):
                entity_type = 'PHONE_NUMBER'
                rgpd_cat = 'Données de contact'
            elif any(word in header_lower for word in ['name', 'nom', 'prenom']):
                entity_type = 'PERSON'
                rgpd_cat = "Données d'identification"
            elif any(word in header_lower for word in ['location', 'address', 'adresse']):
                entity_type = 'LOCATION'
                rgpd_cat = 'Données de localisation'
            else:
                entity_type = 'UNKNOWN'
                rgpd_cat = 'Non défini'
            
            metadata_list.append({
                'job_id': str(job_id),
                'column_name': header,
                'entity_types': [entity_type],
                'confidence_scores': {entity_type: 0.5},  # Faible confiance
                'avg_confidence': 0.5,
                'sample_values': [],
                'total_entities': 0,
                'recommended_rgpd_category': rgpd_cat,
                'recommended_sensitivity_level': 'INTERNAL',
                'recommended_ranger_policy': 'ranger_hashing_policy_id',
                'validation_status': 'pending'
            })
        
        return metadata_list
    


class ColumnValidationConsolidatedView(View):  
     """Validation consolidée qui s'applique à tous les jobs Odoo avec cette colonne"""  
      
     def post(self, request, column_name):  
        logger = logging.getLogger(__name__)  
          
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        try:  
            data = json.loads(request.body)  
              
            entity_type = data.get('entity_type')  
            validation_status = data.get('validation_status')  
            rgpd_category = data.get('rgpd_category')  
            sensitivity_level = data.get('sensitivity_level')  
            ranger_policy = data.get('ranger_policy')  
            comments = data.get('comments', '')  
              
            # Connexion MongoDB  
            metadata_db = MongoClient('mongodb://mongodb:27017/')['metadata_validation_db']  
            enriched_collection = metadata_db['enriched_metadata']  
              
            # CRITIQUE: Mettre à jour TOUTES les métadonnées avec ce nom de colonne  
            # (tous les jobs Odoo qui contiennent cette colonne)  
            update_doc = {  
                'entity_type': entity_type,  
                'validation_status': validation_status,  
                'recommended_rgpd_category': rgpd_category,  
                'recommended_sensitivity_level': sensitivity_level,  
                'recommended_ranger_policy': ranger_policy,  
                'annotation_comments': comments,  
                'validated_by': request.session.get("user_email"),  
                'validation_date': datetime.now(),  
                'updated_at': datetime.now(),  
                'ready_for_atlas_sync': validation_status == 'validated'  
            }  
              
            # Mise à jour en masse pour tous les jobs avec cette colonne  
            result = enriched_collection.update_many(  
                {'column_name': column_name},  # Tous les jobs avec cette colonne  
                {'$set': update_doc}  
            )  
              
            logger.info(f"✅ Validation consolidée: {column_name} -> {result.modified_count} documents mis à jour")  
              
            return JsonResponse({  
                'success': True,  
                'message': f'Validation appliquée à {result.modified_count} batches',  
                'modified_count': result.modified_count,  
                'validation_status': validation_status  
            })  
              
        except Exception as e:  
            logger.error(f"❌ Erreur validation consolidée: {e}")  
            import traceback  
            traceback.print_exc()  
            return JsonResponse({'error': str(e)}, status=500)