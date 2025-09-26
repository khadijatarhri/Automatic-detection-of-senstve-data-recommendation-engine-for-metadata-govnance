import asyncio
# Create your views here.
from django.shortcuts import render, redirect  
from django.http import  JsonResponse
from django.views import View  
from .MoteurDeRecommandationAvecDeepSeekML import GeminiClient,  DataQualityEngine
from .models import RecommendationStorage  
import os  
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
          
        # Vérifier que c'est un data steward  
        if not user or user.get('role') != 'user':    
            return redirect('authapp:home')   
          
        # Pas de vérification de propriétaire - tous les data stewards peuvent voir tous les jobs  
        metadata = self._get_enriched_metadata(job_id)    
        return render(request, 'recommendation_engine/metadata.html', {    
            'job_id': job_id,    
            'metadata': metadata
        })
 def _get_enriched_metadata(self, job_id):  
    """Récupère et génère les métadonnées enrichies groupées par colonne"""  
    
    # Définir EXCLUDED_ENTITY_TYPES localement (même valeur que dans views.py)
    EXCLUDED_ENTITY_TYPES = {    
        'IN_PAN', 'URL', 'DOMAIN_NAME', 'NRP', 'US_BANK_NUMBER',    
        'IN_AADHAAR', 'US_DRIVER_LICENSE', 'UK_NHS'    
    }
    
    try:
        print(f"=== GÉNÉRATION MÉTADONNÉES POUR JOB {job_id} ===")
        
        # 1. Récupérer les chunks de données
        chunks_data = list(csv_db['csv_chunks'].find({'job_id': str(job_id)}))
        print(f"Chunks trouvés: {len(chunks_data)}")
        
        if not chunks_data:
            print(f"Aucun chunk trouvé pour job_id: {job_id}")
            return []
        
        # 2. Extraire headers et données d'échantillon
        headers = chunks_data[0].get('headers', [])
        sample_data = []
        
        # Prendre un échantillon des données pour analyse
        for chunk in chunks_data[:1]:  # Un seul chunk pour éviter surcharge
            chunk_rows = chunk.get('data', [])
            print(f"Chunk contient {len(chunk_rows)} lignes")
            
            for row_data in chunk_rows[:10]:  # 10 lignes max
                if isinstance(row_data, dict):
                    sample_data.append(row_data)
                elif isinstance(row_data, list) and len(row_data) == len(headers):
                    row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
                    sample_data.append(row_dict)
        
        print(f"Données d'échantillon: {len(sample_data)} lignes")
        print(f"Headers: {headers}")
        
        if not sample_data:
            print("Aucune donnée d'échantillon trouvée")
            return []
        
        # 3. Initialiser les analyseurs (comme dans votre code existant)
        try:
            analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")
            semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")
            auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)
            print("Analyseurs initialisés avec succès")
        except Exception as e:
            print(f"Erreur initialisation analyseurs: {e}")
            return []
        
        # 4. Récupérer le nom du fichier original
        job_record = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})
        original_filename = job_record.get('original_filename', 'dataset.csv') if job_record else 'dataset.csv'
        print(f"Fichier original: {original_filename}")
        
        # 5. Analyser chaque colonne
        metadata_list = []
        
        for header in headers:
            print(f"\n--- Analyse colonne: {header} ---")
            
            # Extraire les valeurs de cette colonne
            column_values = []
            for row in sample_data:
                if header in row and row[header]:
                    value = str(row[header]).strip()
                    if value:
                        column_values.append(value)
            
            print(f"Valeurs collectées: {len(column_values)}")
            
            if not column_values:
                print(f"Aucune valeur pour la colonne {header}")
                continue
            
            # Analyser les entités dans cette colonne
            detected_entities = set()
            sample_values = []
            total_entities = 0
            
            # Analyser quelques valeurs de la colonne
            for i, value in enumerate(column_values[:5]):
                try:
                    print(f"  Analyse valeur {i+1}: '{value[:50]}...'")
                    entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=original_filename)
                    
                    for entity in entities:
                        if entity.entity_type not in EXCLUDED_ENTITY_TYPES:
                            detected_entities.add(entity.entity_type)
                            total_entities += 1
                            print(f"    -> Entité détectée: {entity.entity_type} = '{entity.entity_value}'")
                            
                            # Garder un échantillon de la valeur
                            if len(sample_values) < 3:
                                sample_values.append(entity.entity_value)
                                
                except Exception as e:
                    print(f"  Erreur analyse valeur '{value[:30]}...': {e}")
                    continue
            
            print(f"Entités détectées pour {header}: {list(detected_entities)} (total: {total_entities})")
            
            # Si des entités ont été détectées, créer l'entrée métadonnées
            if detected_entities:
                # Déterminer les recommandations automatiques
                rgpd_category = self._get_rgpd_category(detected_entities)
                sensitivity_level = self._get_sensitivity_level(detected_entities)  
                ranger_policy = self._get_ranger_policy(detected_entities)
                
                column_metadata = {
                    'column_name': header,
                    'entity_types': list(detected_entities),
                    'sample_values': sample_values,
                    'total_entities': total_entities,
                    'recommended_rgpd_category': rgpd_category,
                    'recommended_sensitivity_level': sensitivity_level,
                    'recommended_ranger_policy': ranger_policy,
                    'validation_status': 'pending'
                }
                
                metadata_list.append(column_metadata)
                print(f"Métadonnées créées pour {header}")
        
        print(f"\n=== RÉSULTAT: Métadonnées générées pour {len(metadata_list)} colonnes ===")
        return metadata_list
        
    except Exception as e:
        print(f"Erreur globale dans _get_enriched_metadata: {e}")
        import traceback
        traceback.print_exc()
        return []

 def _get_rgpd_category(self, detected_entities):
    """Détermine la catégorie RGPD basée sur les entités détectées"""
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
    """Détermine le niveau de sensibilité"""
    if any(entity in detected_entities for entity in ['PERSON', 'ID_MAROC']):
        return "PERSONAL_DATA"
    elif any(entity in detected_entities for entity in ['IBAN_CODE', 'CREDIT_CARD']):
        return "RESTRICTED"
    elif any(entity in detected_entities for entity in ['PHONE_NUMBER', 'EMAIL_ADDRESS']):
        return "CONFIDENTIAL"
    else:
        return "INTERNAL"

 def _get_ranger_policy(self, detected_entities):
    """Détermine la politique Ranger"""
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