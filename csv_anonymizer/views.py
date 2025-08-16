from django.shortcuts import render, redirect        
from django.http import HttpResponse, JsonResponse        
from django.views import View        
from pymongo import MongoClient        
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine        
from presidio_anonymizer.entities import OperatorConfig        
import pandas as pd        
from presidio_structured import StructuredEngine, PandasAnalysisBuilder        
import csv        
import io    
import traceback    
import json        
from db_connections import db as main_db          
import datetime          
from bson import ObjectId        
from semantic_engine import SemanticAnalyzer, IntelligentAutoTagger    
from presidio_custom import create_enhanced_analyzer_engine    
import os
from recommendation_engine.MoteurDeRecommandationAvecDeepSeekML import IntelligentRecommendationEngine, GeminiClient

# Configuration centralisée des entités à exclure    
EXCLUDED_ENTITY_TYPES = {    
    'IN_PAN', 'URL', 'DOMAIN_NAME', 'NRP', 'US_BANK_NUMBER',    
    'IN_AADHAAR', 'US_DRIVER_LICENSE', 'UK_NHS'    
}    
    
from db_connections import db as main_db      
# Ajoutez cette ligne :      
users = main_db["users"]    
# Connexion à MongoDB pour les données CSV        
client = MongoClient('mongodb://mongodb:27017/')        
csv_db = client['csv_anonymizer_db']        
collection = csv_db['csv_data']        
        
        
class UploadCSVView(View):        
    def get(self, request):          
        if not request.session.get("user_email"):          
            return redirect('login_form')      
              
        # Ajouter la vérification du rôle ici aussi      
        user_email = request.session.get("user_email")      
        user = users.find_one({'email': user_email})      
              
        if user and user.get('role') != 'admin':      
            return redirect('authapp:home')      
                  
        return render(request, 'csv_anonymizer/upload.html')    
            
    def post(self, request):        
        # Vérifiez l'authentification        
        user_email = request.session.get("user_email")        
        if not user_email:        
            return redirect('login_form')        
            
    
        user = users.find_one({'email': user_email})      
      
        if user and user.get('role') != 'admin':      
            return redirect('authapp:home')     
        
        csv_file = request.FILES.get('csv_file')        
        if not csv_file:        
            return render(request, 'csv_anonymizer/upload.html', {'error': 'Aucun fichier sélectionné'})        
        
        if not csv_file.name.endswith('.csv'):        
            return render(request, 'csv_anonymizer/upload.html', {'error': 'Le fichier doit être au format CSV'})        
        
        csv_data = []        
        csv_file_data = csv_file.read().decode('utf-8')        
        reader = csv.reader(io.StringIO(csv_file_data))        
        headers = next(reader)        
        
        for row in reader:        
            row_data = {}        
            for i, header in enumerate(headers):        
                row_data[header] = row[i]        
            csv_data.append(row_data)        
        
        # Utiliser PyMongo directement pour créer le job        
        job_data = {          
            'user_email': request.session.get('user_email'),          
            'original_filename': csv_file.name,        
            'upload_date': datetime.datetime.now(),          
            'status': 'pending'          
        }          
        result = main_db.anonymization_jobs.insert_one(job_data)          
        job_id = result.inserted_id        
        
        # Stocker les données CSV avec l'ID du job        
        collection.insert_one({        
            'job_id': str(job_id),        
            'headers': headers,        
            'data': csv_data        
        })        
        
        # Analyser les entités avec le système sémantique amélioré  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")    
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")    
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
        detected_entities = set()        
        
        for row in csv_data[:10]:        
            for header, value in row.items():        
                if isinstance(value, str):        
                    entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=csv_file.name)  
                    for entity in entities:  
                        # Filtrer les entités indésirables    
                        if entity.entity_type not in EXCLUDED_ENTITY_TYPES:    
                            detected_entities.add(entity.entity_type)        
        
        try:  
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDMZedgDV-_ps4T1Pg03JiLgaq_srrlKDo')  
            print(f"API Key: {gemini_api_key}")  
      
            print("RecommendationEngine créé avec succès")
            import asyncio
            async def generate_recommendations_async():  
                  async with GeminiClient(gemini_api_key) as gemini_client:    
                    recommendation_engine = IntelligentRecommendationEngine(gemini_client , gemini_api_key)    
                    print("RecommendationEngine créé avec succès")  # Debug    
              
                    # Créer le profil du dataset    
                    dataset_profile = recommendation_engine.create_dataset_profile_from_presidio(str(job_id), detected_entities, headers, csv_data )    
                    print(f"Dataset profile créé: {dataset_profile}")  # Debug    
              
                    # Générer les recommandations  
                    print("Début génération des recommandations...")  # Debug     
                    comprehensive_recommendations = await recommendation_engine.generate_comprehensive_recommendations(dataset_profile)    
                    return comprehensive_recommendations  
  
          # Exécuter la fonction asynchrone  
            comprehensive_recommendations = asyncio.run(generate_recommendations_async())  
            recommendations = comprehensive_recommendations.recommendations    
            overall_score = comprehensive_recommendations.overall_score    
            improvement_areas = comprehensive_recommendations.improvement_areas  
  
            # Stocker l'ID du job pour les recommandations  
            request.session['current_job_id'] = str(job_id)  
              
        except Exception as e:  
            print(f"Erreur détaillée: {type(e).__name__}: {e}")  # Plus de détails  
            print(f"Erreur lors de la génération des recommandations: {e}")  
            traceback.print_exc()  # Stack trace complète  
            recommendations = []  
  
            print(f"Final recommendations count: {len(recommendations)}")  # Debug final  
            print(f"has_recommendations will be: {len(recommendations) > 0}")  # Debug final  
    


        return render(request, 'csv_anonymizer/select_entities.html', {  
            'job_id': str(job_id),  
            'detected_entities': list(detected_entities),  
            'headers': headers,  
            'has_recommendations': len(recommendations) > 0  
        })     
      
      
class ProcessCSVView(View):        
    def post(self, request, job_id):        
        # Vérifiez l'authentification        
        if not request.session.get("user_email"):        
            return redirect('login_form')        
        user_email = request.session.get("user_email")      
    
        user = users.find_one({'email': user_email})      
      
        if user and user.get('role') != 'admin':      
            return redirect('authapp:home')        
        # Convertir le string job_id en ObjectId MongoDB si nécessaire        
        try:        
            if len(job_id) == 24:  # ObjectId standard length        
                object_id = ObjectId(job_id)        
            else:        
                object_id = job_id        
        except:        
            return JsonResponse({'error': 'ID invalide'}, status=400)        
                
        # Récupérer les entités sélectionnées        
        selected_entities = request.POST.getlist('entities')        
                
        # Récupérer les données de MongoDB        
        job_data = collection.find_one({'job_id': str(object_id)})        
        if not job_data:        
            return JsonResponse({'error': 'Données non trouvées'}, status=404)        
                
        headers = job_data['headers']        
        csv_data = job_data['data']        
                
        # Convertir en DataFrame pandas        
        df = pd.DataFrame(csv_data)        
                
        # Initialiser les moteurs avec système sémantique  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
        anonymizer = AnonymizerEngine()        
                
        # Créer une copie du DataFrame pour la sortie        
        output_df = df.copy()        
          
        # Récupérer le job pour obtenir le nom du fichier original        
        job = main_db.anonymization_jobs.find_one({'_id': object_id})        
        original_filename = job['original_filename'] if job else 'file.csv'  
                
        # Pour chaque colonne du DataFrame        
        for column in df.columns:        
            # Pour chaque ligne dans cette colonne        
            for index, value in df[column].items():        
                # Vérifier si la valeur est une chaîne avant de l'analyser        
                if isinstance(value, str):        
                    entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=original_filename)  
                    # Convertir les EntityMetadata en format Presidio pour l'anonymisation  
                    results = []  
                    for entity in entities:  
                        if entity.entity_type not in EXCLUDED_ENTITY_TYPES and entity.entity_type in selected_entities:  
                            # Créer un objet compatible avec Presidio  
                            result = RecognizerResult(  
                                entity_type=entity.entity_type,  
                                start=entity.start_pos,  
                                end=entity.end_pos,  
                                score=entity.confidence_score  
                            )  
                            results.append(result)  
                            
                    # Si des entités à anonymiser ont été trouvées        
                    if results:        
                        # Configurer l'anonymisation pour remplacer les valeurs        
                        anonymizers = {entity_type: OperatorConfig("replace", {"new_value": "[MASQUÉ]"})        
                                      for entity_type in selected_entities}        
                                
                        # Anonymiser le texte        
                        anonymized_text = anonymizer.anonymize(        
                            text=value,        
                            analyzer_results=results,        
                            operators=anonymizers        
                        ).text        
                                
                        # Remplacer la valeur dans le DataFrame de sortie        
                        output_df.at[index, column] = anonymized_text        
                
        # Mettre à jour le statut du job avec PyMongo        
        main_db.anonymization_jobs.update_one(        
            {'_id': object_id},        
            {'$set': {'status': 'completed'}}        
        )        
                
        # Préparer le fichier CSV à télécharger        
        output = io.StringIO()        
        output_df.to_csv(output, index=False)        
                
        # Créer la réponse HTTP avec le fichier CSV        
        response = HttpResponse(output.getvalue(), content_type='text/csv')        
        response['Content-Disposition'] = f'attachment; filename="anonymized_{original_filename}"'        
                
        # Sauvegarder les données anonymisées pour les utilisateurs      
        anonymized_collection = csv_db['anonymized_files']      
        anonymized_collection.insert_one({      
           'job_id': str(object_id),      
           'original_filename': original_filename,      
           'anonymized_data': output_df.to_dict('records'),      
           'headers': list(output_df.columns),      
           'anonymized_date': datetime.datetime.now()      
        })      
      
        # Puis nettoyer les données temporaires      
        collection.delete_one({'job_id': str(object_id)})       
                
        return response      
      
      
class StatisticsView(View):      
    def get(self, request):      
        if not request.session.get("user_email"):      
            return redirect('login_form')      
              
        user_email = request.session.get("user_email")      
              
        # Vérifier le rôle de l'utilisateur      
        user = users.find_one({'email': user_email})      
      
        if user and user.get('role') != 'admin':      
            return redirect('authapp:home')     
                
        # Récupérer tous les jobs complétés de l'utilisateur      
        completed_jobs = list(main_db.anonymization_jobs.find({      
            'user_email': user_email,      
            'status': 'completed'      
        }))      
              
        total_files = len(completed_jobs)      
        entity_stats = {}      
        total_entities_detected = 0      
              
        # Utiliser les données anonymisées au lieu des données temporaires    
        anonymized_collection = csv_db['anonymized_files']    
          
        # Initialiser le système sémantique pour les statistiques  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
            
        for job in completed_jobs:      
            job_id = str(job['_id'])      
                  
            # Chercher dans anonymized_files au lieu de csv_data    
            anonymized_record = anonymized_collection.find_one({'job_id': job_id})      
                  
            if anonymized_record:      
                # Utiliser les données anonymisées pour les statistiques    
                csv_data = anonymized_record['anonymized_data']      
                      
                for row in csv_data:      
                    for header, value in row.items():      
                        if isinstance(value, str) and value.strip():      
                            entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=anonymized_record['original_filename'])  
                            for entity in entities:  
                                entity_type = entity.entity_type  
                                # Exclure les entités indésirables des statistiques    
                                if entity_type not in EXCLUDED_ENTITY_TYPES:    
                                    if entity_type in entity_stats:      
                                        entity_stats[entity_type] += 1      
                                    else:      
                                        entity_stats[entity_type] = 1      
                                    total_entities_detected += 1      
              
        # Calculer les niveaux de risque basés sur les types d'entités détectées      
        risk_levels = self.calculate_risk_levels(entity_stats, total_entities_detected)      
              
        context = {      
            'total_files': total_files,      
            'entity_stats': entity_stats,      
            'risk_levels': risk_levels,      
        }      
              
        return render(request, 'csv_anonymizer/statistics.html', context)      
          
    def calculate_risk_levels(self, entity_stats, total_entities):      
        """Calculer les pourcentages de niveaux de risque"""      
        if total_entities == 0:      
            return {      
                'critique': 0,      
                'élevé': 0,      
                'moyen': 0,      
                'faible': 0      
            }      
              
        # Définir les niveaux de risque par type d'entité      
        high_risk_entities = ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD', 'IBAN_CODE', 'ID_MAROC']
        medium_risk_entities = ['DATE_TIME', 'LOCATION', 'IP_ADDRESS']      
              
        critique_count = 0      
        eleve_count = 0      
        moyen_count = 0      
        faible_count = 0      
              
        for entity_type, count in entity_stats.items():      
            if entity_type in high_risk_entities:      
                critique_count += count      
            elif entity_type in medium_risk_entities:      
                eleve_count += count      
            else:      
                moyen_count += count      
              
        # Calculer les pourcentages      
        critique_percent = round((critique_count / total_entities) * 100, 1)      
        eleve_percent = round((eleve_count / total_entities) * 100, 1)      
        moyen_percent = round((moyen_count / total_entities) * 100, 1)      
        faible_percent = round(100 - critique_percent - eleve_percent - moyen_percent, 1)      
              
        return {      
            'critique': critique_percent,      
            'élevé': eleve_percent,      
            'moyen': moyen_percent,      
            'faible': max(0, faible_percent)  # S'assurer que ce n'est pas négatif      
        }