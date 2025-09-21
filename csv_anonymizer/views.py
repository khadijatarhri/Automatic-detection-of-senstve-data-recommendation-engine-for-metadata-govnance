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
import asyncio  
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
from gridfs import GridFS  
from recommendation_engine.recommendation_engine_core import EnterpriseRecommendationEngine  


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
        
fs = GridFS(csv_db)
        
class UploadCSVView(View):  
    def get(self, request):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        # Vérification du rôle  
        user_email = request.session.get("user_email")  
        user = users.find_one({'email': user_email})  
          
        if user and user.get('role') != 'admin':  
            return redirect('authapp:home')  
          
        return render(request, 'csv_anonymizer/upload.html')  
  
    def post(self, request):  
    # Vérifications d'authentification existantes  
     print("=== DÉBUT UPLOAD CSV ===")  
     user_email = request.session.get("user_email")  
     print(f"User email: {user_email}")  
      
     if not user_email:  
        print("ERREUR: Aucun email utilisateur trouvé")  
        return redirect('login_form')  
      
     user = users.find_one({'email': user_email})  
     print(f"Utilisateur trouvé: {user}")  
      
     if user and user.get('role') != 'admin':  
        print(f"ERREUR: Rôle utilisateur incorrect: {user.get('role')}")  
        return redirect('authapp:home')  
      
     csv_file = request.FILES.get('csv_file')  
     if not csv_file:  
        print("ERREUR: Aucun fichier CSV fourni")  
        return render(request, 'csv_anonymizer/upload.html', {'error': 'Aucun fichier sélectionné'})  
      
     print(f"Fichier reçu: {csv_file.name}, taille: {csv_file.size} bytes")  
      
     if not csv_file.name.endswith('.csv'):  
        print(f"ERREUR: Format de fichier incorrect: {csv_file.name}")  
        return render(request, 'csv_anonymizer/upload.html', {'error': 'Le fichier doit être au format CSV'})  
      
    # Traitement par streaming  
     print("Début traitement par streaming...")  
     content_chunks = []  
     chunk_count = 0  
      
     for chunk in csv_file.chunks():  
        content_chunks.append(chunk.decode('utf-8'))  
        chunk_count += 1  
      
     print(f"Streaming terminé: {chunk_count} chunks traités")  
      
     content = ''.join(content_chunks)  
     print(f"Contenu total: {len(content)} caractères")  
      
     reader = csv.reader(io.StringIO(content))  
     headers = next(reader)  
     print(f"Headers détectés: {headers}")  
      
    # Stocker le fichier avec GridFS  
     print("Stockage GridFS...")  
     try:  
        file_id = fs.put(  
            io.BytesIO(content.encode('utf-8')),  
            filename=csv_file.name,  
            content_type='text/csv',  
            upload_date=datetime.datetime.now()  
        )  
        print(f"Fichier stocké dans GridFS avec ID: {file_id}")  
     except Exception as e:  
        print(f"ERREUR GridFS: {e}")  
        return render(request, 'csv_anonymizer/upload.html', {'error': f'Erreur de stockage: {e}'})  
      
    # Récupérer tous les data stewards  
     print("Récupération des data stewards...")  
     data_stewards = list(users.find({'role': 'user'}))  
     authorized_emails = [ds['email'] for ds in data_stewards]  
     print(f"Data stewards trouvés: {len(data_stewards)}, emails: {authorized_emails}")  
      
    # Créer le job  
     print("Création du job...")  
     job_data = {  
        'user_email': request.session.get('user_email'),  
        'original_filename': csv_file.name,  
        'gridfs_file_id': file_id,  
        'upload_date': datetime.datetime.now(),  
        'status': 'pending',  
        'shared_with_data_stewards': True,  
        'authorized_users': authorized_emails  
     }  
      
     try:  
        result = main_db.anonymization_jobs.insert_one(job_data)  
        job_id = result.inserted_id  
        print(f"Job créé avec ID: {job_id}")  
     except Exception as e:  
        print(f"ERREUR création job: {e}")  
        return render(request, 'csv_anonymizer/upload.html', {'error': f'Erreur de création du job: {e}'})  
      
    # Échantillonnage : analyser seulement 10 lignes  
     print("Début échantillonnage (10 lignes)...")  
     sample_rows = []  
     row_count = 0  
      
     for i, row in enumerate(reader):  
        if i >= 10:  
            break  
        row_data = {}  
        for j, header in enumerate(headers):  
            if j < len(row):  
                row_data[header] = row[j]  
        sample_rows.append(row_data)  
        row_count += 1  
      
     print(f"Échantillonnage terminé: {row_count} lignes analysées")  
     print(f"Exemple de données: {sample_rows[0] if sample_rows else 'Aucune donnée'}")  
      
    # Analyser seulement l'échantillon  
     print("Début analyse des entités...")  
     try:  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
        print("Analyseurs initialisés avec succès")  
     except Exception as e:  
        print(f"ERREUR initialisation analyseurs: {e}")  
        return render(request, 'csv_anonymizer/upload.html', {'error': f'Erreur d\'initialisation: {e}'})  
      
     detected_entities = set()  
     entity_count = 0  
      
     for row_idx, row in enumerate(sample_rows):  
        print(f"Analyse ligne {row_idx + 1}/{len(sample_rows)}")  
        for header, value in row.items():  
            if isinstance(value, str) and value.strip():  
                try:  
                    entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=csv_file.name)  
                    for entity in entities:  
                        if entity.entity_type not in EXCLUDED_ENTITY_TYPES:  
                            detected_entities.add(entity.entity_type)  
                            entity_count += 1  
                except Exception as e:  
                    print(f"ERREUR analyse entité '{value}': {e}")  
      
     print(f"Analyse terminée: {entity_count} entités trouvées")  
     print(f"Types d'entités détectées: {list(detected_entities)}")  
      
    # NOUVEAU : Après l'analyse des entités, sauvegarder tout le fichier en chunks  
     print("Début sauvegarde en chunks...")  
     reader = csv.reader(io.StringIO(content))  # Re-créer le reader  
     headers = next(reader)  # Skip headers again  
      
    # Sauvegarder toutes les données en chunks  
     chunk_size = 1000  
     chunk_number = 0  
     current_chunk = []  
     total_rows = 0  
      
     for row in reader:  
        current_chunk.append(row)  
        total_rows += 1  
          
        if len(current_chunk) >= chunk_size:  
            try:  
                self._save_chunk(job_id, chunk_number, headers, current_chunk)  
                print(f"Chunk {chunk_number} sauvegardé: {len(current_chunk)} lignes")  
                current_chunk = []  
                chunk_number += 1  
            except Exception as e:  
                print(f"ERREUR sauvegarde chunk {chunk_number}: {e}")  
      
    # Sauvegarder le dernier chunk  
     if current_chunk:  
        try:  
            self._save_chunk(job_id, chunk_number, headers, current_chunk)  
            print(f"Dernier chunk {chunk_number} sauvegardé: {len(current_chunk)} lignes")  
        except Exception as e:  
            print(f"ERREUR sauvegarde dernier chunk: {e}")  
      
     print(f"Sauvegarde chunks terminée: {chunk_number + 1} chunks, {total_rows} lignes totales")  
      
    # Générer les recommandations IA (logique existante conservée)  
     print("=== DÉBUT GÉNÉRATION RECOMMANDATIONS ===")  
     try:  
        gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCnVTAxyObTDo4fNYeElF49EMvCGz6pLXQ')  
        print(f"API Key Gemini: {gemini_api_key[:10]}...{gemini_api_key[-5:] if gemini_api_key else 'VIDE'}")  
          
        if not gemini_api_key or gemini_api_key == 'AIzaSyCnVTAxyObTDo4fNYeElF49EMvCGz6pLXQ':  
            print("ATTENTION: Utilisation de l'API key par défaut")  
          
        async def generate_recommendations_async():  
            print("Initialisation GeminiClient...")  
            async with GeminiClient(gemini_api_key) as gemini_client:  
                print("GeminiClient initialisé")  
                recommendation_engine = EnterpriseRecommendationEngine(gemini_client)  
                print("EnterpriseRecommendationEngine créé")  
                  
                # Créer le profil du dataset  
                print("Création du profil dataset...")  
                dataset_profile = recommendation_engine.create_dataset_profile_from_presidio(  
                    str(job_id), detected_entities, headers, sample_rows  
                )  
                print(f"Profil dataset créé: {len(dataset_profile)} éléments")  
                  
                # Générer les recommandations structurées  
                print("Génération des recommandations...")  
                comprehensive_recommendations = await recommendation_engine.generate_structured_recommendations(dataset_profile)  
                print(f"Recommandations générées: {type(comprehensive_recommendations)}")  
                return comprehensive_recommendations  
          
        print(f"Début génération recommandations pour job_id: {job_id}")  
        print(f"Entités détectées: {detected_entities}")  
        print(f"Headers: {headers}")  
          
        comprehensive_recommendations = asyncio.run(generate_recommendations_async())  
          
        if hasattr(comprehensive_recommendations, 'recommendations'):  
            recommendations = comprehensive_recommendations.recommendations  
            print(f"Recommandations extraites: {len(recommendations)} items")  
        else:  
            print(f"ATTENTION: Format de recommandations inattendu: {type(comprehensive_recommendations)}")  
            recommendations = []  
          
        print(f"Recommandations finales: {len(recommendations)}")  
          
        request.session['current_job_id'] = str(job_id)  
        print(f"Job ID sauvegardé en session: {str(job_id)}")  
          
     except Exception as e:  
        print(f"ERREUR DÉTAILLÉE RECOMMANDATIONS: {type(e).__name__}: {e}")  
        import traceback  
        traceback.print_exc()  
        recommendations = []  
        print("Recommandations définies comme liste vide suite à l'erreur")  
      
     print(f"=== RÉSULTAT FINAL ===")  
     print(f"Job ID: {job_id}")  
     print(f"Entités détectées: {len(detected_entities)} types")  
     print(f"Headers: {len(headers)} colonnes")  
     print(f"Recommandations: {len(recommendations)} items")  
     print(f"has_recommendations sera: {len(recommendations) > 0}")  

      
     return render(request, 'csv_anonymizer/select_entities.html', {  
        'job_id': str(job_id),  
        'detected_entities': list(detected_entities),  
        'headers': headers,  
        'has_recommendations': True  
     })

    def _process_file_by_chunks(self, file_id, job_id, filename, request):  
        """Traite le fichier CSV par chunks pour éviter les problèmes de mémoire"""  
          
        # Récupérer le fichier depuis GridFS  
        grid_file = fs.get(file_id)  
          
        # Lire par chunks de 1000 lignes  
        chunk_size = 1000  
        headers = None  
        detected_entities = set()  
        chunk_number = 0  
          
        # Initialiser les analyseurs (même logique qu'avant)  
        analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
        semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
        auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
          
        # Lire le fichier ligne par ligne  
        content = grid_file.read().decode('utf-8')  
        reader = csv.reader(io.StringIO(content))  
          
        # Traiter par chunks  
        current_chunk = []  
        for i, row in enumerate(reader):  
            if i == 0:  # Headers  
                headers = row  
                continue  
              
            current_chunk.append(row)  
              
            # Traiter le chunk quand il atteint la taille limite  
            if len(current_chunk) >= chunk_size:  
                entities = self._analyze_chunk(current_chunk, headers, auto_tagger, filename)  
                detected_entities.update(entities)  
                  
                # Sauvegarder le chunk  
                self._save_chunk(job_id, chunk_number, headers, current_chunk)  
                  
                current_chunk = []  
                chunk_number += 1  
          
        # Traiter le dernier chunk  
        if current_chunk:  
            entities = self._analyze_chunk(current_chunk, headers, auto_tagger, filename)  
            detected_entities.update(entities)  
            self._save_chunk(job_id, chunk_number, headers, current_chunk)  
          
        # Générer les recommandations IA (préserver la fonctionnalité existante)  
        recommendations = []  
        try:  
            gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCnVTAxyObTDo4fNYeElF49EMvCGz6pLXQ')  
              
            import asyncio  
            async def generate_recommendations_async():  
                async with GeminiClient(gemini_api_key) as gemini_client:  
                    recommendation_engine = IntelligentRecommendationEngine(gemini_client, gemini_api_key)  
                      
                    # Créer le profil du dataset avec les données par chunks  
                    sample_data = self._get_sample_data_from_chunks(job_id, headers)  
                    dataset_profile = recommendation_engine.create_dataset_profile_from_presidio(  
                        str(job_id), detected_entities, headers, sample_data  
                    )  
                      
                    comprehensive_recommendations = await recommendation_engine.generate_comprehensive_recommendations(dataset_profile)  
                    return comprehensive_recommendations  
              
            comprehensive_recommendations = asyncio.run(generate_recommendations_async())  
            recommendations = comprehensive_recommendations.recommendations  
              
            # Stocker l'ID du job pour les recommandations  
            request.session['current_job_id'] = str(job_id)  
              
        except Exception as e:  
            print(f"Erreur lors de la génération des recommandations: {e}")  
            recommendations = []  
          
        # Retour identique à l'original  
        return render(request, 'csv_anonymizer/select_entities.html', {  
            'job_id': str(job_id),  
            'detected_entities': list(detected_entities),  
            'headers': headers,  
            'has_recommendations': len(recommendations) > 0  
        })  
  
    def _analyze_chunk(self, chunk_data, headers, auto_tagger, filename):  
        """Analyse un chunk de données"""  
        detected_entities = set()  
          
        for row in chunk_data:  
            for i, value in enumerate(row):  
                if isinstance(value, str) and i < len(headers):  
                    entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=filename)  
                    for entity in entities:  
                        if entity.entity_type not in EXCLUDED_ENTITY_TYPES:  
                            detected_entities.add(entity.entity_type)  
          
        return detected_entities  
  
    def _save_chunk(self, job_id, chunk_number, headers, chunk_data):  
        """Sauvegarde un chunk dans MongoDB"""  
        chunk_doc = {  
            'job_id': str(job_id),  
            'chunk_number': chunk_number,  
            'headers': headers,  
            'data': [dict(zip(headers, row)) for row in chunk_data],  
            'created_at': datetime.datetime.now()  
        }  
        csv_db['csv_chunks'].insert_one(chunk_doc)  
  
    def _get_sample_data_from_chunks(self, job_id, headers):  
        """Récupère un échantillon de données depuis les chunks pour les recommandations IA"""  
        # Prendre les 10 premières lignes du premier chunk pour l'analyse IA  
        first_chunk = csv_db['csv_chunks'].find_one({'job_id': str(job_id), 'chunk_number': 0})  
        if first_chunk and first_chunk['data']:  
            return first_chunk['data'][:10]  # Limiter à 10 lignes comme dans l'original  
        return []
class ProcessCSVView(View):  
    def post(self, request, job_id):  
     # Vérifications d'authentification existantes  
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
      
    # Récupérer les données depuis les chunks  
     chunks = list(csv_db['csv_chunks'].find({'job_id': str(object_id)}).sort('chunk_number', 1))  
     if not chunks:  
        return JsonResponse({'error': 'Données non trouvées'}, status=404)  
      
     headers = chunks[0]['headers']  
      
    # Combiner toutes les données  
     all_data = []  
     for chunk in chunks:  
        all_data.extend(chunk['data'])  
      
     df = pd.DataFrame(all_data)  
      
    # NOUVELLE APPROCHE : Analyser les colonnes une seule fois  
     analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")  
     semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
     auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)  
      
     job = main_db.anonymization_jobs.find_one({'_id': object_id})  
     original_filename = job['original_filename'] if job else 'file.csv'  
      
    # Analyser les headers pour identifier les colonnes sensibles  
     sensitive_columns = {}  
     for header in headers:  
        if header in df.columns:  
            column_data = df[header].dropna().astype(str).tolist()  
            if column_data:  # Si la colonne a des données  
                entities = self._analyze_column_sample(header, column_data, auto_tagger, original_filename)  
                if entities:  
                    # Filtrer seulement les entités sélectionnées par l'utilisateur  
                    filtered_entities = entities.intersection(set(selected_entities))  
                    if filtered_entities:  
                        sensitive_columns[header] = filtered_entities  
      
    # Anonymiser seulement les colonnes identifiées comme sensibles  
     output_df = df.copy()  
     anonymizer = AnonymizerEngine()  
      
     for column, column_entities in sensitive_columns.items():  
        for index, value in df[column].items():  
            if isinstance(value, str) and value.strip():  
                # Analyser cette valeur spécifique pour obtenir les positions exactes  
                entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=original_filename)  
                  
                results = []  
                for entity in entities:  
                    if entity.entity_type in column_entities:  
                        result = RecognizerResult(  
                            entity_type=entity.entity_type,  
                            start=entity.start_pos,  
                            end=entity.end_pos,  
                            score=entity.confidence_score  
                        )  
                        results.append(result)  
                  
                if results:  
                    anonymizers = {entity_type: OperatorConfig("replace", {"new_value": "[MASQUÉ]"})  
                                  for entity_type in column_entities}  
                      
                    anonymized_text = anonymizer.anonymize(  
                        text=value,  
                        analyzer_results=results,  
                        operators=anonymizers  
                    ).text  
                      
                    output_df.at[index, column] = anonymized_text  
      
    # Mettre à jour le statut du job  
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
      
     return response  
    
    def _process_chunk_anonymization(self, df_chunk, selected_entities, auto_tagger, anonymizer, original_filename):  
        """Traite l'anonymisation d'un chunk de données"""  
        output_df = df_chunk.copy()  
          
        # Pour chaque colonne du DataFrame chunk  
        for column in df_chunk.columns:  
            # Pour chaque ligne dans cette colonne  
            for index, value in df_chunk[column].items():  
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
          
        return output_df

    def _analyze_column_sample(self, header, column_data, auto_tagger, original_filename):  
     """Analyse un échantillon de données d'une colonne pour identifier les entités"""  
     detected_entities = set()  
      
    # Prendre un échantillon de 10-20 valeurs de la colonne  
     sample_size = min(20, len(column_data))  
     sample_values = column_data[:sample_size]  
       
     for value in sample_values:  
        if isinstance(value, str) and value.strip():  
            entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=original_filename)  
            for entity in entities:  
                if entity.entity_type not in EXCLUDED_ENTITY_TYPES:  
                    detected_entities.add(entity.entity_type)  
      
     return detected_entities
      
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
            #anonymized_record = anonymized_collection.find_one({'job_id': job_id})      
                  
            #if anonymized_record:      
                # Utiliser les données anonymisées pour les statistiques    
                #csv_data = anonymized_record['anonymized_data']    

            chunks = list(csv_db['csv_chunks'].find({'job_id': job_id}).sort('chunk_number', 1))  
            if chunks:  
                csv_data = []  
                for chunk in chunks:  
                    csv_data.extend(chunk['data'])  
                      
                for row in csv_data:      
                    for header, value in row.items():      
                        if isinstance(value, str) and value.strip():  
                            job_record = main_db.anonymization_jobs.find_one({'_id': ObjectId(job_id)})  
    
                            filename = job_record['original_filename'] if job_record else 'unknown'  
                            entities, tags = auto_tagger.analyze_and_tag(value, dataset_name=filename)
                              
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
    


class RemoveDuplicatesView(View):  
    def post(self, request):  
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        user_email = request.session.get("user_email")  
        user = users.find_one({'email': user_email})  
          
        if user and user.get('role') != 'admin':  
            return JsonResponse({'error': 'Accès refusé'}, status=403)  
          
        try:  
         data = json.loads(request.body)  
         job_id = data.get('job_id')  
          
         if not job_id:  
            return JsonResponse({'error': 'job_id manquant'}, status=400)  
          
        # 1. Lire tous les chunks du job  
         chunks = list(csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))  
         if not chunks:  
            return JsonResponse({'error': 'Données non trouvées'}, status=404)  
          
        # 2. Combiner toutes les données  
         all_data = []  
         headers = chunks[0]['headers']  
         for chunk in chunks:  
            all_data.extend(chunk['data'])  
          
        # 3. Supprimer les doublons  
         df = pd.DataFrame(all_data)  
         original_count = len(df)  
         df_cleaned = df.drop_duplicates()  
          
        # 4. Re-sauvegarder en chunks  
         self._save_cleaned_data_as_chunks(job_id, headers, df_cleaned.to_dict('records'))  
          
         return JsonResponse({  
            'success': True,  
            'message': f'{original_count - len(df_cleaned)} doublons supprimés',  
            'rows_removed': original_count - len(df_cleaned)  
         })  
      
        except Exception as e:  
         return JsonResponse({'error': str(e)}, status=500)  
  
    def _save_cleaned_data_as_chunks(self, job_id, headers, cleaned_data):  
     """Re-sauvegarde les données nettoyées en chunks"""  
     # Supprimer les anciens chunks  
     csv_db['csv_chunks'].delete_many({'job_id': str(job_id)})  
      
    # Re-créer les chunks avec les données nettoyées  
     chunk_size = 1000  
     for i in range(0, len(cleaned_data), chunk_size):  
        chunk_data = cleaned_data[i:i + chunk_size]  
        chunk_doc = {  
            'job_id': str(job_id),  
            'chunk_number': i // chunk_size,  
            'headers': headers,  
            'data': chunk_data,  
            'created_at': datetime.datetime.now()  
        }  
        csv_db['csv_chunks'].insert_one(chunk_doc)

class RemoveMissingValuesView(View):  
    def post(self, request):  
        if not request.session.get("user_email"):  
            return JsonResponse({'error': 'Non autorisé'}, status=401)  
          
        user_email = request.session.get("user_email")  
        user = users.find_one({'email': user_email})  
          
        if user and user.get('role') != 'admin':  
            return JsonResponse({'error': 'Accès refusé'}, status=403)  
          
        try:  
         data = json.loads(request.body)  
         job_id = data.get('job_id')  
          
         if not job_id:  
            return JsonResponse({'error': 'job_id manquant'}, status=400)  
          
        # Lire depuis les chunks  
         chunks = list(csv_db['csv_chunks'].find({'job_id': str(job_id)}).sort('chunk_number', 1))  
         if not chunks:  
            return JsonResponse({'error': 'Données non trouvées'}, status=404)  
          
        # Combiner et nettoyer  
         all_data = []  
         headers = chunks[0]['headers']  
         for chunk in chunks:  
            all_data.extend(chunk['data'])  
          
         df = pd.DataFrame(all_data)  
         original_count = len(df)  
         df_cleaned = df.dropna()  
          
        # Re-sauvegarder en chunks  
         self._save_cleaned_data_as_chunks(job_id, headers, df_cleaned.to_dict('records'))  
          
         return JsonResponse({  
            'success': True,  
            'message': f'{original_count - len(df_cleaned)} lignes avec valeurs manquantes supprimées',  
            'rows_removed': original_count - len(df_cleaned)  
         })  
      
        except Exception as e:  
         return JsonResponse({'error': str(e)}, status=500)

class DataStewardDashboardView(View):  
    def get(self, request):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        user_email = request.session.get("user_email")  
        user = users.find_one({'email': user_email})  
          
        # Vérifier que c'est un data steward  
        if not user or user.get('role') != 'user':  
            return redirect('authapp:home')  
          
        # Récupérer les jobs récents de l'utilisateur  
        recent_jobs = list(main_db.anonymization_jobs.find({  
            'user_email': user_email,  
            'status': 'completed'  
        }).sort('upload_date', -1).limit(10))  
          
        return render(request, 'csv_anonymizer/datasteward_dashboard.html', {  
            'recent_jobs': recent_jobs,  
            'user': user  
        })