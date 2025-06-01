import csv  
import io  
import os  
from bson import ObjectId  
from django.conf import settings  
from django.shortcuts import render, redirect  
from django.views import View  
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern  
from presidio_anonymizer import AnonymizerEngine  
from pymongo import MongoClient  
from django.http import HttpResponse, JsonResponse  
from presidio_anonymizer.entities import OperatorConfig  
  
  
# === Connexion MongoDB ===  
client = MongoClient("mongodb://localhost:27017/")  
db = client["anonymisation_db"]  
collection = db["csv_files"]  
jobs_collection = db["anonymisation_jobs"]  
  
# === Détection personnalisée pour le Maroc ===  
class CustomPresidioAnalyzer:  
    def __init__(self):  
        self.analyzer = AnalyzerEngine()  
        self.add_custom_recognizers()  
  
    def add_custom_recognizers(self):  
        recognizers = [  
            {  
                "name": "Moroccan CIN",  
                "entity": "CIN",  
                "pattern": r'\b([A-Z]{1,2}\d{4,6})\b'  
            },  
            {  
                "name": "Moroccan Phone",  
                "entity": "PHONE_MA",  
                "pattern": r'\b(0[5-7]\d{8})\b'  
            },  
            {  
                "name": "Moroccan IBAN",  
                "entity": "IBAN_MA",  
                "pattern": r'\bMA\d{24}\b'  
            },  
            {  
                "name": "Moroccan RIB",  
                "entity": "RIB_MA",  
                "pattern": r'\b\d{24}\b'  
            },  
        ]  
        for rec in recognizers:  
            pattern = Pattern(name=rec["name"], regex=rec["pattern"], score=0.8)  
            recognizer = PatternRecognizer(supported_entity=rec["entity"], patterns=[pattern])  
            self.analyzer.registry.add_recognizer(recognizer)  
  
    def analyze_text(self, text):  
        return self.analyzer.analyze(text=text, language="en")  
  
  
custom_analyzer = CustomPresidioAnalyzer()  
anonymizer = AnonymizerEngine()  
  
  
# === Vue Upload ===  
class UploadCSVView(View):  
    def get(self, request):  
        return render(request, 'csv_anonymizer/upload.html')  
  
    def post(self, request):  
        uploaded_file = request.FILES['csv_file']  
          
        # Lire le contenu du fichier une seule fois  
        csv_content = uploaded_file.read().decode('utf-8')  
          
        # Parser le CSV  
        csv_reader = csv.reader(io.StringIO(csv_content))  
        header = next(csv_reader)  
        sample_rows = [row for _, row in zip(range(10), csv_reader)]  
  
        # Analyser les entités sensibles  
        sensitive_columns = {}  
        for col_index, column_name in enumerate(header):  
            column_data = ' '.join(row[col_index] for row in sample_rows if len(row) > col_index)  
            results = custom_analyzer.analyze_text(column_data)  
            entities = list({result.entity_type for result in results})  
            if entities:  
                sensitive_columns[column_name] = entities  
  
        # Stocker dans MongoDB avec le contenu  
        job_id = jobs_collection.insert_one({  
            "filename": uploaded_file.name,  
            "csv_content": csv_content,  
            "sensitive_columns": sensitive_columns,  
            "status": "uploaded"  
        }).inserted_id  
          
        # Filtrer les entités non pertinentes  
        excluded_entities = {'US_DRIVER_LICENSE', 'US_BANK_NUMBER', 'US_PASSPORT', 'UK_NHS', 'IN_AADHAAR'}  
          
        detected_entities = set()  
        for entities_list in sensitive_columns.values():  
            for entity in entities_list:  
                if entity not in excluded_entities:  
                    detected_entities.add(entity)  
          
        return render(request, 'csv_anonymizer/select_entities.html', {  
            "job_id": str(job_id),  
            "detected_entities": list(detected_entities),  
            "sensitive_columns": sensitive_columns  
        })  
  
  
# === Vue Sélection des entités ===  
class ProcessCSVView(View):  
    def get(self, request, job_id):  
        try:  
            job = jobs_collection.find_one({"_id": ObjectId(job_id)})  
        except Exception:  
            return render(request, 'csv_anonymizer/error.html', {"message": "ID de tâche invalide."})  
  
        return render(request, 'csv_anonymizer/select_entities.html', {  
            "job_id": job_id,  
            "sensitive_columns": job.get("sensitive_columns", {})  
        })  
  
    def post(self, request, job_id):  
        try:  
            job = jobs_collection.find_one({"_id": ObjectId(job_id)})  
        except Exception:  
            return render(request, 'csv_anonymizer/error.html', {"message": "ID de tâche invalide."})  
  
        selected_entities = request.POST.getlist('entities')  
        method = request.POST.get('method', 'replace')  
  
        # Lire le contenu CSV depuis MongoDB  
        csv_data = job["csv_content"]  
  
        csv_reader = csv.reader(io.StringIO(csv_data))  
        header = next(csv_reader)  
        output = io.StringIO()  
        csv_writer = csv.writer(output)  
        csv_writer.writerow(header)  
  
        entity_counts = {}  
  
        for row in csv_reader:  
            new_row = []  
            for i, cell in enumerate(row):  
                cell_entities = custom_analyzer.analyze_text(cell)  
                filtered = [e for e in cell_entities if e.entity_type in selected_entities]  
                if filtered:  
                    entity_counts.setdefault(header[i], 0)  
                    entity_counts[header[i]] += len(filtered)  
                    anonymized_result = anonymizer.anonymize(  
                        text=cell,  
                        analyzer_results=filtered,  
                        operators={e.entity_type: OperatorConfig("replace", {"new_value": "[MASQUÉ]"}) for e in filtered}  
                    )  
                    new_row.append(anonymized_result.text)  
                else:  
                    new_row.append(cell)  
            csv_writer.writerow(new_row)  
  
        anonymized_data = output.getvalue()  
  
        # Stocker le résultat anonymisé dans MongoDB  
        jobs_collection.update_one(  
            {"_id": ObjectId(job_id)},  
            {  
                "$set": {  
                    "status": "anonymized",  
                    "anonymized_content": anonymized_data,  
                    "entity_counts": entity_counts,  
                    "method": method  
                }  
            }  
        )  
  
        # Retourner le fichier anonymisé directement  
        response = HttpResponse(anonymized_data, content_type='text/csv')  
        response['Content-Disposition'] = f'attachment; filename="anonymized_{job["filename"]}"'  
        return response
    

class StatisticsView(View):  
    def get(self, request):  
        if not request.session.get("user_email"):  
            return redirect('authapp:login_form')  
          
        user_email = request.session.get("user_email")  
          

        from db_connections import db
        # Récupérer tous les jobs complétés de l'utilisateur  
        completed_jobs = list(db.anonymization_jobs.find({  
            'user_email': user_email,  
            'status': 'completed'  
        }))  
          
        # Analyser les statistiques  
        entity_stats = {}  
        risk_levels = {'critique': 0, 'élevé': 0, 'moyen': 0, 'faible': 0}  
        total_files = len(completed_jobs)  
          
        # Exemple de données pour tester  
        entity_stats = {  
            'PERSON': 25,  
            'EMAIL_ADDRESS': 15,  
            'PHONE_NUMBER': 10,  
            'DATE_TIME': 8  
        }  
        risk_levels = {'critique': 30, 'élevé': 25, 'moyen': 35, 'faible': 10}  
          
        context = {  
            'entity_stats': entity_stats,  
            'risk_levels': risk_levels,  
            'total_files': total_files,  
            'completed_jobs': completed_jobs  
        }  
          
        return render(request, 'csv_anonymizer/statistics.html', context)