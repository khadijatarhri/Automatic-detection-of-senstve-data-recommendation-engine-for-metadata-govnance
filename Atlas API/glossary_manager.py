from pymongo import MongoClient  
from typing import Dict, List, Optional  
from dataclasses import dataclass  
from datetime import datetime  
import json  
from semantic_engine import SemanticAnalyzer  
  
@dataclass  
class GlossaryTerm:  
    """Structure d'un terme du glossaire"""  
    name: str  
    definition: str  
    category: str  
    sensitivity_level: str  
    anonymization_method: str  
    source: str = "system"  
    created_at: datetime = None  
    updated_at: datetime = None  
    validation_count: int = 0  
  
class GlossaryManager:  
    """Gestionnaire du glossaire RGPD"""  
      
    def __init__(self):  
        self.client = MongoClient('mongodb://mongodb:27017/')  
        self.glossary_db = self.client['glossary_db']  
        self.terms_collection = self.glossary_db['terms']  
        self.semantic_analyzer = SemanticAnalyzer("moroccan_entities_model_v2")  
      
    def initialize_base_glossary(self):  
        """Initialise le glossaire avec les mappings existants du SemanticAnalyzer"""  
        base_terms = self._extract_from_semantic_analyzer()  
          
        for term_data in base_terms:  
            existing_term = self.terms_collection.find_one({'name': term_data['name']})  
            if not existing_term:  
                term_data['created_at'] = datetime.now()  
                term_data['updated_at'] = datetime.now()  
                self.terms_collection.insert_one(term_data)  
                print(f"Terme ajouté au glossaire: {term_data['name']}")  
            else:  
                print(f"Terme déjà existant: {term_data['name']}")  
      
    def _extract_from_semantic_analyzer(self) -> List[Dict]:  
        """Extrait les termes de base depuis SemanticAnalyzer"""  
        base_terms = []  
          
        # Utiliser les mappings RGPD existants  
        for entity_type, rgpd_category in self.semantic_analyzer.rgpd_mapping.items():  
            anonymization_method = self.semantic_analyzer.anonymization_methods.get(entity_type, "masquage")  
              
            term = {  
                'name': entity_type,  
                'definition': self._generate_definition(entity_type),  
                'category': rgpd_category,  
                'sensitivity_level': self._determine_sensitivity(entity_type),  
                'anonymization_method': anonymization_method,  
                'source': 'semantic_analyzer',  
                'validation_count': 0  
            }  
            base_terms.append(term)  
          
        return base_terms  
      
    def _generate_definition(self, entity_type: str) -> str:  
        """Génère une définition pour un type d'entité"""  
        definitions = {  
            'PERSON': 'Nom et prénom d\'une personne physique',  
            'ID_MAROC': 'Carte d\'Identité Nationale marocaine - Identifiant unique des citoyens',  
            'PHONE_NUMBER': 'Numéro de téléphone personnel ou professionnel',  
            'EMAIL_ADDRESS': 'Adresse électronique personnelle ou professionnelle',  
            'LOCATION': 'Données de géolocalisation ou adresse physique',  
            'IBAN_CODE': 'International Bank Account Number - Numéro de compte bancaire international',  
            'DATE_TIME': 'Information temporelle (date, heure, timestamp)'  
        }  
        return definitions.get(entity_type, f"Entité de type {entity_type}")  
      
    def _determine_sensitivity(self, entity_type: str) -> str:  
        """Détermine le niveau de sensibilité d'un type d'entité"""  
        if entity_type in ['PERSON', 'ID_MAROC']:  
            return 'PERSONAL_DATA'  
        elif entity_type in ['IBAN_CODE']:  
            return 'RESTRICTED'  
        elif entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS', 'LOCATION']:  
            return 'CONFIDENTIAL'  
        else:  
            return 'INTERNAL'  
      
    def get_term(self, name: str) -> Optional[Dict]:  
        """Récupère un terme du glossaire"""  
        return self.terms_collection.find_one({'name': name})  
      
    def get_all_terms(self) -> List[Dict]:  
        """Récupère tous les termes du glossaire"""  
        return list(self.terms_collection.find())  
      
    def create_term(self, term_data: Dict) -> str:  
        """Crée un nouveau terme dans le glossaire"""  
        term_data['created_at'] = datetime.now()  
        term_data['updated_at'] = datetime.now()  
        result = self.terms_collection.insert_one(term_data)  
        return str(result.inserted_id)  
      
    def update_term(self, name: str, updates: Dict) -> bool:  
        """Met à jour un terme existant"""  
        updates['updated_at'] = datetime.now()  
        result = self.terms_collection.update_one(  
            {'name': name},  
            {'$set': updates}  
        )  
        return result.modified_count > 0

class GlossaryTermExtractor:  
    """Extrait les termes validés des annotations pour enrichir le glossaire"""  
      
    def __init__(self, glossary_manager: GlossaryManager):  
        self.client = MongoClient('mongodb://mongodb:27017/')  
        self.metadata_db = self.client['metadata_validation_db']  
        self.annotations_collection = self.metadata_db['column_annotations']  
        self.glossary_manager = glossary_manager  
      
    def extract_validated_terms(self) -> List[Dict]:  
        """Extrait les termes validés des annotations"""  
        validated_annotations = self.annotations_collection.find({  
            'validation_status': 'validated'  
        })  
          
        terms_data = {}  
        for annotation in validated_annotations:  
            entity_type = annotation['entity_type']  
              
            if entity_type not in terms_data:  
                # Récupérer le terme existant ou créer un nouveau  
                existing_term = self.glossary_manager.get_term(entity_type)  
                if existing_term:  
                    terms_data[entity_type] = existing_term.copy()  
                else:  
                    terms_data[entity_type] = {  
                        'name': entity_type,  
                        'definition': self._generate_enhanced_definition(entity_type, annotation),  
                        'category': annotation.get('rgpd_category', 'Non classifié'),  
                        'sensitivity_level': 'CONFIDENTIAL',  
                        'anonymization_method': annotation.get('anonymization_method', 'masquage'),  
                        'source': 'user_validation',  
                        'validation_count': 0,  
                        'user_comments': []  
                    }  
              
            # Enrichir avec les données de validation  
            terms_data[entity_type]['validation_count'] += 1  
            if annotation.get('annotation_comments'):  
                if 'user_comments' not in terms_data[entity_type]:  
                    terms_data[entity_type]['user_comments'] = []  
                terms_data[entity_type]['user_comments'].append({  
                    'comment': annotation['annotation_comments'],  
                    'validated_by': annotation.get('validated_by'),  
                    'date': annotation.get('validation_date')  
                })  
          
        return list(terms_data.values())  
      
    def _generate_enhanced_definition(self, entity_type: str, annotation: Dict) -> str:  
        """Génère une définition enrichie basée sur les annotations"""  
        base_definition = self.glossary_manager._generate_definition(entity_type)  
          
        if annotation.get('annotation_comments'):  
            return f"{base_definition}. Note: {annotation['annotation_comments']}"  
          
        return base_definition  
      
    def enrich_glossary_from_annotations(self) -> Dict:  
        """Enrichit le glossaire avec les termes validés"""  
        validated_terms = self.extract_validated_terms()  
          
        enriched_count = 0  
        created_count = 0  
          
        for term_data in validated_terms:  
            existing_term = self.glossary_manager.get_term(term_data['name'])  
              
            if existing_term:  
                # Mettre à jour le terme existant  
                updates = {  
                    'validation_count': term_data['validation_count'],  
                    'user_comments': term_data.get('user_comments', []),  
                    'category': term_data['category'],  
                    'anonymization_method': term_data['anonymization_method']  
                }  
                self.glossary_manager.update_term(term_data['name'], updates)  
                enriched_count += 1  
            else:  
                # Créer un nouveau terme  
                self.glossary_manager.create_term(term_data)  
                created_count += 1  
          
        return {  
            'enriched_terms': enriched_count,  
            'created_terms': created_count,  
            'total_processed': len(validated_terms)  
        }
    


    def extract_with_column_samples(self, job_id: str) -> List[Dict]:  
       """Extrait les termes avec échantillons de valeurs des colonnes"""  
       validated_terms = self.extract_validated_terms()  
      
       # Récupérer les données CSV pour enrichir avec des échantillons  
       from pymongo import MongoClient  
       client = MongoClient('mongodb://mongodb:27017/')  
       csv_db = client['csv_anonymizer_db']  
       csv_collection = csv_db['csv_data']  
      
       csv_data = csv_collection.find_one({'job_id': job_id})  
      
       if csv_data:  
        for term in validated_terms:  
            entity_type = term['name']  
            # Trouver des échantillons de valeurs pour ce type d'entité  
            samples = []  
            for row in csv_data.get('data', [])[:10]:  # Limiter à 10 échantillons  
                for column, value in row.items():  
                    if isinstance(value, str) and value.strip():  
                        # Ici vous pourriez utiliser votre détecteur d'entités  
                        # pour vérifier si cette valeur correspond au type d'entité  
                        samples.append({  
                            'column': column,  
                            'value': value[:50],  # Tronquer pour la sécurité  
                            'masked_value': self._mask_sensitive_value(value, entity_type)  
                        })  
              
            term['column_samples'] = samples[:5]  # Garder 5 échantillons max  
      
       return validated_terms  
  
    def _mask_sensitive_value(self, value: str, entity_type: str) -> str:  
        """Masque les valeurs sensibles pour l'affichage"""  
        if entity_type in ['PERSON', 'ID_MAROC']:  
           return f"{value[:2]}***{value[-2:]}" if len(value) > 4 else "***"  
        elif entity_type == 'EMAIL_ADDRESS':  
            parts = value.split('@')  
            if len(parts) == 2:  
               return f"{parts[0][:2]}***@{parts[1]}"  
        return value[:3] + "***"