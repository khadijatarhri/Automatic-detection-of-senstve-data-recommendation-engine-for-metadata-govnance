# Installation des dépendances nécessaires
!pip install spacy presidio-analyzer presidio-anonymizer pandas
!python -m spacy download fr_core_news_sm

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import pandas as pd
import json
import random
from pathlib import Path
import re
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizero
from presidio_analyzer.nlp_engine import NlpEngineProvider



from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv("entites_nettoye1 (4).csv")  # avec guillemets ✔️

# =============================================================================
# ÉTAPE 1: PRÉPARATION DU DATASET D'ENTRAÎNEMENT
# =============================================================================

def prepare_training_data(csv_file_path):
    """
    Convertit le dataset CSV en format d'entraînement spaCy
    
    Args:
        csv_file_path: Chemin vers le fichier CSV
        
    Returns:
        Liste des exemples d'entraînement au format spaCy
    """
    
    # Charger le dataset CSV
    df = pd.read_csv("entites_nettoye1 (4).csv")
    
    training_data = []
    
    for _, row in df.iterrows():
        text = row['text']
        entities = []
        
        # Fonction pour ajouter une entité si elle existe dans le texte
        def add_entity_if_exists(entity_value, entity_type):
            if pd.notna(entity_value) and str(entity_value).strip():
                # Rechercher toutes les occurrences de l'entité dans le texte
                for match in re.finditer(re.escape(str(entity_value)), text):
                    start, end = match.span()
                    entities.append((start, end, entity_type))
        
        # Ajouter chaque type d'entité
        add_entity_if_exists(row['person'], 'PERSON')
        add_entity_if_exists(row['id_maroc'], 'ID_MAROC')
        add_entity_if_exists(row['phone_number'], 'PHONE_NUMBER')
        add_entity_if_exists(row['email_address'], 'EMAIL_ADDRESS')
        add_entity_if_exists(row['iban_code'], 'IBAN_CODE')
        add_entity_if_exists(row['date_time'], 'DATE_TIME')
        add_entity_if_exists(row['location'], 'LOCATION')
        
        # Ajouter à la liste d'entraînement
        training_data.append((text, {"entities": entities}))
    
    return training_data

# =============================================================================
# ÉTAPE 2: CRÉATION ET ENTRAÎNEMENT DU MODÈLE SPACY
# =============================================================================

def create_spacy_model(training_data, model_name="moroccan_entities_model"):
    """
    Crée et entraîne un modèle spaCy personnalisé
    
    Args:
        training_data: Données d'entraînement
        model_name: Nom du modèle à créer
        
    Returns:
        Modèle spaCy entraîné
    """
    
    # Charger le modèle français de base
    nlp = spacy.load("fr_core_news_sm")
    
    # Ajouter le composant NER si il n'existe pas
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Ajouter les nouveaux labels d'entités
    custom_labels = ["ID_MAROC", "PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "DATE_TIME", "LOCATION", "PERSON"]
    for label in custom_labels:
        ner.add_label(label)
    
    # Préparer les données d'entraînement
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    # Désactiver les autres composants pendant l'entraînement
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    # Entraîner le modèle
    with nlp.disable_pipes(*other_pipes):
        # Initialiser l'entraînement
        nlp.begin_training()
        
        # Boucle d'entraînement
        for iteration in range(30):  # 30 itérations d'entraînement
            print(f"Itération {iteration + 1}/30")
            
            # Mélanger les exemples à chaque itération
            random.shuffle(examples)
            
            # Traiter les exemples par lots
            losses = {}
            for batch in spacy.util.minibatch(examples, size=2):
                nlp.update(batch, losses=losses, drop=0.3)
            
            print(f"Pertes: {losses}")
    
    # Sauvegarder le modèle
    nlp.to_disk(model_name)
    print(f"Modèle sauvegardé dans {model_name}")
    
    return nlp
    """
# Dans votre application Django, vous pouvez utiliser le modèle comme suit:

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Charger le modèle une seule fois au démarrage
ANALYZER = create_custom_analyzer_engine("path/to/moroccan_entities_model")

@csrf_exempt

# =============================================================================
# ÉTAPE 3: CRÉATION DES RECOGNIZERS PERSONNALISÉS POUR PRESIDIO
# =============================================================================
"""
class MoroccanIdRecognizer(PatternRecognizer):
    """
    Recognizer personnalisé pour les CIN marocaines
    Format: 2 lettres + 5-6 chiffres
    """
    
    PATTERNS = [
        Pattern("CIN Maroc", r"\b[A-Z]{2}[0-9]{5,6}\b", 0.8),
    ]
    
    CONTEXT = ["cin", "carte", "identité", "cni"]
    
    def __init__(self):
        super().__init__(
            supported_entity="ID_MAROC",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class MoroccanPhoneRecognizer(PatternRecognizer):
    """
    Recognizer personnalisé pour les numéros de téléphone marocains
    Formats: +212XXXXXXXXX, 06XXXXXXXX, 07XXXXXXXX
    """
    
    PATTERNS = [
        Pattern("Téléphone Maroc +212", r"\+212[0-9]{9}", 0.9),
        Pattern("Téléphone Maroc 06", r"\b0[67][0-9]{8}\b", 0.8),
    ]
    
    CONTEXT = ["téléphone", "phone", "mobile", "tel"]
    
    def __init__(self):
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class MoroccanEmailRecognizer(PatternRecognizer):
    """
    Recognizer personnalisé pour les emails
    """
    
    PATTERNS = [
        Pattern("Email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.9),
    ]
    
    CONTEXT = ["email", "mail", "@", "courriel"]
    
    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class IBANRecognizer(PatternRecognizer):
    """
    Recognizer personnalisé pour les codes IBAN
    """
    
    PATTERNS = [
        Pattern("IBAN", r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b", 0.9),
    ]
    
    CONTEXT = ["iban", "compte", "bancaire", "banque"]
    
    def __init__(self):
        super().__init__(
            supported_entity="IBAN_CODE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

# =============================================================================
# ÉTAPE 4: CONFIGURATION DE L'ANALYZER ENGINE PRESIDIO
# =============================================================================

def create_custom_analyzer_engine(model_path):
    """
    Crée un AnalyzerEngine personnalisé avec nos recognizers
    
    Args:
        model_path: Chemin vers le modèle spaCy entraîné
        
    Returns:
        AnalyzerEngine configuré
    """
    
    # Configuration du provider NLP avec notre modèle personnalisé
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "fr",
                "model_name": model_path,
            }
        ],
    }
    
    # Créer le provider NLP
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    
    # Créer les recognizers personnalisés
    moroccan_id_recognizer = MoroccanIdRecognizer()
    moroccan_phone_recognizer = MoroccanPhoneRecognizer()
    moroccan_email_recognizer = MoroccanEmailRecognizer()
    iban_recognizer = IBANRecognizer()
    
    # Créer l'analyzer engine avec seulement nos recognizers
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["fr"]
    )
    
    # Supprimer tous les recognizers par défaut
    analyzer.registry.remove_recognizer("CreditCardRecognizer")
    analyzer.registry.remove_recognizer("CryptoRecognizer")
    analyzer.registry.remove_recognizer("DateRecognizer")
    analyzer.registry.remove_recognizer("EmailRecognizer")
    analyzer.registry.remove_recognizer("IbanRecognizer")
    analyzer.registry.remove_recognizer("IpRecognizer")
    analyzer.registry.remove_recognizer("PhoneRecognizer")
    analyzer.registry.remove_recognizer("UrlRecognizer")
    analyzer.registry.remove_recognizer("UsItinRecognizer")
    analyzer.registry.remove_recognizer("UsLicenseRecognizer")
    analyzer.registry.remove_recognizer("UsPassportRecognizer")
    analyzer.registry.remove_recognizer("UsSsnRecognizer")
    analyzer.registry.remove_recognizer("UsBankRecognizer")
    
    # Ajouter nos recognizers personnalisés
    analyzer.registry.add_recognizer(moroccan_id_recognizer)
    analyzer.registry.add_recognizer(moroccan_phone_recognizer)
    analyzer.registry.add_recognizer(moroccan_email_recognizer)
    analyzer.registry.add_recognizer(iban_recognizer)
    
    return analyzer

# =============================================================================
# ÉTAPE 5: FONCTION DE TEST
# =============================================================================

def test_model(analyzer, test_texts):
    """
    Teste le modèle sur des textes d'exemple
    
    Args:
        analyzer: AnalyzerEngine configuré
        test_texts: Liste de textes à tester
    """
    
    print("=== TESTS DU MODÈLE ===")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Texte: {text}")
        
        # Analyser le texte
        results = analyzer.analyze(text=text, language="fr")
        
        if results:
            print("Entités détectées:")
            for result in results:
                entity_text = text[result.start:result.end]
                print(f"  - {result.entity_type}: '{entity_text}' (Score: {result.score:.2f})")
        else:
            print("Aucune entité détectée")

# =============================================================================
# FONCTION PRINCIPALE D'EXÉCUTION
# =============================================================================

def main():
    """
    Fonction principale qui orchestre tout le processus
    """
    
    # Créer les données d'exemple (remplacez par votre CSV)
    sample_data = """text_id,text,PERSON,ID_MAROC,PHONE_NUMBER,EMAIL_ADDRESS,IBAN_CODE,DATE_TIME,LOCATION
1,"Contrat de location : - Locataire : Grégoire-Julien Chauvet - CIN : HK56938 - Téléphone : +212637506497 - Email : henriettecourtois@example.org",Grégoire-Julien Chauvet,HK56938,+212637506497,henriettecourtois@example.org,,,
2,"Transaction #772013: - Client : Laurence Le Delahaye - IBAN FR1463648284987662786876248 - Date : 2025-04-04",Laurence Le Delahaye,,,,,FR1463648284987662786876248,2025-04-04,"""
    
    # Sauvegarder les données d'exemple
    with open("sample_data.csv", "w", encoding="utf-8") as f:
        f.write(sample_data)
    
    print("=== ÉTAPE 1: PRÉPARATION DES DONNÉES ===")
    training_data = prepare_training_data("sample_data.csv")
    print(f"Nombre d'exemples d'entraînement: {len(training_data)}")
    
    print("\n=== ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE SPACY ===")
    model = create_spacy_model(training_data)
    
    print("\n=== ÉTAPE 3: CRÉATION DE L'ANALYZER ENGINE ===")
    analyzer = create_custom_analyzer_engine("moroccan_entities_model")
    
    print("\n=== ÉTAPE 4: TESTS ===")
    test_texts = [
        "Voici mes informations: CIN HK56938, téléphone +212637506497, email test@example.org",
        "Transaction vers IBAN FR1463648284987662786876248 pour Laurence Le Delahaye",
        "Contact: Ahmed Benali, CIN: AB12345, mobile: +212661234567"
    ]
    
    test_model(analyzer, test_texts)
    
    print("\n=== MODÈLE PRÊT ===")
    print("Votre modèle est maintenant prêt à être utilisé!")
    print("Vous pouvez l'intégrer dans votre application Django.")

# Exécuter le programme principal
if __name__ == "__main__":
    main()
