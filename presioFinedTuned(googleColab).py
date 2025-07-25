
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



!pip install faker

from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta
import csv

fake = Faker('fr_FR')
Faker.seed(42)

# Données spécifiquement marocaines
MOROCCAN_FIRST_NAMES_MALE = [
    "Mohammed", "Ahmed", "Hassan", "Youssef", "Omar", "Khalid", "Abdellah", "Rachid",
    "Abdelaziz", "Mustapha", "Karim", "Said", "Abderrahim", "Noureddine", "Larbi",
    "Brahim", "Driss", "Hamid", "Abdelkader", "Fouad", "Jamal", "Aziz", "Othman",
    "Tarik", "Hicham", "Amine", "Mehdi", "Samir", "Adil", "Reda", "Zakaria"
]

MOROCCAN_FIRST_NAMES_FEMALE = [
    "Fatima", "Aicha", "Khadija", "Zineb", "Amina", "Latifa", "Hayat", "Najat",
    "Malika", "Zohra", "Samira", "Nadia", "Houria", "Rajae", "Salma", "Leila",
    "Karima", "Nawal", "Souad", "Hafida", "Widad", "Btissam", "Sanae", "Ghizlane",
    "Siham", "Imane", "Laila", "Nezha", "Asma", "Houda", "Meriem", "Yassmine"
]

MOROCCAN_LAST_NAMES = [
    "Alaoui", "Bennani", "El Fassi", "Berrada", "Tazi", "Benali", "Sefrioui", "Benkirane",
    "Lahlou", "Benjelloun", "Filali", "Zniber", "Kettani", "Chraibi", "Mekouar", "Benslimane",
    "Bouchentouf", "Idrissi", "Lamrini", "Benhima", "Guerraoui", "Alami", "Benabdellah",
    "El Malki", "Bensouda", "Tahiri", "Benaissa", "Chorfi", "Lamrani", "Benomar", "Skalli",
    "Bahnini", "Hajji", "Nassiri", "Benkirane", "Zemmouri", "Abouelali", "Bentaleb",
    "Bensalah", "Mansouri", "Bennaceur", "Kabbaj", "Benkirane", "Fassi Fihri", "Bennouna"
]

MOROCCAN_CITIES = [
    "Casablanca", "Rabat", "Fès", "Marrakech", "Agadir", "Tanger", "Meknès", "Oujda",
    "Kenitra", "Tétouan", "Safi", "Mohammedia", "Khouribga", "Béni Mellal", "Jadida",
    "Larache", "Ksar El Kebir", "Khémisset", "Guelmim", "Berrechid", "Taourirt",
    "Nador", "Settat", "Essaouira", "Tiznit", "Dakhla", "Laâyoune", "Ouarzazate",
    "Ifrane", "Azrou", "Midelt", "Errachidia", "Zagora", "Chefchaouen", "Asilah"
]

MOROCCAN_STREETS = [
    "Avenue Mohammed V", "Rue Hassan II", "Boulevard Zerktouni", "Avenue des FAR",
    "Rue Allal Ben Abdellah", "Avenue Moulay Youssef", "Boulevard Abdelmoumen",
    "Rue Ibn Sina", "Avenue Hassan Ibn Youssef", "Boulevard Al Massira",
    "Rue Imam Malik", "Avenue Lalla Yacout", "Boulevard Mohammed VI",
    "Rue Oukaimeden", "Avenue Annakhil", "Boulevard Ghandi", "Rue Atlas",
    "Avenue Hay Riad", "Boulevard Bir Anzarane", "Rue Sourya", "Avenue Agdal",
    "Boulevard Ar Razi", "Rue Sebou", "Avenue Moulay Rachid", "Boulevard Mly Abdellah",
    "Rue Tarik Ibn Ziad", "Avenue Anfa", "Boulevard Abdelkrim Khattabi",
    "Rue Yaacoub Al Mansour", "Avenue Mers Sultan", "Boulevard Sidi Belyout"
]

def generate_moroccan_name():
    """Génère un nom marocain authentique"""
    first_name = random.choice(MOROCCAN_FIRST_NAMES_MALE + MOROCCAN_FIRST_NAMES_FEMALE)
    last_name = random.choice(MOROCCAN_LAST_NAMES)

    # Parfois ajouter un préfixe traditionnel
    prefixes = ["", "", "", "El ", "Al ", "Ait ", "Ben ", "Bou ", "Abd "]
    prefix = random.choice(prefixes)

    return f"{first_name} {prefix}{last_name}".strip()

def generate_moroccan_cin():
    """Génère un CIN marocain (format: 2 lettres + 5-6 chiffres)"""
    letters = ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2)])
    digits = ''.join([str(random.randint(0, 9)) for _ in range(random.choice([5, 6]))])
    return letters + digits

def generate_moroccan_phone():
    """Génère un numéro de téléphone marocain"""
    # Préfixes mobiles marocains: 06, 07, 05 (récemment)
    prefix = random.choice(["06", "07", "05"])
    number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    return prefix + number

def generate_moroccan_iban():
    """Génère un IBAN marocain (format: MA + 2 chiffres + 24 caractères alphanumériques)"""
    control_digits = ''.join([str(random.randint(0, 9)) for _ in range(2)])
    account_identifier = ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(24)])
    return f"MA{control_digits}{account_identifier}"

def generate_moroccan_address():
    """Génère une adresse marocaine authentique"""
    street = random.choice(MOROCCAN_STREETS)
    number = random.randint(1, 999)
    city = random.choice(MOROCCAN_CITIES)
    postal_code = random.randint(10000, 99999)

    # Différents formats d'adresse
    formats = [
        f"{number}, {street}; {postal_code} {city}",
        f"{street}, {number}; {postal_code} {city}",
        f"{number} {street}; {postal_code} {city}",
        f"{street} {number}; {postal_code} {city}"
    ]

    return random.choice(formats)

def generate_moroccan_email(name):
    """Génère un email basé sur le nom"""
    # Nettoyer le nom pour l'email
    clean_name = name.lower().replace(' ', '').replace('é', 'e').replace('è', 'e').replace('ç', 'c')
    clean_name = ''.join(c for c in clean_name if c.isalnum())

    # Domaines marocains populaires
    domains = [
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "menara.ma", "maroc.ma", "iam.ma", "orange.ma"
    ]

    # Formats d'email
    formats = [
        f"{clean_name}@{random.choice(domains)}",
        f"{clean_name}{random.randint(1, 99)}@{random.choice(domains)}",
        f"{clean_name[0]}{clean_name.split()[0] if len(clean_name.split()) > 1 else clean_name}@{random.choice(domains)}"
    ]

    return random.choice(formats)

def generate_date_range(start_year=1950, end_year=2024):
    """Génère une date aléatoire dans une plage"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)

    return start_date + timedelta(days=random_days)

def generate_transaction_date():
    """Génère une date de transaction récente"""
    return generate_date_range(2020, 2024).strftime('%Y-%m-%d')

def generate_birth_date():
    """Génère une date de naissance réaliste"""
    return generate_date_range(1940, 2010).strftime('%d/%m/%Y')

def generate_entry(entry_id):
    """Génère une entrée complète"""
    type_entry = random.choice(['contrat', 'transaction', 'identite'])

    person = generate_moroccan_name()
    cin = generate_moroccan_cin() if type_entry in ['contrat', 'identite'] else ''
    phone = generate_moroccan_phone() if type_entry == 'contrat' else ''
    phone_with_prefix = f"+212{phone}" if phone else ""
    email = generate_moroccan_email(person) if type_entry == 'contrat' else ''
    iban = generate_moroccan_iban() if type_entry == 'transaction' else ''
    date = generate_transaction_date() if type_entry == 'transaction' else (generate_birth_date() if type_entry == 'identite' else '')
    location = generate_moroccan_address() if type_entry != 'transaction' else ''

    if type_entry == 'contrat':
        text = f"""Contrat de location :
- Locataire : {person}
- CIN : {cin}
- Téléphone : {phone_with_prefix}
- Adresse : {location}
- Email : {email}"""
    elif type_entry == 'transaction':
        montant = random.randint(500, 15000)
        transaction_id = random.randint(100000, 999999)
        text = f"""Transaction #{transaction_id}:
- Client : {person}
- Montant : {montant} MAD
- Destinataire : IBAN {iban}
- Date : {date}"""
    else:  # identite
        text = f"""Document d'identité :
- Nom : {person}
- CIN : {cin}
- Date de naissance : {date}
- Adresse : {location}"""

    return {
        "text_id": entry_id,
        "text": text,
        "person": person,
        "id_maroc": cin,
        "phone_number": phone_with_prefix,
        "email_address": email,
        "iban_code": iban,
        "date_time": date,
        "location": location
    }

# Génération des données
print("Génération des données marocaines...")
data = [generate_entry(i) for i in range(1, 10001)]  # 1500 entrées pour un entraînement efficace

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde en CSV
df.to_csv("entites_marocaines_authentiques.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

print("✅ Fichier CSV généré avec succès !")
print(f"📊 {len(data)} entrées générées")
print("📁 Fichier sauvegardé: entites_marocaines_authentiques.csv")

# Affichage d'un échantillon
print("\n🔍 Aperçu des données générées:")
print(df.head())

# Statistiques
print(f"\n📈 Répartition des types d'entrées:")
type_counts = df['text'].str.contains('Contrat').sum(), df['text'].str.contains('Transaction').sum(), df['text'].str.contains('Document').sum()
print(f"- Contrats de location: {type_counts[0]}")
print(f"- Transactions: {type_counts[1]}")
print(f"- Documents d'identité: {type_counts[2]}")



!pip install spacy presidio-analyzer presidio-anonymizer pandas

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import pandas as pd
import json
import random
from pathlib import Path
import re
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sklearn.model_selection import train_test_split
import numpy as np
from spacy.scorer import Scorer


from google.colab import files
uploaded = files.upload()





import pandas as pd

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import pandas as pd
import json
import random
from pathlib import Path
import re
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from google.colab import files
import numpy as np
from sklearn.model_selection import train_test_split


# Charger le dataset
df = pd.read_csv("entites_marocaines_diversifiees.csv")

def prepare_training_data_improved(df):
    """
    Version améliorée de la préparation des données avec validation
    """
    training_data = []
    validation_errors = []

    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        if not text or text == 'nan':
            continue

        entities = []

        # Fonction pour ajouter une entité si elle existe dans le texte
        def add_entity_if_exists(entity_value, entity_type):
            if pd.notna(entity_value) and str(entity_value).strip():
                entity_str = str(entity_value).strip()
                found_entities = []

                # Gestion spéciale pour les téléphones
                if entity_type == 'PHONE_NUMBER':
                    # Essayer d'abord avec le préfixe +212
                    phone_with_prefix = f"+212{entity_str}"
                    for match in re.finditer(re.escape(phone_with_prefix), text):
                        start, end = match.span()
                        found_entities.append((start, end, entity_type))

                    # Si pas trouvé avec +212, essayer le format original
                    if not found_entities:
                        for match in re.finditer(re.escape(entity_str), text):
                            start, end = match.span()
                            found_entities.append((start, end, entity_type))
                else:
                    # Recherche normale pour les autres entités
                    for match in re.finditer(re.escape(entity_str), text):
                        start, end = match.span()
                        found_entities.append((start, end, entity_type))

                # Ajouter toutes les entités trouvées
                entities.extend(found_entities)

        # Ajouter chaque type d'entité (CETTE PARTIE DOIT ÊTRE AU MÊME NIVEAU QUE LA DÉFINITION DE LA FONCTION)
        add_entity_if_exists(row['person'], 'PERSON')
        add_entity_if_exists(row['id_maroc'], 'ID_MAROC')
        add_entity_if_exists(row['phone_number'], 'PHONE_NUMBER')
        add_entity_if_exists(row['email_address'], 'EMAIL_ADDRESS')
        add_entity_if_exists(row['iban_code'], 'IBAN_CODE')
        add_entity_if_exists(row['date_time'], 'DATE_TIME')
        add_entity_if_exists(row['location'], 'LOCATION')

        # Validation des entités
        entities = sorted(set(entities))  # Supprimer les doublons

        # Vérifier les chevauchements
        valid_entities = []
        for i, (start, end, label) in enumerate(entities):
            overlap = False
            for j, (other_start, other_end, other_label) in enumerate(entities):
                if i != j and not (end <= other_start or start >= other_end):
                    overlap = True
                    break
            if not overlap:
                valid_entities.append((start, end, label))

        if valid_entities:
            training_data.append((text, {"entities": valid_entities}))
        else:
            validation_errors.append(f"Ligne {idx}: Aucune entité valide trouvée")

    print(f"Données d'entraînement préparées: {len(training_data)} exemples")
    if validation_errors:
        print(f"Erreurs de validation: {len(validation_errors)}")
        for error in validation_errors[:5]:  # Afficher les 5 premières erreurs
            print(f"  - {error}")

    return training_data

def create_improved_spacy_model(training_data, model_name="moroccan_entities_model_v2"):
    """
    Version améliorée avec régularisation contre l'overfitting
    """
    import random  # Import local pour éviter les conflits

    # Augmenter la taille du set de validation pour mieux détecter l'overfitting
    train_data, val_data = train_test_split(training_data, test_size=0.3, random_state=42)  # 30% au lieu de 20%

    print(f"Données divisées: {len(train_data)} train, {len(val_data)} validation")

    # Charger le modèle français de base
    nlp = spacy.load("fr_core_news_sm")

    # Configuration du NER
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Ajouter les labels personnalisés
    custom_labels = ["ID_MAROC", "PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE", "DATE_TIME", "LOCATION", "PERSON"]
    for label in custom_labels:
        ner.add_label(label)

    # Préparer les exemples d'entraînement
    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)

    val_examples = []
    for text, annotations in val_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        val_examples.append(example)

    # Entraînement avec régularisation forte
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        nlp.begin_training()

        best_score = 0
        patience = 3  # Patience réduite pour early stopping plus strict
        patience_counter = 0

        # RÉDUIRE les itérations pour éviter l'overfitting
        for iteration in range(10):  # 10 au lieu de 15
            print(f"Itération {iteration + 1}/10")

            # Mélanger les exemples
            random.shuffle(train_examples)

            # Entraînement avec DROPOUT TRÈS ÉLEVÉ
            losses = {}
            for batch in spacy.util.minibatch(train_examples, size=2):
                # Augmenter drastiquement le dropout
                nlp.update(batch, losses=losses, drop=0.8)  # 0.8 au lieu de 0.6

            # Évaluation plus fréquente sur validation
            if iteration % 2 == 0:  # Évaluer tous les 2 epochs au lieu de 5
                scores = nlp.evaluate(val_examples)
                f1_score = scores['ents_f']
                print(f"F1-Score validation: {f1_score:.3f}, Pertes: {losses}")

                # Early stopping plus strict
                if f1_score > best_score:
                    best_score = f1_score
                    patience_counter = 0
                    # Sauvegarder le meilleur modèle
                    nlp.to_disk(f"{model_name}_best")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping à l'itération {iteration}")
                        break

    # Charger le meilleur modèle
    try:
        nlp = spacy.load(f"{model_name}_best")
        print("Meilleur modèle chargé")
    except:
        print("Utilisation du modèle actuel")

    nlp.to_disk(model_name)
    print(f"Modèle final sauvegardé dans {model_name}")

    return nlp

# Recognizers améliorés
class ImprovedMoroccanIdRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les CIN marocaines"""

    PATTERNS = [
        Pattern("CIN Maroc", r"\b[A-Z]{2}[0-9]{5,6}\b", 0.9),
        Pattern("CIN Maroc avec espaces", r"\b[A-Z]{2}\s*[0-9]{5,6}\b", 0.8),
    ]

    CONTEXT = ["cin", "carte", "identité", "cni", "numéro"]

    def __init__(self):
        super().__init__(
            supported_entity="ID_MAROC",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class ImprovedMoroccanPhoneRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les téléphones marocains"""

    PATTERNS = [
        Pattern("Téléphone Maroc +212", r"\+212[0-9]{9}", 0.95),
        Pattern("Téléphone Maroc 0X", r"\b0[5-7][0-9]{8}\b", 0.9),
        Pattern("Téléphone compact", r"\b[0-9]{10}\b", 0.7),
    ]

    CONTEXT = ["téléphone", "phone", "mobile", "tel", "appel", "contact"]

    def __init__(self):
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class ImprovedIBANRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les codes IBAN"""

    PATTERNS = [
        Pattern("IBAN Maroc", r"\bMA[0-9]{2}[A-Z0-9]{20}\b", 0.95),
        Pattern("IBAN avec lettres", r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[A-Z0-9]{7}[A-Z0-9]{0,16}\b", 0.8),
        Pattern("IBAN général", r"\b[A-Z]{2}[0-9A-Z]{22,34}\b", 0.7),
    ]

    CONTEXT = ['banque', 'iban', 'compte', 'bancaire', 'virement', 'destinataire']  # Contexte élargi

    def __init__(self):
        super().__init__(
            supported_entity="IBAN_CODE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class ImprovedEmailRecognizer(PatternRecognizer):
    """Recognizer amélioré pour les adresses email"""

    PATTERNS = [
        Pattern("Email standard", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.9),
        Pattern("Email avec sous-domaines", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.85),  # Amélioré
        Pattern("Email domaines marocains", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*\.ma\b", 0.95),  # Nouveau pour .ma
    ]

    CONTEXT = ['email', 'mail', 'courriel', '@', 'adresse', 'contact']  # Corrigé et élargi

    def __init__(self):
        super().__init__(
            supported_entity="EMAIL_ADDRESS",  # Corrigé: EMAIL_ADDRESS au lieu de EMAIL_ADRESS
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )




def detect_overfitting(train_data, val_data, analyzer):
    """
    Détecte l'overfitting en comparant les performances sur train vs validation
    """
    print("=== DÉTECTION D'OVERFITTING ===")

    # Évaluer sur les données d'entraînement
    train_correct = 0
    train_total = 0

    for text, annotations in train_data[:100]:  # Échantillon pour éviter la lenteur
        predictions = analyzer.analyze(text=text, language="fr")
        expected_entities = annotations.get('entities', [])

        train_total += len(expected_entities)

        # Compter les correspondances exactes
        for start, end, label in expected_entities:
            for pred in predictions:
                if (pred.start == start and pred.end == end and pred.entity_type == label):
                    train_correct += 1
                    break

    # Évaluer sur les données de validation
    val_correct = 0
    val_total = 0

    for text, annotations in val_data[:100]:  # Échantillon
        predictions = analyzer.analyze(text=text, language="fr")
        expected_entities = annotations.get('entities', [])

        val_total += len(expected_entities)

        for start, end, label in expected_entities:
            for pred in predictions:
                if (pred.start == start and pred.end == end and pred.entity_type == label):
                    val_correct += 1
                    break

    # Calculer les scores
    train_accuracy = train_correct / train_total if train_total > 0 else 0
    val_accuracy = val_correct / val_total if val_total > 0 else 0

    print(f"Précision train: {train_accuracy:.3f}")
    print(f"Précision validation: {val_accuracy:.3f}")
    print(f"Écart: {train_accuracy - val_accuracy:.3f}")

    # Détecter l'overfitting
    if train_accuracy - val_accuracy > 0.15:  # Seuil de 15%
        print("🚨 OVERFITTING DÉTECTÉ!")
        print("Recommandations:")
        print("- Augmenter le dropout")
        print("- Réduire le nombre d'itérations")
        print("- Ajouter plus de données d'entraînement")
        return True
    else:
        print("✅ Pas d'overfitting détecté")
        return False



def create_enhanced_analyzer_engine(model_path):
    """
    Analyzer engine amélioré avec meilleure configuration
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "fr",
                "model_name": model_path,
            }
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Recognizers améliorés
    moroccan_id_recognizer = ImprovedMoroccanIdRecognizer()
    moroccan_phone_recognizer = ImprovedMoroccanPhoneRecognizer()
    email_recognizer = ImprovedEmailRecognizer()
    iban_recognizer = ImprovedIBANRecognizer()



    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["fr"]
    )

    # Nettoyer les recognizers par défaut
    default_recognizers = [
        "CreditCardRecognizer", "CryptoRecognizer", "DateRecognizer",
        "EmailRecognizer", "IbanRecognizer", "IpRecognizer",
        "PhoneRecognizer", "UrlRecognizer"
    ]

    for recognizer in default_recognizers:
        try:
            analyzer.registry.remove_recognizer(recognizer)
        except:
            pass

    # Ajouter nos recognizers
    analyzer.registry.add_recognizer(moroccan_id_recognizer)
    analyzer.registry.add_recognizer(moroccan_phone_recognizer)
    analyzer.registry.add_recognizer(email_recognizer)
    analyzer.registry.add_recognizer(iban_recognizer)


    return analyzer

def comprehensive_test(analyzer, test_cases):  
    """  
    Tests complets avec métriques détaillées  
    """  
    print("=== TESTS COMPLETS ===")  
  
    total_entities = 0  
    detected_entities = 0  
  
    for i, (text, expected_count) in enumerate(test_cases, 1):  
        print(f"\n--- Test {i} ---")  
        print(f"Texte: {text}")  
  
        results = analyzer.analyze(text=text, language="fr")  
  
        print(f"Entités attendues: {expected_count}")  
        print(f"Entités détectées: {len(results)}")  
  
        total_entities += expected_count   
        detected_entities += len(results)  
  
        if results:  
            for result in results:  
                entity_text = text[result.start:result.end]  
                print(f"  ✓ {result.entity_type}: '{entity_text}' (Score: {result.score:.2f})")  
        else:  
            print("  ✗ Aucune entité détectée")  
  
    print(f"\n=== STATISTIQUES GLOBALES ===")  
    print(f"Taux de détection: {detected_entities/total_entities*100:.1f}%")

# EXÉCUTION PRINCIPALE
def main():
    print("=== PRÉPARATION DES DONNÉES ===")
    training_data = prepare_training_data_improved(df)

    if len(training_data) < 5:
        print("⚠️ Dataset trop petit. Ajoutez plus d'exemples pour un meilleur entraînement.")
        return

    # Diviser les données pour la détection d'overfitting
    train_data, temp_data = train_test_split(training_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Données divisées: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
    model = create_improved_spacy_model(train_data + val_data)  # Utiliser train+val pour l'entraînement

    print("\n=== CRÉATION DE L'ANALYZER ===")
    analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")

    print("\n=== DÉTECTION D'OVERFITTING ===")
    is_overfitting = detect_overfitting(train_data, val_data, analyzer)

    if not is_overfitting:
        print("\n=== TESTS FINAUX ===")
        test_cases = [
            ("Larbi Al Lahlou, CIN: QP131924, téléphone: +2120591816049", 3),
            ("Email: larbiallahlou@menara.ma, adresse: Rue Yaacoub Al Mansour", 2),
            ("Transaction IBAN MA91J4JAZAY0K19RMWF0C4MZDZUE le 2021-07-07", 2),
        ]

        comprehensive_test(analyzer, test_cases)

        print("\n=== MODÈLE PRÊT ===")
        print("✅ Votre modèle amélioré est prêt!")
        print("📁 Téléchargez le dossier 'moroccan_entities_model_v2'")
    else:
        print("\n⚠️ Modèle non recommandé pour la production à cause de l'overfitting")


if __name__ == "__main__":  
    main()





from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_with_metrics(analyzer, test_data):
    """
    Évalue le modèle avec des métriques détaillées

    Args:
        analyzer: AnalyzerEngine configuré
        test_data: Liste de tuples (text, expected_entities)

    Returns:
        Dict contenant toutes les métriques
    """

    # Structures pour stocker les résultats
    true_entities = []
    predicted_entities = []
    entity_level_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    print("=== ÉVALUATION DÉTAILLÉE DU MODÈLE ===\n")

    for i, (text, expected_annotations) in enumerate(test_data):
        print(f"--- Exemple {i+1} ---")
        print(f"Texte: {text[:100]}...")

        # Prédictions du modèle
        predictions = analyzer.analyze(text=text, language="fr")

        # Convertir les annotations attendues en format comparable
        expected_entities = []
        if 'entities' in expected_annotations:
            for start, end, label in expected_annotations['entities']:
                expected_entities.append({
                    'start': start,
                    'end': end,
                    'label': label,
                    'text': text[start:end]
                })

        # Convertir les prédictions
        pred_entities = []
        for pred in predictions:
            pred_entities.append({
                'start': pred.start,
                'end': pred.end,
                'label': pred.entity_type,
                'text': text[pred.start:pred.end],
                'score': pred.score
            })

        print(f"Attendu: {len(expected_entities)} entités")
        print(f"Prédit: {len(pred_entities)} entités")

        # Évaluation au niveau des entités
        for expected in expected_entities:
            true_entities.append(expected['label'])

            # Chercher une correspondance exacte
            found_match = False
            for pred in pred_entities:
                if (pred['start'] == expected['start'] and
                    pred['end'] == expected['end'] and
                    pred['label'] == expected['label']):
                    found_match = True
                    break

            if found_match:
                entity_level_results[expected['label']]['tp'] += 1
                print(f"  ✓ {expected['label']}: '{expected['text']}'")
            else:
                entity_level_results[expected['label']]['fn'] += 1
                print(f"  ✗ Manqué {expected['label']}: '{expected['text']}'")

        # Faux positifs
        for pred in pred_entities:
            predicted_entities.append(pred['label'])

            found_match = False
            for expected in expected_entities:
                if (pred['start'] == expected['start'] and
                    pred['end'] == expected['end'] and
                    pred['label'] == expected['label']):
                    found_match = True
                    break

            if not found_match:
                entity_level_results[pred['label']]['fp'] += 1
                print(f"  ⚠ Faux positif {pred['label']}: '{pred['text']}'")

        print()

    # Calculer les métriques par type d'entité
    metrics_by_entity = {}
    overall_tp = overall_fp = overall_fn = 0

    print("=== MÉTRIQUES PAR TYPE D'ENTITÉ ===")
    for entity_type, counts in entity_level_results.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_by_entity[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': tp + fn,
            'tp': tp, 'fp': fp, 'fn': fn
        }

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        print(f"{entity_type}:")
        print(f"  Précision: {precision:.3f}")
        print(f"  Rappel: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Support: {tp + fn}")
        print()

    # Métriques globales
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("=== MÉTRIQUES GLOBALES ===")
    print(f"Précision globale: {overall_precision:.3f}")
    print(f"Rappel global: {overall_recall:.3f}")
    print(f"F1-Score global: {overall_f1:.3f}")
    print(f"Entités totales: {overall_tp + overall_fn}")
    print(f"Entités détectées: {overall_tp + overall_fp}")
    print(f"Entités correctes: {overall_tp}")

    # Matrice de confusion
    plot_confusion_matrix(metrics_by_entity)

    return {
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'accuracy': overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) > 0 else 0
        },
        'entity_metrics': metrics_by_entity,
        'detailed_results': entity_level_results
    }

def plot_confusion_matrix(metrics_by_entity):
    """Affiche une matrice de confusion visuelle"""

    entity_types = list(metrics_by_entity.keys())
    if not entity_types:
        return

    # Créer les données pour le graphique
    precision_scores = [metrics_by_entity[et]['precision'] for et in entity_types]
    recall_scores = [metrics_by_entity[et]['recall'] for et in entity_types]
    f1_scores = [metrics_by_entity[et]['f1_score'] for et in entity_types]

    # Graphique des métriques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique en barres des métriques
    x = range(len(entity_types))
    width = 0.25

    ax1.bar([i - width for i in x], precision_scores, width, label='Précision', alpha=0.8)
    ax1.bar(x, recall_scores, width, label='Rappel', alpha=0.8)
    ax1.bar([i + width for i in x], f1_scores, width, label='F1-Score', alpha=0.8)

    ax1.set_xlabel('Types d\'entités')
    ax1.set_ylabel('Score')
    ax1.set_title('Métriques par type d\'entité')
    ax1.set_xticks(x)
    ax1.set_xticklabels(entity_types, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Heatmap des scores F1
    f1_matrix = [[f1_scores[i]] for i in range(len(entity_types))]
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='Blues',
                yticklabels=entity_types, xticklabels=['F1-Score'], ax=ax2)
    ax2.set_title('Heatmap des F1-Scores')

    plt.tight_layout()
    plt.show()

def create_test_dataset_from_training(training_data, test_ratio=0.3):
    """
    Crée un dataset de test à partir des données d'entraînement
    """
    random.shuffle(training_data)
    split_idx = int(len(training_data) * (1 - test_ratio))

    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]

    print(f"Dataset divisé: {len(train_data)} entraînement, {len(test_data)} test")
    return train_data, test_data




def detect_overfitting(train_metrics, val_metrics):
    """Détecte l'overfitting en comparant train vs validation"""

    overfitting_indicators = {}

    for entity_type in train_metrics['entity_metrics']:
        train_f1 = train_metrics['entity_metrics'][entity_type]['f1_score']
        val_f1 = val_metrics['entity_metrics'][entity_type]['f1_score']

        # Si la différence est > 0.1, c'est suspect
        overfitting_indicators[entity_type] = {
            'train_f1': train_f1,
            'val_f1': val_f1,
            'gap': train_f1 - val_f1,
            'overfitting': (train_f1 - val_f1) > 0.1
        }

    return overfitting_indicators

def main_with_overfitting_detection():
    print("=== PRÉPARATION DES DONNÉES ===")
    all_training_data = prepare_training_data_improved(df)

    if len(all_training_data) < 10:
        print("⚠️ Dataset trop petit pour l'évaluation.")
        return

    # Diviser en train/validation/test (60/20/20)
    train_data, temp_data = train_test_split(all_training_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Données divisées: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")

    print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
    model = create_improved_spacy_model(train_data)

    print("\n=== CRÉATION DE L'ANALYZER ===")
    analyzer = create_enhanced_analyzer_engine("moroccan_entities_model_v2")

    print("\n=== ÉVALUATION SUR DONNÉES D'ENTRAÎNEMENT ===")
    train_metrics = evaluate_model_with_metrics(analyzer, train_data)

    print("\n=== ÉVALUATION SUR DONNÉES DE VALIDATION ===")
    val_metrics = evaluate_model_with_metrics(analyzer, val_data)

    print("\n=== DÉTECTION D'OVERFITTING ===")
    overfitting_results = detect_overfitting(train_metrics, val_metrics)

    # Afficher les résultats d'overfitting
    print("=== ANALYSE D'OVERFITTING PAR ENTITÉ ===")
    for entity_type, indicators in overfitting_results.items():
        status = "🚨 OVERFITTING" if indicators['overfitting'] else "✅ OK"
        print(f"{entity_type}: {status}")
        print(f"  Train F1: {indicators['train_f1']:.3f}")
        print(f"  Val F1: {indicators['val_f1']:.3f}")
        print(f"  Écart: {indicators['gap']:.3f}")
        print()

    # Évaluation finale sur test si pas d'overfitting majeur
    overfitting_entities = [et for et, ind in overfitting_results.items() if ind['overfitting']]

    if len(overfitting_entities) < len(overfitting_results) / 2:
        print("\n=== ÉVALUATION FINALE SUR DONNÉES DE TEST ===")
        test_metrics = evaluate_model_with_metrics(analyzer, test_data)
    else:
        print(f"\n⚠️ Overfitting détecté sur {len(overfitting_entities)} entités: {overfitting_entities}")
        print("Recommandation: Augmentez la régularisation ou collectez plus de données")

    return train_metrics, val_metrics, overfitting_results

# Exécuter avec détection d'overfitting
if __name__ == "__main__":
    train_metrics, val_metrics, overfitting_results = main_with_overfitting_detection()



import shutil  
# Créer une archive zip du modèle  
shutil.make_archive('moroccan_entities_model_v2', 'zip', 'moroccan_entities_model_v2')  
  
# Télécharger l'archive  
from google.colab import files  
files.download('moroccan_entities_model_v2.zip')