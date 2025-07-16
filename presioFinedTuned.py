
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