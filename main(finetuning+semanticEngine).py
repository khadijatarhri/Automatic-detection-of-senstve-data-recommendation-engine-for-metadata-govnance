def main():
    # 1. Préparation des données
    training_data = prepare_training_data("entites_nettoye1 (4).csv")
    
    # 2. Fine-tuning du modèle spaCy
    create_spacy_model(training_data, model_name="moroccan_entities_model")

    # 3. Charger le modèle spaCy entraîné dans Presidio
    analyzer = create_custom_analyzer_engine("moroccan_entities_model")

    # 4. Créer l'analyseur sémantique
    semantic_analyzer = SemanticAnalyzer("moroccan_entities_model")

    # 5. Initialiser l’autotagging
    auto_tagger = IntelligentAutoTagger(analyzer, semantic_analyzer)

    # 6. Texte à analyser
    test_text = "Ahmed Benali, CIN: AB12345, téléphone: +212661234567, email: ahmed@example.com, IBAN: FR1463648284987662786876248"

    # 7. Lancer l'analyse et le tag automatique
    entities, tags = auto_tagger.analyze_and_tag(test_text, dataset_name="client_data")

    # 8. Afficher les résultats
    print("\n=== ENTITÉS DÉTECTÉES ET ENRICHIES ===")
    for entity in entities:
        print(entity)

    print("\n=== TAGS GÉNÉRÉS ===")
    print(tags)
