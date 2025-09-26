# recommendation_engine/recommendation_templates.py

ENTERPRISE_TEMPLATES = {  
    'COMPLIANCE': """  
    Analysez la conformité RGPD du dataset et générez des recommandations PRIORITAIRES.  
      
    Dataset: {dataset_name}  
    Entités personnelles détectées: {personal_entities}  
    Score de conformité actuel: {compliance_score}
      
    Répondez UNIQUEMENT en JSON valide:  
    {{  
        "compliance_score": 7.5,  
        "critical_gaps": ["Registre des traitements manquant", "Information des personnes insuffisante"],  
        "immediate_actions": ["Créer registre RGPD", "Rédiger mentions d'information"],  
        "regulatory_risk": "HIGH"  
    }}  
    """,  
      
    'SECURITY': """  
    Évaluez les risques de sécurité des données sensibles.  
      
    Entités sensibles: {sensitive_entities}  
    Méthodes de protection actuelles: {current_protection}  
    Niveau de risque détecté: {risk_level}
      
    Répondez UNIQUEMENT en JSON valide:  
    {{  
        "risk_level": "HIGH",  
        "encryption_needs": ["EMAIL_ADDRESS", "PHONE_NUMBER"],  
        "access_controls": ["Authentification multi-facteurs", "Contrôle d'accès basé sur les rôles"]  
    }}  
    """,  
      
    'QUALITY': """  
    Analysez la qualité des données du dataset.  
      
    Dataset: {dataset_name}  
    Pourcentage de valeurs manquantes: {missing_percentage}%
    Nombre de doublons: {duplicate_count}
    Cohérence des données: {data_consistency}
      
    Répondez UNIQUEMENT en JSON valide:  
    {{  
        "quality_score": 6.5,  
        "data_issues": ["Valeurs manquantes élevées", "Doublons détectés"],  
        "improvement_actions": ["Nettoyage des données", "Validation des formats"]  
    }}  
    """,  
      
    'GOVERNANCE': """  
    Évaluez la gouvernance des données.  
      
    Dataset: {dataset_name}  
    Nombre total de lignes: {total_rows}
    Nombre de colonnes: {headers_count}
    Statut des métadonnées: {metadata_status}
      
    Répondez UNIQUEMENT en JSON valide:  
    {{  
        "governance_score": 5.0,  
        "missing_metadata": ["Description des colonnes", "Propriétaires des données"],  
        "governance_actions": ["Créer catalogue de données", "Documenter métadonnées"]  
    }}  
    """  
}