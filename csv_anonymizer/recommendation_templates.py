ENTERPRISE_TEMPLATES = {  
    'COMPLIANCE': """  
    Analysez la conformité RGPD du dataset et générez des recommandations PRIORITAIRES.  
      
    Dataset: {dataset_name}  
    Entités personnelles détectées: {personal_entities}  
    Score de conformité actuel: {compliance_score}
      
    Répondez en JSON avec:  
    {{  
        "compliance_score": score sur 10,  
        "critical_gaps": ["lacune1", "lacune2"],  
        "immediate_actions": ["action1", "action2"],  
        "regulatory_risk": "LOW|MEDIUM|HIGH"  
    }}  
    """,  
      
    'SECURITY': """  
    Évaluez les risques de sécurité des données sensibles.  
      
    Entités sensibles: {sensitive_entities}  
    Méthodes de protection actuelles: {current_protection}  
    Niveau de risque détecté: {risk_level}
      
    Générez des recommandations de sécurité en JSON:  
    {{  
        "risk_level": "LOW|MEDIUM|HIGH",  
        "encryption_needs": ["colonne1", "colonne2"],  
        "access_controls": ["contrôle1", "contrôle2"]  
    }}  
    """,  
      
    'QUALITY': """  
    Analysez la qualité des données du dataset.  
      
    Dataset: {dataset_name}  
    Pourcentage de valeurs manquantes: {missing_percentage}%
    Nombre de doublons: {duplicate_count}
    Cohérence des données: {data_consistency}
      
    Répondez en JSON:  
    {{  
        "quality_score": score sur 10,  
        "data_issues": ["problème1", "problème2"],  
        "improvement_actions": ["action1", "action2"]  
    }}  
    """,  
      
    'GOVERNANCE': """  
    Évaluez la gouvernance des données.  
      
    Dataset: {dataset_name}  
    Nombre total de lignes: {total_rows}
    Nombre de colonnes: {headers_count}
    Statut des métadonnées: {metadata_status}
      
    Répondez en JSON:  
    {{  
        "governance_score": score sur 10,  
        "missing_metadata": ["métadonnée1", "métadonnée2"],  
        "governance_actions": ["action1", "action2"]  
    }}  
    """  
}