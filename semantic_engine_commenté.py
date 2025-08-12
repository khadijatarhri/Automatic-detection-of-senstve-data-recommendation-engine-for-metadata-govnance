# =============================================================================
# SYSTÈME D'AUTOTAGGING ET NLP SÉMANTIQUE AMÉLIORÉ
# =============================================================================
"""
Ce module implémente un système intelligent d'analyse et de classification automatique
des données sensibles, avec un focus sur la conformité RGPD.

Le système combine :
- Traitement du langage naturel (spaCy)
- Détection d'entités sensibles (Presidio)
- Classification sémantique contextuelle
- Génération automatique de tags et métadonnées

Auteur: Système d'analyse de données
Version: 1.0
"""

# =============================================================================
# IMPORTS ET DÉPENDANCES
# =============================================================================

# Librairies de traitement du langage naturel
import spacy  # Framework NLP principal pour l'analyse linguistique

# Librairies de manipulation de données
import pandas as pd  # Manipulation de dataframes
import numpy as np   # Calculs numériques et matrices

# Librairies de typage Python (améliore la lisibilité et détection d'erreurs)
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass  # Création de classes de données simplifiées
from enum import Enum              # Création d'énumérations pour les constantes

# Librairies utilitaires
import json                        # Sérialisation/désérialisation JSON
import re                         # Expressions régulières
from collections import Counter, defaultdict  # Structures de données avancées

# Librairies de sécurité et anonymisation (Microsoft Presidio)
from presidio_analyzer import AnalyzerEngine      # Détection d'entités sensibles
from presidio_anonymizer import AnonymizerEngine  # Anonymisation des données

# Librairies de persistance et utilitaires
import sqlite3                    # Base de données locale
from datetime import datetime     # Gestion des dates/heures
import hashlib                   # Fonctions de hachage cryptographique

# =============================================================================
# DÉFINITION DES STRUCTURES DE DONNÉES
# =============================================================================

class SensitivityLevel(Enum):
    """
    Énumération des niveaux de sensibilité des données selon RGPD.
    
    Cette classification aide à déterminer le niveau de protection requis
    pour chaque type de donnée identifiée.
    
    Niveaux (du moins sensible au plus sensible) :
    - PUBLIC : Données publiques, aucune restriction
    - INTERNAL : Données internes à l'organisation
    - CONFIDENTIAL : Données confidentielles nécessitant une protection
    - RESTRICTED : Données à accès très restreint
    - PERSONAL_DATA : Données personnelles soumises au RGPD
    """
    PUBLIC = "PUBLIC"           # Données publiques - ex: informations sur site web
    INTERNAL = "INTERNAL"       # Données internes - ex: procédures internes
    CONFIDENTIAL = "CONFIDENTIAL"  # Données confidentielles - ex: emails, téléphones
    RESTRICTED = "RESTRICTED"   # Données restreintes - ex: données financières
    PERSONAL_DATA = "PERSONAL_DATA"  # Données personnelles RGPD - ex: nom, CIN

class DataCategory(Enum):
    """
    Énumération des catégories métier de données.
    
    Cette classification permet de regrouper les données par domaine fonctionnel
    pour faciliter la gouvernance et l'application de politiques spécifiques.
    """
    IDENTITY = "IDENTITY"           # Données d'identité - nom, prénom, CIN
    FINANCIAL = "FINANCIAL"         # Données financières - IBAN, transactions
    CONTACT = "CONTACT"             # Données de contact - email, téléphone
    LOCATION = "LOCATION"           # Données de localisation - adresses, coordonnées
    TRANSACTION = "TRANSACTION"     # Données transactionnelles - achats, paiements
    BEHAVIORAL = "BEHAVIORAL"       # Données comportementales - historiques, préférences

@dataclass
class EntityMetadata:
    """
    Structure de données pour stocker les métadonnées enrichies d'une entité détectée.
    
    Cette classe utilise le décorateur @dataclass qui génère automatiquement
    les méthodes __init__, __repr__, __eq__, etc.
    
    Attributes:
        entity_type (str): Type d'entité (ex: 'PERSON', 'EMAIL_ADDRESS')
        entity_value (str): Valeur réelle de l'entité détectée
        start_pos (int): Position de début dans le texte original
        end_pos (int): Position de fin dans le texte original
        confidence_score (float): Score de confiance de la détection (0.0 à 1.0)
        sensitivity_level (SensitivityLevel): Niveau de sensibilité évalué
        data_category (DataCategory): Catégorie métier de la donnée
        semantic_context (List[str]): Contextes sémantiques identifiés
        rgpd_category (Optional[str]): Catégorie RGPD applicable (peut être None)
        anonymization_method (Optional[str]): Méthode d'anonymisation recommandée
    """
    entity_type: str                    # Type de l'entité détectée
    entity_value: str                   # Valeur concrète trouvée dans le texte
    start_pos: int                      # Index de début dans le texte source
    end_pos: int                        # Index de fin dans le texte source
    confidence_score: float             # Niveau de confiance (0.0 - 1.0)
    sensitivity_level: SensitivityLevel # Niveau de sensibilité évalué
    data_category: DataCategory         # Catégorie fonctionnelle
    semantic_context: List[str]         # Liste des contextes sémantiques
    rgpd_category: Optional[str] = None # Catégorie RGPD (optionnelle)
    anonymization_method: Optional[str] = None  # Méthode anonymisation recommandée

@dataclass
class DatasetProfile:
    """
    Structure pour stocker le profil complet d'analyse d'un jeu de données.
    
    Cette classe représente le résultat final de l'analyse d'un dataset,
    incluant statistiques, scores de qualité et recommandations.
    """
    dataset_id: str                     # Identifiant unique du dataset
    name: str                          # Nom descriptif du dataset
    total_entities: int                # Nombre total d'entités détectées
    entity_distribution: Dict[str, int] # Distribution par type d'entité
    sensitivity_distribution: Dict[str, int]  # Distribution par niveau sensibilité
    semantic_tags: List[str]           # Tags sémantiques générés
    quality_score: float               # Score de qualité global (0.0 - 1.0)
    rgpd_compliance_score: float       # Score de conformité RGPD (0.0 - 1.0)
    recommendations: List[str]         # Liste des recommandations d'amélioration
    created_at: datetime               # Timestamp de création du profil

# =============================================================================
# MOTEUR D'ANALYSE SÉMANTIQUE
# =============================================================================

class SemanticAnalyzer:
    """
    Analyseur sémantique pour comprendre le contexte des données.
    
    Cette classe utilise spaCy pour analyser le contexte linguistique et sémantique
    des textes, permettant une classification plus précise des entités détectées.
    
    Fonctionnalités principales :
    - Analyse du contexte sémantique par domaine
    - Classification automatique du niveau de sensibilité
    - Détermination de la catégorie de données
    - Mapping vers les exigences RGPD
    """
    
    def __init__(self, model_path: str):
        """
        Initialise l'analyseur sémantique avec un modèle spaCy.
        
        Args:
            model_path (str): Chemin vers le modèle spaCy pré-entraîné
                            (ex: 'fr_core_news_sm' pour le français)
        
        Note:
            Le modèle doit être installé au préalable :
            python -m spacy download fr_core_news_sm
        """
        # Charger le modèle de langue spaCy
        self.nlp = spacy.load(model_path)
        
        # Dictionnaires sémantiques pour la classification contextuelle
        # Ces mots-clés aident à identifier le domaine d'usage des données
        self.semantic_contexts = {
            # Contexte financier : banques, transactions, paiements
            'financial': [
                'banque', 'compte', 'iban', 'transaction', 'virement', 'carte',
                'crédit', 'débit', 'solde', 'facture', 'paiement', 'montant',
                'euros', 'dirham', 'devise', 'taux', 'intérêt'
            ],
            
            # Contexte identitaire : informations personnelles
            'identity': [
                'nom', 'prénom', 'cin', 'identité', 'naissance', 'âge',
                'nationalité', 'profession', 'statut', 'titre', 'genre',
                'état civil', 'situation familiale'
            ],
            
            # Contexte contact : moyens de communication
            'contact': [
                'email', 'téléphone', 'adresse', 'domicile', 'bureau',
                'contact', 'joindre', 'appeler', 'écrire', 'courrier',
                'mobile', 'fixe', 'fax'
            ],
            
            # Contexte géographique : localisation
            'location': [
                'rue', 'avenue', 'boulevard', 'ville', 'quartier', 'région',
                'pays', 'code postal', 'géolocalisation', 'coordonnées',
                'latitude', 'longitude', 'GPS'
            ],
            
            # Contexte comportemental : analyse des habitudes
            'behavioral': [
                'historique', 'comportement', 'préférence', 'habitude',
                'fréquence', 'pattern', 'analyse', 'profil', 'segmentation',
                'scoring', 'tendance'
            ]
        }
        
        # Mapping des types d'entités vers les catégories RGPD
        # Conforme aux articles 4 et 9 du RGPD
        self.rgpd_mapping = {
            'PERSON': 'Données d\'identification directe',  # Art. 4(1) RGPD
            'ID_MAROC': 'Données d\'identification directe',  # Numéro CIN
            'PHONE_NUMBER': 'Données de contact',  # Moyens de communication
            'EMAIL_ADDRESS': 'Données de contact',  # Adresses électroniques
            'LOCATION': 'Données de localisation',  # Données de géolocalisation
            'IBAN_CODE': 'Données financières',  # Informations bancaires
            'DATE_TIME': 'Données temporelles',  # Horodatage
            'IP_ADDRESS': 'Identifiants en ligne'  # Identificateurs réseau
        }
        
        # Méthodes d'anonymisation recommandées par type d'entité
        # Basées sur les recommandations CNIL et bonnes pratiques RGPD
        self.anonymization_methods = {
            'PERSON': 'pseudonymisation',      # Remplacer par un pseudonyme
            'ID_MAROC': 'hachage',            # Hash cryptographique irreversible
            'PHONE_NUMBER': 'masquage partiel', # Masquer les derniers chiffres
            'EMAIL_ADDRESS': 'masquage partiel', # Masquer le domaine ou utilisateur
            'LOCATION': 'généralisation',      # Réduire la précision géographique
            'IBAN_CODE': 'chiffrement',       # Chiffrement symétrique/asymétrique
            'DATE_TIME': 'généralisation temporelle', # Réduire précision temporelle
            'IP_ADDRESS': 'suppression'       # Suppression complète
        }
    
    def analyze_semantic_context(self, text: str) -> Dict[str, float]:
        """
        Analyse le contexte sémantique d'un texte pour identifier les domaines d'usage.
        
        Cette méthode utilise l'analyse linguistique de spaCy combinée à des
        dictionnaires sémantiques pour évaluer dans quels contextes métier
        le texte s'inscrit.
        
        Args:
            text (str): Texte à analyser (peut contenir plusieurs phrases)
            
        Returns:
            Dict[str, float]: Dictionnaire des scores de contexte par domaine.
                            Clés : noms des domaines ('financial', 'identity', etc.)
                            Valeurs : scores de 0.0 à 1.0 indiquant la pertinence
                            
        Exemple:
            >>> analyzer.analyze_semantic_context("Virement de 1000€ sur compte IBAN")
            {'financial': 0.75, 'identity': 0.0, 'contact': 0.0, ...}
        """
        # Traitement NLP avec spaCy : tokenisation, lemmatisation, POS tagging
        doc = self.nlp(text.lower())
        
        # Extraire les tokens significatifs (exclure stop words et ponctuation)
        # La lemmatisation réduit les mots à leur forme canonique
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        
        # Si pas de tokens significatifs, retourner des scores nuls
        if not tokens:
            return {context: 0.0 for context in self.semantic_contexts.keys()}
        
        # Calculer les scores pour chaque contexte sémantique
        context_scores = {}
        for context_name, keywords in self.semantic_contexts.items():
            # Compter les correspondances entre tokens et mots-clés du contexte
            matches = 0
            for token in tokens:
                # Recherche de sous-chaînes pour une correspondance flexible
                if any(keyword in token or token in keyword for keyword in keywords):
                    matches += 1
            
            # Normaliser par le nombre total de tokens pour obtenir un score [0,1]
            context_scores[context_name] = matches / len(tokens)
        
        return context_scores
    
    def determine_sensitivity_level(self, entity_type: str, context_scores: Dict[str, float]) -> SensitivityLevel:
        """
        Détermine le niveau de sensibilité d'une entité basé sur son type et contexte.
        
        Cette fonction applique des règles de classification inspirées du RGPD
        et des bonnes pratiques de sécurité des données.
        
        Args:
            entity_type (str): Type de l'entité détectée (ex: 'PERSON', 'EMAIL_ADDRESS')
            context_scores (Dict[str, float]): Scores de contexte sémantique
            
        Returns:
            SensitivityLevel: Niveau de sensibilité déterminé
            
        Règles appliquées :
        - Données personnelles directes → PERSONAL_DATA
        - Données financières → RESTRICTED  
        - Données de contact → CONFIDENTIAL
        - Autres → Classification contextuelle
        """
        # Règle 1 : Données personnelles directes (RGPD Art. 4)
        if entity_type in ['PERSON', 'ID_MAROC']:
            return SensitivityLevel.PERSONAL_DATA
        
        # Règle 2 : Données financières ou contexte financier fort
        elif entity_type in ['IBAN_CODE'] or context_scores.get('financial', 0) > 0.3:
            return SensitivityLevel.RESTRICTED
        
        # Règle 3 : Données de contact personnelles
        elif entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS']:
            return SensitivityLevel.CONFIDENTIAL
        
        # Règle 4 : Données de localisation (peuvent être sensibles selon contexte)
        elif entity_type == 'LOCATION':
            # Si contexte comportemental fort → plus sensible (géolocalisation précise)
            if context_scores.get('behavioral', 0) > 0.4:
                return SensitivityLevel.RESTRICTED
            else:
                return SensitivityLevel.CONFIDENTIAL
        
        # Règle 5 : Classification par défaut
        else:
            return SensitivityLevel.INTERNAL
    
    def determine_data_category(self, entity_type: str, context_scores: Dict[str, float]) -> DataCategory:
        """
        Détermine la catégorie fonctionnelle d'une entité.
        
        La catégorisation permet d'appliquer des politiques métier spécifiques
        et de faciliter la gouvernance des données.
        
        Args:
            entity_type (str): Type de l'entité
            context_scores (Dict[str, float]): Scores de contexte sémantique
            
        Returns:
            DataCategory: Catégorie fonctionnelle déterminée
        """
        # Classification primaire basée sur le type d'entité
        if entity_type in ['PERSON', 'ID_MAROC']:
            return DataCategory.IDENTITY
        
        elif entity_type == 'IBAN_CODE' or context_scores.get('financial', 0) > 0.2:
            return DataCategory.FINANCIAL
        
        elif entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS']:
            return DataCategory.CONTACT
        
        elif entity_type == 'LOCATION':
            return DataCategory.LOCATION
        
        # Classification secondaire basée sur le contexte sémantique
        elif context_scores.get('behavioral', 0) > 0.2:
            return DataCategory.BEHAVIORAL
        
        # Classification par défaut
        else:
            return DataCategory.TRANSACTION

# =============================================================================
# MOTEUR D'AUTOTAGGING INTELLIGENT
# =============================================================================

class IntelligentAutoTagger:
    """
    Moteur d'autotagging basé sur l'analyse sémantique et la détection de patterns.
    
    Cette classe orchestre l'analyse complète d'un texte en combinant :
    - Détection d'entités avec Presidio
    - Analyse sémantique contextuelle
    - Génération automatique de tags métier
    - Production de métadonnées enrichies
    
    Le système génère automatiquement des étiquettes pertinentes pour
    faciliter la classification et la gouvernance des datasets.
    """
    
    def __init__(self, analyzer_engine: AnalyzerEngine, semantic_analyzer: SemanticAnalyzer):
        """
        Initialise le moteur d'autotagging.
        
        Args:
            analyzer_engine (AnalyzerEngine): Instance de Presidio pour détecter les PII
            semantic_analyzer (SemanticAnalyzer): Analyseur sémantique personnalisé
        """
        self.analyzer = analyzer_engine          # Moteur Presidio pour détection PII
        self.semantic_analyzer = semantic_analyzer  # Notre analyseur sémantique
        self.tag_rules = self._initialize_tag_rules()  # Règles d'étiquetage
    
    def _initialize_tag_rules(self) -> Dict[str, List[str]]:
        """
        Initialise les règles d'étiquetage automatique.
        
        Ces règles définissent quels tags sont générés selon les caractéristiques
        détectées dans le texte analysé.
        
        Returns:
            Dict[str, List[str]]: Dictionnaire des conditions → tags générés
        """
        return {
            # Tags pour présence de données personnelles
            'contains_personal_data': ['PII', 'RGPD', 'PERSONAL', 'GDPR_APPLICABLE'],
            
            # Tags pour données financières
            'contains_financial_data': ['FINANCIAL', 'BANKING', 'PAYMENT', 'MONETARY'],
            
            # Tags pour données de contact
            'contains_contact_data': ['CONTACT', 'COMMUNICATION', 'REACHABILITY'],
            
            # Tags pour données de localisation
            'contains_location_data': ['LOCATION', 'GEOGRAPHIC', 'GEOLOCATION'],
            
            # Tags par niveau de sensibilité
            'high_sensitivity': ['RESTRICTED', 'CONFIDENTIAL', 'HIGH_RISK'],
            'medium_sensitivity': ['INTERNAL', 'MEDIUM_RISK'],
            'low_sensitivity': ['PUBLIC', 'LOW_RISK'],
            
            # Tags contextuels
            'behavioral_context': ['PROFILING', 'ANALYTICS', 'BEHAVIORAL'],
            'transactional_context': ['TRANSACTION', 'BUSINESS_DATA', 'OPERATIONAL']
        }
    
    def analyze_and_tag(self, text: str, dataset_name: str = "") -> Tuple[List[EntityMetadata], List[str]]:
        """
        Analyse un texte et génère automatiquement les tags et métadonnées.
        
        Cette méthode constitue le point d'entrée principal du système.
        Elle orchestre toutes les analyses et produit un résultat enrichi.
        
        Args:
            text (str): Texte à analyser (peut être une ligne de dataset, document, etc.)
            dataset_name (str, optional): Nom du dataset pour contextualisation
            
        Returns:
            Tuple[List[EntityMetadata], List[str]]: 
                - Liste des entités détectées avec métadonnées enrichies
                - Liste des tags générés automatiquement
                
        Process:
            1. Détection d'entités PII avec Presidio
            2. Analyse du contexte sémantique
            3. Enrichissement des métadonnées
            4. Génération des tags contextuels
        """
        # ÉTAPE 1 : Détection des entités sensibles avec Presidio
        # Presidio scanne le texte à la recherche de patterns PII connus
        presidio_results = self.analyzer.analyze(text=text, language="fr")
        
        # ÉTAPE 2 : Analyse du contexte sémantique global du texte
        # Notre analyseur évalue les domaines métier présents
        context_scores = self.semantic_analyzer.analyze_semantic_context(text)
        
        # ÉTAPE 3 : Enrichissement des entités avec métadonnées complètes
        enriched_entities = []
        for result in presidio_results:
            # Extraire la valeur réelle de l'entité du texte original
            entity_value = text[result.start:result.end]
            
            # Déterminer le niveau de sensibilité en croisant type et contexte
            sensitivity = self.semantic_analyzer.determine_sensitivity_level(
                result.entity_type, context_scores
            )
            
            # Déterminer la catégorie fonctionnelle métier
            category = self.semantic_analyzer.determine_data_category(
                result.entity_type, context_scores
            )
            
            # Créer l'objet métadonnées enrichi
            metadata = EntityMetadata(
                entity_type=result.entity_type,               # Type Presidio
                entity_value=entity_value,                    # Valeur extraite
                start_pos=result.start,                       # Position début
                end_pos=result.end,                          # Position fin
                confidence_score=result.score,                # Confiance Presidio
                sensitivity_level=sensitivity,                # Notre évaluation
                data_category=category,                       # Catégorie métier
                semantic_context=list(context_scores.keys()), # Contextes détectés
                rgpd_category=self.semantic_analyzer.rgpd_mapping.get(result.entity_type),
                anonymization_method=self.semantic_analyzer.anonymization_methods.get(result.entity_type)
            )
            
            enriched_entities.append(metadata)
        
        # ÉTAPE 4 : Génération automatique des tags
        generated_tags = self._generate_tags(enriched_entities, context_scores, dataset_name)
        
        return enriched_entities, generated_tags
    
    def _generate_tags(self, entities: List[EntityMetadata], context_scores: Dict[str, float], dataset_name: str) -> List[str]:
        """
        Génère automatiquement les tags basés sur l'analyse complète.
        
        Cette méthode applique des règles métier pour produire des tags
        pertinents facilitant la classification et la recherche de datasets.
        
        Args:
            entities (List[EntityMetadata]): Liste des entités enrichies détectées
            context_scores (Dict[str, float]): Scores de contexte sémantique
            dataset_name (str): Nom du dataset pour tags contextuels
            
        Returns:
            List[str]: Liste unique et triée des tags générés
        """
        tags = set()  # Utiliser un set pour éviter les doublons
        
        # RÈGLES BASÉES SUR LES TYPES D'ENTITÉS DÉTECTÉES
        entity_types = [entity.entity_type for entity in entities]
        
        # Règle : Présence de données personnelles directes
        if any(entity_type in ['PERSON', 'ID_MAROC'] for entity_type in entity_types):
            tags.update(['PII', 'RGPD', 'PERSONAL_DATA', 'GDPR_APPLICABLE'])
        
        # Règle : Présence de données financières
        if 'IBAN_CODE' in entity_types:
            tags.update(['FINANCIAL', 'BANKING', 'PAYMENT_DATA'])
        
        # Règle : Présence de données de contact
        if any(entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS'] for entity_type in entity_types):
            tags.update(['CONTACT', 'COMMUNICATION', 'CONTACT_INFO'])
        
        # Règle : Présence de données de localisation
        if 'LOCATION' in entity_types:
            tags.update(['LOCATION', 'GEOGRAPHIC', 'GEOLOCATION'])
        
        # RÈGLES BASÉES SUR LES NIVEAUX DE SENSIBILITÉ
        sensitivity_levels = [entity.sensitivity_level for entity in entities]
        
        # Classification par niveau de sensibilité maximal détecté
        if SensitivityLevel.PERSONAL_DATA in sensitivity_levels:
            tags.update(['HIGH_SENSITIVITY', 'PERSONAL_DATA', 'RGPD_CRITICAL'])
        elif SensitivityLevel.RESTRICTED in sensitivity_levels:
            tags.update(['RESTRICTED_ACCESS', 'HIGH_SECURITY'])
        elif SensitivityLevel.CONFIDENTIAL in sensitivity_levels:
            tags.update(['CONFIDENTIAL', 'MEDIUM_SECURITY'])
        else:
            tags.add('INTERNAL_USE')
        
        # RÈGLES BASÉES SUR LE CONTEXTE SÉMANTIQUE
        # Seuil de 0.3 = contexte significativement présent
        for context, score in context_scores.items():
            if score > 0.3:
                tags.add(f'CONTEXT_{context.upper()}')
            # Seuil de 0.5 = contexte très dominant
            if score > 0.5:
                tags.add(f'PRIMARY_{context.upper()}')
        
        # RÈGLES BASÉES SUR LE NOM DU DATASET
        # Analyse du nom pour tags contextuels supplémentaires
        if dataset_name:
            dataset_lower = dataset_name.lower()
            
            # Patterns courants dans les noms de datasets
            if any(keyword in dataset_lower for keyword in ['client', 'customer', 'user']):
                tags.update(['CLIENT_DATA', 'CUSTOMER_INFO'])
            
            elif any(keyword in dataset_lower for keyword in ['transaction', 'payment', 'order']):
                tags.update(['TRANSACTION_DATA', 'BUSINESS_DATA'])
            
            elif any(keyword in dataset_lower for keyword in ['employee', 'staff', 'hr']):
                tags.update(['EMPLOYEE_DATA', 'HR_DATA'])
            
            elif any(keyword in dataset_lower for keyword in ['log', 'event', 'activity']):
                tags.update(['LOG_DATA', 'EVENT_DATA', 'ACTIVITY_TRACKING'])
        
        # RÈGLES BASÉES SUR LA COMBINAISON D'ENTITÉS
        # Détection de patterns complexes
        if len(set(entity_types)) > 3:  # Dataset avec beaucoup de types d'entités
            tags.add('MULTI_ENTITY')
            tags.add('COMPLEX_DATASET')
        
        if ('PERSON' in entity_types and 'LOCATION' in entity_types and 
            'PHONE_NUMBER' in entity_types):
            tags.add('COMPLETE_PROFILE')  # Profil complet de personne
        
        # Retourner la liste triée pour cohérence
        return sorted(list(tags))

