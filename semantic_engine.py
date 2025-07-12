# =============================================================================
# SYSTÈME D'AUTOTAGGING ET NLP SÉMANTIQUE AMÉLIORÉ -
# =============================================================================

import spacy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re
from collections import Counter, defaultdict
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import sqlite3
from datetime import datetime
import hashlib

# =============================================================================
# DÉFINITION DES STRUCTURES DE DONNÉES
# =============================================================================

class SensitivityLevel(Enum):
    """Niveaux de sensibilité des données selon RGPD"""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    PERSONAL_DATA = "PERSONAL_DATA"

class DataCategory(Enum):
    """Catégories de données métier"""
    IDENTITY = "IDENTITY"
    FINANCIAL = "FINANCIAL"
    CONTACT = "CONTACT"
    LOCATION = "LOCATION"
    TRANSACTION = "TRANSACTION"
    BEHAVIORAL = "BEHAVIORAL"

@dataclass
class EntityMetadata:
    """Métadonnées enrichies d'une entité détectée"""
    entity_type: str
    entity_value: str
    start_pos: int
    end_pos: int
    confidence_score: float
    sensitivity_level: SensitivityLevel
    data_category: DataCategory
    semantic_context: List[str]
    rgpd_category: Optional[str] = None
    anonymization_method: Optional[str] = None

@dataclass
class DatasetProfile:
    """Profil complet d'un jeu de données"""
    dataset_id: str
    name: str
    total_entities: int
    entity_distribution: Dict[str, int]
    sensitivity_distribution: Dict[str, int]
    semantic_tags: List[str]
    quality_score: float
    rgpd_compliance_score: float
    recommendations: List[str]
    created_at: datetime

# =============================================================================
# MOTEUR D'ANALYSE SÉMANTIQUE
# =============================================================================

class SemanticAnalyzer:
    """Analyseur sémantique pour comprendre le contexte des données"""
    
    def __init__(self, model_path: str):
        """
        Initialise l'analyseur sémantique
        
        Args:
            model_path: Chemin vers le modèle spaCy entraîné
        """
        self.nlp = spacy.load(model_path)
        
        # Dictionnaires sémantiques pour la classification contextuelle
        self.semantic_contexts = {
            'financial': [
                'banque', 'compte', 'iban', 'transaction', 'virement', 'carte',
                'crédit', 'débit', 'solde', 'facture', 'paiement', 'montant'
            ],
            'identity': [
                'nom', 'prénom', 'cin', 'identité', 'naissance', 'âge',
                'nationalité', 'profession', 'statut', 'titre'
            ],
            'contact': [
                'email', 'téléphone', 'adresse', 'domicile', 'bureau',
                'contact', 'joindre', 'appeler', 'écrire'
            ],
            'location': [
                'rue', 'avenue', 'boulevard', 'ville', 'quartier', 'région',
                'pays', 'code postal', 'géolocalisation', 'coordonnées'
            ],
            'behavioral': [
                'historique', 'comportement', 'préférence', 'habitude',
                'fréquence', 'pattern', 'analyse', 'profil'
            ]
        }
        
        # Mapping des entités vers les catégories RGPD
        self.rgpd_mapping = {
            'PERSON': 'Données d\'identification',
            'ID_MAROC': 'Données d\'identification',
            'PHONE_NUMBER': 'Données de contact',
            'EMAIL_ADDRESS': 'Données de contact',
            'LOCATION': 'Données de localisation',
            'IBAN_CODE': 'Données financières',
            'DATE_TIME': 'Données temporelles'
        }
        
        # Méthodes d'anonymisation recommandées
        self.anonymization_methods = {
            'PERSON': 'pseudonymisation',
            'ID_MAROC': 'hachage',
            'PHONE_NUMBER': 'masquage partiel',
            'EMAIL_ADDRESS': 'masquage partiel',
            'LOCATION': 'généralisation',
            'IBAN_CODE': 'chiffrement',
            'DATE_TIME': 'généralisation temporelle'
        }
    
    def analyze_semantic_context(self, text: str) -> Dict[str, float]:
        """
        Analyse le contexte sémantique d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des scores de contexte par domaine
        """
        doc = self.nlp(text.lower())
        
        # Extraire les tokens significatifs
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Calculer les scores pour chaque contexte
        context_scores = {}
        for context, keywords in self.semantic_contexts.items():
            # Compter les correspondances
            matches = sum(1 for token in tokens if any(keyword in token for keyword in keywords))
            # Normaliser par le nombre total de tokens
            context_scores[context] = matches / len(tokens) if tokens else 0
        
        return context_scores
    
    def determine_sensitivity_level(self, entity_type: str, context_scores: Dict[str, float]) -> SensitivityLevel:
        """
        Détermine le niveau de sensibilité d'une entité
        
        Args:
            entity_type: Type d'entité détectée
            context_scores: Scores de contexte sémantique
            
        Returns:
            Niveau de sensibilité
        """
        # Règles de classification de sensibilité
        if entity_type in ['PERSON', 'ID_MAROC']:
            return SensitivityLevel.PERSONAL_DATA
        elif entity_type in ['IBAN_CODE'] or context_scores.get('financial', 0) > 0.3:
            return SensitivityLevel.RESTRICTED
        elif entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS']:
            return SensitivityLevel.CONFIDENTIAL
        elif entity_type == 'LOCATION':
            return SensitivityLevel.CONFIDENTIAL
        else:
            return SensitivityLevel.INTERNAL
    
    def determine_data_category(self, entity_type: str, context_scores: Dict[str, float]) -> DataCategory:
        """
        Détermine la catégorie de données
        
        Args:
            entity_type: Type d'entité
            context_scores: Scores de contexte
            
        Returns:
            Catégorie de données
        """
        # Mapping basé sur le type d'entité et le contexte
        if entity_type in ['PERSON', 'ID_MAROC']:
            return DataCategory.IDENTITY
        elif entity_type == 'IBAN_CODE' or context_scores.get('financial', 0) > 0.2:
            return DataCategory.FINANCIAL
        elif entity_type in ['PHONE_NUMBER', 'EMAIL_ADDRESS']:
            return DataCategory.CONTACT
        elif entity_type == 'LOCATION':
            return DataCategory.LOCATION
        elif context_scores.get('behavioral', 0) > 0.2:
            return DataCategory.BEHAVIORAL
        else:
            return DataCategory.TRANSACTION

# =============================================================================
# MOTEUR D'AUTOTAGGING INTELLIGENT
# =============================================================================

class IntelligentAutoTagger:
    """Moteur d'autotagging basé sur l'analyse sémantique et les patterns"""
    
    def __init__(self, analyzer_engine: AnalyzerEngine, semantic_analyzer: SemanticAnalyzer):
        """
        Initialise le moteur d'autotagging
        
        Args:
            analyzer_engine: Moteur d'analyse Presidio
            semantic_analyzer: Analyseur sémantique
        """
        self.analyzer = analyzer_engine
        self.semantic_analyzer = semantic_analyzer
        self.tag_rules = self._initialize_tag_rules()
    
    def _initialize_tag_rules(self) -> Dict[str, List[str]]:
        """
        Initialise les règles d'étiquetage automatique
        
        Returns:
            Dictionnaire des règles d'étiquetage
        """
        return {
            'contains_personal_data': ['PII', 'RGPD', 'PERSONAL'],
            'contains_financial_data': ['FINANCIAL', 'BANKING', 'PAYMENT'],
            'contains_contact_data': ['CONTACT', 'COMMUNICATION'],
            'contains_location_data': ['LOCATION', 'GEOGRAPHIC'],
            'high_sensitivity': ['RESTRICTED', 'CONFIDENTIAL'],
            'medium_sensitivity': ['INTERNAL'],
            'low_sensitivity': ['PUBLIC']
        }
    
    def analyze_and_tag(self, text: str, dataset_name: str = "") -> Tuple[List[EntityMetadata], List[str]]:
        """
        Analyse un texte et génère les tags automatiquement
        
        Args:
            text: Texte à analyser
            dataset_name: Nom du jeu de données
            
        Returns:
            Tuple (entités_enrichies, tags_générés)
        """
        # Analyser avec Presidio
        presidio_results = self.analyzer.analyze(text=text, language="fr")
        
        # Analyser le contexte sémantique
        context_scores = self.semantic_analyzer.analyze_semantic_context(text)
        
        # Enrichir les entités avec les métadonnées
        enriched_entities = []
        for result in presidio_results:
            entity_value = text[result.start:result.end]
            
            # Déterminer la sensibilité et la catégorie
            sensitivity = self.semantic_analyzer.determine_sensitivity_level(
                result.entity_type, context_scores
            )
            category = self.semantic_analyzer.determine_data_category(
                result.entity_type, context_scores
            )
            
            # Créer les métadonnées enrichies
            metadata = EntityMetadata(
                entity_type=result.entity_type,
                entity_value=entity_value,
                start_pos=result.start,
                end_pos=result.end,
                confidence_score=result.score,
                sensitivity_level=sensitivity,
                data_category=category,
                semantic_context=list(context_scores.keys()),
                rgpd_category=self.semantic_analyzer.rgpd_mapping.get(result.entity_type),
                anonymization_method=self.semantic_analyzer.anonymization_methods.get(result.entity_type)
            )
            
            enriched_entities.append(metadata)
        
        # Générer les tags automatiques
        generated_tags = self._generate_tags(enriched_entities, context_scores, dataset_name)
        
        return enriched_entities, generated_tags
    
    def _generate_tags(self, entities: List[EntityMetadata], context_scores: Dict[str, float], dataset_name: str) -> List[str]:
        """
        Génère les tags automatiquement basés sur l'analyse
        
        Args:
            entities: Liste des entités enrichies
            context_scores: Scores de contexte sémantique
            dataset_name: Nom du jeu de données
            
        Returns:
            Liste des tags générés
        """
        tags = set()
        
        # Tags basés sur les types d'entités
        entity_types = [entity.entity_type for entity in entities]
        if any(t in ['PERSON', 'ID_MAROC'] for t in entity_types):
            tags.update(['PII', 'RGPD', 'PERSONAL_DATA'])
        
        if 'IBAN_CODE' in entity_types:
            tags.update(['FINANCIAL', 'BANKING'])
        
        if any(t in ['PHONE_NUMBER', 'EMAIL_ADDRESS'] for t in entity_types):
            tags.update(['CONTACT', 'COMMUNICATION'])
        
        if 'LOCATION' in entity_types:
            tags.update(['LOCATION', 'GEOGRAPHIC'])
        
        # Tags basés sur la sensibilité
        sensitivity_levels = [entity.sensitivity_level for entity in entities]
        if SensitivityLevel.PERSONAL_DATA in sensitivity_levels:
            tags.add('HIGH_SENSITIVITY')
        elif SensitivityLevel.RESTRICTED in sensitivity_levels:
            tags.add('RESTRICTED_ACCESS')
        elif SensitivityLevel.CONFIDENTIAL in sensitivity_levels:
            tags.add('CONFIDENTIAL')
        
        # Tags basés sur le contexte sémantique
        for context, score in context_scores.items():
            if score > 0.3:
                tags.add(f'CONTEXT_{context.upper()}')
        
        # Tags basés sur le nom du dataset
        if dataset_name:
            if 'client' in dataset_name.lower():
                tags.add('CLIENT_DATA')
            elif 'transaction' in dataset_name.lower():
                tags.add('TRANSACTION_DATA')
            elif 'employee' in dataset_name.lower():
                tags.add('EMPLOYEE_DATA')
        
        return sorted(list(tags))

