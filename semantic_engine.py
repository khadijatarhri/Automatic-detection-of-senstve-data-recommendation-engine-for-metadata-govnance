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

# =============================================================================
# MOTEUR DE QUALITÉ DES DONNÉES
# =============================================================================

class DataQualityEngine:
    """Moteur d'évaluation de la qualité des données"""
    
    def __init__(self):
        """Initialise le moteur de qualité"""
        self.quality_metrics = {
            'completeness': 0.0,
            'accuracy': 0.0,
            'consistency': 0.0,
            'validity': 0.0,
            'uniqueness': 0.0
        }
    
    def evaluate_quality(self, entities: List[EntityMetadata], text: str) -> Dict[str, float]:
        """
        Évalue la qualité des données détectées
        
        Args:
            entities: Liste des entités détectées
            text: Texte original
            
        Returns:
            Dictionnaire des métriques de qualité
        """
        metrics = {}
        
        # Complétude : ratio d'entités détectées vs attendues
        total_tokens = len(text.split())
        detected_entities = len(entities)
        metrics['completeness'] = min(detected_entities / max(total_tokens * 0.1, 1), 1.0)
        
        # Précision : score de confiance moyen
        if entities:
            metrics['accuracy'] = sum(entity.confidence_score for entity in entities) / len(entities)
        else:
            metrics['accuracy'] = 0.0
        
        # Cohérence : cohérence des types d'entités
        entity_types = [entity.entity_type for entity in entities]
        type_consistency = len(set(entity_types)) / len(entity_types) if entity_types else 1.0
        metrics['consistency'] = 1.0 - type_consistency
        
        # Validité : format des entités
        valid_entities = sum(1 for entity in entities if self._is_valid_format(entity))
        metrics['validity'] = valid_entities / len(entities) if entities else 1.0
        
        # Unicité : pas de doublons
        unique_values = len(set(entity.entity_value for entity in entities))
        metrics['uniqueness'] = unique_values / len(entities) if entities else 1.0
        
        return metrics
    
    def _is_valid_format(self, entity: EntityMetadata) -> bool:
        """
        Vérifie si le format d'une entité est valide
        
        Args:
            entity: Entité à vérifier
            
        Returns:
            True si le format est valide
        """
        patterns = {
            'ID_MAROC': r'^[A-Z]{2}[0-9]{5,6}$',
            'PHONE_NUMBER': r'^(\+212|0)[0-9]{9}$',
            'EMAIL_ADDRESS': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'IBAN_CODE': r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}$'
        }
        
        pattern = patterns.get(entity.entity_type)
        if pattern:
            return bool(re.match(pattern, entity.entity_value))
        return True

# =============================================================================
# SYSTÈME DE PROFILAGE DE DONNÉES
# =============================================================================

class DataProfiler:
    """Système de profilage complet des données"""
    
    def __init__(self, auto_tagger: IntelligentAutoTagger, quality_engine: DataQualityEngine):
        """
        Initialise le profileur de données
        
        Args:
            auto_tagger: Moteur d'autotagging
            quality_engine: Moteur de qualité
        """
        self.auto_tagger = auto_tagger
        self.quality_engine = quality_engine
        self.profiles_db = "data_profiles.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialise la base de données des profils"""
        conn = sqlite3.connect(self.profiles_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE,
                name TEXT,
                profile_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_profile(self, dataset_name: str, texts: List[str]) -> DatasetProfile:
        """
        Crée un profil complet pour un jeu de données
        
        Args:
            dataset_name: Nom du jeu de données
            texts: Liste des textes à analyser
            
        Returns:
            Profil complet du jeu de données
        """
        all_entities = []
        all_tags = set()
        quality_scores = []
        
        # Analyser chaque texte
        for text in texts:
            entities, tags = self.auto_tagger.analyze_and_tag(text, dataset_name)
            all_entities.extend(entities)
            all_tags.update(tags)
            
            # Évaluer la qualité
            quality_metrics = self.quality_engine.evaluate_quality(entities, text)
            quality_scores.append(quality_metrics)
        
        # Calculer les distributions
        entity_distribution = Counter(entity.entity_type for entity in all_entities)
        sensitivity_distribution = Counter(entity.sensitivity_level.value for entity in all_entities)
        
        # Calculer les scores globaux
        overall_quality = self._calculate_overall_quality(quality_scores)
        rgpd_compliance = self._calculate_rgpd_compliance(all_entities)
        
        # Générer les recommandations
        recommendations = self._generate_recommendations(all_entities, overall_quality, rgpd_compliance)
        
        # Créer le profil
        dataset_id = hashlib.md5(dataset_name.encode()).hexdigest()
        profile = DatasetProfile(
            dataset_id=dataset_id,
            name=dataset_name,
            total_entities=len(all_entities),
            entity_distribution=dict(entity_distribution),
            sensitivity_distribution=dict(sensitivity_distribution),
            semantic_tags=sorted(list(all_tags)),
            quality_score=overall_quality,
            rgpd_compliance_score=rgpd_compliance,
            recommendations=recommendations,
            created_at=datetime.now()
        )
        
        # Sauvegarder le profil
        self._save_profile(profile)
        
        return profile
    
    def _calculate_overall_quality(self, quality_scores: List[Dict[str, float]]) -> float:
        """Calcule le score de qualité global"""
        if not quality_scores:
            return 0.0
        
        # Moyenner tous les scores
        total_scores = defaultdict(list)
        for score_dict in quality_scores:
            for metric, value in score_dict.items():
                total_scores[metric].append(value)
        
        # Calculer la moyenne pondérée
        weights = {'completeness': 0.3, 'accuracy': 0.3, 'consistency': 0.2, 'validity': 0.1, 'uniqueness': 0.1}
        overall_score = 0.0
        
        for metric, values in total_scores.items():
            avg_value = sum(values) / len(values)
            overall_score += avg_value * weights.get(metric, 0.1)
        
        return round(overall_score, 2)
    
    def _calculate_rgpd_compliance(self, entities: List[EntityMetadata]) -> float:
        """Calcule le score de conformité RGPD"""
        if not entities:
            return 1.0
        
        # Vérifier les critères RGPD
        personal_data_entities = [e for e in entities if e.sensitivity_level == SensitivityLevel.PERSONAL_DATA]
        compliance_score = 0.0
        
        if personal_data_entities:
            # Vérifier si les données personnelles ont des méthodes d'anonymisation
            with_anonymization = sum(1 for e in personal_data_entities if e.anonymization_method)
            compliance_score += (with_anonymization / len(personal_data_entities)) * 0.5
            
            # Vérifier si les catégories RGPD sont définies
            with_rgpd_category = sum(1 for e in personal_data_entities if e.rgpd_category)
            compliance_score += (with_rgpd_category / len(personal_data_entities)) * 0.5
        else:
            compliance_score = 1.0
        
        return round(compliance_score, 2)
    
    def _generate_recommendations(self, entities: List[EntityMetadata], quality_score: float, rgpd_score: float) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        # Recommandations basées sur la qualité
        if quality_score < 0.7:
            recommendations.append("⚠️ Améliorer la qualité des données - Score actuel: {:.1%}".format(quality_score))
        
        # Recommandations RGPD
        if rgpd_score < 0.8:
            recommendations.append("🛡️ Renforcer la conformité RGPD - Score actuel: {:.1%}".format(rgpd_score))
        
        # Recommandations spécifiques par type d'entité
        entity_types = set(entity.entity_type for entity in entities)
        
        if 'PERSON' in entity_types:
            recommendations.append("👤 Appliquer la pseudonymisation pour les noms de personnes")
        
        if 'ID_MAROC' in entity_types:
            recommendations.append("🆔 Chiffrer ou hacher les numéros CIN")
        
        if 'IBAN_CODE' in entity_types:
            recommendations.append("💳 Sécuriser les données bancaires avec un chiffrement fort")
        
        if 'LOCATION' in entity_types:
            recommendations.append("📍 Généraliser les données de localisation")
        
        # Recommandations de gouvernance
        recommendations.append("📋 Définir un propriétaire de données (Data Owner)")
        recommendations.append("🔄 Mettre en place un processus de révision périodique")
        recommendations.append("📊 Créer des métriques de suivi de la qualité")
        
        return recommendations
    
    def _save_profile(self, profile: DatasetProfile):
        """Sauvegarde le profil en base de données"""
        conn = sqlite3.connect(self.profiles_db)
        cursor = conn.cursor()
        
        profile_json = json.dumps({
            'dataset_id': profile.dataset_id,
            'name': profile.name,
            'total_entities': profile.total_entities,
            'entity_distribution': profile.entity_distribution,
            'sensitivity_distribution': profile.sensitivity_distribution,
            'semantic_tags': profile.semantic_tags,
            'quality_score': profile.quality_score,
            'rgpd_compliance_score': profile.rgpd_compliance_score,
            'recommendations': profile.recommendations,
            'created_at': profile.created_at.isoformat()
        })
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_profiles (dataset_id, name, profile_data)
            VALUES (?, ?, ?)
        ''', (profile.dataset_id, profile.name, profile_json))
        
        conn.commit()
        conn.close()

# =============================================================================
# SYSTÈME INTÉGRÉ - ORCHESTRATEUR PRINCIPAL
# =============================================================================

class IntegratedDataGovernanceSystem:
    """Système intégré de gouvernance des données"""
    
    def __init__(self, model_path: str, analyzer_engine: AnalyzerEngine):
        """
        Initialise le système intégré
        
        Args:
            model_path: Chemin vers le modèle spaCy
            analyzer_engine: Moteur d'analyse Presidio
        """
        self.semantic_analyzer = SemanticAnalyzer(model_path)
        self.auto_tagger = IntelligentAutoTagger(analyzer_engine, self.semantic_analyzer)
        self.quality_engine = DataQualityEngine()
        self.profiler = DataProfiler(self.auto_tagger, self.quality_engine)
        self.anonymizer = AnonymizerEngine()
    
    def process_dataset(self, dataset_name: str, data: List[str]) -> Dict[str, Any]:
        """
        Traite un jeu de données complet
        
        Args:
            dataset_name: Nom du jeu de données
            data: Liste des textes à analyser
            
        Returns:
            Résultats complets de l'analyse
        """
        print(f"🔍 Traitement du jeu de données: {dataset_name}")
        print(f"📊 Nombre d'enregistrements: {len(data)}")
        
        # Créer le profil complet
        profile = self.profiler.create_profile(dataset_name, data)
        
        # Analyser quelques exemples en détail
        sample_analyses = []
        for i, text in enumerate(data[:3]):  # Analyser les 3 premiers exemples
            entities, tags = self.auto_tagger.analyze_and_tag(text, dataset_name)
            quality_metrics = self.quality_engine.evaluate_quality(entities, text)
            
            sample_analyses.append({
                'text': text,
                'entities': [
                    {
                        'type': e.entity_type,
                        'value': e.entity_value,
                        'start': e.start_pos,
                        'end': e.end_pos,
                        'sensitivity': e.sensitivity_level.value,
                        'category': e.data_category.value,
                        'confidence': round(e.confidence_score, 2),
                        'anonymization': e.anonymization_method
                    } for e in entities
                ],
                'tags': tags,
                'quality': quality_metrics
            })

        # Résumé global
        return {
            'dataset_name': profile.name,
            'total_entities': profile.total_entities,
            'entity_distribution': profile.entity_distribution,
            'sensitivity_distribution': profile.sensitivity_distribution,
            'semantic_tags': profile.semantic_tags,
            'quality_score': profile.quality_score,
            'rgpd_compliance_score': profile.rgpd_compliance_score,
            'recommendations': profile.recommendations,
            'samples': sample_analyses,
            'created_at': profile.created_at.isoformat()
        }
