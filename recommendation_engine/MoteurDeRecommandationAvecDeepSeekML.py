# =============================================================================
# MOTEUR DE RECOMMANDATION AVEC DEEPSEEK ML
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3
from datetime import datetime
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STRUCTURES DE DONNÉES POUR LES RECOMMANDATIONS
# =============================================================================

@dataclass
class RecommendationItem:
    """Item de recommandation avec métadonnées"""
    id: str
    title: str
    description: str
    category: str
    priority: float
    confidence: float
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class DatasetRecommendation:
    """Recommandation spécifique à un dataset"""
    dataset_id: str
    recommendations: List[RecommendationItem]
    overall_score: float
    improvement_areas: List[str]
    compliance_gaps: List[str]

class RecommendationType(Enum):
    """Types de recommandations"""
    QUALITY_IMPROVEMENT = "QUALITY_IMPROVEMENT"
    SECURITY_ENHANCEMENT = "SECURITY_ENHANCEMENT"
    COMPLIANCE_RGPD = "COMPLIANCE_RGPD"
    METADATA_ENRICHMENT = "METADATA_ENRICHMENT"
    CLASSIFICATION_OPTIMIZATION = "CLASSIFICATION_OPTIMIZATION"
    ANONYMIZATION_STRATEGY = "ANONYMIZATION_STRATEGY"

# =============================================================================
# CLIENT DEEPSEEK ML
# =============================================================================

class DeepSeekClient:
    """Client pour interagir avec l'API DeepSeek"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_recommendations(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Génère des recommandations via l'API DeepSeek
        
        Args:
            prompt: Prompt pour la génération
            max_tokens: Nombre maximum de tokens
            
        Returns:
            Réponse générée par DeepSeek
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """Tu es un expert en gouvernance des données et en conformité RGPD. 
                    Tu dois analyser les profils de datasets et générer des recommandations précises 
                    pour améliorer la qualité, la sécurité et la conformité des données."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Erreur API DeepSeek: {response.status}")

# =============================================================================
# MOTEUR DE RECOMMANDATION INTELLIGENT
# =============================================================================

class IntelligentRecommendationEngine:
    """Moteur de recommandation basé sur DeepSeek ML et l'analyse sémantique"""
    
    def __init__(self, deepseek_client: DeepSeekClient, database_path: str = "recommendations.db"):
        self.deepseek_client = deepseek_client
        self.database_path = database_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.recommendation_templates = self._load_recommendation_templates()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialise la base de données pour stocker les recommandations"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                dataset_id TEXT,
                type TEXT,
                title TEXT,
                description TEXT,
                priority REAL,
                confidence REAL,
                metadata TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_analysis (
                dataset_id TEXT PRIMARY KEY,
                profile_data TEXT,
                analysis_results TEXT,
                last_updated TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_recommendation_templates(self) -> Dict[str, str]:
        """Charge les templates de recommandations"""
        return {
            "quality_analysis": """
            Analyse le profil de dataset suivant et génère des recommandations pour améliorer la qualité des données:
            
            Dataset: {dataset_name}
            Entités détectées: {entities}
            Distribution de sensibilité: {sensitivity_distribution}
            Score de qualité actuel: {quality_score}
            
            Génère des recommandations SPÉCIFIQUES en format JSON avec les clés:
            - type: type de recommandation
            - priority: priorité (1-10)
            - title: titre court
            - description: description détaillée
            - actions: liste d'actions concrètes
            - impact: impact estimé
            """,
            
            "security_analysis": """
            Analyse de sécurité pour le dataset:
            
            Dataset: {dataset_name}
            Données sensibles: {sensitive_data}
            Niveaux de sensibilité: {sensitivity_levels}
            Méthodes d'anonymisation actuelles: {anonymization_methods}
            
            Génère des recommandations de sécurité en format JSON pour:
            - Améliorer la protection des données sensibles
            - Optimiser les méthodes d'anonymisation
            - Renforcer l'accès aux données
            """,
            
            "compliance_analysis": """
            Analyse de conformité RGPD pour:
            
            Dataset: {dataset_name}
            Catégories RGPD détectées: {rgpd_categories}
            Score de conformité: {compliance_score}
            Lacunes identifiées: {compliance_gaps}
            
            Génère des recommandations de conformité RGPD en format JSON pour:
            - Combler les lacunes de conformité
            - Améliorer la gouvernance des données
            - Optimiser la gestion des droits des personnes
            """,
            
            "metadata_enrichment": """
            Analyse des métadonnées pour:
            
            Dataset: {dataset_name}
            Métadonnées actuelles: {current_metadata}
            Tags générés: {generated_tags}
            Contexte sémantique: {semantic_context}
            
            Génère des recommandations d'enrichissement en format JSON pour:
            - Améliorer la qualité des métadonnées
            - Optimiser l'étiquetage automatique
            - Enrichir le contexte sémantique
            """
        }
    
    async def generate_comprehensive_recommendations(self, dataset_profile: dict) -> DatasetRecommendation:
        """
        Génère des recommandations complètes pour un dataset
        
        Args:
            dataset_profile: Profil complet du dataset
            
        Returns:
            Recommandations complètes
        """
        dataset_id = dataset_profile.get('dataset_id', 'unknown')
        recommendations = []
        
        # 1. Analyse de qualité
        quality_recs = await self._generate_quality_recommendations(dataset_profile)
        recommendations.extend(quality_recs)
        
        # 2. Analyse de sécurité
        security_recs = await self._generate_security_recommendations(dataset_profile)
        recommendations.extend(security_recs)
        
        # 3. Analyse de conformité RGPD
        compliance_recs = await self._generate_compliance_recommendations(dataset_profile)
        recommendations.extend(compliance_recs)
        
        # 4. Enrichissement des métadonnées
        metadata_recs = await self._generate_metadata_recommendations(dataset_profile)
        recommendations.extend(metadata_recs)
        
        # 5. Calcul du score global et identification des domaines d'amélioration
        overall_score = self._calculate_overall_score(dataset_profile, recommendations)
        improvement_areas = self._identify_improvement_areas(recommendations)
        compliance_gaps = self._identify_compliance_gaps(dataset_profile)
        
        # 6. Sauvegarder les recommandations
        await self._save_recommendations(dataset_id, recommendations)
        
        return DatasetRecommendation(
            dataset_id=dataset_id,
            recommendations=recommendations,
            overall_score=overall_score,
            improvement_areas=improvement_areas,
            compliance_gaps=compliance_gaps
        )
    
    async def _generate_quality_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:
        """Génère des recommandations de qualité"""
        prompt = self.recommendation_templates["quality_analysis"].format(
            dataset_name=dataset_profile.get('name', 'Dataset'),
            entities=dataset_profile.get('entity_distribution', {}),
            sensitivity_distribution=dataset_profile.get('sensitivity_distribution', {}),
            quality_score=dataset_profile.get('quality_score', 0.0)
        )
        
        response = await self.deepseek_client.generate_recommendations(prompt)
        
        # Parser la réponse JSON
        recommendations = []
        try:
            # Extraire le JSON de la réponse
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_recs = json.loads(json_str)
                
                for rec in parsed_recs:
                    recommendation = RecommendationItem(
                        id=f"quality_{dataset_profile.get('dataset_id', 'unknown')}_{len(recommendations)}",
                        title=rec.get('title', 'Amélioration de qualité'),
                        description=rec.get('description', ''),
                        category=RecommendationType.QUALITY_IMPROVEMENT.value,
                        priority=float(rec.get('priority', 5.0)),
                        confidence=0.85,
                        metadata=rec,
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)
        except Exception as e:
            print(f"Erreur lors du parsing des recommandations qualité: {e}")
            
        return recommendations
    
    async def _generate_security_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:
        """Génère des recommandations de sécurité"""
        prompt = self.recommendation_templates["security_analysis"].format(
            dataset_name=dataset_profile.get('name', 'Dataset'),
            sensitive_data=dataset_profile.get('sensitive_entities', []),
            sensitivity_levels=dataset_profile.get('sensitivity_distribution', {}),
            anonymization_methods=dataset_profile.get('anonymization_methods', [])
        )
        
        response = await self.deepseek_client.generate_recommendations(prompt)
        
        recommendations = []
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_recs = json.loads(json_str)
                
                for rec in parsed_recs:
                    recommendation = RecommendationItem(
                        id=f"security_{dataset_profile.get('dataset_id', 'unknown')}_{len(recommendations)}",
                        title=rec.get('title', 'Amélioration sécurité'),
                        description=rec.get('description', ''),
                        category=RecommendationType.SECURITY_ENHANCEMENT.value,
                        priority=float(rec.get('priority', 8.0)),
                        confidence=0.90,
                        metadata=rec,
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)
        except Exception as e:
            print(f"Erreur lors du parsing des recommandations sécurité: {e}")
            
        return recommendations
    
    async def _generate_compliance_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:
        """Génère des recommandations de conformité RGPD"""
        prompt = self.recommendation_templates["compliance_analysis"].format(
            dataset_name=dataset_profile.get('name', 'Dataset'),
            rgpd_categories=dataset_profile.get('rgpd_categories', []),
            compliance_score=dataset_profile.get('rgpd_compliance_score', 0.0),
            compliance_gaps=dataset_profile.get('compliance_gaps', [])
        )
        
        response = await self.deepseek_client.generate_recommendations(prompt)
        
        recommendations = []
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_recs = json.loads(json_str)
                
                for rec in parsed_recs:
                    recommendation = RecommendationItem(
                        id=f"compliance_{dataset_profile.get('dataset_id', 'unknown')}_{len(recommendations)}",
                        title=rec.get('title', 'Conformité RGPD'),
                        description=rec.get('description', ''),
                        category=RecommendationType.COMPLIANCE_RGPD.value,
                        priority=float(rec.get('priority', 9.0)),
                        confidence=0.88,
                        metadata=rec,
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)
        except Exception as e:
            print(f"Erreur lors du parsing des recommandations conformité: {e}")
            
        return recommendations
    
    async def _generate_metadata_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:
        """Génère des recommandations d'enrichissement des métadonnées"""
        prompt = self.recommendation_templates["metadata_enrichment"].format(
            dataset_name=dataset_profile.get('name', 'Dataset'),
            current_metadata=dataset_profile.get('metadata', {}),
            generated_tags=dataset_profile.get('semantic_tags', []),
            semantic_context=dataset_profile.get('semantic_context', {})
        )
        
        response = await self.deepseek_client.generate_recommendations(prompt)
        
        recommendations = []
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_recs = json.loads(json_str)
                
                for rec in parsed_recs:
                    recommendation = RecommendationItem(
                        id=f"metadata_{dataset_profile.get('dataset_id', 'unknown')}_{len(recommendations)}",
                        title=rec.get('title', 'Enrichissement métadonnées'),
                        description=rec.get('description', ''),
                        category=RecommendationType.METADATA_ENRICHMENT.value,
                        priority=float(rec.get('priority', 6.0)),
                        confidence=0.82,
                        metadata=rec,
                        created_at=datetime.now()
                    )
                    recommendations.append(recommendation)
        except Exception as e:
            print(f"Erreur lors du parsing des recommandations métadonnées: {e}")
            
        return recommendations
    
    def _calculate_overall_score(self, dataset_profile: dict, recommendations: List[RecommendationItem]) -> float:
        """Calcule le score global du dataset"""
        quality_score = dataset_profile.get('quality_score', 0.0)
        compliance_score = dataset_profile.get('rgpd_compliance_score', 0.0)
        
        # Pénalité basée sur le nombre de recommandations critiques
        critical_recs = [r for r in recommendations if r.priority >= 8.0]
        penalty = len(critical_recs) * 0.1
        
        overall_score = max(0.0, min(10.0, (quality_score + compliance_score) / 2 - penalty))
        return overall_score
    
    def _identify_improvement_areas(self, recommendations: List[RecommendationItem]) -> List[str]:
        """Identifie les domaines d'amélioration prioritaires"""
        areas = {}
        for rec in recommendations:
            if rec.category not in areas:
                areas[rec.category] = []
            areas[rec.category].append(rec.priority)
        
        # Trier par priorité moyenne
        sorted_areas = sorted(areas.items(), key=lambda x: np.mean(x[1]), reverse=True)
        return [area[0] for area in sorted_areas[:3]]
    
    def _identify_compliance_gaps(self, dataset_profile: dict) -> List[str]:
        """Identifie les lacunes de conformité"""
        gaps = []
        
        # Vérifier la présence de données personnelles sans protection
        if dataset_profile.get('has_personal_data', False):
            if not dataset_profile.get('has_anonymization', False):
                gaps.append("Données personnelles non anonymisées")
        
        # Vérifier la documentation des traitements
        if dataset_profile.get('rgpd_compliance_score', 0.0) < 7.0:
            gaps.append("Documentation des traitements insuffisante")
        
        # Vérifier la gestion des droits
        if not dataset_profile.get('has_consent_management', False):
            gaps.append("Gestion des consentements manquante")
        
        return gaps
    
    async def _save_recommendations(self, dataset_id: str, recommendations: List[RecommendationItem]):
        """Sauvegarde les recommandations dans la base de données"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for rec in recommendations:
            cursor.execute('''
                INSERT OR REPLACE INTO recommendations 
                (id, dataset_id, type, title, description, priority, confidence, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rec.id,
                dataset_id,
                rec.category,
                rec.title,
                rec.description,
                rec.priority,
                rec.confidence,
                json.dumps(rec.metadata),
                rec.created_at
            ))
        
        conn.commit()
        conn.close()
    
    async def get_dataset_recommendations(self, dataset_id: str) -> Optional[DatasetRecommendation]:
        """Récupère les recommandations pour un dataset"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, type, title, description, priority, confidence, metadata, created_at
            FROM recommendations
            WHERE dataset_id = ?
            ORDER BY priority DESC
        ''', (dataset_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return None
        
        recommendations = []
        for row in results:
            rec = RecommendationItem(
                id=row[0],
                title=row[2],
                description=row[3],
                category=row[1],
                priority=row[4],
                confidence=row[5],
                metadata=json.loads(row[6]),
                created_at=datetime.fromisoformat(row[7])
            )
            recommendations.append(rec)
        
        return DatasetRecommendation(
            dataset_id=dataset_id,
            recommendations=recommendations,
            overall_score=8.0,  # Calculé dynamiquement
            improvement_areas=['QUALITY_IMPROVEMENT', 'SECURITY_ENHANCEMENT'],
            compliance_gaps=['Documentation manquante']
        )

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def example_usage():
    """Exemple d'utilisation du moteur de recommandation"""
    
    # Configuration DeepSeek
    DEEPSEEK_API_KEY = "votre_clé_api_deepseek"
    
    # Profil d'exemple d'un dataset
    sample_dataset_profile = {
        'dataset_id': 'clients_2024',
        'name': 'Base clients 2024',
        'entity_distribution': {
            'PERSON': 1250,
            'EMAIL_ADDRESS': 1200,
            'PHONE_NUMBER': 1180,
            'ID_MAROC': 1250,
            'LOCATION': 890
        },
        'sensitivity_distribution': {
            'PERSONAL_DATA': 2450,
            'CONFIDENTIAL': 1200,
            'INTERNAL': 500
        },
        'quality_score': 6.5,
        'rgpd_compliance_score': 7.2,
        'semantic_tags': ['CLIENT_DATA', 'PII', 'CONTACT'],
        'has_personal_data': True,
        'has_anonymization': False,
        'has_consent_management': True,
        'compliance_gaps': ['Documentation insuffisante', 'Anonymisation manquante']
    }
    
    # Créer le moteur de recommandation
    async with DeepSeekClient(DEEPSEEK_API_KEY) as client:
        engine = IntelligentRecommendationEngine(client)
        
        # Générer les recommandations
        print("🔍 Génération des recommandations...")
        recommendations = await engine.generate_comprehensive_recommendations(sample_dataset_profile)
        
        # Afficher les résultats
        print(f"\n📊 Recommandations pour {sample_dataset_profile['name']}")
        print(f"Score global: {recommendations.overall_score:.1f}/10")
        print(f"Domaines d'amélioration: {', '.join(recommendations.improvement_areas)}")
        print(f"Lacunes de conformité: {', '.join(recommendations.compliance_gaps)}")
        
        print(f"\n📋 Recommandations détaillées ({len(recommendations.recommendations)} items):")
        for i, rec in enumerate(recommendations.recommendations, 1):
            print(f"\n{i}. {rec.title} (Priorité: {rec.priority}/10)")
            print(f"   Catégorie: {rec.category}")
            print(f"   Description: {rec.description}")
            print(f"   Confiance: {rec.confidence:.2f}")

# Pour tester le code
if __name__ == "__main__":
    asyncio.run(example_usage())
