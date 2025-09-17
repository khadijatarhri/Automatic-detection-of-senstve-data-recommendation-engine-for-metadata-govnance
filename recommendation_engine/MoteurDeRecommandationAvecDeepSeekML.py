# =============================================================================  
# MOTEUR DE RECOMMANDATION AVEC GEMINI ML + KMEANS + PCA (ANALYSE INTRA-FICHIER)  
# =============================================================================  
from pymongo import MongoClient        
import pandas as pd  
import numpy as np  
from typing import Dict, List, Tuple, Optional, Any  
from dataclasses import dataclass  
from enum import Enum  
import json  
from datetime import datetime  
import asyncio  
import aiohttp  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.cluster import KMeans  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import StandardScaler  
import re

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
    column_clusters: Optional[Dict[str, Any]] = None  
    row_clusters: Optional[Dict[str, Any]] = None  
    pca_analysis: Optional[Dict[str, Any]] = None  
  
@dataclass  
class ColumnAnalysis:  
    """Analyse d'une colonne spécifique"""  
    column_name: str  
    cluster_id: int  
    sensitivity_score: float  
    entity_types: List[str]  
    anomaly_score: float  
  
class RecommendationType(Enum):  
    """Types de recommandations"""  
    QUALITY_IMPROVEMENT = "QUALITY_IMPROVEMENT"  
    SECURITY_ENHANCEMENT = "SECURITY_ENHANCEMENT"  
    COMPLIANCE_RGPD = "COMPLIANCE_RGPD"  
    METADATA_ENRICHMENT = "METADATA_ENRICHMENT"  
    CLASSIFICATION_OPTIMIZATION = "CLASSIFICATION_OPTIMIZATION"  
    ANONYMIZATION_STRATEGY = "ANONYMIZATION_STRATEGY"  
    COLUMN_BASED = "COLUMN_BASED"  
    ROW_BASED = "ROW_BASED"  
    ANOMALY_DETECTION = "ANOMALY_DETECTION"  
  
# =============================================================================  
# CLIENT GEMINI ML  
# =============================================================================  
  
import google.generativeai as genai  
  
class GeminiClient:    
    """Client pour interagir avec l'API Gemini"""    
        
    def __init__(self, api_key: str):    
        self.api_key = api_key    
        genai.configure(api_key=api_key)    
        self.model = genai.GenerativeModel('gemini-1.5-flash')    
      
    async def __aenter__(self):  
        """Async context manager entry"""  
        return self  
      
    async def __aexit__(self, exc_type, exc_val, exc_tb):  
        """Async context manager exit"""  
        # Clean up any resources here if needed  
        pass  
            
    async def generate_recommendations(self, prompt: str, max_tokens: int = 1500) -> str:    
        """Génère des recommandations via l'API Gemini"""    
        try:    
            response = self.model.generate_content(prompt)    
            return response.text    
        except Exception as e:    
            raise Exception(f"Erreur API Gemini: {e}")
# =============================================================================  
# MOTEUR DE RECOMMANDATION INTELLIGENT AVEC ML INTRA-FICHIER  
# =============================================================================  
  
class IntelligentRecommendationEngine:  
    """Moteur de recommandation basé sur Gemini ML, KMeans et PCA pour analyse intra-fichier"""  
      
    def __init__(self,gemini_client: GeminiClient, gemini_api_key: str, database_path: str = "recommendations.db"):  
         self.gemini_client = GeminiClient(gemini_api_key)  
         self.database_path = database_path 
         self.gemini_client = gemini_client 
         self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  
         self.scaler = StandardScaler()   
         self.recommendation_templates = self._load_recommendation_templates()  
          


    def get_column_annotations(self, job_id, column_name):  
       """Récupère les annotations existantes pour une colonne"""  
       client = MongoClient('mongodb://mongodb:27017/')  
       metadata_db = client['metadata_validation_db']  
       annotations_collection = metadata_db['column_annotations']  
      
       return annotations_collection.find_one({  
        'job_id': job_id,  
        'column_name': column_name  
       })
    


    def _extract_column_features(self, column_data: pd.Series, column_name: str, detected_entities: set) -> np.ndarray:  
        """Extrait les features d'une colonne spécifique"""  
        features = []  
          
        # Statistiques de base  
        total_values = len(column_data)  
        non_null_values = column_data.notna().sum()  
        unique_values = column_data.nunique()  
          
        features.extend([  
            non_null_values / total_values if total_values > 0 else 0,  # Taux de remplissage  
            unique_values / total_values if total_values > 0 else 0,    # Taux d'unicité  
        ])  
          
        # Détection d'entités dans la colonne  
        entity_counts = {}  
        for entity in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'ID_MAROC', 'CREDIT_CARD', 'IBAN_CODE', 'LOCATION']:  
            count = sum(1 for value in column_data.dropna().astype(str) if entity in detected_entities and entity in str(value))  
            entity_counts[entity] = count / total_values if total_values > 0 else 0  
            features.append(entity_counts[entity])  
          
        # Features textuelles du nom de colonne  
        column_name_lower = column_name.lower()  
        text_features = [  
            1.0 if any(word in column_name_lower for word in ['email', 'mail', 'e-mail']) else 0.0,  
            1.0 if any(word in column_name_lower for word in ['phone', 'tel', 'telephone']) else 0.0,  
            1.0 if any(word in column_name_lower for word in ['name', 'nom', 'prenom']) else 0.0,  
            1.0 if any(word in column_name_lower for word in ['id', 'identifier', 'cin']) else 0.0,  
            1.0 if any(word in column_name_lower for word in ['address', 'adresse', 'location']) else 0.0,  
            1.0 if any(word in column_name_lower for word in ['date', 'time', 'timestamp']) else 0.0,  
        ]  
        features.extend(text_features)  
          
        return np.array(features)  
      
    def _extract_row_features(self, row_data: pd.Series, detected_entities: set) -> np.ndarray:  
        """Extrait les features d'une ligne spécifique"""  
        features = []  
          
        # Compter les champs sensibles dans la ligne  
        sensitive_count = 0  
        total_fields = len(row_data)  
          
        for value in row_data.dropna():  
            value_str = str(value)  
            for entity in detected_entities:  
                if entity in value_str:  
                    sensitive_count += 1  
                    break  
          
        features.extend([  
            sensitive_count / total_fields if total_fields > 0 else 0,  # Ratio de champs sensibles  
            len(row_data.dropna()) / total_fields if total_fields > 0 else 0,  # Taux de remplissage  
            sensitive_count,  # Nombre absolu de champs sensibles  
        ])  
          
        # Features par type d'entité  
        for entity in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'ID_MAROC', 'CREDIT_CARD']:  
            has_entity = any(entity in str(value) for value in row_data.dropna() if entity in detected_entities)  
            features.append(1.0 if has_entity else 0.0)  
          
        return np.array(features)  
      
    def _analyze_columns_with_ml(self, csv_data: list, headers: list, detected_entities: set) -> Dict[str, Any]:  
        """Analyse les colonnes avec KMeans et PCA"""  
        if not csv_data or not headers:  
            return {}  
          
        # Convertir en DataFrame pour faciliter l'analyse  
        df = pd.DataFrame(csv_data)  
        df.columns = headers[:len(df.columns)]  
          
        # Extraire les features pour chaque colonne  
        column_features = []  
        column_names = []  
          
        for col in df.columns:  
            features = self._extract_column_features(df[col], col, detected_entities)  
            column_features.append(features)  
            column_names.append(col)  
          
        if len(column_features) < 2:  
            return {'column_names': column_names, 'clusters': [0] * len(column_names)}  
          
        column_features = np.array(column_features)  
          
        # Normaliser les features  
        features_scaled = self.scaler.fit_transform(column_features)  
          
        # Appliquer KMeans (nombre de clusters adaptatif)  
        n_clusters = min(3, max(2, len(column_names) // 2))  
        self.column_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  
        column_clusters = self.column_kmeans.fit_predict(features_scaled)  
          
        # Appliquer PCA pour visualisation  
        n_components = min(2, features_scaled.shape[1])  
        self.column_pca = PCA(n_components=n_components)  
        pca_coords = self.column_pca.fit_transform(features_scaled)  
          
        # Détecter les anomalies  
        anomaly_scores = []  
        center = np.mean(pca_coords, axis=0)  
        for coord in pca_coords:  
            distance = np.linalg.norm(coord - center)  
            anomaly_scores.append(distance)  
          
        return {  
            'column_names': column_names,  
            'clusters': column_clusters.tolist(),  
            'pca_coordinates': pca_coords.tolist(),  
            'anomaly_scores': anomaly_scores,  
            'cluster_characteristics': self._analyze_column_clusters(column_names, column_clusters, detected_entities)  
        }  
      
    def _analyze_rows_with_ml(self, csv_data: list, detected_entities: set) -> Dict[str, Any]:  
        """Analyse les lignes avec KMeans"""  
        if not csv_data:  
            return {}  
          
        # Convertir en DataFrame  
        df = pd.DataFrame(csv_data)  
          
        # Extraire les features pour chaque ligne  
        row_features = []  
        for idx, row in df.iterrows():  
            features = self._extract_row_features(row, detected_entities)  
            row_features.append(features)  
          
        if len(row_features) < 2:  
            return {'row_count': len(row_features), 'clusters': [0] * len(row_features)}  
          
        row_features = np.array(row_features)  
          
        # Normaliser les features  
        features_scaled = self.scaler.fit_transform(row_features)  
          
        # Appliquer KMeans  
        n_clusters = min(3, max(2, len(row_features) // 10))  # Moins de clusters pour les lignes  
        self.row_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  
        row_clusters = self.row_kmeans.fit_predict(features_scaled)  
          
        return {  
            'row_count': len(row_features),  
            'clusters': row_clusters.tolist(),  
            'cluster_characteristics': self._analyze_row_clusters(row_clusters, row_features)  
        }  
      
    def _analyze_column_clusters(self, column_names: list, clusters: np.ndarray, detected_entities: set) -> Dict[int, Dict]:  
        """Analyse les caractéristiques des clusters de colonnes"""  
        cluster_info = {}  
          
        for cluster_id in set(clusters):  
            cluster_columns = [column_names[i] for i, c in enumerate(clusters) if c == cluster_id]  
                
              # Analyser les types d'entités communes  
            common_entities = []  
            for entity in detected_entities:  
                entity_columns = [col for col in cluster_columns   
                                if any(word in col.lower() for word in self._get_entity_keywords(entity))]  
                if len(entity_columns) > 0:  
                    common_entities.append(entity)  
              
            # Déterminer le type de cluster  
            cluster_type = self._determine_cluster_type(cluster_columns, common_entities)  
              
            cluster_info[cluster_id] = {  
                'columns': cluster_columns,  
                'size': len(cluster_columns),  
                'common_entities': common_entities,  
                'cluster_type': cluster_type,  
                'risk_level': self._calculate_cluster_risk_level(common_entities)  
            }  
          
        return cluster_info  
      
    def _analyze_row_clusters(self, clusters: np.ndarray, row_features: np.ndarray) -> Dict[int, Dict]:  
        """Analyse les caractéristiques des clusters de lignes"""  
        cluster_info = {}  
          
        for cluster_id in set(clusters):  
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]  
            cluster_features = row_features[cluster_indices]  
              
            # Calculer les statistiques du cluster  
            avg_sensitive_ratio = np.mean(cluster_features[:, 0])  # Ratio de champs sensibles  
            avg_completeness = np.mean(cluster_features[:, 1])     # Taux de remplissage  
            avg_sensitive_count = np.mean(cluster_features[:, 2])  # Nombre de champs sensibles  
              
            cluster_info[cluster_id] = {  
                'size': len(cluster_indices),  
                'avg_sensitive_ratio': avg_sensitive_ratio,  
                'avg_completeness': avg_completeness,  
                'avg_sensitive_count': avg_sensitive_count,  
                'risk_level': 'HIGH' if avg_sensitive_ratio > 0.5 else 'MEDIUM' if avg_sensitive_ratio > 0.2 else 'LOW'  
            }  
          
        return cluster_info  
      
    def _get_entity_keywords(self, entity: str) -> List[str]:  
        """Retourne les mots-clés associés à un type d'entité"""  
        keywords_map = {  
            'EMAIL_ADDRESS': ['email', 'mail', 'e-mail', 'courriel'],  
            'PHONE_NUMBER': ['phone', 'tel', 'telephone', 'mobile', 'gsm'],  
            'PERSON': ['name', 'nom', 'prenom', 'firstname', 'lastname'],  
            'ID_MAROC': ['cin', 'id', 'identifier', 'carte'],  
            'CREDIT_CARD': ['card', 'carte', 'credit', 'visa', 'mastercard'],  
            'IBAN_CODE': ['iban', 'rib', 'bank', 'banque', 'compte'],  
            'LOCATION': ['address', 'adresse', 'location', 'ville', 'city']  
        }  
        return keywords_map.get(entity, [])  
      
    def _determine_cluster_type(self, columns: List[str], entities: List[str]) -> str:  
        """Détermine le type d'un cluster de colonnes"""  
        if any(entity in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'] for entity in entities):  
            return 'IDENTIFICATION'  
        elif any(entity in ['CREDIT_CARD', 'IBAN_CODE'] for entity in entities):  
            return 'FINANCIAL'  
        elif 'LOCATION' in entities:  
            return 'GEOGRAPHICAL'  
        else:  
            return 'METADATA'  
      
    def _calculate_cluster_risk_level(self, entities: List[str]) -> str:  
        """Calcule le niveau de risque d'un cluster"""  
        high_risk_entities = ['CREDIT_CARD', 'IBAN_CODE', 'ID_MAROC']  
        medium_risk_entities = ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'PERSON']  
          
        if any(entity in high_risk_entities for entity in entities):  
            return 'HIGH'  
        elif any(entity in medium_risk_entities for entity in entities):  
            return 'MEDIUM'  
        else:  
            return 'LOW'  
      
    async def _generate_column_based_recommendations(self, column_analysis: Dict[str, Any]) -> List[RecommendationItem]:  
        """Génère des recommandations basées sur l'analyse des colonnes"""  
        recommendations = []  
          
        cluster_characteristics = column_analysis.get('cluster_characteristics', {})  
        column_names = column_analysis.get('column_names', [])  
        clusters = column_analysis.get('clusters', [])  
        anomaly_scores = column_analysis.get('anomaly_scores', [])  
          
        for cluster_id, characteristics in cluster_characteristics.items():  
            cluster_type = characteristics.get('cluster_type', 'UNKNOWN')  
            risk_level = characteristics.get('risk_level', 'LOW')  
            columns = characteristics.get('columns', [])  
              
            # Recommandations spécifiques par type de cluster  
            if cluster_type == 'IDENTIFICATION':  
                rec = RecommendationItem(  
                    id=f"column_cluster_{cluster_id}_identification",  
                    title=f"Anonymisation recommandée pour les colonnes d'identification",  
                    description=f"Les colonnes {', '.join(columns)} contiennent des données d'identification. Appliquez un hachage SHA-256 ou un masquage.",  
                    category=RecommendationType.COLUMN_BASED.value,  
                    priority=8.0 if risk_level == 'HIGH' else 6.0,  
                    confidence=0.85,  
                    metadata={  
                        'cluster_id': cluster_id,  
                        'cluster_type': cluster_type,  
                        'affected_columns': columns,  
                        'recommended_method': 'hashing'  
                    },  
                    created_at=datetime.now()  
                )  
                recommendations.append(rec)  
              
            elif cluster_type == 'FINANCIAL':  
                rec = RecommendationItem(  
                    id=f"column_cluster_{cluster_id}_financial",  
                    title=f"Chiffrement requis pour les données financières",  
                    description=f"Les colonnes {', '.join(columns)} contiennent des données financières sensibles. Utilisez un chiffrement AES-256.",  
                    category=RecommendationType.COLUMN_BASED.value,  
                    priority=9.0,  
                    confidence=0.95,  
                    metadata={  
                        'cluster_id': cluster_id,  
                        'cluster_type': cluster_type,  
                        'affected_columns': columns,  
                        'recommended_method': 'encryption'  
                    },  
                    created_at=datetime.now()  
                )  
                recommendations.append(rec)  
          
        # Recommandations pour les colonnes anomales  
        for i, (column_name, anomaly_score) in enumerate(zip(column_names, anomaly_scores)):  
            if anomaly_score > 1.5:  # Seuil d'anomalie  
                rec = RecommendationItem(  
                    id=f"column_anomaly_{i}",  
                    title=f"Colonne atypique détectée: {column_name}",  
                    description=f"La colonne '{column_name}' présente un profil inhabituel (score: {anomaly_score:.2f}). Vérifiez son contenu.",  
                    category=RecommendationType.ANOMALY_DETECTION.value,  
                    priority=7.0,  
                    confidence=0.75,  
                    metadata={  
                        'column_name': column_name,  
                        'anomaly_score': anomaly_score,  
                        'threshold': 1.5  
                    },  
                    created_at=datetime.now()  
                )  
                recommendations.append(rec)  
          
        return recommendations  
      
    async def _generate_row_based_recommendations(self, row_analysis: Dict[str, Any]) -> List[RecommendationItem]:  
        """Génère des recommandations basées sur l'analyse des lignes"""  
        recommendations = []  
          
        cluster_characteristics = row_analysis.get('cluster_characteristics', {})  
          
        for cluster_id, characteristics in cluster_characteristics.items():  
            risk_level = characteristics.get('risk_level', 'LOW')  
            size = characteristics.get('size', 0)  
            avg_sensitive_ratio = characteristics.get('avg_sensitive_ratio', 0)  
              
            if risk_level == 'HIGH':  
                rec = RecommendationItem(  
                    id=f"row_cluster_{cluster_id}_high_risk",  
                    title=f"Lignes à haut risque identifiées",  
                    description=f"{size} lignes contiennent un taux élevé de données sensibles ({avg_sensitive_ratio:.1%}). Anonymisation prioritaire recommandée.",  
                    category=RecommendationType.ROW_BASED.value,  
                    priority=8.5,  
                    confidence=0.80,  
                    metadata={  
                        'cluster_id': cluster_id,  
                        'risk_level': risk_level,  
                        'affected_rows': size,  
                        'sensitive_ratio': avg_sensitive_ratio  
                    },  
                    created_at=datetime.now()  
                )  
                recommendations.append(rec)  
          
        return recommendations  
      
    
      
    def create_dataset_profile_from_presidio(self, job_id: str, detected_entities: set, headers: list, csv_data: list) -> dict:  
        """Crée un profil de dataset enrichi à partir des données Presidio"""  
        entity_distribution = {}  
        for entity in detected_entities:  
            entity_distribution[entity] = sum(1 for row in csv_data for value in row.values()   
                                            if isinstance(value, str) and entity in str(value))  
          
        # Calculer la distribution de sensibilité  
        sensitivity_distribution = self._calculate_sensitivity_distribution(detected_entities, entity_distribution)  
          
        # Générer des tags sémantiques  
        semantic_tags = self._generate_semantic_tags(detected_entities, headers)  
          
        return {  
            'dataset_id': job_id,  
            'name': f'Dataset_{job_id}',  
            'entity_distribution': entity_distribution,  
            'detected_entities': list(detected_entities),  
            'headers': headers,  
            'csv_data': csv_data,  # Ajouter les données pour l'analyse ML  
            'total_rows': len(csv_data),  
            'quality_score': self._calculate_quality_score(csv_data),  
            'has_personal_data': bool(detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}),  
            'rgpd_compliance_score': self._calculate_rgpd_score(detected_entities),  
            'sensitivity_distribution': sensitivity_distribution,  
            'semantic_tags': semantic_tags,  
            'has_anonymization': False,  
            'has_consent_management': False,  
            'compliance_gaps': self._identify_initial_compliance_gaps(detected_entities)  
        }  
      
    async def generate_comprehensive_recommendations(self, dataset_profile: dict) -> DatasetRecommendation:  
        """Génère des recommandations complètes avec analyse ML intra-fichier"""  
        dataset_id = dataset_profile.get('dataset_id', 'unknown')  
        recommendations = []  
          
        # Extraire les données nécessaires  
        csv_data = dataset_profile.get('csv_data', [])  
        headers = dataset_profile.get('headers', [])  
        detected_entities = set(dataset_profile.get('detected_entities', []))  
          
        # 1. Analyse des colonnes avec ML  
        column_analysis = self._analyze_columns_with_ml(csv_data, headers, detected_entities)  
          
        # 2. Analyse des lignes avec ML  
        row_analysis = self._analyze_rows_with_ml(csv_data, detected_entities)  
          
        # 3. Générer des recommandations basées sur les colonnes  
        column_recs = await self._generate_column_based_recommendations(column_analysis)  
        recommendations.extend(column_recs)  
          
        # 4. Générer des recommandations basées sur les lignes  
        row_recs = await self._generate_row_based_recommendations(row_analysis)  
        recommendations.extend(row_recs)  
          
        # 5. Analyses traditionnelles (qualité, sécurité, conformité, métadonnées)  
        quality_recs = await self._generate_quality_recommendations(dataset_profile)  
        recommendations.extend(quality_recs)  
          
        #security_recs = await self._generate_security_recommendations(dataset_profile)  
        #recommendations.extend(security_recs)  
          
        #compliance_recs = await self._generate_compliance_recommendations(dataset_profile)  
        #recommendations.extend(compliance_recs)  
          
        #metadata_recs = await self._generate_metadata_recommendations(dataset_profile)  
        #recommendations.extend(metadata_recs)  
          
        # 6. Calcul du score global et identification des domaines d'amélioration  
        overall_score = self._calculate_overall_score(dataset_profile, recommendations)  
        improvement_areas = self._identify_improvement_areas(recommendations)  
        compliance_gaps = self._identify_compliance_gaps(dataset_profile)  
          
        # 7. Sauvegarder les recommandations avec informations ML  
        await self._save_recommendations(dataset_id, recommendations)  
          
        return DatasetRecommendation(  
            dataset_id=dataset_id,  
            recommendations=recommendations,  
            overall_score=overall_score,  
            improvement_areas=improvement_areas,  
            compliance_gaps=compliance_gaps,  
            column_clusters=column_analysis,  
            row_clusters=row_analysis,  
            pca_analysis={  
                'column_pca_coordinates': column_analysis.get('pca_coordinates', []),  
                'explained_variance': self.column_pca.explained_variance_ratio_.tolist() if self.column_pca else []  
            }  
        )  
      

      

      
    def _calculate_sensitivity_distribution(self, detected_entities: set, entity_distribution: dict) -> dict:  
        """Calcule la distribution de sensibilité des données"""  
        sensitivity_map = {  
            'PERSON': 'PERSONAL_DATA',  
            'EMAIL_ADDRESS': 'PERSONAL_DATA',  
            'PHONE_NUMBER': 'PERSONAL_DATA',  
            'ID_MAROC': 'PERSONAL_DATA',  
            'CREDIT_CARD': 'CONFIDENTIAL',  
            'IBAN_CODE': 'CONFIDENTIAL',  
            'LOCATION': 'INTERNAL',  
            'DATE_TIME': 'INTERNAL',  
            'URL': 'PUBLIC',  
            'IP_ADDRESS': 'INTERNAL'  
        }  
          
        distribution = {'PERSONAL_DATA': 0, 'CONFIDENTIAL': 0, 'INTERNAL': 0, 'PUBLIC': 0}  
          
        for entity, count in entity_distribution.items():  
            sensitivity = sensitivity_map.get(entity, 'INTERNAL')  
            distribution[sensitivity] += count  
          
        return distribution  
      
    def _generate_semantic_tags(self, detected_entities: set, headers: list) -> List[str]:  
        """Génère des tags sémantiques basés sur les entités et headers"""  
        tags = []  
          
        # Tags basés sur les entités  
        if detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}:  
            tags.append('PII')  
        if detected_entities & {'CREDIT_CARD', 'IBAN_CODE'}:  
            tags.append('FINANCIAL')  
        if 'LOCATION' in detected_entities:  
            tags.append('GEOLOCATION')  
          
        # Tags basés sur les headers (analyse simple)  
        header_text = ' '.join(headers).lower()  
        if any(word in header_text for word in ['client', 'customer', 'user']):  
            tags.append('CLIENT_DATA')  
        if any(word in header_text for word in ['contact', 'phone', 'email']):  
            tags.append('CONTACT')  
        if any(word in header_text for word in ['health', 'medical', 'patient']):  
            tags.append('HEALTH')  
          
        return list(set(tags))  # Supprimer les doublons  
      
    def _calculate_quality_score(self, csv_data: list) -> float:  
        """Calcule un score de qualité basique"""  
        if not csv_data:  
            return 0.0  
          
        total_cells = sum(len(row) for row in csv_data)  
        empty_cells = sum(1 for row in csv_data for value in row.values() if not value or str(value).strip() == '')  
          
        return max(0.0, (total_cells - empty_cells) / total_cells * 10)  
      
    def _calculate_rgpd_score(self, detected_entities: set) -> float:  
        """Calcule un score RGPD basique"""  
        sensitive_entities = {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD', 'ID_MAROC'}  
        found_sensitive = len(detected_entities & sensitive_entities)  
          
        if found_sensitive == 0:  
            return 10.0  
        elif found_sensitive <= 2:  
            return 7.0  
        else:  
            return 4.0  
      
    def _identify_initial_compliance_gaps(self, detected_entities: set) -> List[str]:  
        """Identifie les lacunes de conformité initiales"""  
        gaps = []  
          
        if detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}:  
            gaps.append("Documentation des traitements requise")  
            gaps.append("Anonymisation recommandée")  
          
        if 'CREDIT_CARD' in detected_entities:  
            gaps.append("Chiffrement des données financières requis")  
          
        return gaps  
      
    # Inclure toutes les méthodes existantes du fichier original  
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
      
    # Ajouter toutes les autres méthodes du fichier original (_generate_quality_recommendations, etc.)  
    async def _generate_quality_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations de qualité"""  
        prompt = self.recommendation_templates["quality_analysis"].format(  
            dataset_name=dataset_profile.get('name', 'Dataset'),  
            entities=dataset_profile.get('entity_distribution', {}),  
            sensitivity_distribution=dataset_profile.get('sensitivity_distribution', {}),  
            quality_score=dataset_profile.get('quality_score', 0.0)  
        )  
          
        response = await self.gemini_client.generate_recommendations(prompt)  
          
        # Parser la réponse JSON  
        recommendations = []  
        try:  
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
      
    # Continuer avec toutes les autres méthodes (_generate_security_recommendations, _generate_compliance_recommendations, etc.)  
    # selon le fichier original...  
      
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
      from .models import RecommendationStorage  
      storage = RecommendationStorage()  
      storage.save_recommendations(dataset_id, recommendations)
      
    
  
class DataQualityEngine:  
    def __init__(self, gemini_client: GeminiClient):  
        self.gemini_client = gemini_client  
          
    async def analyze_data_quality(self, csv_data: list, headers: list) -> Dict[str, Any]:  
        """Analyse complète de la qualité des données"""  
        return {  
            'completeness': self._analyze_completeness(csv_data, headers),  
            'consistency': self._analyze_consistency(csv_data, headers),  
            'duplicates': self._detect_duplicates(csv_data, headers),  
            'ai_recommendations':await self._generate_ai_quality_recommendations(csv_data, headers)  
        }  
      
    def _detect_duplicates(self, csv_data: list, headers: list) -> Dict[str, Any]:  
        """Détection des doublons avec analyse par colonne"""  
        df = pd.DataFrame(csv_data)  
        df.columns = headers[:len(df.columns)]  
          
        # Doublons exacts  
        exact_duplicates = df.duplicated().sum()  
        duplicate_rows = df[df.duplicated(keep=False)].index.tolist()  
          
        # Doublons par colonne clé (ID, email, etc.)  
        key_columns = self._identify_key_columns(headers)  
        column_duplicates = {}  
          
        for col in key_columns:  
            if col in df.columns:  
                col_duplicates = df[df[col].duplicated(keep=False)]  
                column_duplicates[col] = {  
                    'count': len(col_duplicates),  
                    'rows': col_duplicates.index.tolist(),  
                    'values': col_duplicates[col].tolist()  
                }  
          
        return {  
            'exact_duplicates': exact_duplicates,  
            'duplicate_rows': duplicate_rows,  
            'column_duplicates': column_duplicates,  
            'total_issues': exact_duplicates + sum(len(v['rows']) for v in column_duplicates.values())  
        }  
      
    def _analyze_consistency(self, csv_data: list, headers: list) -> Dict[str, Any]:  
        """Analyse de cohérence des données"""  
        df = pd.DataFrame(csv_data)  
        df.columns = headers[:len(df.columns)]  
          
        consistency_issues = {}  
          
        for col in df.columns:  
            issues = []  
              
            # Vérifier les formats incohérents (dates, emails, téléphones)  
            if 'email' in col.lower():  
                invalid_emails = self._validate_email_format(df[col])  
                if invalid_emails:  
                    issues.append({'type': 'invalid_email', 'count': len(invalid_emails), 'rows': invalid_emails})  
              
            if 'phone' in col.lower() or 'tel' in col.lower():  
                invalid_phones = self._validate_phone_format(df[col])  
                if invalid_phones:  
                    issues.append({'type': 'invalid_phone', 'count': len(invalid_phones), 'rows': invalid_phones})  
              
            if 'date' in col.lower():  
                invalid_dates = self._validate_date_format(df[col])  
                if invalid_dates:  
                    issues.append({'type': 'invalid_date', 'count': len(invalid_dates), 'rows': invalid_dates})  
              
            if issues:  
                consistency_issues[col] = issues  
          
        return consistency_issues  
    

    def _identify_key_columns(self, headers):  
        """Identifie les colonnes clés potentielles"""  
        key_indicators = ['id', 'email', 'phone', 'tel', 'cin', 'passport']  
        key_columns = []  
          
        for header in headers:  
            header_lower = header.lower()  
            if any(indicator in header_lower for indicator in key_indicators):  
                key_columns.append(header)  
          
        return key_columns  


    def _validate_email_format(self, csv_data, column):  
        """Valide le format des emails"""  
        import re  
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'  
        invalid_rows = []  
          
        for idx, row in enumerate(csv_data):  
            value = row.get(column, '')  
            if value and not re.match(email_pattern, str(value)):  
                invalid_rows.append(idx)  
          
        return invalid_rows  
      
    def _validate_phone_format(self, csv_data, column):  
        """Valide le format des téléphones"""  
        import re  
        # Pattern pour téléphones marocains et internationaux  
        phone_pattern = r'^(\+212|0)[5-7][0-9]{8}$|^\+?[1-9]\d{1,14}$'  
        invalid_rows = []  
          
        for idx, row in enumerate(csv_data):  
            value = row.get(column, '')  
            if value and not re.match(phone_pattern, str(value).replace(' ', '').replace('-', '')):  
                invalid_rows.append(idx)  
          
        return invalid_rows  
      
    

    def _analyze_completeness(self, csv_data: list, headers: list) -> Dict[str, Any]:  
     """Analyse de la complétude des données"""  
     if not csv_data or not headers:  
        return {'score': 0, 'missing_by_column': {}, 'total_missing': 0}  
      
     total_cells = len(csv_data) * len(headers)  
     missing_cells = 0  
     missing_by_column = {}  
      
     for header in headers:  
        column_missing = 0  
        for row in csv_data:  
            value = row.get(header, '')  
            # Considérer comme manquant : None, chaîne vide, ou espaces uniquement  
            if value is None or str(value).strip() == '' or str(value).lower() in ['nan', 'null', 'none']:  
                column_missing += 1  
                missing_cells += 1  
          
        missing_by_column[header] = {  
            'missing_count': column_missing,  
            'total_count': len(csv_data),  
            'completeness_rate': ((len(csv_data) - column_missing) / len(csv_data)) * 100 if len(csv_data) > 0 else 0,  
            'missing_percentage': (column_missing / len(csv_data)) * 100 if len(csv_data) > 0 else 0  
        }  
      
     overall_score = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0  
      
     return {  
        'score': round(overall_score, 1),  
        'missing_by_column': missing_by_column,  
        'total_missing': missing_cells,  
        'total_cells': total_cells  
     }
      
    async def _generate_ai_quality_recommendations(self, csv_data: list, headers: list) -> List[Dict]:  
        """Génération de recommandations IA pour la qualité"""  
        prompt = f"""  
        Analysez la qualité des données suivantes et proposez des corrections spécifiques:  
          
        Colonnes: {headers}  
        Nombre de lignes: {len(csv_data)}  
          
        Répondez en format JSON avec la structure:  
        [{{  
            "column": "nom_colonne",  
            "issue_type": "completeness|consistency|duplicate",  
            "description": "description du problème",  
            "auto_fix_suggestion": "suggestion de correction",  
            "priority": 8,  
            "confidence": 0.9  
        }}]  
        """  
          
        try:  
            response = await self.gemini_client.generate_recommendations(prompt)  
            # Parser la réponse JSON  
            import json  
            json_start = response.find('[')  
            json_end = response.rfind(']') + 1  
            if json_start != -1 and json_end != -1:  
                json_str = response[json_start:json_end]  
                return json.loads(json_str)  
            return []  
        except Exception as e:  
            print(f"Erreur lors de la génération des recommandations IA: {e}")  
            return []

    def _analyze_consistency(self, csv_data: list, headers: list) -> Dict[str, Any]:  
     """Analyse de cohérence générale des données"""  
     if not csv_data or not headers:  
        return {'issues_by_column': {}, 'total_issues': 0}  
      
     consistency_issues = {}  
     total_issues = 0  
      
     for header in headers:  
        issues = []  
          
        # 1. Analyse des patterns automatiques  
        pattern_issues = self._analyze_column_patterns(csv_data, header)  
        issues.extend(pattern_issues)  
          
        # 2. Analyse statistique des valeurs aberrantes  
        outlier_issues = self._detect_statistical_outliers(csv_data, header)  
        issues.extend(outlier_issues)  
          
        # 3. Analyse de la cohérence des types de données  
        type_issues = self._analyze_data_type_consistency(csv_data, header)  
        issues.extend(type_issues)  
          
        # 4. Analyse des formats spécifiques (votre code existant)  
        specific_issues = self._analyze_specific_formats(csv_data, header)  
        issues.extend(specific_issues)  
          
        if issues:  
            consistency_issues[header] = issues  
            total_issues += sum(issue.get('count', 0) for issue in issues)  
      
     return {  
        'issues_by_column': consistency_issues,  
        'total_issues': total_issues  
     }  


    def _analyze_specific_formats(self, csv_data: list, column: str) -> list:
     """
     Analyse des formats spécifiques pour certaines colonnes (emails, téléphones, dates, CIN).
    
     Args:
        csv_data (list): Les données CSV sous forme de liste de dictionnaires
        column (str): Le nom de la colonne à analyser

     Returns:
        list: Liste des problèmes de format détectés
     """
     issues = []
     col_lower = column.lower()
     values = [(idx, str(row.get(column, '')).strip()) for idx, row in enumerate(csv_data)]

    # Vérification email
     if "email" in col_lower:
        invalid_emails = []
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for idx, value in values:
            if value and not re.match(email_pattern, value):
                invalid_emails.append(idx)
        if invalid_emails:
            issues.append({
                "type": "invalid_email",
                "count": len(invalid_emails),
                "rows": invalid_emails,
                "description": f"{len(invalid_emails)} emails invalides détectés",
                "severity": "medium"
            })

    # Vérification téléphone
     if "phone" in col_lower or "tel" in col_lower:
        invalid_phones = []
        phone_pattern = r'^(\+212|0)[5-7][0-9]{8}$|^\+?[1-9]\d{1,14}$'
        for idx, value in values:
            cleaned = value.replace(" ", "").replace("-", "")
            if cleaned and not re.match(phone_pattern, cleaned):
                invalid_phones.append(idx)
        if invalid_phones:
            issues.append({
                "type": "invalid_phone",
                "count": len(invalid_phones),
                "rows": invalid_phones,
                "description": f"{len(invalid_phones)} numéros de téléphone invalides détectés",
                "severity": "medium"
            })

    # Vérification date
     if "date" in col_lower:
        invalid_dates = []
        for idx, value in values:
            if value and not self._is_date_like(value):  # utilise ta fonction existante
                invalid_dates.append(idx)
        if invalid_dates:
            issues.append({
                "type": "invalid_date",
                "count": len(invalid_dates),
                "rows": invalid_dates,
                "description": f"{len(invalid_dates)} dates invalides détectées",
                "severity": "medium"
            })

    # Vérification CIN (si colonne contient "cin")
     if "cin" in col_lower:
        invalid_cins = []
        cin_pattern = r'^[A-Z]{1,2}[0-9]{3,6}$'  # ex: AB123456
        for idx, value in values:
            if value and not re.match(cin_pattern, value, re.IGNORECASE):
                invalid_cins.append(idx)
        if invalid_cins:
            issues.append({
                "type": "invalid_cin",
                "count": len(invalid_cins),
                "rows": invalid_cins,
                "description": f"{len(invalid_cins)} CIN invalides détectés",
                "severity": "low"
            })

     return issues

    

    def _analyze_data_type_consistency(self , csv_data: list, column: str) -> list:
     """
     Analyse la cohérence des types de données dans une colonne spécifique.
     Détecte les incohérences de types (mélange de nombres et texte, dates mal formatées, etc.)

     Args:
        csv_data (list): Les données CSV sous forme de liste de dictionnaires
        column (str): Le nom de la colonne à analyser

     Returns:
        list: Liste des problèmes de cohérence de type détectés
     """
     issues = []

    # Extraire toutes les valeurs non vides de la colonne
     values = []
     for idx, row in enumerate(csv_data):
        value = row.get(column, '')
        if value is not None and str(value).strip() != '':
            values.append({
                'value': str(value).strip(),
                'row_index': idx
            })

     if len(values) < 2:  # Pas assez de données pour l'analyse
        return issues

     # ---- Fonctions internes ----
     def is_date_like(val: str) -> bool:
        """Vérifie si la valeur ressemble à une date selon plusieurs formats."""
        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
        for fmt in date_formats:
            try:
                datetime.strptime(val, fmt)
                return True
            except ValueError:
                continue
        return False

     def is_phone_like(val: str) -> bool:
        """Vérifie si la valeur ressemble à un numéro de téléphone."""
        cleaned = re.sub(r"[\s\-\(\)]", "", val)
        return bool(re.match(r"^\+?\d{6,15}$", cleaned))

     def analyze_date_formats(values: list) -> dict:
        """Retourne les formats de date détectés dans les données."""
        date_formats = {}
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
        for v in values:
            for fmt in formats:
                try:
                    datetime.strptime(v['value'], fmt)
                    date_formats.setdefault(fmt, 0)
                    date_formats[fmt] += 1
                    break
                except ValueError:
                    continue
        return date_formats

    # 1. Classification des types de données
     type_classifications = {
        'integer': [],
        'float': [],
        'date': [],
        'email': [],
        'phone': [],
        'boolean': [],
        'text': [],
        'mixed': []
     }

     for item in values:
        value = item['value']
        row_idx = item['row_index']
        detected_types = []

        # Test integer
        if re.match(r'^-?\d+$', value):
            detected_types.append('integer')

        # Test float
        elif re.match(r'^-?\d*\.\d+$', value) or re.match(r'^-?\d+,\d+$', value):
            detected_types.append('float')

        # Test date
        elif is_date_like(value):
            detected_types.append('date')

        # Test email
        elif re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            detected_types.append('email')

        # Test téléphone
        elif is_phone_like(value):
            detected_types.append('phone')

        # Test boolean
        elif value.lower() in ['true', 'false', 'yes', 'no', 'oui', 'non', '1', '0', 'vrai', 'faux']:
            detected_types.append('boolean')

        # Sinon texte
        if not detected_types:
            detected_types.append('text')

        # Si plusieurs types détectés → mixte
        if len(detected_types) > 1:
            type_classifications['mixed'].append(row_idx)
        else:
            type_classifications[detected_types[0]].append(row_idx)

    # 2. Cohérence des types
     non_empty_types = {k: v for k, v in type_classifications.items() if v}

     if len(non_empty_types) > 2:
        majority_type = max(non_empty_types.keys(), key=lambda k: len(non_empty_types[k]))
        minority_types = {k: v for k, v in non_empty_types.items() if k != majority_type}
        total_minority_count = sum(len(v) for v in minority_types.values())

        issues.append({
            'type': 'mixed_data_types',
            'count': total_minority_count,
            'rows': [idx for indices in minority_types.values() for idx in indices],
            'description': f"Types de données incohérents. Type majoritaire: {majority_type}, minoritaires: {list(minority_types.keys())}",
            'severity': 'high' if total_minority_count > len(values) * 0.1 else 'medium'
        })

    # 3. Problèmes spécifiques
     if 'integer' in non_empty_types and 'float' in non_empty_types:
        mixed_numeric_count = len(non_empty_types['integer']) + len(non_empty_types['float'])
        if mixed_numeric_count < len(values) * 0.9:
            issues.append({
                'type': 'mixed_numeric_text',
                'count': len(values) - mixed_numeric_count,
                'rows': [idx for k, indices in non_empty_types.items()
                        if k not in ['integer', 'float'] for idx in indices],
                'description': "Mélange de données numériques et non-numériques",
                'severity': 'medium'
            })

     if 'date' in non_empty_types:
        date_formats = analyze_date_formats(values)
        if len(date_formats) > 2:
            issues.append({
                'type': 'inconsistent_date_formats',
                'count': len(non_empty_types['date']),
                'rows': non_empty_types['date'],
                'description': f"Formats de date incohérents: {list(date_formats.keys())}",
                'severity': 'medium'
            })

    # 4. Valeurs nulles déguisées
     null_like_values = ['null', 'none', 'nan', 'n/a', 'na', '#n/a', 'nil', '', ' ']
     null_like_rows = []
     for idx, row in enumerate(csv_data):
        value = str(row.get(column, '')).strip().lower()
        if value in null_like_values:
            null_like_rows.append(idx)

     if null_like_rows and len(null_like_rows) > 1:
        issues.append({
            'type': 'disguised_null_values',
            'count': len(null_like_rows),
            'rows': null_like_rows,
            'description': "Valeurs nulles déguisées détectées",
            'severity': 'low'
        })

     return issues

    
    
  
    def _analyze_column_patterns(self, csv_data: list, column: str) -> list:  
     """Analyse automatique des patterns dans une colonne"""  
     import re  
     from collections import Counter  
      
     issues = []  
     values = [str(row.get(column, '')).strip() for row in csv_data if row.get(column)]  
      
     if not values:  
        return issues  
      
     # Analyser les patterns de longueur  
     lengths = [len(v) for v in values]  
     length_counter = Counter(lengths)  
     most_common_length = length_counter.most_common(1)[0][0]  
      
     # Détecter les valeurs avec des longueurs inhabituelles  
     unusual_lengths = []  
     for idx, row in enumerate(csv_data):  
        value = str(row.get(column, '')).strip()  
        if value and abs(len(value) - most_common_length) > 3:  # Seuil configurable  
            unusual_lengths.append(idx)  
      
     if unusual_lengths and len(unusual_lengths) < len(values) * 0.1:  # Moins de 10% des valeurs  
        issues.append({  
            'type': 'unusual_length',  
            'count': len(unusual_lengths),  
            'rows': unusual_lengths,  
            'description': f"Longueur inhabituelle détectée dans {len(unusual_lengths)} valeurs (longueur attendue: ~{most_common_length})"  
        })  
      
     # Analyser les patterns de caractères  
     char_patterns = []  
     for value in values[:100]:  # Échantillon pour performance  
        pattern = re.sub(r'[a-zA-Z]', 'A', value)  
        pattern = re.sub(r'[0-9]', '9', pattern)  
        char_patterns.append(pattern)  
      
     pattern_counter = Counter(char_patterns)  
     most_common_pattern = pattern_counter.most_common(1)[0][0] if pattern_counter else None  
      
     if most_common_pattern:  
        inconsistent_patterns = []  
        for idx, row in enumerate(csv_data):  
            value = str(row.get(column, '')).strip()  
            if value:  
                current_pattern = re.sub(r'[a-zA-Z]', 'A', value)  
                current_pattern = re.sub(r'[0-9]', '9', current_pattern)  
                if current_pattern != most_common_pattern and len(pattern_counter) > 1:  
                    inconsistent_patterns.append(idx)  
          
        if inconsistent_patterns and len(inconsistent_patterns) < len(values) * 0.2:  
            issues.append({  
                'type': 'inconsistent_pattern',  
                'count': len(inconsistent_patterns),  
                'rows': inconsistent_patterns,  
                'description': f"Pattern incohérent détecté (pattern attendu: {most_common_pattern})"  
            })  
      
     return issues  
  
    def _detect_statistical_outliers(self, csv_data: list, column: str) -> list:  
     """Détection des valeurs aberrantes statistiques"""  
     import numpy as np  
      
     issues = []  
     numeric_values = []  
     numeric_indices = []  
      
     # Extraire les valeurs numériques  
     for idx, row in enumerate(csv_data):  
        value = row.get(column, '')  
        try:  
            numeric_val = float(str(value).replace(',', '.'))  
            numeric_values.append(numeric_val)  
            numeric_indices.append(idx)  
        except (ValueError, TypeError):  
            continue  
      
     if len(numeric_values) < 10:  # Pas assez de données numériques  
        return issues  
      
     # Méthode IQR pour détecter les outliers  
     q1 = np.percentile(numeric_values, 25)  
     q3 = np.percentile(numeric_values, 75)  
     iqr = q3 - q1  
     lower_bound = q1 - 1.5 * iqr  
     upper_bound = q3 + 1.5 * iqr  
      
     outlier_indices = []  
     for i, value in enumerate(numeric_values):  
        if value < lower_bound or value > upper_bound:  
            outlier_indices.append(numeric_indices[i])  
      
     if outlier_indices:  
        issues.append({  
            'type': 'statistical_outlier',  
            'count': len(outlier_indices),  
            'rows': outlier_indices,  
            'description': f"Valeurs aberrantes détectées (hors de l'intervalle [{lower_bound:.2f}, {upper_bound:.2f}])"  
        })  
      
     return issues  
    

    def _is_float(self, value: str) -> bool:  
     """Vérifie si une valeur est un nombre décimal"""  
     try:  
        float(value.replace(',', '.'))  
        return '.' in value or ',' in value  
     except ValueError:  
        return False  
  
    def _is_date_like(self, value: str) -> bool:  
     """Vérifie si une valeur ressemble à une date"""  
     import re  
     date_patterns = [  
        r'\d{4}-\d{2}-\d{2}',  
        r'\d{2}/\d{2}/\d{4}',  
        r'\d{2}-\d{2}-\d{4}'  
     ]  
     return any(re.match(pattern, value) for pattern in date_patterns)  
  
  
 
