# =============================================================================  
# MOTEUR DE RECOMMANDATION AVEC DEEPSEEK ML + KMEANS + PCA (ANALYSE INTRA-FICHIER)  
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
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns  
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
        """Génère des recommandations via l'API DeepSeek"""  
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
# MOTEUR DE RECOMMANDATION INTELLIGENT AVEC ML INTRA-FICHIER  
# =============================================================================  
  
class IntelligentRecommendationEngine:  
    """Moteur de recommandation basé sur DeepSeek ML, KMeans et PCA pour analyse intra-fichier"""  
      
    def __init__(self, deepseek_client: DeepSeekClient, database_path: str = "recommendations.db"):  
        self.deepseek_client = deepseek_client  
        self.database_path = database_path  
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  
        self.scaler = StandardScaler()  
        self.column_kmeans = None  
        self.row_kmeans = None  
        self.column_pca = None  
        self.row_pca = None  
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
            CREATE TABLE IF NOT EXISTS column_analysis (  
                dataset_id TEXT,  
                column_name TEXT,  
                cluster_id INTEGER,  
                sensitivity_score REAL,  
                entity_types TEXT,  
                anomaly_score REAL,  
                pca_coordinates TEXT,  
                PRIMARY KEY (dataset_id, column_name)  
            )  
        ''')  
          
        cursor.execute('''  
            CREATE TABLE IF NOT EXISTS row_analysis (  
                dataset_id TEXT,  
                row_index INTEGER,  
                cluster_id INTEGER,  
                risk_score REAL,  
                sensitive_fields_count INTEGER,  
                PRIMARY KEY (dataset_id, row_index)  
            )  
        ''')  
          
        conn.commit()  
        conn.close()  
      
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
      
    def visualize_column_clusters(self, column_analysis: Dict[str, Any], output_path: str = "column_clusters.png"):  
        """Visualise les clusters de colonnes avec PCA"""  
        if not column_analysis or 'pca_coordinates' not in column_analysis:  
            print("Pas de données PCA disponibles pour la visualisation")  
            return  
          
        pca_coords = np.array(column_analysis['pca_coordinates'])  
        clusters = column_analysis['clusters']  
        column_names = column_analysis['column_names']  
          
        plt.figure(figsize=(12, 8))  
          
        # Scatter plot des coordonnées PCA  
        scatter = plt.scatter(  
            pca_coords[:, 0],   
            pca_coords[:, 1] if pca_coords.shape[1] > 1 else np.zeros(len(pca_coords)),   
            c=clusters,   
            cmap='viridis',   
            alpha=0.7,  
            s=100  
        )  
          
        plt.colorbar(scatter, label='Cluster ID')  
        plt.xlabel('PC1')  
        plt.ylabel('PC2' if pca_coords.shape[1] > 1 else 'Constant')  
        plt.title('Clustering des Colonnes par Profil de Sensibilité')  
          
        # Ajouter les labels des colonnes  
        for i, col_name in enumerate(column_names):  
            plt.annotate(  
                col_name,   
                (pca_coords[i, 0], pca_coords[i, 1] if pca_coords.shape[1] > 1 else 0),  
                xytext=(5, 5),   
                textcoords='offset points',  
                fontsize=8,  
                alpha=0.7  
            )  
          
        plt.tight_layout()  
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  
        plt.close()  
          
        print(f"Visualisation des colonnes sauvegardée: {output_path}")  
      
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
          
        security_recs = await self._generate_security_recommendations(dataset_profile)  
        recommendations.extend(security_recs)  
          
        compliance_recs = await self._generate_compliance_recommendations(dataset_profile)  
        recommendations.extend(compliance_recs)  
          
        metadata_recs = await self._generate_metadata_recommendations(dataset_profile)  
        recommendations.extend(metadata_recs)  
          
        # 6. Calcul du score global et identification des domaines d'amélioration  
        overall_score = self._calculate_overall_score(dataset_profile, recommendations)  
        improvement_areas = self._identify_improvement_areas(recommendations)  
        compliance_gaps = self._identify_compliance_gaps(dataset_profile)  
          
        # 7. Sauvegarder les recommandations avec informations ML  
        await self._save_recommendations(dataset_id, recommendations)  
        await self._save_column_analysis(dataset_id, column_analysis)  
        await self._save_row_analysis(dataset_id, row_analysis)  
          
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
      
    async def _save_column_analysis(self, dataset_id: str, column_analysis: Dict[str, Any]):  
        """Sauvegarde l'analyse des colonnes"""  
        conn = sqlite3.connect(self.database_path)  
        cursor = conn.cursor()  
          
        column_names = column_analysis.get('column_names', [])  
        clusters = column_analysis.get('clusters', [])  
        anomaly_scores = column_analysis.get('anomaly_scores', [])  
        pca_coords = column_analysis.get('pca_coordinates', [])  
          
        for i, column_name in enumerate(column_names):  
            cursor.execute('''  
                INSERT OR REPLACE INTO column_analysis   
                (dataset_id, column_name, cluster_id, sensitivity_score, entity_types, anomaly_score, pca_coordinates)  
                VALUES (?, ?, ?, ?, ?, ?, ?)  
            ''', (  
                dataset_id,  
                column_name,  
                clusters[i] if i < len(clusters) else 0,  
                0.0,  # À calculer selon vos besoins  
                json.dumps([]),  # À enrichir avec les entités détectées  
                anomaly_scores[i] if i < len(anomaly_scores) else 0.0,  
                json.dumps(pca_coords[i] if i < len(pca_coords) else [])  
            ))  
          
        conn.commit()  
        conn.close()  
      
    async def _save_row_analysis(self, dataset_id: str, row_analysis: Dict[str, Any]):  
        """Sauvegarde l'analyse des lignes"""  
        conn = sqlite3.connect(self.database_path)  
        cursor = conn.cursor()  
          
        clusters = row_analysis.get('clusters', [])  
          
        for row_index, cluster_id in enumerate(clusters):  
            cursor.execute('''  
                INSERT OR REPLACE INTO row_analysis   
                (dataset_id, row_index, cluster_id, risk_score, sensitive_fields_count)  
                VALUES (?, ?, ?, ?, ?)  
            ''', (  
                dataset_id,  
                row_index,  
                cluster_id,  
                0.0,  # À calculer selon le profil de risque  
                0     # À calculer selon le nombre de champs sensibles  
            ))  
          
        conn.commit()  
        conn.close()  
      
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
          
        response = await self.deepseek_client.generate_recommendations(prompt)  
          
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
  
# =============================================================================  
# EXEMPLE D'UTILISATION AVEC ANALYSE INTRA-FICHIER  
# =============================================================================  
  
async def example_usage_single_file():  
    """Exemple d'utilisation du moteur de recommandation pour un seul fichier"""  
      
    # Configuration DeepSeek  
    DEEPSEEK_API_KEY = "votre_clé_api_deepseek"  
      
    # Profil d'exemple d'un dataset avec données CSV  
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
        'compliance_gaps': ['Documentation insuffisante', 'Anonymisation manquante'],  
        'headers': ['nom', 'email', 'telephone', 'cin', 'adresse'],  
        'csv_data': [  
            {'nom': 'Ahmed Ben Ali', 'email': 'ahmed@email.com', 'telephone': '0612345678', 'cin': 'AB123456', 'adresse': 'Casablanca'},  
            {'nom': 'Fatima Zahra', 'email': 'fatima@email.com', 'telephone': '0687654321', 'cin': 'FZ789012', 'adresse': 'Rabat'}  
        ]  
    }  
      
    # Créer le moteur de recommandation  
    async with DeepSeekClient(DEEPSEEK_API_KEY) as client:  
        engine = IntelligentRecommendationEngine(client)  
          
        # Générer les recommandations avec analyse ML intra-fichier  
        print("🔍 Génération des recommandations avec analyse ML...")  
        recommendations = await engine.generate_comprehensive_recommendations(sample_dataset_profile)  
          
        # Afficher les résultats  
        print(f"\n📊 Recommandations pour {sample_dataset_profile['name']}")  
        print(f"Score global: {recommendations.overall_score:.1f}/10")  
        print(f"Domaines d'amélioration: {', '.join(recommendations.improvement_areas)}")  
        print(f"Lacunes de conformité: {', '.join(recommendations.compliance_gaps)}")  
          
        # Afficher l'analyse des colonnes  
        if recommendations.column_clusters:  
            print(f"\n🔍 Analyse des colonnes:")  
            column_analysis = recommendations.column_clusters  
            print(f"Colonnes analysées: {', '.join(column_analysis.get('column_names', []))}")  
            print(f"Clusters identifiés: {len(set(column_analysis.get('clusters', [])))}")  
          
        # Afficher l'analyse des lignes  
        if recommendations.row_clusters:  
            print(f"\n📋 Analyse des lignes:")  
            row_analysis = recommendations.row_clusters  
            print(f"Lignes analysées: {row_analysis.get('row_count', 0)}")  
            print(f"Clusters de risque: {len(set(row_analysis.get('clusters', [])))}")  
          
        print(f"\n📋 Recommandations détaillées ({len(recommendations.recommendations)} items):")  
        for i, rec in enumerate(recommendations.recommendations, 1):  
            print(f"\n{i}. {rec.title} (Priorité: {rec.priority}/10)")  
            print(f"   Catégorie: {rec.category}")  
            print(f"   Description: {rec.description}")  
            print(f"   Confiance: {rec.confidence:.2f}")  
          
        # Générer la visualisation des colonnes  
        if recommendations.column_clusters:  
            engine.visualize_column_clusters(recommendations.column_clusters, "column_analysis.png")  
  
# Pour tester le code  
if __name__ == "__main__":  
    asyncio.run(example_usage_single_file())