from typing import Dict, List, Any    
from datetime import datetime    
import json    
from .models import RecommendationItem    
from .recommendation_templates import ENTERPRISE_TEMPLATES    
    
class EnterpriseRecommendationEngine:    
    """Version simplifiée pour entreprises"""    
        
    def __init__(self, gemini_client):    
        self.gemini_client = gemini_client    
        self.categories = {    
            'COMPLIANCE': {'priority_base': 9.0, 'color': 'red'},    
            'SECURITY': {'priority_base': 8.0, 'color': 'orange'},       
            'QUALITY': {'priority_base': 6.0, 'color': 'yellow'},    
            'GOVERNANCE': {'priority_base': 5.0, 'color': 'blue'}    
        }    
      
    def create_dataset_profile_from_presidio(self, job_id, detected_entities, headers, csv_data):  
        """Crée un profil de dataset à partir des données Presidio"""  
        quality_issues = self._analyze_basic_quality_issues(csv_data, headers)  
        compliance_status = self._assess_compliance(detected_entities)  
        security_risks = self._assess_security_risks(detected_entities)  
      
        return {  
            'dataset_id': job_id,  
            'name': f'Dataset_{job_id}',  
            'detected_entities': list(detected_entities),  
            'headers': headers,  
            'csv_data': csv_data,  
            'total_rows': len(csv_data),  
            'has_personal_data': bool(detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}),  
            'quality_issues': quality_issues,  # Ajout de cette clé manquante
            'compliance_status': compliance_status,  
            'security_risks': security_risks  
        }
      
    def _analyze_basic_quality_issues(self, csv_data: list, headers: list) -> dict:  
        """Analyse basique des problèmes de qualité"""  
        if not csv_data or not headers:  
            return {'missing_values': 0, 'duplicate_rows': 0, 'data_consistency': 'good'}  
          
        # Compter les valeurs manquantes  
        missing_count = 0  
        total_cells = len(csv_data) * len(headers)  
          
        for row in csv_data:  
            for header in headers:  
                value = row.get(header, '')  
                if not value or str(value).strip() == '':  
                    missing_count += 1  
          
        # Détecter les doublons simples  
        seen_rows = set()  
        duplicate_count = 0  
        for row in csv_data:  
            row_str = str(sorted(row.items()))  
            if row_str in seen_rows:  
                duplicate_count += 1  
            seen_rows.add(row_str)  
          
        return {  
            'missing_values': missing_count,  
            'missing_percentage': (missing_count / total_cells * 100) if total_cells > 0 else 0,  
            'duplicate_rows': duplicate_count,  
            'data_consistency': 'poor' if missing_count > total_cells * 0.2 else 'good'  
        }  
      
    def _assess_compliance(self, detected_entities: set) -> dict:  
        """Évalue le statut de conformité RGPD"""  
        personal_data_entities = {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'ID_MAROC'}  
        has_personal_data = bool(detected_entities & personal_data_entities)  
          
        compliance_score = 10.0  
        if has_personal_data:  
            compliance_score -= len(detected_entities & personal_data_entities) * 2.0  
          
        return {  
            'has_personal_data': has_personal_data,  
            'compliance_score': max(0.0, compliance_score),  
            'requires_consent': has_personal_data,  
            'requires_anonymization': has_personal_data  
        }  
      
    def _assess_security_risks(self, detected_entities: set) -> dict:  
        """Évalue les risques de sécurité"""  
        high_risk_entities = {'CREDIT_CARD', 'IBAN_CODE', 'ID_MAROC'}  
        medium_risk_entities = {'EMAIL_ADDRESS', 'PHONE_NUMBER'}  
          
        high_risk_count = len(detected_entities & high_risk_entities)  
        medium_risk_count = len(detected_entities & medium_risk_entities)  
          
        if high_risk_count > 0:  
            risk_level = 'HIGH'  
        elif medium_risk_count > 0:  
            risk_level = 'MEDIUM'  
        else:  
            risk_level = 'LOW'  
          
        return {  
            'risk_level': risk_level,  
            'high_risk_entities': list(detected_entities & high_risk_entities),  
            'medium_risk_entities': list(detected_entities & medium_risk_entities),  
            'encryption_recommended': high_risk_count > 0  
        }  
      
    async def generate_comprehensive_recommendations(self, dataset_profile: dict):  
        """Point d'entrée principal pour générer des recommandations complètes"""  
        return await self.generate_structured_recommendations(dataset_profile)  
      
    async def generate_structured_recommendations(self, dataset_profile):    
        """Génère des recommandations structurées par catégorie"""    
        recommendations_by_category = {}    
            
        for category in self.categories:    
            recs = await self._generate_category_recommendations(    
                dataset_profile, category    
            )    
            recommendations_by_category[category] = recs    
                
        return self._format_enterprise_output(recommendations_by_category)    
        
    async def _generate_category_recommendations(self, dataset_profile, category):    
     """Génère des recommandations pour une catégorie spécifique"""    
     template = ENTERPRISE_TEMPLATES.get(category, "")    
    
    # Toujours utiliser les recommandations par défaut si pas de template ou quota épuisé
     if not template:    
        return self._create_default_recommendations(category, dataset_profile)  
        
    # Préparer les données pour le template    
     template_data = self._prepare_template_data(dataset_profile, category)    
     prompt = template.format(**template_data)    
        
     try:  
        # Appeler Gemini    
        response = await self.gemini_client.generate_recommendations(prompt)    
        # Parser la réponse JSON    
        return self._parse_gemini_response(response, category, dataset_profile)  
     except Exception as e:  
        print(f"Erreur lors de l'appel à Gemini pour {category}: {e}")  
        # IMPORTANT : Retourner les recommandations par défaut au lieu d'une liste vide
        return self._create_default_recommendations(category, dataset_profile)  

    def _create_default_recommendations(self, category, dataset_profile):
     """Crée des recommandations par défaut si Gemini échoue"""
     print(f"Génération des recommandations par défaut pour {category}")
    
     if category == 'COMPLIANCE':
        return self._generate_compliance_recommendations(dataset_profile)
     elif category == 'SECURITY':
        return self._generate_security_recommendations(dataset_profile)
     elif category == 'QUALITY':
        return self._generate_quality_recommendations(dataset_profile)
     elif category == 'GOVERNANCE':
        return self._generate_governance_recommendations(dataset_profile)
    
    # Fallback ultime : au moins une recommandation générique
     return [RecommendationItem(
        id=f"{category.lower()}_{dataset_profile.get('dataset_id', 'unknown')}_default",
        title=f"Analyse {category} requise",
        description=f"Une analyse {category} approfondie est recommandée pour ce dataset.",
        category=category,
        priority=self.categories[category]['priority_base'],
        confidence=0.70,
        metadata={'color': self.categories[category]['color'], 'default': True},
        created_at=datetime.now()
     )]
        
    def _prepare_template_data(self, dataset_profile, category):      
        """Prépare les données pour les templates"""      
        detected_entities = dataset_profile.get('detected_entities', [])    
        quality_issues = dataset_profile.get('quality_issues', {  
            'missing_values': 0,  
            'missing_percentage': 0.0,  
            'duplicate_rows': 0,  
            'data_consistency': 'good'  
        })      
        compliance_status = dataset_profile.get('compliance_status', {})    
        security_risks = dataset_profile.get('security_risks', {})    
      
        # Vérifications de sécurité pour éviter les erreurs  
        if not isinstance(quality_issues, dict):  
            quality_issues = {  
                'missing_values': 0,  
                'missing_percentage': 0.0,  
                'duplicate_rows': 0,  
                'data_consistency': 'good'  
            }  
        if not isinstance(compliance_status, dict):  
            compliance_status = {}  
        if not isinstance(security_risks, dict):  
            security_risks = {}  
          
        if category == 'COMPLIANCE':      
            personal_entities = [e for e in detected_entities if e in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']]      
            return {      
                'dataset_name': dataset_profile.get('name', 'Dataset'),      
                'personal_entities': ', '.join(personal_entities),    
                'compliance_score': compliance_status.get('compliance_score', 5.0),    
                'compliance_status': compliance_status.get('status', 'UNKNOWN'),  
                'requires_consent': compliance_status.get('requires_consent', False)    
            }      
        elif category == 'SECURITY':      
            sensitive_entities = [e for e in detected_entities if e in ['CREDIT_CARD', 'ID_MAROC', 'IBAN_CODE']]      
            return {      
                'sensitive_entities': ', '.join(sensitive_entities),      
                'current_protection': 'Aucune protection détectée',    
                'risk_level': security_risks.get('risk_level', 'MEDIUM'),    
                'encryption_recommended': security_risks.get('encryption_recommended', False)    
            }    
        elif category == 'QUALITY':  
            return {  
                'dataset_name': dataset_profile.get('name', 'Dataset'),  
                'quality_issues': quality_issues,  # Utiliser l'objet complet
                'missing_percentage': quality_issues.get('missing_percentage', 0),  
                'duplicate_count': quality_issues.get('duplicate_rows', 0),  
                'data_consistency': quality_issues.get('data_consistency', 'good')
            }  
        elif category == 'GOVERNANCE':    
            return {    
                'dataset_name': dataset_profile.get('name', 'Dataset'),    
                'total_rows': dataset_profile.get('total_rows', 0),    
                'headers_count': len(dataset_profile.get('headers', [])),
                'metadata_status': 'Métadonnées manquantes'
            }   
            
        # Fallback par défaut      
        return {  
            'dataset_name': dataset_profile.get('name', 'Dataset'),  
            'detected_entities': detected_entities,  
            'headers': dataset_profile.get('headers', []),  
            'total_rows': dataset_profile.get('total_rows', 0),  
            'quality_issues': quality_issues,  
            'compliance_status': compliance_status.get('status', 'UNKNOWN'),  
            'security_risks': security_risks.get('risk_level', 'LOW'),  
            'missing_percentage': quality_issues.get('missing_percentage', 0),  
            'duplicate_count': quality_issues.get('duplicate_rows', 0),  
            'data_consistency': quality_issues.get('data_consistency', 'good'),  
            'headers_count': len(dataset_profile.get('headers', [])),
            'metadata_status': 'Métadonnées manquantes'
        }
        
    def _parse_gemini_response(self, response, category, dataset_profile):    
        """Parse la réponse de Gemini en objets RecommendationItem"""    
        recommendations = []    
            
        try:    
            # Extraire le JSON de la réponse    
            json_start = response.find('{')    
            json_end = response.rfind('}') + 1    
            if json_start != -1 and json_end != -1:    
                json_str = response[json_start:json_end]    
                parsed_data = json.loads(json_str)    
                    
                # Créer des recommandations basées sur la réponse    
                if category == 'COMPLIANCE':    
                    recommendations.extend(self._create_compliance_recommendations(parsed_data, dataset_profile))    
                elif category == 'SECURITY':    
                    recommendations.extend(self._create_security_recommendations(parsed_data, dataset_profile))  
                elif category == 'QUALITY':  
                    recommendations.extend(self._create_quality_recommendations(parsed_data, dataset_profile))  
                elif category == 'GOVERNANCE':  
                    recommendations.extend(self._create_governance_recommendations(parsed_data, dataset_profile))  
                        
        except Exception as e:    
            print(f"Erreur parsing Gemini pour {category}: {e}")    
            # Fallback vers des recommandations par défaut    
            recommendations = self._create_default_recommendations(category, dataset_profile)    
            
        return recommendations    

    def _create_default_recommendations(self, category, dataset_profile):
        """Crée des recommandations par défaut si Gemini échoue"""
        if category == 'COMPLIANCE':
            return self._generate_compliance_recommendations(dataset_profile)
        elif category == 'SECURITY':
            return self._generate_security_recommendations(dataset_profile)
        elif category == 'QUALITY':
            return self._generate_quality_recommendations(dataset_profile)
        elif category == 'GOVERNANCE':
            return self._generate_governance_recommendations(dataset_profile)
        return []

    def _format_enterprise_output(self, recommendations_by_category):
        """Formate la sortie pour l'entreprise"""
        all_recommendations = []
        for category_recs in recommendations_by_category.values():
            all_recommendations.extend(category_recs)
        
        # Calculer score global
        if all_recommendations:
            avg_priority = sum(rec.priority for rec in all_recommendations) / len(all_recommendations)
            overall_score = max(0, min(10, 10 - (avg_priority - 5)))
        else:
            overall_score = 5.0
        
        # Summary par catégorie
        category_summary = {}
        for category, recs in recommendations_by_category.items():
            category_summary[category] = {
                'count': len(recs),
                'avg_priority': sum(rec.priority for rec in recs) / len(recs) if recs else 0,
                'color': self.categories[category]['color']
            }
        
        return {
            'recommendations': all_recommendations,
            'recommendations_by_category': recommendations_by_category,
            'category_summary': category_summary,
            'overall_score': overall_score,
            'total_count': len(all_recommendations)
        }
        
    def _create_compliance_recommendations(self, parsed_data, dataset_profile):    
        """Crée des recommandations de conformité"""    
        recommendations = []    
        compliance_status = dataset_profile.get('compliance_status', {})  
          
        compliance_score = parsed_data.get('compliance_score', compliance_status.get('compliance_score', 5.0))  
        critical_gaps = parsed_data.get('critical_gaps', [])    
            
        if compliance_score < 7.0:    
            rec = RecommendationItem(    
                id=f"compliance_{dataset_profile.get('dataset_id', 'unknown')}_critical",    
                title="Conformité RGPD critique",    
                description=f"Score de conformité faible ({compliance_score}/10). Actions immédiates requises.",    
                category="COMPLIANCE",    
                priority=self.categories['COMPLIANCE']['priority_base'],    
                confidence=0.90,    
                metadata={    
                    'compliance_score': compliance_score,    
                    'critical_gaps': critical_gaps,    
                    'color': self.categories['COMPLIANCE']['color']    
                },    
                created_at=datetime.now()    
            )    
            recommendations.append(rec)  
          
        if compliance_status.get('requires_consent', False):  
            rec = RecommendationItem(  
                id=f"compliance_{dataset_profile.get('dataset_id', 'unknown')}_consent",  
                title="Gestion des consentements requise",  
                description="Le dataset contient des données personnelles nécessitant une gestion des consentements.",  
                category="COMPLIANCE",  
                priority=self.categories['COMPLIANCE']['priority_base'] - 1.0,  
                confidence=0.85,  
                metadata={'color': self.categories['COMPLIANCE']['color']},  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
            
        return recommendations    
        
    def _create_security_recommendations(self, parsed_data, dataset_profile):    
        """Crée des recommandations de sécurité"""    
        recommendations = []    
        security_risks = dataset_profile.get('security_risks', {})  
          
        risk_level = parsed_data.get('risk_level', security_risks.get('risk_level', 'MEDIUM'))  
        encryption_needs = parsed_data.get('encryption_needs', [])    
            
        if risk_level == 'HIGH':    
            rec = RecommendationItem(    
                id=f"security_{dataset_profile.get('dataset_id', 'unknown')}_high_risk",    
                title="Risque de sécurité élevé",    
                description="Données sensibles détectées sans protection adéquate. Chiffrement recommandé.",    
                category="SECURITY",    
                priority=self.categories['SECURITY']['priority_base'],    
                confidence=0.85,    
                metadata={    
                    'risk_level': risk_level,    
                    'encryption_needs': encryption_needs,    
                    'color': self.categories['SECURITY']['color']    
                },    
                created_at=datetime.now()    
            )    
            recommendations.append(rec)  
          
        if security_risks.get('encryption_recommended', False):  
            rec = RecommendationItem(  
                id=f"security_{dataset_profile.get('dataset_id', 'unknown')}_encryption",  
                title="Chiffrement des données sensibles",  
                description="Chiffrement AES-256 recommandé pour les données financières détectées.",  
                category="SECURITY",  
                priority=self.categories['SECURITY']['priority_base'] - 0.5,  
                confidence=0.90,  
                metadata={'color': self.categories['SECURITY']['color']},  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
            
        return recommendations  
      
    def _generate_quality_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations de qualité basées sur l'analyse des données"""  
        recommendations = []  
        quality_issues = dataset_profile.get('quality_issues', {})  
          
        # Recommandation pour les valeurs manquantes  
        missing_percentage = quality_issues.get('missing_percentage', 0)  
        if missing_percentage > 10:  
            rec = RecommendationItem(  
                id=f"quality_{dataset_profile.get('dataset_id', 'unknown')}_missing",  
                title="Valeurs manquantes détectées",  
                description=f"{missing_percentage:.1f}% de valeurs manquantes détectées. Un nettoyage des données est recommandé pour améliorer la qualité.",  
                category="QUALITY",  
                priority=self.categories['QUALITY']['priority_base'] + (missing_percentage / 10),  
                confidence=0.85,  
                metadata={  
                    'color': self.categories['QUALITY']['color'],  
                    'missing_percentage': missing_percentage,  
                    'action_type': 'data_cleaning'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Recommandation pour les doublons  
        duplicate_count = quality_issues.get('duplicate_rows', 0)  
        if duplicate_count > 0:  
            rec = RecommendationItem(  
                id=f"quality_{dataset_profile.get('dataset_id', 'unknown')}_duplicates",  
                title="Lignes dupliquées identifiées",  
                description=f"{duplicate_count} lignes dupliquées trouvées. Suppression recommandée pour optimiser le stockage.",  
                category="QUALITY",  
                priority=self.categories['QUALITY']['priority_base'],  
                confidence=0.90,  
                metadata={  
                    'color': self.categories['QUALITY']['color'],  
                    'duplicate_count': duplicate_count,  
                    'action_type': 'deduplication'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        return recommendations  
  
    def _generate_security_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations de sécurité"""  
        recommendations = []  
        detected_entities = set(dataset_profile.get('detected_entities', []))  
          
        # Recommandation pour les données personnelles  
        if detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}:  
            rec = RecommendationItem(  
                id=f"security_{dataset_profile.get('dataset_id', 'unknown')}_pii",  
                title="Données personnelles détectées",  
                description="Des informations personnelles identifiables ont été trouvées. Chiffrement et contrôles d'accès recommandés.",  
                category="SECURITY",  
                priority=self.categories['SECURITY']['priority_base'] + 1.0,  
                confidence=0.95,  
                metadata={  
                    'color': self.categories['SECURITY']['color'],  
                    'entity_types': list(detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'}),  
                    'action_type': 'encryption'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Recommandation pour les données financières  
        if detected_entities & {'CREDIT_CARD', 'IBAN_CODE'}:  
            rec = RecommendationItem(  
                id=f"security_{dataset_profile.get('dataset_id', 'unknown')}_financial",  
                title="Données financières sensibles",  
                description="Données financières critiques détectées. Chiffrement fort et audit des accès obligatoires.",  
                category="SECURITY",  
                priority=self.categories['SECURITY']['priority_base'] + 1.5,  
                confidence=0.98,  
                metadata={  
                    'color': self.categories['SECURITY']['color'],  
                    'entity_types': list(detected_entities & {'CREDIT_CARD', 'IBAN_CODE'}),  
                    'action_type': 'strong_encryption'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        return recommendations  
  
    def _generate_compliance_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations de conformité RGPD"""  
        recommendations = []  
        detected_entities = set(dataset_profile.get('detected_entities', []))  
          
        # Vérification RGPD pour données personnelles  
        if detected_entities & {'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'ID_MAROC'}:  
            rec = RecommendationItem(  
                id=f"compliance_{dataset_profile.get('dataset_id', 'unknown')}_gdpr",  
                title="Conformité RGPD requise",  
                description="Données personnelles détectées. Documentation des traitements et mise en place des droits des personnes concernées nécessaires.",  
                category="COMPLIANCE",  
                priority=self.categories['COMPLIANCE']['priority_base'],  
                confidence=0.92,  
                metadata={  
                    'color': self.categories['COMPLIANCE']['color'],  
                    'regulation': 'GDPR',  
                    'action_type': 'documentation'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Recommandation pour la minimisation des données  
        total_entities = len(detected_entities)  
        if total_entities > 5:  
            rec = RecommendationItem(  
                id=f"compliance_{dataset_profile.get('dataset_id', 'unknown')}_minimization",  
                title="Principe de minimisation des données",  
                description=f"{total_entities} types d'entités détectés. Évaluez si toutes ces données sont nécessaires selon le principe RGPD de minimisation.",  
                category="COMPLIANCE",  
                priority=self.categories['COMPLIANCE']['priority_base'] - 1.0,  
                confidence=0.75,  
                metadata={  
                    'color': self.categories['COMPLIANCE']['color'],  
                    'entity_count': total_entities,  
                    'action_type': 'data_minimization'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        return recommendations  
  
    def _generate_governance_recommendations(self, dataset_profile: dict) -> List[RecommendationItem]:  
        """Génère des recommandations de gouvernance des données"""  
        recommendations = []  
        headers = dataset_profile.get('headers', [])  
          
        # Recommandation pour la documentation des métadonnées  
        if len(headers) > 10:  
            rec = RecommendationItem(  
                id=f"governance_{dataset_profile.get('dataset_id', 'unknown')}_metadata",  
                title="Documentation des métadonnées",  
                description=f"Dataset avec {len(headers)} colonnes. Documentation détaillée des métadonnées recommandée pour faciliter la gouvernance.",  
                category="GOVERNANCE",  
                priority=self.categories['GOVERNANCE']['priority_base'],  
                confidence=0.80,  
                metadata={  
                    'color': self.categories['GOVERNANCE']['color'],  
                    'column_count': len(headers),  
                    'action_type': 'metadata_documentation'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        # Recommandation pour la classification des données  
        detected_entities = set(dataset_profile.get('detected_entities', []))  
        if detected_entities:  
            rec = RecommendationItem(  
                id=f"governance_{dataset_profile.get('dataset_id', 'unknown')}_classification",  
                title="Classification des données",  
                description="Entités sensibles détectées. Mise en place d'un système de classification des données recommandée.",  
                category="GOVERNANCE",  
                priority=self.categories['GOVERNANCE']['priority_base'] + 0.5,  
                confidence=0.85,  
                metadata={  
                    'color': self.categories['GOVERNANCE']['color'],  
                    'entity_types': list(detected_entities),  
                    'action_type': 'data_classification'  
                },  
                created_at=datetime.now()  
            )  
            recommendations.append(rec)  
          
        return recommendations

    def _create_quality_recommendations(self, parsed_data, dataset_profile):
        """Crée des recommandations de qualité à partir de la réponse Gemini"""
        recommendations = []
        quality_issues = dataset_profile.get('quality_issues', {})
        
        quality_score = parsed_data.get('quality_score', 5.0)
        data_issues = parsed_data.get('data_issues', [])
        
        if quality_score < 7.0:
            rec = RecommendationItem(
                id=f"quality_{dataset_profile.get('dataset_id', 'unknown')}_score",
                title="Qualité des données insuffisante",
                description=f"Score de qualité: {quality_score}/10. Amélioration recommandée.",
                category="QUALITY",
                priority=self.categories['QUALITY']['priority_base'] + (7 - quality_score),
                confidence=0.80,
                metadata={
                    'quality_score': quality_score,
                    'data_issues': data_issues,
                    'color': self.categories['QUALITY']['color']
                },
                created_at=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations

    def _create_governance_recommendations(self, parsed_data, dataset_profile):
        """Crée des recommandations de gouvernance à partir de la réponse Gemini"""
        recommendations = []
        
        governance_score = parsed_data.get('governance_score', 5.0)
        missing_metadata = parsed_data.get('missing_metadata', [])
        
        if governance_score < 7.0:
            rec = RecommendationItem(
                id=f"governance_{dataset_profile.get('dataset_id', 'unknown')}_score",
                title="Gouvernance des données à améliorer",
                description=f"Score de gouvernance: {governance_score}/10. Mise en place de processus recommandée.",
                category="GOVERNANCE",
                priority=self.categories['GOVERNANCE']['priority_base'],
                confidence=0.75,
                metadata={
                    'governance_score': governance_score,
                    'missing_metadata': missing_metadata,
                    'color': self.categories['GOVERNANCE']['color']
                },
                created_at=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations