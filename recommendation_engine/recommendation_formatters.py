from typing import Dict, List, Any  
  
class EnterpriseFormatter:  
    """Formate les recommandations pour l'affichage enterprise"""  
      
    def format_dashboard_view(self, recommendations_data):  
        """Format pour tableau de bord exécutif"""  
        recommendations = recommendations_data.get('recommendations', [])  
        category_summary = recommendations_data.get('category_summary', {})  
          
        # Top 5 recommandations critiques  
        critical_recs = [r for r in recommendations if r.priority >= 8.0][:5]  
          
        return {  
            'executive_summary': {  
                'total_recommendations': len(recommendations),  
                'critical_count': len(critical_recs),  
                'overall_score': recommendations_data.get('overall_score', 0),  
                'top_priorities': [  
                    {  
                        'title': rec.title,  
                        'priority': rec.priority,  
                        'category': rec.category  
                    } for rec in critical_recs  
                ]  
            },  
            'category_overview': category_summary  
        }  
      
    def format_technical_view(self, recommendations_data):  
        """Format pour équipes techniques"""  
        recommendations_by_category = recommendations_data.get('recommendations_by_category', {})  
          
        technical_format = {}  
        for category, recs in recommendations_by_category.items():  
            technical_format[category] = [  
                {  
                    'id': rec.id,  
                    'title': rec.title,  
                    'description': rec.description,  
                    'priority': rec.priority,  
                    'confidence': rec.confidence,  
                    'metadata': rec.metadata,  
                    'actions': self._extract_technical_actions(rec)  
                } for rec in recs  
            ]  
          
        return technical_format  
      
    def format_compliance_report(self, recommendations_data):  
        """Format pour rapports de conformité"""  
        compliance_recs = []  
        all_recs = recommendations_data.get('recommendations', [])  
          
        for rec in all_recs:  
            if rec.category == 'COMPLIANCE':  
                compliance_recs.append({  
                    'requirement': rec.title,  
                    'status': 'NON_CONFORME' if rec.priority >= 8.0 else 'ATTENTION',  
                    'description': rec.description,  
                    'remediation': self._extract_remediation_actions(rec),  
                    'risk_level': rec.metadata.get('regulatory_risk', 'MEDIUM')  
                })  
          
        return {  
            'compliance_status': 'NON_CONFORME' if any(r['status'] == 'NON_CONFORME' for r in compliance_recs) else 'CONFORME',  
            'requirements': compliance_recs,  
            'overall_compliance_score': recommendations_data.get('overall_score', 0)  
        }  
      
    def _extract_technical_actions(self, recommendation):  
        """Extrait les actions techniques d'une recommandation"""  
        actions = []  
          
        if recommendation.category == 'SECURITY':  
            encryption_needs = recommendation.metadata.get('encryption_needs', [])  
            if encryption_needs:  
                actions.append(f"Chiffrer les colonnes: {', '.join(encryption_needs)}")  
          
        elif recommendation.category == 'COMPLIANCE':  
            critical_gaps = recommendation.metadata.get('critical_gaps', [])  
            for gap in critical_gaps:  
                actions.append(f"Corriger: {gap}")  
          
        return actions  
      
    def _extract_remediation_actions(self, recommendation):  
        """Extrait les actions de remédiation"""  
        if recommendation.category == 'COMPLIANCE':  
            return recommendation.metadata.get('immediate_actions', [])  
        return []
