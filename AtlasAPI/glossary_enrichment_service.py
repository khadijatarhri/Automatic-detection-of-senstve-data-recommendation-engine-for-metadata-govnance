from glossary_manager import GlossaryManager, GlossaryTermExtractor  
from recommendation_engine.MoteurDeRecommandationAvecDeepSeekML import IntelligentRecommendationEngine  
from typing import Dict, List  
from datetime import datetime
  
class GlossaryEnrichmentService:  
    """Service d'enrichissement automatique du glossaire"""  
      
    def __init__(self):  
        self.glossary_manager = GlossaryManager()  
        self.term_extractor = GlossaryTermExtractor(self.glossary_manager)  
      
    def run_enrichment_cycle(self) -> Dict:  
        """Exécute un cycle complet d'enrichissement du glossaire"""  
        print("Démarrage de l'enrichissement du glossaire...")  
          
        # 1. Enrichir depuis les annotations validées  
        enrichment_results = self.term_extractor.enrich_glossary_from_annotations()  
          
        # 2. Générer des recommandations d'amélioration  
        improvement_recommendations = self._generate_improvement_recommendations()  
          
        # 3. Calculer les statistiques  
        stats = self._calculate_glossary_stats()  
          
        results = {  
            'enrichment_results': enrichment_results,  
            'improvement_recommendations': improvement_recommendations,  
            'glossary_stats': stats,  
            'timestamp': datetime.now().isoformat()  
        }  
          
        print(f"Enrichissement terminé: {enrichment_results}")  
        return results  
      
    def _generate_improvement_recommendations(self) -> List[Dict]:  
        """Génère des recommandations d'amélioration du glossaire"""  
        recommendations = []  
        terms = self.glossary_manager.get_all_terms()  
          
        for term in terms:  
            # Recommandation pour les termes peu validés  
            if term.get('validation_count', 0) < 3:  
                recommendations.append({  
                    'type': 'validation_needed',  
                    'term': term['name'],  
                    'message': f"Le terme '{term['name']}' nécessite plus de validations utilisateur",  
                    'priority': 'medium'  
                })  
              
            # Recommandation pour les définitions courtes  
            if len(term.get('definition', '')) < 50:  
                recommendations.append({  
                    'type': 'definition_enhancement',  
                    'term': term['name'],  
                    'message': f"La définition du terme '{term['name']}' pourrait être enrichie",  
                    'priority': 'low'  
                })  
          
        return recommendations  
      
    def _calculate_glossary_stats(self) -> Dict:  
        """Calcule les statistiques du glossaire"""  
        terms = self.glossary_manager.get_all_terms()  
          
        total_terms = len(terms)  
        validated_terms = len([t for t in terms if t.get('validation_count', 0) > 0])  
        categories = {}  
          
        for term in terms:  
            category = term.get('category', 'Non classifié')  
            categories[category] = categories.get(category, 0) + 1  
          
        return {  
            'total_terms': total_terms,  
            'validated_terms': validated_terms,  
            'validation_rate': (validated_terms / total_terms * 100) if total_terms > 0 else 0,  
            'categories_distribution': categories  
        }