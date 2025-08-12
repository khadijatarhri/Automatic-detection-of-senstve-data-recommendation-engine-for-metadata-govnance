from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect  
from django.http import JsonResponse  
from django.views import View  
from .MoteurDeRecommandationAvecDeepSeekML import IntelligentRecommendationEngine
from .models import RecommendationStorage  
import os  
  
class RecommendationView(View):  
    def get(self, request, job_id):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        return render(request, 'recommendation_engine/recommendations.html', {  
            'job_id': job_id,  
            'recommendations': recommendations  
        })  
  
class RecommendationAPIView(View):  
    def get(self, request, job_id):  
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        data = []  
        for rec in recommendations:  
            data.append({  
                'id': rec.id,  
                'title': rec.title,  
                'description': rec.description,  
                'category': rec.category,  
                'priority': rec.priority,  
                'confidence': rec.confidence,  
                'created_at': rec.created_at.isoformat()  
            })  
          
        return JsonResponse({'recommendations': data})
    
class RecommendationView(View):  
    def get(self, request, job_id):  
        if not request.session.get("user_email"):  
            return redirect('login_form')  
          
        storage = RecommendationStorage()  
        recommendations = storage.get_recommendations(job_id)  
          
        # Calculer les scores dynamiquement  
        overall_score = self._calculate_overall_score(recommendations)  
        rgpd_score = self._calculate_rgpd_score(recommendations)  
          
        return render(request, 'recommendation_engine/recommendations.html', {  
            'job_id': job_id,  
            'recommendations': recommendations,  
            'overall_score': overall_score,  
            'rgpd_score': rgpd_score  
        })  
      
    def _calculate_overall_score(self, recommendations):  
        if not recommendations:  
            return 10.0  
          
        # Score basé sur la priorité moyenne des recommandations  
        avg_priority = sum(rec.priority for rec in recommendations) / len(recommendations)  
        return max(0.0, 10.0 - avg_priority)  
      
    def _calculate_rgpd_score(self, recommendations):  
        rgpd_recs = [rec for rec in recommendations if rec.category == "COMPLIANCE_RGPD"]  
        if not rgpd_recs:  
            return 9.0  
          
        # Score RGPD basé sur le nombre et la priorité des recommandations RGPD  
        avg_rgpd_priority = sum(rec.priority for rec in rgpd_recs) / len(rgpd_recs)  
        return max(0.0, 10.0 - avg_rgpd_priority)