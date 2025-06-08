# moroccan_detectors.py
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
import re

class MoroccanCINRecognizer(PatternRecognizer):
    """
    Détecteur pour les numéros de Carte d'Identité Nationale marocaine
    Format: 1-2 lettres + 6-8 chiffres
    Exemples: AB123456, X1234567, CD12345678
    """
    
    PATTERNS = [
        Pattern(
            "CIN_MOROCCO",
            r"\b[A-Z]{1,2}\d{6,8}\b",
            0.8
        ),
        # Pattern plus strict avec validation
        Pattern(
            "CIN_MOROCCO_STRICT",
            r"\b[A-Z]{1,2}[0-9]{6,8}\b",
            0.9
        )
    ]
    
    CONTEXT = [
        "cin", "carte identité", "identité nationale", "cni",
        "بطاقة التعريف", "رقم البطاقة", "هوية"
    ]
    
    def __init__(self):
        super().__init__(
            supported_entity="MOROCCAN_CIN",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"  # ou "ar" pour l'arabe
        )
        
    def validate_result(self, pattern_text):
        """Validation supplémentaire du format CIN"""
        # Vérifications supplémentaires si nécessaire
        if len(pattern_text) < 7 or len(pattern_text) > 10:
            return False
        return True

class MoroccanPhoneRecognizer(PatternRecognizer):
    """
    Détecteur pour les numéros de téléphone marocains
    Formats: +212 6XX XX XX XX, 06XX-XX-XX-XX, 0661234567, etc.
    """
    
    PATTERNS = [
        # Format international
        Pattern(
            "PHONE_MOROCCO_INTL",
            r"\+212[\s-]?[5-7]\d{2}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}",
            0.9
        ),
        # Format national
        Pattern(
            "PHONE_MOROCCO_NAT",
            r"\b0[5-7]\d{2}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}\b",
            0.8
        ),
        # Format compact
        Pattern(
            "PHONE_MOROCCO_COMPACT",
            r"\b0[5-7]\d{8}\b",
            0.7
        )
    ]
    
    CONTEXT = [
        "téléphone", "mobile", "portable", "tél", "tel",
        "هاتف", "جوال", "رقم الهاتف"
    ]
    
    def __init__(self):
        super().__init__(
            supported_entity="MOROCCAN_PHONE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class MoroccanRIBRecognizer(PatternRecognizer):
    """
    Détecteur pour les RIB (Relevé d'Identité Bancaire) marocains
    Format: 24 chiffres (généralement groupés par 4)
    """
    
    PATTERNS = [
        Pattern(
            "RIB_MOROCCO",
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            0.8
        ),
        Pattern(
            "RIB_MOROCCO_COMPACT",
            r"\b\d{24}\b",
            0.7
        )
    ]
    
    CONTEXT = [
        "rib", "compte bancaire", "numéro compte", "relevé identité bancaire",
        "حساب بنكي", "رقم الحساب"
    ]
    
    def __init__(self):
        super().__init__(
            supported_entity="MOROCCAN_RIB",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )

class MoroccanAddressRecognizer(PatternRecognizer):
    """
    Détecteur pour les adresses marocaines
    """
    
    PATTERNS = [
        # Villes principales
        Pattern(
            "MOROCCO_CITY",
            r"\b(Casablanca|Rabat|Fès|Marrakech|Agadir|Tanger|Meknès|Oujda|Kénitra|Tétouan|Safi|Mohammedia|El Jadida|Beni Mellal|Nador|Taza|Settat|Berrechid|Khémisset|Inezgane|Salé|Temara|Larache|Khouribga|Guelmim|Tiznit|Essaouira)\b",
            0.8
        ),
        # Quartiers/Districts
        Pattern(
            "MOROCCO_DISTRICT",
            r"\b(Hay\s+\w+|Quartier\s+\w+|Médina|Gueliz|Hivernage|Agdal|Souissi)\b",
            0.6
        )
    ]
    
    CONTEXT = [
        "adresse", "domicile", "résidence", "habite",
        "عنوان", "سكن", "منزل"
    ]
    
    def __init__(self):
        super().__init__(
            supported_entity="MOROCCAN_ADDRESS",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr"
        )