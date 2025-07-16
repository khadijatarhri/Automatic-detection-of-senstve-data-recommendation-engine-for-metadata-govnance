import re  
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer  
from presidio_analyzer.nlp_engine import NlpEngineProvider  
  
# Recognizers améliorés (copiés de votre script Colab)  
class ImprovedMoroccanIdRecognizer(PatternRecognizer):  
    """Recognizer amélioré pour les CIN marocaines"""  
  
    PATTERNS = [  
        Pattern("CIN Maroc", r"\b[A-Z]{2}[0-9]{5,6}\b", 0.9),  
        Pattern("CIN Maroc avec espaces", r"\b[A-Z]{2}\s*[0-9]{5,6}\b", 0.8),  
    ]  
  
    CONTEXT = ["cin", "carte", "identité", "cni", "numéro"]  
  
    def __init__(self):  
        super().__init__(  
            supported_entity="ID_MAROC",  
            patterns=self.PATTERNS,  
            context=self.CONTEXT,  
            supported_language="fr"  
        )  
  
class ImprovedMoroccanPhoneRecognizer(PatternRecognizer):  
    """Recognizer amélioré pour les téléphones marocains"""  
  
    PATTERNS = [  
        Pattern("Téléphone Maroc +212", r"\+212[0-9]{9}", 0.95),  
        Pattern("Téléphone Maroc 0X", r"\b0[5-7][0-9]{8}\b", 0.9),  
        Pattern("Téléphone compact", r"\b[0-9]{10}\b", 0.7),  
    ]  
  
    CONTEXT = ["téléphone", "phone", "mobile", "tel", "appel", "contact"]  
  
    def __init__(self):  
        super().__init__(  
            supported_entity="PHONE_NUMBER",  
            patterns=self.PATTERNS,  
            context=self.CONTEXT,  
            supported_language="fr"  
        )  
  
class ImprovedIBANRecognizer(PatternRecognizer):  
    """Recognizer amélioré pour les codes IBAN"""  
  
    PATTERNS = [  
        Pattern("IBAN Maroc", r"\bMA[0-9]{2}[A-Z0-9]{20}\b", 0.95),  
        Pattern("IBAN avec lettres", r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[A-Z0-9]{7}[A-Z0-9]{0,16}\b", 0.8),  
        Pattern("IBAN général", r"\b[A-Z]{2}[0-9A-Z]{22,34}\b", 0.7),  
    ]  
  
    CONTEXT = ['banque', 'iban', 'compte', 'bancaire', 'virement', 'destinataire']  
  
    def __init__(self):  
        super().__init__(  
            supported_entity="IBAN_CODE",  
            patterns=self.PATTERNS,  
            context=self.CONTEXT,  
            supported_language="fr"  
        )  
  
class ImprovedEmailRecognizer(PatternRecognizer):  
    """Recognizer amélioré pour les adresses email"""  
  
    PATTERNS = [  
        Pattern("Email standard", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.9),  
        Pattern("Email avec sous-domaines", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.85),  
        Pattern("Email domaines marocains", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*\.ma\b", 0.95),  
    ]  
  
    CONTEXT = ['email', 'mail', 'courriel', '@', 'adresse', 'contact']  
  
    def __init__(self):  
        super().__init__(  
            supported_entity="EMAIL_ADDRESS",  
            patterns=self.PATTERNS,  
            context=self.CONTEXT,  
            supported_language="fr"  
        )  
  
def create_enhanced_analyzer_engine(model_path):  
    """  
    Analyzer engine amélioré avec meilleure configuration  
    """  
    configuration = {  
        "nlp_engine_name": "spacy",  
        "models": [  
            {  
                "lang_code": "fr",  
                "model_name": model_path,  
            }  
        ],  
    }  
  
    provider = NlpEngineProvider(nlp_configuration=configuration)  
    nlp_engine = provider.create_engine()  
  
    # Recognizers améliorés  
    moroccan_id_recognizer = ImprovedMoroccanIdRecognizer()  
    moroccan_phone_recognizer = ImprovedMoroccanPhoneRecognizer()  
    email_recognizer = ImprovedEmailRecognizer()  
    iban_recognizer = ImprovedIBANRecognizer()  
  
    analyzer = AnalyzerEngine(  
        nlp_engine=nlp_engine,  
        supported_languages=["fr"]  
    )  
  
    # Nettoyer les recognizers par défaut  
    default_recognizers = [  
        "CreditCardRecognizer", "CryptoRecognizer", "DateRecognizer",  
        "EmailRecognizer", "IbanRecognizer", "IpRecognizer",  
        "PhoneRecognizer", "UrlRecognizer"  
    ]  
  
    for recognizer in default_recognizers:  
        try:  
            analyzer.registry.remove_recognizer(recognizer)  
        except:  
            pass  
  
    # Ajouter nos recognizers  
    analyzer.registry.add_recognizer(moroccan_id_recognizer)  
    analyzer.registry.add_recognizer(moroccan_phone_recognizer)  
    analyzer.registry.add_recognizer(email_recognizer)  
    analyzer.registry.add_recognizer(iban_recognizer)  
  
    return analyzer