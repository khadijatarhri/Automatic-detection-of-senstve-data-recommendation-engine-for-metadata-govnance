from glossary_manager import GlossaryManager  
  
def main():  
    """Initialise le glossaire de base"""  
    print("Initialisation du glossaire RGPD...")  
      
    glossary_manager = GlossaryManager()  
    glossary_manager.initialize_base_glossary()  
      
    # Afficher les termes créés  
    terms = glossary_manager.get_all_terms()  
    print(f"\nGlossaire initialisé avec {len(terms)} termes:")  
    for term in terms:  
        print(f"- {term['name']}: {term['definition']}")  
  
if __name__ == "__main__":  
    main()