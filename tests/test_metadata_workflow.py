from playwright.sync_api import sync_playwright  
import pytest  
import os  
import time  
  
def test_complete_metadata_workflow():  
    """Test complet du workflow métadonnées enrichies"""  
    with sync_playwright() as p:  
        # Lancer le navigateur en mode visible pour débugger  
        browser = p.chromium.launch(headless=True, slow_mo=1000)  
        page = browser.new_page()  
          
        try:  
            # 1. CONNEXION avec les identifiants par défaut  
            print("🔐 Étape 1: Connexion...")  
            page.goto("http://web:8000/login/")  # Utiliser le nom du service Docker  
            # Debug : vérifier le contenu de la page  
            print(f"Page title: {page.title()}")  
            page.screenshot(path="/tmp/login_page.png")  
  
            # Vérifier si les éléments existent  
            email_input = page.locator("input[name='email']")  
            if email_input.count() == 0:  
                 print("❌ Champ email non trouvé avec input[name='email']")  
                 # Essayer d'autres sélecteurs  
                 all_inputs = page.locator("input").all()  
                 for i, input_elem in enumerate(all_inputs):  
                     print(f"Input {i}: {input_elem.get_attribute('name')} - {input_elem.get_attribute('type')}")


            page.fill("input[name='email']", "admin@example.com")  
            page.fill("input[name='password']", "admin123")
 
            page.click("button[type='submit']")  
              
            # Attendre la redirection vers home  
            page.wait_for_url("****/csv-anonymizer/upload/", timeout=10000)  
            print("✅ Connexion réussie")  
              
            # 2. UPLOAD CSV  
            print("📁 Étape 2: Upload du fichier CSV...")  
              
            # Créer un fichier CSV de test avec des données sensibles  
            test_csv_content = "nom,email,telephone,cin\nJohn Doe,john@example.com,0612345678,AB123456\nMarie Martin,marie@test.fr,0687654321,CD789012"  
            with open("/tmp/test_data.csv", "w") as f:  
                f.write(test_csv_content)  
              
            # Upload du fichier  
            page.set_input_files("input[type='file']", "/tmp/test_data.csv")  
            page.click("button:has-text('Démarrer l\\'analyse')")  
              
            # Attendre la page de sélection d'entités  
            print("✅ Analyse terminée, entités détectées")  
              
            # 3. NAVIGATION VERS RECOMMANDATIONS  
            print("🤖 Étape 3: Accès aux recommandations...")  
            page.click("a:has-text('Voir les recommandations')")  
            page.wait_for_url("**/recommendations/**", timeout=15000)  
            print("✅ Page recommandations chargée")  
              
            # 4. TEST DU BOUTON MÉTADONNÉES  
            print("📊 Étape 4: Test du bouton métadonnées...")  
              
            # Attendre que le bouton soit visible  
            metadata_button = page.locator("a:has-text('Voir les métadonnées enrichi')")  
            metadata_button.wait_for(state="visible", timeout=10000)  
              
            if metadata_button.is_visible():  
                metadata_button.click()  
                page.wait_for_url("**/metadata/**", timeout=15000)  
                  
                # Attendre que le tableau se charge  
                page.wait_for_selector("table", timeout=10000)  
                  
                # Vérifier que le tableau n'est pas vide  
                table_rows = page.locator("tbody tr")  
                row_count = table_rows.count()  
                  
                print(f"📈 Nombre de lignes dans le tableau: {row_count}")  
                  
                if row_count > 0:  
                    print("✅ Métadonnées affichées avec succès")  
                      
                    # Vérifier les colonnes attendues  
                    assert page.is_visible("th:has-text('Type d\\'Entité')")  
                    assert page.is_visible("th:has-text('Niveau de Sensibilité')")  
                    assert page.is_visible("th:has-text('Catégorie RGPD')")  
                      
                    # Vérifier qu'il y a des données dans la première ligne  
                    first_row = table_rows.first  
                    assert first_row.is_visible()  
                      
                    print("✅ Test réussi - Métadonnées correctement affichées")  
                else:  
                    print("❌ Tableau vide - problème dans _get_enriched_metadata()")  
                    # Prendre une capture d'écran pour débugger  
                    page.screenshot(path="/tmp/empty_table_debug.png")  
                    raise AssertionError("Le tableau des métadonnées est vide")  
                      
            else:  
                print("❌ Bouton métadonnées non trouvé")  
                page.screenshot(path="/tmp/button_not_found.png")  
                raise AssertionError("Bouton 'Voir les métadonnées enrichi' non trouvé")  
                  
        except Exception as e:  
            print(f"❌ Erreur: {e}")  
            # Prendre une capture d'écran pour débugger  
            page.screenshot(path="/tmp/error_screenshot.png")  
            raise  
              
        finally:  
            # Nettoyer les fichiers temporaires  
            if os.path.exists("/tmp/test_data.csv"):  
                os.remove("/tmp/test_data.csv")  
            browser.close()  
  
if __name__ == "__main__":  
    test_complete_metadata_workflow()