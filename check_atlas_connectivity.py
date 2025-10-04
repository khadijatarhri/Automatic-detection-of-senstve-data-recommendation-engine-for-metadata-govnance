#!/usr/bin/env python3
"""
Script de diagnostic réseau pour Apache Atlas
Exécutez ce script AVANT entity_migration.py pour identifier les problèmes
"""

import socket
import requests
import subprocess
import sys
from datetime import datetime

# Configuration
ATLAS_HOST = "172.19.0.2"
ATLAS_PORT = 21000
ATLAS_URL = f"http://{ATLAS_HOST}:{ATLAS_PORT}"
ATLAS_USER = "admin"
ATLAS_PASS = "ensias123@"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_dns_resolution():
    """Test 1: Résolution DNS"""
    print_header("TEST 1: Résolution DNS")
    
    # Tester avec l'IP
    try:
        socket.inet_aton(ATLAS_HOST)
        print(f"✅ IP valide: {ATLAS_HOST}")
    except:
        print(f"❌ IP invalide: {ATLAS_HOST}")
        return False
    
    # Tester avec le hostname
    try:
        ip = socket.gethostbyname("sandbox-hdp.hortonworks.com")
        print(f"✅ Hostname résolu: sandbox-hdp.hortonworks.com -> {ip}")
        return True
    except:
        print("⚠️  Hostname sandbox-hdp.hortonworks.com non résolu")
        print("   Utilisation de l'IP directe recommandée")
        return True

def test_tcp_connection():
    """Test 2: Connexion TCP"""
    print_header("TEST 2: Connexion TCP")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ATLAS_HOST, ATLAS_PORT))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {ATLAS_PORT} OUVERT sur {ATLAS_HOST}")
            return True
        else:
            print(f"❌ Port {ATLAS_PORT} FERMÉ sur {ATLAS_HOST}")
            print("\n🔧 Solutions:")
            print("   1. Vérifiez que HDP Sandbox est démarré:")
            print("      docker ps | grep sandbox")
            print("   2. Vérifiez l'IP du conteneur:")
            print("      docker inspect sandbox-hdp | grep IPAddress")
            print("   3. Vérifiez que Atlas est démarré dans le conteneur:")
            print("      docker exec -it sandbox-hdp bash")
            print("      systemctl status atlas-metadata")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return False

def test_http_connection():
    """Test 3: Connexion HTTP"""
    print_header("TEST 3: Connexion HTTP Atlas API")
    
    try:
        print(f"Tentative connexion à: {ATLAS_URL}/api/atlas/v2/types/typedefs")
        
        response = requests.get(
            f"{ATLAS_URL}/api/atlas/v2/types/typedefs",
            auth=(ATLAS_USER, ATLAS_PASS),
            timeout=(10, 30)
        )
        
        if response.status_code == 200:
            print(f"✅ API Atlas accessible (HTTP {response.status_code})")
            types_data = response.json()
            print(f"   Types définis: {len(types_data.get('entityDefs', []))} entités")
            return True
        elif response.status_code == 401:
            print(f"❌ Authentification échouée (HTTP {response.status_code})")
            print("   Vérifiez ATLAS_USER et ATLAS_PASS")
            return False
        else:
            print(f"⚠️  API répond avec code {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Atlas ne répond pas dans les délais")
        print("\n🔧 Solutions:")
        print("   1. Atlas est peut-être en cours de démarrage")
        print("   2. Attendez 2-3 minutes après le démarrage du conteneur")
        print("   3. Vérifiez les logs Atlas:")
        print("      docker exec sandbox-hdp tail -f /var/log/atlas/application.log")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Erreur connexion HTTP: {e}")
        print("\n🔧 Le service Atlas n'est probablement pas démarré")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def test_docker_network():
    """Test 4: Configuration réseau Docker"""
    print_header("TEST 4: Configuration Réseau Docker")
    
    try:
        # Lister les réseaux Docker
        result = subprocess.run(
            ['docker', 'network', 'ls'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Réseaux Docker:")
            print(result.stdout)
        
        # Inspecter le réseau du conteneur Django
        result = subprocess.run(
            ['docker', 'inspect', '-f', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}', 'django_container'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            django_ip = result.stdout.strip()
            print(f"\n✅ IP conteneur Django: {django_ip}")
        
        # Inspecter le conteneur HDP
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=sandbox', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            containers = result.stdout.strip().split('\n')
            print(f"\n✅ Conteneurs HDP trouvés: {containers}")
            
            for container in containers:
                if container:
                    result = subprocess.run(
                        ['docker', 'inspect', '-f', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}', container],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print(f"   {container}: {result.stdout.strip()}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Impossible de vérifier Docker: {e}")
        return True

def test_atlas_health():
    """Test 5: Santé d'Atlas"""
    print_header("TEST 5: Santé du Service Atlas")
    
    try:
        # Tester l'endpoint admin
        response = requests.get(
            f"{ATLAS_URL}/api/atlas/admin/status",
            auth=(ATLAS_USER, ATLAS_PASS),
            timeout=(5, 15)
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Atlas Status: {status.get('Status', 'UNKNOWN')}")
            return True
        else:
            print(f"⚠️  Status endpoint: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Impossible de vérifier le statut: {e}")
        return False

def provide_recommendations(results):
    """Afficher les recommandations finales"""
    print_header("RECOMMANDATIONS")
    
    dns_ok, tcp_ok, http_ok, docker_ok, health_ok = results
    
    if all([tcp_ok, http_ok]):
        print("✅ TOUT FONCTIONNE!")
        print("\nVous pouvez exécuter entity_migration.py")
        print("\nCommande:")
        print("  python entity_migration.py")
        return True
    
    print("❌ PROBLÈMES DÉTECTÉS\n")
    
    if not tcp_ok:
        print("🔴 PROBLÈME CRITIQUE: Port TCP fermé")
        print("\nActions immédiates:")
        print("  1. Démarrer HDP Sandbox:")
        print("     docker start sandbox-hdp")
        print("\n  2. Vérifier que le conteneur tourne:")
        print("     docker ps | grep sandbox")
        print("\n  3. Attendre 2-3 minutes le démarrage complet")
        print("\n  4. Vérifier l'IP réelle:")
        print("     docker inspect sandbox-hdp | grep IPAddress")
        print("\n  5. Mettre à jour ATLAS_HOST dans entity_migration.py")
    
    elif not http_ok:
        print("🔴 PROBLÈME: Atlas API inaccessible")
        print("\nActions:")
        print("  1. Connectez-vous au conteneur:")
        print("     docker exec -it sandbox-hdp bash")
        print("\n  2. Vérifiez le statut Atlas:")
        print("     systemctl status atlas-metadata")
        print("\n  3. Si arrêté, démarrez Atlas:")
        print("     systemctl start atlas-metadata")
        print("\n  4. Surveillez les logs:")
        print("     tail -f /var/log/atlas/application.log")
        print("\n  5. Atlas peut prendre 5-10 minutes pour démarrer")
    
    print("\n" + "="*60)
    print("Relancez ce script après avoir appliqué les corrections")
    print("="*60)
    
    return False

def main():
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     DIAGNOSTIC RÉSEAU APACHE ATLAS                        ║")
    print("║     Script de vérification avant migration                ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cible: {ATLAS_URL}")
    
    # Exécution des tests
    results = [
        test_dns_resolution(),
        test_tcp_connection(),
        test_http_connection(),
        test_docker_network(),
        test_atlas_health()
    ]
    
    # Recommandations finales
    success = provide_recommendations(results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()