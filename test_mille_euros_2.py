import feedparser
import requests
from faster_whisper import WhisperModel
import re
import os
import subprocess

# --- CONFIGURATION ---
RSS_URL = "https://radiofrance-podcast.net/podcast09/rss_10206.xml"
NOM_AUDIO = "emission_temp.mp3"

def preparer_audio():
    print("📡 Consultation du flux France Inter...")
    flux = feedparser.parse(RSS_URL)
    derniere = flux.entries[0]
    url = derniere.enclosures[0].href
    
    print(f"📥 Téléchargement de : {derniere.title}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    r = requests.get(url, headers=headers)
    with open("temp_brut.mp3", 'wb') as f:
        f.write(r.content)
    
    print("🛠️ Conversion en WAV pour analyse complète...")
    # Conversion en WAV mono 16kHz pour une lecture linéaire sans saut de pub
    subprocess.run([
        'ffmpeg', '-y', '-i', 'temp_brut.mp3', 
        '-ar', '16000', '-ac', '1', NOM_AUDIO
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    return derniere.title

def extraire_texte():
    print("🧠 Analyse de l'audio (Modèle tiny)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(
        NOM_AUDIO, 
        beam_size=1, 
        language="fr",
        condition_on_previous_text=False,
        vad_filter=True 
    )
    
    print(f"⏱️ Durée traitée : {info.duration / 60:.1f} minutes")
    
    texte_final = ""
    for segment in segments:
        texte_final += segment.text + " "
    return texte_final

def classer_questions_detaille(texte):
    print("\n🔍 Extraction détaillée des questions (Candidats, Questions, Réponses)...")
    
    # On définit les types de questions à chercher
    categories = ["BLEUE", "BLANCHE", "ROUGE", "BANCO", "SUPER BANCO"]
    resultats_finaux = []

    # Liste des mots-clés pour repérer les réponses après la question
    # Nicolas Stoufflet dit souvent "La réponse est..." ou "C'est..."
    
    for cat in categories:
        # Regex plus complexe pour capturer le contexte autour du mot-clé
        pattern = r"(?i)" + cat + r"(.*?)(?=" + "|".join(categories) + r"|$)"
        blocs = re.findall(pattern, texte, re.DOTALL)
        
        for bloc in blocs:
            # Nettoyage de base
            bloc = bloc.strip()
            if len(bloc) < 20: continue

            # Tentative de séparation Question | Réponse
            # On cherche souvent le point d'interrogation comme séparateur
            if "?" in bloc:
                parties = bloc.split("?", 1)
                question_brute = parties[0].strip() + " ?"
                reponse_brute = parties[1].split(".")[0].strip() # On prend la phrase juste après
            else:
                question_brute = bloc
                reponse_brute = "Non détectée"

            # Pour le nom/provenance : l'IA "tiny" peut parfois mélanger les noms propres
            # Dans cette version, on met "À identifier" car l'extraction automatique 
            # du nom du candidat demande une analyse de texte très avancée (NLP).
            infos_candidat = "Candidat et Provenance" 

            ligne = f"{cat} | {infos_candidat} | {question_brute} | {reponse_brute}"
            resultats_finaux.append(ligne)

    return resultats_finaux

if __name__ == "__main__":
    try:
        titre = preparer_audio()
        texte_brut = extraire_texte()
        lignes_finales = classer_questions_detaille(texte_brut)
        
        print(f"\n🏆 EXTRACTION : {titre}")
        print("-" * 50)
        
        for i, ligne in enumerate(lignes_finales, 1):
            print(f"{i}° {ligne}")
            
        # Sauvegarde propre
        with open("jeu_1000_euros_complet.txt", "w") as f:
            for i, ligne in enumerate(lignes_finales, 1):
                f.write(f"{i}° {ligne}\n")
        print(f"\n✅ Fichier 'jeu_1000_euros_complet.txt' généré.")

    except Exception as e:
        print(f"❌ Erreur : {e}")
