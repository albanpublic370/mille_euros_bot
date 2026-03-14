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
    if not flux.entries: return None
    derniere = flux.entries[0]
    url = derniere.enclosures[0].href
    
    print(f"📥 Téléchargement de : {derniere.title}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    r = requests.get(url, headers=headers)
    with open("temp_brut.mp3", 'wb') as f:
        f.write(r.content)
    
    print("🛠️ Conversion en WAV (Optimisation pour modèle Small)...")
    subprocess.run([
        'ffmpeg', '-y', '-i', 'temp_brut.mp3', 
        '-ar', '16000', '-ac', '1', NOM_AUDIO
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    return derniere.title

def extraire_texte():
    print("🧠 Chargement du modèle 'small' (Précision ++)...")
    # Utilisation de int8 pour que les 8 Go de RAM (standard sur i5) suffisent
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print("🎙️ Analyse en cours... Le processeur travaille dur, patience.")
    
    segments, info = model.transcribe(
        NOM_AUDIO, 
        beam_size=5, # Analyse plusieurs hypothèses pour éviter les erreurs de mots
        language="fr",
        condition_on_previous_text=False,
        vad_filter=True 
    )
    
    print(f"⏱️ Durée de l'émission : {info.duration / 60:.1f} minutes")
    
    texte_final = ""
    for segment in segments:
        # On affiche pour surveiller la qualité
        print(f"[{segment.start:.0f}s] {segment.text}")
        texte_final += segment.text + " "
    return texte_final

def classer_questions_detaille(texte):
    print("\n🔍 Extraction et formatage final...")
    
    categories = ["BLEUE", "BLANCHE", "ROUGE", "BANCO", "SUPER BANCO"]
    resultats_finaux = []

    # Extraction des candidats (Prénom + Ville)
    candidats_raw = re.findall(r"(?:avec|accueillons)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:qui nous vient de|à)\s+([A-Z][a-z]+)", texte)
    
    for cat in categories:
        # Recherche du bloc entre deux couleurs
        pattern = r"(?i)" + cat + r"(.*?)(?=" + "|".join(categories) + r"|$)"
        blocs = re.findall(pattern, texte, re.DOTALL)
        
        for idx, bloc in enumerate(blocs):
            bloc = " ".join(bloc.split()) # Nettoie les espaces et retours ligne
            if len(bloc) < 25: continue

            # --- LOGIQUE QUESTION | RÉPONSE ---
            if "?" in bloc:
                parties = bloc.split("?", 1)
                question = parties[0].strip() + " ?"
                reste = parties[1].strip()
                
                # On cherche la réponse (souvent introduite par "C'est", "La réponse est", ou juste après le ?)
                # On prend la première phrase complète après le point d'interrogation
                reponse_match = re.search(r"(?:C'est |La réponse est |C'était |)[ ]?([^.!?]{2,})[.!?]", reste)
                reponse = reponse_match.group(1).strip() if reponse_match else "À valider"
            else:
                question = bloc
                reponse = "Non détectée"

            # --- ASSOCIATION CANDIDAT ---
            info_c = "Finalistes / Collectif"
            if cat in ["BLEUE", "BLANCHE"] and idx < len(candidats_raw):
                info_c = f"{candidats_raw[idx][0]} ({candidats_raw[idx][1]})"

            ligne = f"{cat} | {info_c} | {question} | {reponse}"
            resultats_finaux.append(ligne)

    return resultats_finaux

if __name__ == "__main__":
    try:
        titre = preparer_audio()
        if titre:
            texte_brut = extraire_texte()
            lignes = classer_questions_detaille(texte_brut)
            
            print(f"\n🏆 RÉSULTATS : {titre}")
            print("-" * 80)
            
            with open("jeu_1000_euros_complet.txt", "w") as f:
                for i, l in enumerate(lignes, 1):
                    output = f"{i}° {l}"
                    print(output)
                    f.write(output + "\n")
            
            print(f"\n✅ Terminé ! Fichier 'jeu_1000_euros_complet.txt' généré.")

    except Exception as e:
        print(f"❌ Erreur critique : {e}")
