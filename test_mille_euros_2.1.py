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
    
    print("🛠️ Conversion en WAV (Haute Qualité pour modèle Small)...")
    subprocess.run([
        'ffmpeg', '-y', '-i', 'temp_brut.mp3', 
        '-ar', '16000', '-ac', '1', NOM_AUDIO
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    return derniere.title

def extraire_texte():
    print("🧠 Chargement du modèle 'small' (Précision accrue)...")
    # On reste en int8 pour ne pas saturer la RAM de votre i5
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print("🎙️ Analyse en cours... (C'est le moment de prendre un café)")
    
    segments, info = model.transcribe(
        NOM_AUDIO, 
        beam_size=5, # Meilleure recherche de mots que beam_size=1
        language="fr",
        condition_on_previous_text=False,
        vad_filter=True 
    )
    
    print(f"⏱️ Durée de l'audio : {info.duration / 60:.1f} minutes")
    
    texte_final = ""
    for segment in segments:
        # On affiche pour voir la qualité s'améliorer en direct
        print(f"[{segment.start:.0f}s] {segment.text}")
        texte_final += segment.text + " "
    return texte_final

def classer_questions_detaille(texte):
    print("\n🔍 Extraction haute précision...")
    
    categories = ["BLEUE", "BLANCHE", "ROUGE", "BANCO", "SUPER BANCO"]
    resultats_finaux = []

    # Tentative d'extraction des candidats (plus précise avec Small)
    candidats_match = re.findall(r"(?:avec|accueillons)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:qui nous vient de|à)\s+([A-Z][a-z]+)", texte)
    
    for cat in categories:
        # On cherche le mot clé et on capture jusqu'au prochain type de question
        pattern = r"(?i)" + cat + r"(.*?)(?=" + "|".join(categories) + r"|$)"
        blocs = re.findall(pattern, texte, re.DOTALL)
        
        for idx, bloc in enumerate(blocs):
            bloc = bloc.strip()
            if len(bloc) < 20: continue

            # Logique Question | Réponse
            # Nicolas donne souvent la réponse après "C'est..." ou "La réponse est..."
            if "?" in bloc:
                parties = bloc.split("?", 1)
                question = parties[0].strip() + " ?"
                suite = parties[1].strip()
                # On cherche la première phrase affirmative après le ?
                reponse_match = re.search(r"([^.!?]{3,})[.!?]", suite)
                reponse = reponse_match.group(1).strip() if reponse_match else "À identifier"
            else:
                question = bloc
                reponse = "Non détectée"

            # Association candidat
            nom_c = "Candidat"
            if idx < len(candidats_match) and cat in ["BLEUE", "BLANCHE"]:
                nom_c = f"{candidats_match[idx][0]} ({candidats_match[idx][1]})"

            ligne = f"{cat} | {nom_c} | {question} | {reponse}"
            resultats_finaux.append(" ".join(ligne.split()))

    return resultats_finaux

if __name__ == "__main__":
    try:
        titre = preparer_audio()
        texte_brut = extraire_texte()
        lignes = classer_questions_detaille(texte_brut)
        
        print(f"\n🏆 EXTRACTION TERMINÉE : {titre}")
        with open("resultats_mille_euros.txt", "w") as f:
            for i, l in enumerate(lignes, 1):
                output = f"{i}° {l}"
                print(output)
                f.write(output + "\n")
        
        print("\n✅ Fichier 'resultats_mille_euros.txt' prêt.")

    except Exception as e:
        print(f"❌ Erreur : {e}")
