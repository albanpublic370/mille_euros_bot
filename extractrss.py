import feedparser
import requests
from faster_whisper import WhisperModel
import re
import os

# --- CONFIGURATION ---
RSS_URL = "https://radiofrance-podcast.net/podcast09/rss_10351.xml"
FICHIER_AUDIO = "derniere_emission.mp3"

def telecharger_audio():
    print("1. Vérification du flux RSS...")
    flux = feedparser.parse(RSS_URL)
    derniere_emission = flux.entries[0]
    url_audio = derniere_emission.enclosures[0].href
    
    if not os.path.exists(FICHIER_AUDIO):
        print(f"   -> Téléchargement de : {derniere_emission.title}")
        r = requests.get(url_audio)
        with open(FICHIER_AUDIO, 'wb') as f:
            f.write(r.content)
    return derniere_emission.title

def transcrire_optimise():
    print("\n2. Initialisation de Faster-Whisper (Optimisé CPU)...")
    
    # Configuration spécifique pour processeur i5 (int8 = moins de RAM utilisée)
    # On utilise le modèle "small" qui est le meilleur compromis pour vous
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("   -> Transcription lancée. Votre i5 travaille...")
    # beam_size=1 accélère encore le processus au détriment d'un poil de précision
    segments, info = model.transcribe(FICHIER_AUDIO, beam_size=5, language="fr")

    texte_complet = ""
    for segment in segments:
        print(f"[{segment.start:.0f}s] {segment.text}") # Affiche la progression en direct
        texte_complet += segment.text + " "
        
    return texte_complet

def trier_questions(texte):
    print("\n3. Analyse et tri des questions...")
    resultats = {"Bleues": [], "Blanches": [], "Rouges": []}
    
    # Recherche textuelle simplifiée
    patterns = {
        "Bleues": r"(?:bleue)(.*?)(?=\?|bleue|blanche|rouge|$)",
        "Blanches": r"(?:blanche)(.*?)(?=\?|bleue|blanche|rouge|$)",
        "Rouges": r"(?:rouge)(.*?)(?=\?|bleue|blanche|rouge|$)"
    }
    
    for couleur, regex in patterns.items():
        trouvailles = re.findall(regex, texte, re.IGNORECASE)
        for t in trouvailles:
            phrase = t.strip(" :-\n") + "?"
            if len(phrase) > 25:
                resultats[couleur].append(phrase)
    return resultats

# --- LANCEMENT ---
titre = telecharger_audio()
transcription = transcrire_optimise()
questions = trier_questions(transcription)

print(f"\n✅ Terminé pour : {titre}")
print(questions)
