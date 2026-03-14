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
    
    # Téléchargement du MP3 original
    r = requests.get(url, headers=headers)
    with open(NOM_AUDIO, 'wb') as f:
        f.write(r.content)
    
    print("🛠️ Conversion FORCÉE en WAV (pour briser le blocage de la pub)...")
    NOM_WAV = "emission_fixe.wav"
    
    # On convertit en WAV 16kHz mono (le format préféré de Whisper)
    # Cela va "aplatir" les 12Mo en un seul bloc audio continu de 15 min
    subprocess.run([
        'ffmpeg', '-y', '-i', NOM_AUDIO, 
        '-ar', '16000', '-ac', '1', NOM_WAV
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    # On utilise maintenant le WAV pour la suite
    if os.path.exists(NOM_WAV):
        # On remplace le MP3 par le WAV pour que Whisper travaille sur du propre
        os.replace(NOM_WAV, NOM_AUDIO) 
        print(f"✅ Fichier converti et prêt. Taille : {os.path.getsize(NOM_AUDIO)/(1024*1024):.1f} Mo")
    
    return derniere.title

def extraire_texte():
    print("🧠 Chargement du modèle 'tiny' (Optimal pour i5-6300U)...")
    # Utilisation du CPU et précision int8 pour économiser la RAM
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    print("🎙️ Analyse complète lancée. Ne fermez pas votre terminal.")
    
    # Paramètres pour ne pas rester bloqué sur la pub
    segments, info = model.transcribe(
        NOM_AUDIO, 
        beam_size=1, 
        language="fr",
        condition_on_previous_text=False,
        vad_filter=True 
    )
    
    print(f"⏱️ Durée réelle détectée : {info.duration / 60:.1f} minutes")
    
    texte_final = ""
    for segment in segments:
        # Affiche la progression en temps réel
        print(f"[{segment.start:.0f}s] {segment.text}")
        texte_final += segment.text + " "
        
    return texte_final

def classer_questions(texte):
    print("\n🔍 Analyse et tri des questions par couleur...")
    db_questions = {"Bleue": [], "Blanche": [], "Rouge": []}
    
    # Regex pour capturer la question après la couleur
    patterns = {
        "Bleue": r"(?:bleue|bleu)(.*?)(?=\?|bleue|blanche|rouge|$)",
        "Blanche": r"(?:blanche|blanc)(.*?)(?=\?|bleue|blanche|rouge|$)",
        "Rouge": r"(?:rouge)(.*?)(?=\?|bleue|blanche|rouge|$)"
    }
    
    for couleur, pattern in patterns.items():
        matches = re.findall(pattern, texte, re.IGNORECASE | re.DOTALL)
        for m in matches:
            nettoye = m.strip(" :-\n")
            if len(nettoye) > 15: # On ignore les bruits de moins de 15 caractères
                db_questions[couleur].append(nettoye + " ?")
                
    return db_questions

# --- EXECUTION PRINCIPALE ---
if __name__ == "__main__":
    try:
        titre_emission = preparer_audio()
        if titre_emission:
            texte_complet = extraire_texte()
            resultat = classer_questions(texte_complet)
            
            print(f"\n🏆 RÉSULTATS : {titre_emission}")
            for couleur, q_list in resultat.items():
                print(f"\n--- {couleur.upper()} ---")
                if not q_list:
                    print("(Aucune question détectée)")
                else:
                    for i, q in enumerate(q_list, 1):
                        print(f"{i}. {q}")
                        
            # Optionnel : Sauvegarde dans un fichier texte pour vos archives
            with open("questions_extraites.txt", "w") as f:
                f.write(f"Émission : {titre_emission}\n\n" + str(resultat))
                
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")
