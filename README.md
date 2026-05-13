# transcribe

Pipeline 100 % local pour transcrire et corriger des fichiers audio sensibles.
Aucun appel cloud pendant l'usage. Les modèles sont téléchargés une fois depuis
Hugging Face / Ollama, puis l'inférence est entièrement offline.

Stack :
- **WhisperX** (`large-v3`) — transcription + alignement word-level + VAD Silero
- **pyannote/speaker-diarization-community-1** — diarisation (qui dit quoi)
- **Ollama + mistral-nemo:12b** — correction LLM des erreurs de speech-to-text

Deux scripts :
- `./transcribe <audio>` — audio → transcript multi-format (`.txt`, `.srt`, `.vtt`, `.json`, `.tsv`)
- `./correct <transcript.json>` — transcript → version corrigée par un LLM local

Chaque étape est mise en cache. Si un run plante (ou si tu fais Ctrl-C), relance
la même commande : seules les étapes manquantes seront refaites.

## Setup

### 1. WhisperX (transcription + diarisation)

```sh
uv tool install whisperx
```

### 2. Token Hugging Face pour la diarisation

La diarisation pyannote est gated : il faut un token HF et accepter les CGU des
deux modèles (un clic chacun, gratuit) :

1. Créer un token "Read" sur https://huggingface.co/settings/tokens
2. Accepter les termes :
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Sauvegarder le token :
   ```sh
   mkdir -p ~/.config/whisperx
   echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.config/whisperx/token
   ```

Le token sert **uniquement à télécharger les modèles**. Une fois en cache local
(`~/.cache/huggingface/`), tu peux couper le réseau et tout continue de marcher.

Pour transcrire sans diarisation, supprime ce fichier — le script détectera
l'absence et sautera l'étape.

### 3. Ollama (correction LLM)

```sh
brew install ollama
ollama serve &
ollama pull mistral-nemo:12b
```

Modèles alternatifs selon ta RAM :
- 8 GB → `mistral:7b`
- 16 GB → `mistral-nemo:12b` (défaut)
- 32 GB+ → `qwen2.5:32b`
- 64 GB+ → `llama3.3:70b`

## Usage

### Transcription

```sh
./transcribe <audio-file> [lang] [output-dir]
```

- `lang` : code langue (`fr` par défaut, `auto` pour détection automatique)
- `output-dir` : dossier de sortie (`.` par défaut)

Exemples :
```sh
./transcribe ~/Downloads/seance.m4a
./transcribe ~/Downloads/talk.mp3 en
./transcribe ~/Downloads/seance.m4a fr ~/transcripts/
```

Outputs dans `output-dir` :
- `seance.txt` — texte plat avec speaker labels
- `seance.srt` — sous-titres avec timestamps
- `seance.vtt` — sous-titres web
- `seance.json` — source canonique (segments + mots + speakers)
- `seance.tsv` — table

Stages cachés dans `<output-dir>/.transcribe-cache/<fingerprint>/` :
- `transcribe.json` — transcription brute
- `align.json` — segments avec timestamps mot-à-mot
- `diarize.json` — segments speaker (start/end/speaker)

### Correction LLM

```sh
./correct <transcript.json> [model]
```

Exemples :
```sh
./correct ~/Downloads/seance.json
./correct ~/Downloads/seance.json qwen2.5:14b
```

Outputs à côté du JSON :
- `seance.corrected.txt` — transcript nettoyé
- `seance.diff` — diff unifié vs original (pour reviewer les changements)

Cache par chunk dans `<json-dir>/.correct-cache/<content-hash>/<model>/`.

## Quelles fichiers utiliser

| Fichier | Usage |
|---|---|
| `.txt` | Lecture rapide |
| `.srt` | Réécoute ciblée (timestamps précis sur l'audio) |
| `.json` | Source canonique, à archiver / passer à d'autres outils |
| `.corrected.txt` | Version corrigée par LLM, à comparer au `.txt` |
| `.diff` | Liste exacte des changements proposés par le LLM |
| `.vtt`, `.tsv` | Redondants pour usage courant |

## Performance (M-series CPU, 1h d'audio)

| Étape | Durée |
|---|---|
| Transcription (`large-v3`) | ~5-10 min |
| Alignement | ~1-2 min |
| Diarisation | ~30-40 min |
| Correction LLM (chunks de 40 segments) | ~5-10 min |

Pas de GPU exploité actuellement (CTranslate2 ne supporte pas Metal). La
diarisation pyannote peut tourner sur MPS mais n'est pas activée dans ce script.

## Privacy

- Audio : reste sur disque, jamais uploadé
- Transcripts : restent sur disque, jamais uploadés
- HF token : passé à pyannote via API Python (pas visible dans `ps`)
- Modèles : téléchargés une fois (HF + Ollama), puis inférence offline

Le `.gitignore` exclut automatiquement les outputs et caches : même un
`git add .` accidentel ne committera pas de contenu sensible.

## Reprendre un run interrompu

Tous les caches sont keyed sur le contenu (audio fingerprint pour
`transcribe`, hash du transcript pour `correct`). Relancer la même commande
saute toutes les étapes déjà complétées.

Pour forcer le re-traitement d'une étape, supprime le fichier de cache
correspondant.
