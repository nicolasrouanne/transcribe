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

Le basename de l'audio est **sanitisé** (ASCII, minuscules, tirets) et tronqué
à 60 caractères. Par exemple `Psychologue - Corps et pas de côté .m4a` →
`psychologue-corps-et-pas-de-cote`.

Structure créée sous `output-dir` :

```
<output-dir>/
├── out/
│   ├── <safe>.txt        # texte plat avec speaker labels
│   ├── <safe>.srt        # sous-titres avec timestamps
│   ├── <safe>.vtt        # sous-titres web
│   ├── <safe>.json       # source canonique (segments + mots + speakers)
│   └── <safe>.tsv        # table
└── cache/
    └── <safe>/
        ├── meta.json     # fingerprint + params (détection stale)
        ├── transcribe.json
        ├── align.json
        └── diarize.json
```

Le `meta.json` contient le fingerprint (path + size + mtime) de l'audio
source. Si tu re-runs avec un audio modifié au même nom, le mismatch est
détecté et le script abort. Pour repartir de zéro : `rm -rf cache/<safe>/`.

### Correction LLM

```sh
./correct <transcript.json> [model]
```

Exemples :
```sh
./correct ~/Downloads/seance.json
./correct ~/Downloads/seance.json qwen2.5:14b
```

Si le JSON est dans `out/`, les outputs sont écrits à côté et le cache à la
position cohérente :

```
<output-dir>/
├── out/
│   ├── <safe>.json              # input
│   ├── <safe>.corrected.txt     # nouveau
│   └── <safe>.diff              # nouveau
└── cache/
    └── <safe>/
        └── correct/
            └── <model>/
                └── <content-hash>/
                    ├── chunk-000-0001-0040.txt
                    └── chunk-001-0041-0080.txt
```

Le hash de contenu est dans le chemin du cache : si le transcript JSON
change (re-transcription avec d'autres params), le cache `correct` n'est
pas réutilisé (nouvel hash → nouveau sous-dossier).

## Quels fichiers utiliser

Tous dans `out/` :

| Fichier | Usage |
|---|---|
| `<safe>.txt` | Lecture rapide |
| `<safe>.srt` | Réécoute ciblée (timestamps précis sur l'audio) |
| `<safe>.json` | Source canonique, à archiver / passer à d'autres outils |
| `<safe>.corrected.txt` | Version corrigée par LLM, à comparer au `.txt` |
| `<safe>.diff` | Liste exacte des changements proposés par le LLM |
| `<safe>.vtt`, `<safe>.tsv` | Redondants pour usage courant |

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

Tous les caches sont keyed sur le contenu :
- `transcribe` : `cache/<safe>/meta.json` valide via fingerprint audio
- `correct` : `cache/<safe>/correct/<model>/<content-hash>/` change dès que le
  transcript change

Relancer la même commande saute toutes les étapes déjà complétées.

Pour forcer le re-traitement d'une étape, supprime le fichier de cache
correspondant :
- Re-transcribe : `rm cache/<safe>/transcribe.json`
- Re-aligne :    `rm cache/<safe>/align.json`
- Re-diarise :   `rm cache/<safe>/diarize.json`
- Re-correct :   `rm -rf cache/<safe>/correct/`
- Tout repartir : `rm -rf cache/<safe>/`
