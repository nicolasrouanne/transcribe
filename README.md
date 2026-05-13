# transcribe

Pipeline 100 % local pour transcrire et corriger des fichiers audio sensibles.
Aucun appel cloud pendant l'usage. Les modèles sont téléchargés une fois depuis
Hugging Face / Ollama, puis l'inférence est entièrement offline.

Stack :
- **ffmpeg + EBU R128 loudnorm** — pré-traitement audio (normalisation -16 LUFS)
- **WhisperX** (`large-v3`) — transcription + alignement word-level + VAD Silero
- **pyannote/speaker-diarization-community-1** — diarisation (qui dit quoi)
- **Ollama + mistral-nemo:12b** — correction LLM contextuelle

Deux scripts (installés dans `~/.local/bin/`, donc appelables depuis n'importe où) :
- `transcribe <audio>` — audio → transcript multi-format (`.txt`, `.srt`, `.vtt`, `.json`, `.tsv`)
- `correct <transcript.json>` — transcript → version corrigée par un LLM local

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
transcribe <audio-file> [lang] [output-dir]
```

- `lang` : code langue (`fr` par défaut, `auto` pour détection automatique)
- `output-dir` : dossier de sortie (`.` par défaut)

Exemples :
```sh
transcribe ~/Downloads/seance.m4a
transcribe ~/Downloads/talk.mp3 en
transcribe ~/Downloads/seance.m4a fr ~/transcripts/
```

Flags avancés (pour le `transcribe.py` directement) :

```sh
transcribe.py audio.m4a \
  --no-loudnorm \                # désactive la normalisation EBU R128 (rare)
  --no-speech-threshold 0.5 \    # défaut 0.3 ; whisper default 0.6 — voir Tuning
  --vad-onset 0.4                # défaut 0.3 ; whisperx default 0.5
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
correct <transcript.json> [model]
```

Exemples :
```sh
correct ~/Downloads/seance.json
correct ~/Downloads/seance.json qwen2.5:14b
```

Le prompt (v3) autorise la correction contextuelle : le LLM utilise les lignes
voisines pour fixer les mots phonétiquement proches qui collent au sujet et
reconstruire les phrases nonsense. Il ne paraphrase pas les phrases claires et
conserve les hésitations orales.

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
                └── <prompt-version>/      # v1, v2, v3... change quand le prompt bouge
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

## Tuning : récupérer les passages manquants

Whisper drop les passages trop bas. Sur un audio à `mean_volume` -35 dB, on
peut perdre 20-30 % du contenu sans le savoir.

Le script applique par défaut trois mitigations :

| Réglage | Valeur ici | Whisper default | Effet |
|---|---|---|---|
| `loudnorm` (ffmpeg) | activé (-16 LUFS) | non appliqué | Pull-up des passages calmes avant transcription |
| `no_speech_threshold` | 0.3 | 0.6 | Whisper drop moins de segments incertains |
| `vad_onset` (Silero) | 0.3 | 0.5 | VAD catch les voix plus calmes |
| `hallucination_silence_threshold` | 2.0 | None | Coupe les boucles répétées d'hallucination |

Si tu vois encore des trous, baisse encore avec `--vad-onset 0.2` ou
`--no-speech-threshold 0.2`. Trop bas → whisper se met à transcrire du
bruit de fond, donc test progressivement.

### Diagnostic des trous

Pour voir où le transcript saute des moments d'audio :

```python
# pip install via uv tool; ou écrire un script
import json, statistics
from pathlib import Path

j = json.loads(Path("out/<safe>.json").read_text())
segs = j["segments"]
print(f"Segments: {len(segs)}, last end: {segs[-1]['end']:.0f}s")
gaps = sorted(
    [(i, segs[i]['end'], segs[i+1]['start'], segs[i+1]['start']-segs[i]['end'])
     for i in range(len(segs)-1)],
    key=lambda x: -x[3],
)
for i, e, s, g in [g for g in gaps if g[3] > 30][:10]:
    print(f"  seg {i}: {e:7.1f} → {s:7.1f}  (gap {g:6.1f}s)")
```

Si la somme des gaps > 30s dépasse 10 % de la durée totale, tune les seuils.

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
