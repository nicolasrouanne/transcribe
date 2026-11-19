# transcribe

Fully local pipeline to transcribe and clean up sensitive audio files. No
cloud calls during use. Models are downloaded once from Hugging Face / Ollama,
then inference is entirely offline.

Stack:
- **ffmpeg + EBU R128 loudnorm** — audio pre-processing (normalised to -16 LUFS)
- **WhisperX** (`large-v3`) — transcription + word-level alignment + Silero VAD
- **pyannote/speaker-diarization-community-1** — speaker diarisation
- **Ollama + mistral-nemo:12b** — contextual LLM correction

Two scripts (symlinked into `~/.local/bin/`, callable from anywhere):
- `transcribe <audio>` — audio → multi-format transcript (`.txt`, `.srt`, `.vtt`, `.json`, `.tsv`)
- `correct <transcript.json>` — transcript → LLM-corrected version

Every stage is cached. If a run crashes (or you Ctrl-C), re-running the same
command resumes from the last completed stage.

## Setup

### 1. WhisperX (transcription + diarisation)

```sh
uv tool install whisperx
```

### 2. Hugging Face token for diarisation

The pyannote diarisation model is gated — you need an HF token and you must
accept the terms for two models (one click each, free):

1. Create a "Read" token at https://huggingface.co/settings/tokens
2. Accept the terms:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
3. Save the token:
   ```sh
   mkdir -p ~/.config/whisperx
   echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.config/whisperx/token
   ```

The token is **only used to download the models**. Once they're cached locally
(`~/.cache/huggingface/`), you can disconnect from the internet and everything
keeps working.

To transcribe without diarisation, delete that file — the script detects the
absence and skips the diarisation stage.

### 3. Ollama (LLM correction)

```sh
brew install ollama
ollama serve &
ollama pull mistral-nemo:12b
```

Alternative models depending on your RAM:
- 8 GB → `mistral:7b`
- 16 GB → `mistral-nemo:12b` (default)
- 32 GB+ → `qwen2.5:32b`
- 64 GB+ → `llama3.3:70b`

## Usage

### Transcription

```sh
transcribe <audio-file> [lang] [output-dir]
```

- `lang`: language code (`fr` by default, `auto` to autodetect)
- `output-dir`: output directory (`.` by default)

Examples:
```sh
transcribe ~/Downloads/meeting.m4a
transcribe ~/Downloads/talk.mp3 en
transcribe ~/Downloads/meeting.m4a fr ~/transcripts/
```

Advanced flags (passed directly to `transcribe.py`):

```sh
transcribe.py audio.m4a \
  --no-loudnorm \                # disable EBU R128 normalisation (rare)
  --no-speech-threshold 0.5 \    # ours 0.3; whisper default 0.6 — see Tuning
  --vad-onset 0.4                # ours 0.3; whisperx default 0.5
```

The audio basename is **sanitised** (ASCII, lowercase, dashes) and truncated
to 60 characters. For example `Réunion d'équipe - Roadmap Q3 .m4a` →
`reunion-d-equipe-roadmap-q3`.

Layout created under `output-dir`:

```
<output-dir>/
├── out/
│   ├── <safe>.txt        # plain text with speaker labels
│   ├── <safe>.srt        # subtitles with timestamps
│   ├── <safe>.vtt        # web-format subtitles
│   ├── <safe>.json       # canonical source (segments + words + speakers)
│   └── <safe>.tsv        # table
└── cache/
    └── <safe>/
        ├── meta.json     # fingerprint + params (stale-cache detection)
        ├── transcribe.json
        ├── align.json
        └── diarize.json
```

`meta.json` records the source audio's fingerprint (path + size + mtime). If
you re-run with a modified audio at the same basename, the mismatch is
detected and the script aborts. To start over: `rm -rf cache/<safe>/`.

### LLM correction

```sh
correct <transcript.json> [model]
```

Examples:
```sh
correct ~/Downloads/meeting.json
correct ~/Downloads/meeting.json qwen2.5:14b
```

The prompt (v3) allows contextual correction: the LLM uses neighbouring lines
to fix phonetically-close mishears that fit the topic and to reconstruct
nonsense phrases. It does not paraphrase clear sentences and preserves oral
hesitations.

If the JSON sits in `out/`, the new outputs are written next to it and the
cache lands in the matching location:

```
<output-dir>/
├── out/
│   ├── <safe>.json              # input
│   ├── <safe>.corrected.txt     # new
│   └── <safe>.diff              # new
└── cache/
    └── <safe>/
        └── correct/
            └── <model>/
                └── <prompt-version>/      # v1, v2, v3... changes when the prompt changes
                    └── <content-hash>/
                        ├── chunk-000-0001-0040.txt
                        └── chunk-001-0041-0080.txt
```

The content hash is in the cache path: if the transcript JSON changes
(re-transcription with different params), the `correct` cache is not reused
(new hash → new subfolder).

## Which files to use

All under `out/`:

| File | Use |
|---|---|
| `<safe>.txt` | Quick reading |
| `<safe>.srt` | Targeted re-listening (precise timestamps on the audio) |
| `<safe>.json` | Canonical source; archive or pipe to other tools |
| `<safe>.corrected.txt` | LLM-corrected version; compare against the `.txt` |
| `<safe>.diff` | Exact list of changes proposed by the LLM |
| `<safe>.vtt`, `<safe>.tsv` | Redundant for typical use |

## Performance (M-series CPU, 1h of audio)

| Stage | Duration |
|---|---|
| Transcription (`large-v3`) | ~5-10 min |
| Alignment | ~1-2 min |
| Diarisation | ~30-40 min |
| LLM correction (40 segments per chunk) | ~5-10 min |

No GPU exploited at the moment (CTranslate2 doesn't support Metal). Pyannote
diarisation can run on MPS but it isn't enabled here.

## Tuning: recovering missing passages

Whisper drops passages that are too quiet. On audio at `mean_volume` -35 dB,
you can lose 20-30% of the content without noticing.

The script applies three mitigations by default:

| Setting | Our value | Whisper default | Effect |
|---|---|---|---|
| `loudnorm` (ffmpeg) | enabled (-16 LUFS) | not applied | Pulls quiet passages up before transcription |
| `no_speech_threshold` | 0.3 | 0.6 | Whisper drops fewer uncertain segments |
| `vad_onset` (Silero) | 0.3 | 0.5 | VAD catches quieter speech |
| `hallucination_silence_threshold` | 2.0 | None | Cuts the repeated-phrase hallucination loops |

If you still see gaps, go lower with `--vad-onset 0.2` or
`--no-speech-threshold 0.2`. Too low → whisper starts transcribing background
noise as text, so step down gradually.

### Diagnosing gaps

To see where the transcript skips moments of audio:

```python
# pip install via uv tool, or just paste into a script
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

If the sum of >30s gaps exceeds 10% of the total duration, retune the
thresholds.

## Privacy

- Audio: stays on disk, never uploaded
- Transcripts: stay on disk, never uploaded
- HF token: passed to pyannote via the Python API (not visible in `ps`)
- Models: downloaded once (HF + Ollama), then inference is offline

The `.gitignore` excludes outputs and caches by default — an accidental
`git add .` won't commit sensitive content.

## Resuming an interrupted run

All caches are keyed on content:
- `transcribe`: `cache/<safe>/meta.json` validates the audio fingerprint
- `correct`: `cache/<safe>/correct/<model>/<prompt-version>/<content-hash>/`
  changes as soon as the transcript changes

Re-running the same command skips every completed stage.

To force re-processing a single stage, delete its cache file:
- Re-transcribe: `rm cache/<safe>/transcribe.json`
- Re-align:     `rm cache/<safe>/align.json`
- Re-diarise:   `rm cache/<safe>/diarize.json`
- Re-correct:   `rm -rf cache/<safe>/correct/`
- Full reset:   `rm -rf cache/<safe>/`
