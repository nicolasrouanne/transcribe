#!/usr/bin/env python3
"""Local WhisperX transcription with stage-level caching.

Each stage (transcribe / align / diarise) is cached to disk so an interrupted
run can be resumed without redoing finished work. The HF token is read from
~/.config/whisperx/token and passed via the Python API (never via argv).

Layout under <output-dir>:
  out/<safe>.{txt,srt,vtt,json,tsv}
  cache/<safe>/{meta,transcribe,align,diarize}.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import get_writer

MAX_BASENAME_LEN = 60


def best_gpu_device() -> str:
    """Best torch device for align/diarize on this machine.

    ASR (faster-whisper / CTranslate2) stays on CPU because CTranslate2
    does not yet support Metal/MPS on macOS.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_audio_normalized(
    path: Path, sr: int = 16000, apply_loudnorm: bool = True
) -> np.ndarray:
    """Decode audio via ffmpeg with optional EBU R128 loudness normalisation.

    Returns float32 mono PCM at the target sample rate. With loudnorm enabled,
    quiet passages are pulled up so whisper's no_speech / VAD thresholds see
    speech instead of background noise.
    """
    audio_filter = (
        "loudnorm=I=-16:TP=-1.5:LRA=11" if apply_loudnorm else "anull"
    )
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(path),
        "-af", audio_filter,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def safe_dirname(name: str, max_len: int = MAX_BASENAME_LEN) -> str:
    """Sanitize a filename stem into a shell-safe, ASCII-only, truncated form."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = ascii_only.lower()
    safe = re.sub(r"[^a-z0-9]+", "-", lowered)
    safe = re.sub(r"-+", "-", safe).strip("-")
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("-")
    return safe or "unnamed"


def fingerprint(path: Path) -> str:
    """Stable identity hash from file (path + size + mtime)."""
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode())
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:16]


def load_hf_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token.strip()
    token_file = Path.home() / ".config/whisperx/token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def stage(label: str, cache_path: Path, loader, compute):
    """Skip-or-run: load cached output if present, else compute and save."""
    if cache_path.exists():
        print(f"✓ {label}: cached → {cache_path.name}", file=sys.stderr)
        return loader(cache_path)
    print(f"→ {label}...", file=sys.stderr)
    t0 = time.time()
    result = compute()
    print(f"  {label} done in {time.time() - t0:.0f}s", file=sys.stderr)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="audio file path")
    p.add_argument("--lang", default="fr", help="language code or 'auto'")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--min-speakers", type=int, default=2)
    p.add_argument("--max-speakers", type=int, default=2)
    p.add_argument(
        "--diarize-model",
        default="pyannote/speaker-diarization-community-1",
    )
    p.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="skip EBU R128 loudness normalisation (default: enabled)",
    )
    p.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.3,
        help="whisper threshold below which a segment is dropped as silence "
        "(default 0.3; whisper default is 0.6)",
    )
    p.add_argument(
        "--vad-onset",
        type=float,
        default=0.3,
        help="Silero VAD onset probability threshold (default 0.3; "
        "whisperx default is 0.5)",
    )
    args = p.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        sys.exit(f"File not found: {audio_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    safe_name = safe_dirname(audio_path.stem)

    out_dir = output_dir / "out"
    cache_dir = output_dir / "cache" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fp = fingerprint(audio_path)
    meta_path = cache_dir / "meta.json"
    new_meta = {
        "audio": str(audio_path),
        "fingerprint": fp,
        "safe_name": safe_name,
        "model": args.model,
        "lang": args.lang,
        "diarize_model": args.diarize_model if not args.no_diarize else None,
        "loudnorm": not args.no_loudnorm,
        "no_speech_threshold": args.no_speech_threshold,
        "vad_onset": args.vad_onset,
    }
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        prev_fp = prev.get("fingerprint")
        # Treat missing fingerprint (legacy / migrated cache) as "unknown,
        # trust it" — only abort on an explicit mismatch.
        if prev_fp is not None and prev_fp != fp:
            print(
                f"⚠ Cache fingerprint mismatch ({prev_fp} vs {fp}).",
                file=sys.stderr,
            )
            print(
                f"  The audio at this basename changed. Clear with: rm -rf {cache_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    meta_path.write_text(json.dumps(new_meta, indent=2))

    asr_device = "cpu"  # CTranslate2 has no Metal/MPS backend yet
    gpu_device = best_gpu_device()  # for alignment + diarisation
    compute_type = "int8"

    print(f"Audio: {audio_path.name}", file=sys.stderr)
    print(f"Safe: {safe_name}", file=sys.stderr)
    print(f"Out:   {out_dir}", file=sys.stderr)
    print(f"Cache: {cache_dir}", file=sys.stderr)
    print(f"Devices: ASR={asr_device}, align/diarize={gpu_device}", file=sys.stderr)
    print(
        f"Loading audio (loudnorm={'on' if not args.no_loudnorm else 'off'})...",
        file=sys.stderr,
    )
    audio = load_audio_normalized(audio_path, apply_loudnorm=not args.no_loudnorm)

    # Stage 1: transcription
    def do_transcribe():
        lang = None if args.lang == "auto" else args.lang
        print("  loading model + VAD (silent ~1-2min)...", file=sys.stderr)
        asr_options = {
            "no_speech_threshold": args.no_speech_threshold,
            "condition_on_previous_text": False,
            "hallucination_silence_threshold": 2.0,
        }
        vad_options = {
            "vad_onset": args.vad_onset,
            "vad_offset": max(args.vad_onset - 0.15, 0.05),
        }
        model = whisperx.load_model(
            args.model,
            asr_device,
            compute_type=compute_type,
            language=lang,
            vad_method="silero",
            asr_options=asr_options,
            vad_options=vad_options,
        )
        res = model.transcribe(audio, language=lang, print_progress=True)
        (cache_dir / "transcribe.json").write_text(json.dumps(res))
        return res

    result = stage(
        "transcribe",
        cache_dir / "transcribe.json",
        lambda path: json.loads(path.read_text()),
        do_transcribe,
    )
    detected_lang = result.get("language", args.lang)

    # Stage 2: alignment
    def do_align():
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=gpu_device
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            gpu_device,
            return_char_alignments=False,
        )
        (cache_dir / "align.json").write_text(json.dumps(aligned))
        return aligned

    result = stage(
        "align",
        cache_dir / "align.json",
        lambda path: json.loads(path.read_text()),
        do_align,
    )
    # whisperx.align drops the top-level language key; the writer needs it.
    result["language"] = detected_lang

    # Stage 3: diarisation (optional)
    if not args.no_diarize:
        token = load_hf_token()
        if not token:
            print(
                "No HF token (~/.config/whisperx/token), skipping diarisation",
                file=sys.stderr,
            )
        else:

            def do_diarize():
                pipeline = DiarizationPipeline(
                    model_name=args.diarize_model,
                    token=token,
                    device=gpu_device,
                )
                segs = pipeline(
                    audio,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                segs.to_json(cache_dir / "diarize.json", orient="records")
                return segs

            diarize_segs = stage(
                "diarize",
                cache_dir / "diarize.json",
                lambda path: pd.read_json(path, orient="records"),
                do_diarize,
            )
            result = assign_word_speakers(diarize_segs, result)

    # Stage 4: write outputs (filenames driven by safe_name)
    print("→ writing outputs...", file=sys.stderr)
    writer_opts = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }
    # The writer uses the stem of this path for output filenames; the file
    # itself never gets opened, so a synthetic path with the safe stem works.
    synthetic_audio_path = audio_path.with_name(safe_name + audio_path.suffix)
    for fmt in ("txt", "srt", "vtt", "json", "tsv"):
        writer = get_writer(fmt, str(out_dir))
        writer(result, str(synthetic_audio_path), writer_opts)

    print(
        f"Done → {out_dir}/{safe_name}.{{txt,srt,vtt,json,tsv}}",
        file=sys.stderr,
    )
    print(f"Cache kept at {cache_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
