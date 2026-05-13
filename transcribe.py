#!/usr/bin/env python3
"""Local WhisperX transcription with stage-level caching.

Each stage (transcribe / align / diarise) is cached to disk so an interrupted
run can be resumed without redoing finished work. The HF token is read from
~/.config/whisperx/token and passed via the Python API (never via argv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import get_writer


def fingerprint(path: Path) -> str:
    """Stable cache key from file identity (path + size + mtime)."""
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
    args = p.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        sys.exit(f"File not found: {audio_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / ".transcribe-cache" / fingerprint(audio_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "model": args.model,
                "lang": args.lang,
                "diarize_model": args.diarize_model if not args.no_diarize else None,
            },
            indent=2,
        )
    )

    device = "cpu"
    compute_type = "int8"

    print(f"Audio: {audio_path.name}", file=sys.stderr)
    print(f"Cache: {cache_dir}", file=sys.stderr)
    audio = whisperx.load_audio(str(audio_path))

    # Stage 1: transcription
    def do_transcribe():
        lang = None if args.lang == "auto" else args.lang
        print("  loading model + VAD (silent ~1-2min)...", file=sys.stderr)
        model = whisperx.load_model(
            args.model,
            device,
            compute_type=compute_type,
            language=lang,
            vad_method="silero",
        )
        result = model.transcribe(audio, language=lang, print_progress=True)
        (cache_dir / "transcribe.json").write_text(json.dumps(result))
        return result

    result = stage(
        "transcribe",
        cache_dir / "transcribe.json",
        lambda p: json.loads(p.read_text()),
        do_transcribe,
    )
    detected_lang = result.get("language", args.lang)

    # Stage 2: alignment
    def do_align():
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        (cache_dir / "align.json").write_text(json.dumps(aligned))
        return aligned

    result = stage(
        "align",
        cache_dir / "align.json",
        lambda p: json.loads(p.read_text()),
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
                    device=device,
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
                lambda p: pd.read_json(p, orient="records"),
                do_diarize,
            )
            result = assign_word_speakers(diarize_segs, result)

    # Stage 4: write outputs
    print("→ writing outputs...", file=sys.stderr)
    writer_opts = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }
    for fmt in ("txt", "srt", "vtt", "json", "tsv"):
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(audio_path), writer_opts)

    basename = audio_path.stem
    print(
        f"Done → {output_dir}/{basename}.{{txt,srt,vtt,json,tsv}}",
        file=sys.stderr,
    )
    print(f"Cache kept at {cache_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
