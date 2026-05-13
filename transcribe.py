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
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import get_writer

MAX_BASENAME_LEN = 60


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
    }
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev.get("fingerprint") != fp:
            print(
                f"⚠ Cache fingerprint mismatch ({prev.get('fingerprint')} vs {fp}).",
                file=sys.stderr,
            )
            print(
                f"  The audio at this basename changed. Clear with: rm -rf {cache_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    meta_path.write_text(json.dumps(new_meta, indent=2))

    device = "cpu"
    compute_type = "int8"

    print(f"Audio: {audio_path.name}", file=sys.stderr)
    print(f"Safe: {safe_name}", file=sys.stderr)
    print(f"Out:   {out_dir}", file=sys.stderr)
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
