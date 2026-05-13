#!/usr/bin/env python3
"""Correct a WhisperX transcript via a local LLM (Ollama).

Reads the canonical <safe>.json produced by transcribe.py, processes it in
chunks through a local LLM, and writes:
  out/<safe>.corrected.txt  (cleaned-up transcript)
  out/<safe>.diff           (unified diff vs the original)

Each chunk's response is cached under
  cache/<safe>/correct/<model>/<content-hash>/chunk-NNN.txt
so re-running skips work already done.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_BASENAME_LEN = 60
PROMPT_VERSION = "v2"  # bump to invalidate cached chunks when the prompt changes

SYSTEM_PROMPT = """Tu es un éditeur de transcription automatique français.

Tu reçois des lignes au format strict: [N] [SPEAKER_XX] texte
Pour chaque ligne d'entrée, tu produis EXACTEMENT UNE ligne de sortie au même format.

Tu corriges UNIQUEMENT:
- mots inexistants en français (mauvaise reconnaissance speech-to-text)
- noms propres mal orthographiés (lieux, personnes)
- ponctuation manquante évidente

Tu NE changes PAS le sens. Tu NE paraphrases PAS. Tu NE résumes PAS. Tu NE \
reformules PAS. Tu conserves les hésitations, répétitions, "euh", coupures, \
phrases inachevées (c'est du français parlé). Si tu ne sais pas comment \
améliorer une ligne, recopie-la telle quelle.

INTERDICTIONS ABSOLUES:
- AUCUNE explication, excuse, préambule, ou commentaire sur la tâche.
- AUCUN texte hors des lignes [N] [SPEAKER_XX] texte.
- Ta réponse commence par "[" (le premier caractère de "[1] [SPEAKER...").
- Ta réponse contient le MÊME nombre de lignes que l'entrée.

Tu agis. Tu n'expliques pas."""

LINE_RE = re.compile(r"^\[(\d+)\]\s+\[([^\]]+)\]\s+(.*)$")


def safe_dirname(name: str, max_len: int = MAX_BASENAME_LEN) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    safe = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())
    safe = re.sub(r"-+", "-", safe).strip("-")
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("-")
    return safe or "unnamed"


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def call_ollama(model: str, system: str, user: str, timeout: int = 600) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8192},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def segments_from_json(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    out = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker", "SPEAKER")
        text = (seg.get("text") or "").strip()
        if text:
            out.append((speaker, text))
    return out


def format_chunk(items: list[tuple[int, str, str]]) -> str:
    return "\n".join(f"[{idx}] [{spk}] {txt}" for idx, spk, txt in items)


def parse_chunk_output(text: str) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for raw in text.splitlines():
        m = LINE_RE.match(raw.strip())
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3).strip())
    return out


def ollama_alive() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def resolve_layout(json_path: Path) -> tuple[Path, Path, str]:
    """Return (out_dir, cache_root, safe_name) following the project layout.

    Expected: <root>/out/<safe>.json → out_dir = <root>/out, cache_root = <root>/cache.
    Otherwise: fall back to writing alongside the JSON.
    """
    safe_name = safe_dirname(json_path.stem)
    parent = json_path.parent
    if parent.name == "out" and (parent.parent / "cache").exists():
        return parent, parent.parent / "cache", safe_name
    return parent, parent / "cache", safe_name


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_file", help="<safe>.json from transcribe.py")
    p.add_argument("--model", default="mistral-nemo:12b")
    p.add_argument("--chunk-size", type=int, default=40, help="segments per chunk")
    p.add_argument("--out-dir", default=None, help="override output directory")
    p.add_argument("--cache-root", default=None, help="override cache root directory")
    args = p.parse_args()

    if not ollama_alive():
        sys.exit("Ollama not reachable at localhost:11434. Run: ollama serve")

    json_path = Path(args.json_file).expanduser().resolve()
    if not json_path.is_file():
        sys.exit(f"File not found: {json_path}")

    out_dir, cache_root, safe_name = resolve_layout(json_path)
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    if args.cache_root:
        cache_root = Path(args.cache_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = segments_from_json(json_path)
    if not segments:
        sys.exit("No segments in JSON.")
    print(f"Loaded {len(segments)} segments", file=sys.stderr)

    indexed: list[tuple[int, str, str]] = [
        (i + 1, spk, txt) for i, (spk, txt) in enumerate(segments)
    ]
    original_numbered = "\n".join(f"[{i}] [{spk}] {txt}" for i, spk, txt in indexed)
    content_hash = short_hash(original_numbered)

    model_safe = args.model.replace(":", "-").replace("/", "-")
    cache_dir = (
        cache_root / safe_name / "correct" / model_safe / PROMPT_VERSION / content_hash
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out:   {out_dir}", file=sys.stderr)
    print(f"Cache: {cache_dir}", file=sys.stderr)

    chunks = [
        indexed[i : i + args.chunk_size]
        for i in range(0, len(indexed), args.chunk_size)
    ]
    print(f"{len(chunks)} chunks of ~{args.chunk_size} segments", file=sys.stderr)

    corrected: dict[int, tuple[str, str]] = {}
    for ci, chunk in enumerate(chunks):
        first_idx, last_idx = chunk[0][0], chunk[-1][0]
        chunk_path = cache_dir / f"chunk-{ci:03d}-{first_idx:04d}-{last_idx:04d}.txt"
        chunk_in = format_chunk(chunk)

        if chunk_path.exists():
            print(f"✓ chunk {ci + 1}/{len(chunks)}: cached", file=sys.stderr)
            chunk_out = chunk_path.read_text()
        else:
            print(
                f"→ chunk {ci + 1}/{len(chunks)} (segments {first_idx}-{last_idx})...",
                file=sys.stderr,
            )
            t0 = time.time()
            chunk_out = call_ollama(args.model, SYSTEM_PROMPT, chunk_in)
            chunk_path.write_text(chunk_out)
            print(f"  done in {time.time() - t0:.0f}s", file=sys.stderr)

        parsed = parse_chunk_output(chunk_out)
        for idx, spk, txt in chunk:
            corrected[idx] = parsed.get(idx, (spk, txt))

    def clean_line(spk: str, txt: str) -> str:
        return f"[{spk}] {txt}"

    original_clean = "\n".join(clean_line(spk, txt) for _, spk, txt in indexed)
    corrected_clean = "\n".join(
        clean_line(corrected[i][0], corrected[i][1]) for i, _, _ in indexed
    )

    out_path = out_dir / f"{safe_name}.corrected.txt"
    diff_path = out_dir / f"{safe_name}.diff"
    out_path.write_text(corrected_clean + "\n")
    diff = difflib.unified_diff(
        original_clean.splitlines(),
        corrected_clean.splitlines(),
        fromfile=f"{safe_name}.txt (original)",
        tofile=f"{safe_name}.corrected.txt",
        lineterm="",
        n=2,
    )
    diff_path.write_text("\n".join(diff) + "\n")

    changed = sum(
        1
        for (_, s0, t0), (_, s1, t1) in zip(
            indexed,
            ((i, *corrected[i]) for i, _, _ in indexed),
        )
        if (s0, t0) != (s1, t1)
    )
    print(f"Done → {out_path}", file=sys.stderr)
    print(
        f"Diff → {diff_path}  ({changed}/{len(indexed)} lines changed)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
