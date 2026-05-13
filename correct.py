#!/usr/bin/env python3
"""Correct a WhisperX transcript via a local LLM (Ollama).

Reads the canonical <basename>.json produced by transcribe.py, processes it
in chunks through a local LLM, and writes:
  - <basename>.corrected.txt  (cleaned-up transcript)
  - <basename>.diff           (unified diff vs the original)

Each chunk's response is cached, so re-running skips work already done.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """Tu es un éditeur de transcription. Voici une portion de transcription \
automatique d'un dialogue en français.

RÈGLES STRICTES — applique-les fidèlement :
1. Corrige UNIQUEMENT les erreurs probables de speech-to-text :
   - mots qui n'existent pas en français (mauvaise reconnaissance)
   - noms propres mal orthographiés (lieux, personnes)
   - ponctuation manquante évidente
2. NE change PAS le sens d'une phrase, même si elle semble bizarre.
3. NE PARAPHRASE PAS. NE RÉSUME PAS. NE REFORMULE PAS.
4. CONSERVE les hésitations, répétitions, "euh", coupures, phrases \
inachevées — c'est du français parlé.
5. CONSERVE EXACTEMENT le préfixe [N] (numéro) et le label [SPEAKER_XX] \
au début de chaque ligne.
6. Une ligne d'entrée = une ligne de sortie. Même nombre de lignes.
7. Si tu ne sais pas comment améliorer une ligne, recopie-la TELLE QUELLE.

Entrée :
{input}

Sortie (même numérotation, mêmes labels) :
"""

LINE_RE = re.compile(r"^\[(\d+)\]\s+\[([^\]]+)\]\s+(.*)$")


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def call_ollama(model: str, prompt: str, timeout: int = 600) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8192},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"]


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_file", help="<basename>.json from transcribe.py")
    p.add_argument("--model", default="mistral-nemo:12b")
    p.add_argument("--chunk-size", type=int, default=40, help="segments per chunk")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if not ollama_alive():
        sys.exit("Ollama not reachable at localhost:11434. Run: ollama serve")

    json_path = Path(args.json_file).expanduser().resolve()
    if not json_path.is_file():
        sys.exit(f"File not found: {json_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else json_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = json_path.stem

    segments = segments_from_json(json_path)
    if not segments:
        sys.exit("No segments in JSON.")
    print(f"Loaded {len(segments)} segments", file=sys.stderr)

    indexed: list[tuple[int, str, str]] = [
        (i + 1, spk, txt) for i, (spk, txt) in enumerate(segments)
    ]
    original_numbered = "\n".join(f"[{i}] [{spk}] {txt}" for i, spk, txt in indexed)

    cache_dir = (
        json_path.parent
        / ".correct-cache"
        / short_hash(original_numbered)
        / args.model.replace(":", "-")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
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
            chunk_out = call_ollama(args.model, PROMPT.format(input=chunk_in))
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

    out_path = output_dir / f"{basename}.corrected.txt"
    diff_path = output_dir / f"{basename}.diff"
    out_path.write_text(corrected_clean + "\n")
    diff = difflib.unified_diff(
        original_clean.splitlines(),
        corrected_clean.splitlines(),
        fromfile=f"{basename}.txt (original)",
        tofile=f"{basename}.corrected.txt",
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
    print(f"Diff → {diff_path}  ({changed}/{len(indexed)} lines changed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
