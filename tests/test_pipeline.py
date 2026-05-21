"""End-to-end regression test against annotated fixtures.

Runs the full pipeline on each fixture audio, then compares the output
against a hand-corrected reference.json (same schema as the pipeline
output, just cleaned up).

Each fixture dir contains:
  - audio.<ext>      source audio (committed)
  - reference.json   ground truth: corrected pipeline output (committed)
  - baseline.json    current measured {"wer": ..., "der": ...} (committed
                     once you validate a run; test skips while absent)

Adding a fixture:
  1. cp <audio> tests/fixtures/<name>/audio.<ext>
  2. ./transcribe tests/fixtures/<name>/audio.<ext> fr tests/fixtures/<name>/
  3. edit out/<safe>.json, fix speakers + text, save as reference.json
  4. rm -rf tests/fixtures/<name>/{out,cache}
  5. UPDATE_BASELINE=1 uv run pytest   # writes baseline.json with measured values
  6. git diff to sanity-check, commit

Updating a baseline after a validated tuning improvement:
  UPDATE_BASELINE=1 uv run pytest      # overwrites baseline.json
  git diff                              # confirm the new values, commit
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from jiwer import wer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

REGRESSION_TOLERANCE = 0.05


def _to_annotation(segments: list[dict]) -> Annotation:
    annotation = Annotation()
    for seg in segments:
        annotation[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return annotation


def _concat_text(segments: list[dict]) -> str:
    return " ".join(seg["text"].strip() for seg in segments)


def _fixture_dirs() -> list[Path]:
    return sorted(d for d in FIXTURES.iterdir() if d.is_dir())


@pytest.mark.parametrize(
    "fixture_dir", _fixture_dirs(), ids=lambda d: d.name
)
def test_pipeline(fixture_dir: Path, tmp_path: Path) -> None:
    audio_path = next(fixture_dir.glob("audio.*"))
    reference = json.loads((fixture_dir / "reference.json").read_text())

    subprocess.run(
        [str(REPO / "transcribe"), str(audio_path), "fr", str(tmp_path)],
        check=True,
    )

    out_json = next((tmp_path / "out").glob("*.json"))
    hypothesis = json.loads(out_json.read_text())

    ref_text = _concat_text(reference["segments"])
    hyp_text = _concat_text(hypothesis["segments"])
    wer_score = wer(ref_text, hyp_text)

    der = DiarizationErrorRate()(
        _to_annotation(reference["segments"]),
        _to_annotation(hypothesis["segments"]),
    )

    print(f"\n[{fixture_dir.name}] WER={wer_score:.3f}  DER={der:.3f}")

    baseline_path = fixture_dir / "baseline.json"
    if os.environ.get("UPDATE_BASELINE") == "1":
        baseline_path.write_text(
            json.dumps({"wer": wer_score, "der": der}, indent=2) + "\n"
        )
        pytest.skip(f"Updated {baseline_path.relative_to(REPO)}")
    if not baseline_path.exists():
        pytest.skip(
            f"No baseline for {fixture_dir.name}. "
            f"Re-run with UPDATE_BASELINE=1 to write {baseline_path.relative_to(REPO)}."
        )
    baseline = json.loads(baseline_path.read_text())

    assert wer_score <= baseline["wer"] + REGRESSION_TOLERANCE, (
        f"WER regressed: {wer_score:.3f} > {baseline['wer']:.3f} + {REGRESSION_TOLERANCE}"
    )
    assert der <= baseline["der"] + REGRESSION_TOLERANCE, (
        f"DER regressed: {der:.3f} > {baseline['der']:.3f} + {REGRESSION_TOLERANCE}"
    )
