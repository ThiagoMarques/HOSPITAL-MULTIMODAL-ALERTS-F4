"""Baixa o modelo MediaPipe PoseLandmarker lite para models/."""
from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEST = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"Já existe: {DEST}")
        return
    print(f"Baixando {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"OK — {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
