"""Download e cache dos modelos .task do MediaPipe."""
from __future__ import annotations

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "face": f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand": f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_lite": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": f"{_BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def get_model(name: str) -> Path:
    """Devolve o caminho local do modelo, descarregando-o se necessário."""
    url = MODEL_URLS[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"[modelos] A descarregar {path.name} ...")
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"\nNão foi possível descarregar o modelo '{path.name}': {exc}\n"
                f"Descarrega-o manualmente (ex.: no browser) e coloca-o em:\n  {path}\n"
                f"URL: {url}\n"
            ) from exc
        tmp.replace(path)
        print(f"[modelos] OK ({path.stat().st_size / 1e6:.1f} MB)")
    return path
