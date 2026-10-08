"""Preferências guardadas entre sessões (settings.json, pessoal como o calibration.json).

O que se afina ao vivo com as teclas (perfil de suavização, intensidade do exagero, partes da
animação ligadas/desligadas, ...) fica guardado ao sair e é reposto no arranque seguinte. As
opções da linha de comandos (estilo, paleta, fundo, ...) não são guardadas: mandam sempre.
"""
from __future__ import annotations

import json
from pathlib import Path

from .tracker import SMOOTHING_PRESETS

VERSION = 1

# nome no ficheiro -> (objeto, atributo); "tracker"/"renderer"/"cartoon" resolvidos em apply/collect.
FIELDS = {
    "smoothing": ("tracker", "smoothing"),
    "rigid_face": ("tracker", "rigid_face"),
    "constraints": ("tracker", "constraints"),
    "lively_eyes": ("renderer", "lively_eyes"),
    "transitions": ("renderer", "transitions"),
    "idle": ("renderer", "idle_enabled"),
    "secondary": ("renderer", "secondary_enabled"),
    "head_3d": ("renderer", "head_3d"),
    "exaggerate": ("renderer", "exaggerate"),
    "exaggeration": ("renderer", "exaggeration"),
    "body_widths": ("cartoon", "body_widths"),
}


def _targets(tracker, renderer) -> dict:
    return {"tracker": tracker, "renderer": renderer, "cartoon": renderer.cartoon}


def collect(tracker, renderer) -> dict:
    """Preferências atuais."""
    objs = _targets(tracker, renderer)
    out = {"version": VERSION, "smoothing_preset": tracker.smoothing_preset}
    for key, (obj, attr) in FIELDS.items():
        out[key] = getattr(objs[obj], attr)
    return out


def apply(data: dict, tracker, renderer) -> list[str]:
    """Repõe as preferências; ignora chaves desconhecidas ou com tipo errado. Devolve as aplicadas."""
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return []
    objs = _targets(tracker, renderer)
    applied = []
    preset = data.get("smoothing_preset")
    if preset in SMOOTHING_PRESETS:
        tracker.set_smoothing_preset(preset)
        applied.append("smoothing_preset")
    for key, (obj, attr) in FIELDS.items():
        value = data.get(key)
        current = getattr(objs[obj], attr)
        if key == "exaggeration":
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 3.0:
                setattr(objs[obj], attr, float(value))
                applied.append(key)
        elif isinstance(value, bool) and isinstance(current, bool):
            setattr(objs[obj], attr, value)
            applied.append(key)
    return applied


def load(path: Path, tracker, renderer) -> list[str]:
    try:
        return apply(json.loads(Path(path).read_text(encoding="utf-8")), tracker, renderer)
    except (OSError, ValueError):
        return []


def save(path: Path, tracker, renderer) -> None:
    try:
        Path(path).write_text(json.dumps(collect(tracker, renderer), indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[settings] could not save {path}: {exc}")
