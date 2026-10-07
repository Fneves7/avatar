"""Peças comuns dos testes: caras, poses e mãos sintéticas (sem webcam).

Correr (na raiz do projeto):
    .venv\\Scripts\\python.exe -m pytest                 # testes rápidos
    .venv\\Scripts\\python.exe -m pytest -m mediapipe    # ciclo completo com o MediaPipe (mais lento)
    .venv\\Scripts\\python.exe -m pytest -m parity       # cartoon igual ao último commit (refatorizações)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avatar_app.landmarks import EYE_A, EYE_B, FINGERS  # noqa: E402
from avatar_app.tracker import BodyState  # noqa: E402

W, H = 1280, 720
_CANON = np.loadtxt(Path(__file__).resolve().parent / "data" / "canonical_face_468.txt")


def make_face(yaw: float = 0.0, roll: float = 0.0, scale: float = 13.0, center=(640.0, 250.0),
              mirror: bool = True) -> np.ndarray:
    """Cara sintética (478 x 3, píxeis) a partir do modelo canónico do MediaPipe.

    mirror=True imita a webcam espelhada. Os pontos 468-477 (íris) são postos no centro
    de cada olho, como faz o Face Mesh com refine_landmarks."""
    tpl = _CANON * np.array([-1.0 if mirror else 1.0, -1.0, -1.0])
    a, b = math.radians(yaw), math.radians(roll)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    f = tpl @ (rz @ ry).T * scale + np.array([center[0], center[1], 0.0])
    extra = []
    for eye in (EYE_A, EYE_B):
        c = f[eye].mean(axis=0)
        extra += [c, c + [0.4 * scale, 0, 0], c, c, c]
    return np.vstack([f, extra])


def make_pose() -> tuple[np.ndarray, np.ndarray]:
    """Pose (33 x 3) de alguém sentado de frente, com o braço 15 levantado; ancas fora de imagem."""
    pose = np.zeros((33, 3))
    vis = np.ones(33)
    points = {0: (640, 250), 2: (620, 230), 5: (660, 230), 7: (590, 240), 8: (690, 240), 9: (628, 280),
              10: (652, 280), 11: (790, 440), 12: (490, 440), 13: (860, 560), 14: (420, 570),
              15: (880, 420), 16: (400, 690), 17: (885, 400), 19: (895, 395), 21: (860, 410),
              18: (395, 710), 20: (405, 715), 22: (390, 695), 23: (750, 720), 24: (530, 720)}
    for i, p in points.items():
        pose[i, :2] = p
    vis[[23, 24]] = 0.2
    return pose, vis


def make_hand(wrist=(880.0, 420.0), spacing: float = 16.0, base: float = 22.0) -> np.ndarray:
    """Mão aberta (21 x 3) com os dedos em leque para cima."""
    hand = np.zeros((21, 3))
    wrist = np.asarray(wrist, float)
    hand[0, :2] = wrist
    for fi, ang in enumerate(np.linspace(-0.9, 0.5, 5)):
        for j in range(4):
            hand[FINGERS[fi][j], :2] = wrist + (base + j * spacing) * np.array([math.sin(ang), -math.cos(ang)])
    return hand


NEUTRAL_BS = {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0,
              "jawOpen": 0.0, "browInnerUp": 0.0, "browDownLeft": 0.0, "browDownRight": 0.0,
              "eyeWideLeft": 0.0, "eyeWideRight": 0.0}


def make_state(face=True, pose=True, hand=True, calibrated=False, **blendshapes) -> BodyState:
    s = BodyState(W, H)
    if face is True:
        face = make_face()
    if face is not None and face is not False:
        s.face = face
    if pose:
        s.pose, s.pose_visibility = make_pose()
    if hand is True:
        hand = make_hand()
    if hand is not None and hand is not False:
        s.hands = {15: hand}
    s.calibrated = calibrated
    s.blendshapes = {**NEUTRAL_BS, **blendshapes}
    return s


@pytest.fixture
def state():
    return make_state()
