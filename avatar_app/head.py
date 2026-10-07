"""Estima a cabeça inteira (crânio, nuca e orelhas) a partir da malha da cara.

A malha do MediaPipe só cobre a cara (testa ao queixo). Usando a profundidade z dos
landmarks calcula-se a orientação 3D da cabeça e encaixa-se, atrás da cara, um crânio
elipsoidal com proporções anatómicas. A projeção desse crânio dá a silhueta da cabeça,
incluindo a nuca que aparece quando a cara roda.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FACE_LEFT, FACE_RIGHT, FOREHEAD, CHIN = 234, 454, 10, 152

# Proporções do crânio relativas à largura (fw) e altura (fh) da cara,
# em coordenadas locais da cabeça (direita, baixo, trás), origem no centro da cara.
SKULL_CENTER = (0.0, -0.18, 0.42)       # (x*fw, y*fh, z*fw)
SKULL_RADII = (0.56, 0.62, 0.64)        # (x*fw, y*fh, z*fw)
SKULL_BOTTOM = 0.12                     # y*fh máximo da nuca
NECK_TOP = (0.05, 0.35)                 # topo do pescoço (y*fh, z*fw), escondido dentro do crânio
EAR_POS = (0.53, 0.02, 0.30)            # lateral, vertical, profundidade
EAR_RADII = (0.15, 0.11)                # altura (*fh), largura (*fw)
EAR_FLARE = 1.3                         # quanto a orelha abre para fora (vs. para trás)


def _unit_sphere(n_lat: int = 14, n_lon: int = 28) -> np.ndarray:
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon)
    return np.stack([np.cos(la) * np.cos(lo), np.sin(la), np.cos(la) * np.sin(lo)], -1).reshape(-1, 3)


_SPHERE = _unit_sphere()
_EAR_T = np.linspace(0, 2 * np.pi, 20, endpoint=False)


@dataclass
class Ear:
    polygon: np.ndarray   # (N, 2) int32
    hidden: bool          # True se está virada para trás (fica atrás do crânio)
    inner: np.ndarray     # (N, 2) int32, concha interior


@dataclass
class HeadGeometry:
    skull: np.ndarray            # (N, 2) int32, silhueta convexa do crânio
    ears: list[Ear]
    rotation: np.ndarray         # 3x3, colunas = direita, baixo, trás
    origin: np.ndarray           # centro da cara (3,)
    fw: float
    fh: float

    def to_world(self, local: np.ndarray) -> np.ndarray:
        local = np.atleast_2d(local)
        return self.origin + local @ self.rotation.T


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def estimate_head(face: np.ndarray, oval_idx: list[int]) -> HeadGeometry | None:
    """face: (N, 3) em píxeis (z na mesma escala que x)."""
    if face is None or len(face) < 468:
        return None
    right = face[FACE_RIGHT] - face[FACE_LEFT]
    fw = float(np.linalg.norm(right))
    down = face[CHIN] - face[FOREHEAD]
    fh = float(np.linalg.norm(down))
    if fw < 1 or fh < 1:
        return None
    right = _normalize(right)
    down = _normalize(down - right * np.dot(down, right))
    back = np.cross(right, down)
    if back[2] < 0:  # garante que "trás" aponta para longe da câmara
        back, right = -back, -right
    rot = np.stack([right, down, back], axis=1)
    origin = face[oval_idx].mean(axis=0)
    scale = np.array([fw, fh, fw])

    def project(local: np.ndarray) -> np.ndarray:
        return (origin + (local * scale) @ rot.T)[:, :2]

    skull_local = np.array(SKULL_CENTER) + _SPHERE * np.array(SKULL_RADII)
    # A nuca acaba à altura do fundo das orelhas; abaixo disso é pescoço.
    # Achata (em vez de cortar) para o fundo ficar arredondado.
    below = skull_local[:, 1] > SKULL_BOTTOM
    skull_local[below, 1] = SKULL_BOTTOM + (skull_local[below, 1] - SKULL_BOTTOM) * 0.25
    skull = cv2.convexHull(np.round(project(skull_local)).astype(np.int32))[:, 0, :]

    ears = []
    for sgn in (-1.0, 1.0):
        out = np.array([sgn, 0.0, 0.0])
        # Plano da orelha: vertical + diagonal para trás/fora (as orelhas abrem para os lados).
        u = np.array([0.0, EAR_RADII[0], 0.0])
        v = _normalize(np.array([sgn * EAR_FLARE, 0.0, 1.0]))
        c = np.array([sgn * EAR_POS[0], EAR_POS[1], EAR_POS[2]])
        pts = c + np.outer(np.cos(_EAR_T), u) + np.outer(np.sin(_EAR_T), v * EAR_RADII[1])
        inner = c + 0.55 * (np.outer(np.cos(_EAR_T), u) + np.outer(np.sin(_EAR_T), v * EAR_RADII[1]))
        out_world = rot @ out
        ears.append(Ear(
            polygon=np.round(project(pts)).astype(np.int32),
            hidden=bool(out_world[2] > 0.55),
            inner=np.round(project(inner)).astype(np.int32),
        ))
    return HeadGeometry(skull=skull, ears=ears, rotation=rot, origin=origin, fw=fw, fh=fh)
