"""Suavização "rígida" da malha da cara (tecla r).

O filtro One Euro ponto a ponto dá a cada ponto um cutoff conforme a sua velocidade: quando a
cabeça roda ou se desloca, pontos diferentes ficam com atrasos diferentes e a malha deforma-se
e treme. Aqui a malha é separada em duas partes, filtradas cada uma à sua maneira:

  * movimento de conjunto: posição, escala e inclinação (ajuste de semelhança 2D sobre pontos
    estáveis da parte de cima da cara: cantos dos olhos, nariz, testa, laterais). É uma média
    de muitos pontos, por isso tem pouco ruído e pode ser filtrado com mais força;
  * forma local: a malha sem esse movimento (expressões, olhar, rotação lateral), filtrada
    como antes, ponto a ponto.

Depois volta a montar a malha. Só muda o filtro; o desenho é o mesmo.
"""
from __future__ import annotations

import math

import numpy as np

from .smoothing import OneEuroFilter

# Pontos que quase não mexem com as expressões (sem boca, maxilar, pálpebras nem sobrancelhas).
ANCHORS = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 10, 109, 338, 67, 297, 234, 454, 127, 356, 93, 323]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RigidFaceFilter:
    """Mesma interface do OneEuroFilter (chamar com (pontos, t); reset())."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 0.08):
        self.local = OneEuroFilter(min_cutoff, beta)       # forma local (px)
        self.motion = OneEuroFilter(0.5, 0.1)               # [[cx, cy], [escala, ângulo * escala_ref]]
        self.depth = OneEuroFilter(0.5, 0.1)                # z médio
        self.reset()

    def filters(self) -> list[OneEuroFilter]:
        return [self.local, self.motion, self.depth]

    def reset(self) -> None:
        for f in (self.local, self.motion, self.depth):
            f.reset()
        self._ref: np.ndarray | None = None   # forma das âncoras no 1.º frame (centrada)
        self._ref_scale = 1.0
        self._angle = 0.0

    def _pose(self, pts: np.ndarray) -> tuple[np.ndarray, float, float]:
        a = pts[ANCHORS, :2]
        c = a.mean(axis=0)
        d = a - c
        s = float(np.sqrt((d ** 2).sum(axis=1).mean())) or 1.0
        if self._ref is None:
            self._ref, self._ref_scale, self._angle = d.copy(), s, 0.0
            return c, s, 0.0
        # Rotação de Procrustes 2D das âncoras face ao 1.º frame.
        ref = self._ref
        num = float((ref[:, 0] * d[:, 1] - ref[:, 1] * d[:, 0]).sum())
        den = float((ref * d).sum())
        angle = self._angle + _wrap(math.atan2(num, den) - self._angle)  # contínuo, sem saltos de 2π
        self._angle = angle
        return c, s, angle

    def __call__(self, pts: np.ndarray, t: float) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] <= max(ANCHORS):
            return self.local(pts, t)
        c, s, angle = self._pose(pts)
        k = self._ref_scale / s
        cos, sin = math.cos(-angle), math.sin(-angle)
        d = pts[:, :2] - c
        local = np.empty_like(pts)
        local[:, 0] = (cos * d[:, 0] - sin * d[:, 1]) * k
        local[:, 1] = (sin * d[:, 0] + cos * d[:, 1]) * k
        z0 = float(pts[:, 2].mean())
        local[:, 2] = (pts[:, 2] - z0) * k

        local = self.local(local, t)
        m = self.motion(np.array([[c[0], c[1]], [s, angle * self._ref_scale]]), t)
        z0 = float(self.depth(np.array([[z0, 0.0]]), t)[0, 0])
        c, s, angle = m[0], float(m[1, 0]), float(m[1, 1]) / self._ref_scale

        k = s / self._ref_scale
        cos, sin = math.cos(angle), math.sin(angle)
        out = np.empty_like(pts)
        out[:, 0] = c[0] + (cos * local[:, 0] - sin * local[:, 1]) * k
        out[:, 1] = c[1] + (sin * local[:, 0] + cos * local[:, 1]) * k
        out[:, 2] = z0 + local[:, 2] * k
        return out
