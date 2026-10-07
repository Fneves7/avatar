"""Filtro One Euro vetorizado para suavizar landmarks sem introduzir muito atraso."""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) aplicado a um array de pontos.

    min_cutoff: menor => mais suave quando parado.
    beta: maior => menos atraso em movimentos rápidos.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._x.shape != x.shape or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        # Velocidade por ponto (norma em x,y) para ajustar o cutoff de cada landmark.
        speed = np.linalg.norm(self._dx[..., :2], axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()
