"""Idle: respiração subtil quando a pessoa está parada.

Mede-se o movimento da cabeça e dos ombros (relativo à largura dos ombros). Depois de
~1 s parado, a respiração entra aos poucos; ao primeiro movimento sai depressa, sem
saltos. A respiração sobe os ombros; cotovelos e cabeça acompanham em parte; mãos e
ancas ficam no sítio (normalmente estão apoiadas). Só altera o estado usado para desenhar.
"""
from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from .tracker import L_SHOULDER, R_SHOULDER, BodyState

BREATH_PERIOD = 4.5        # s por ciclo (≈ 13 respirações por minuto)
BREATH_INHALE = 0.4        # fração do ciclo a inspirar (inspirar é mais rápido)
BREATH_AMPLITUDE = 0.015   # subida dos ombros (* largura dos ombros)
STILL_SPEED = 0.25         # abaixo disto (larguras de ombros por segundo) conta como parado
SPEED_WINDOW = 0.4         # s: velocidade = deslocamento nesta janela (o ruído frame a frame não soma)
STILL_DELAY = 1.0          # s parado antes de começar a respirar
FADE_IN_S = 1.0
FADE_OUT_S = 0.25

# Quanto cada ponto da pose acompanha a respiração (ombros = 1).
_POSE_WEIGHTS = np.zeros(33)
_POSE_WEIGHTS[[L_SHOULDER, R_SHOULDER]] = 1.0
_POSE_WEIGHTS[[13, 14]] = 0.5          # cotovelos
_POSE_WEIGHTS[0:11] = 0.7              # cabeça (nariz, olhos, orelhas, boca)
HEAD_WEIGHT = 0.7                      # malha da cara


def breath(phase: float) -> float:
    """0 (expirado) .. 1 (inspirado) ao longo do ciclo, com inspiração mais rápida."""
    phase %= 1.0
    if phase < BREATH_INHALE:
        x = phase / BREATH_INHALE
    else:
        x = 1.0 - (phase - BREATH_INHALE) / (1.0 - BREATH_INHALE)
    return x * x * (3 - 2 * x)  # suave nos extremos


class IdleAnimator:
    def __init__(self):
        self.weight = 0.0           # 0 = sem idle, 1 = respiração completa
        self._t: float | None = None
        self._hist: deque[tuple[float, np.ndarray]] = deque()
        self._speed = 0.0
        self._still_since: float | None = None

    def _anchor(self, s: BodyState) -> tuple[np.ndarray | None, float]:
        """Pontos de referência do movimento (cabeça + ombros) e a escala (largura dos ombros)."""
        pts = []
        scale = 0.0
        if s.pose is not None:
            pts += [s.pose[L_SHOULDER, :2], s.pose[R_SHOULDER, :2]]
            scale = float(np.linalg.norm(s.pose[L_SHOULDER, :2] - s.pose[R_SHOULDER, :2]))
        if s.face is not None:
            pts.append(s.face[:468, :2].mean(axis=0))
            if scale == 0.0:
                scale = 2.2 * float(np.linalg.norm(s.face[234, :2] - s.face[454, :2]))
        return (np.array(pts) if pts else None), scale

    def apply(self, s: BodyState, t: float) -> BodyState:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        anchor, scale = self._anchor(s)
        if anchor is None or scale < 1:
            self._hist.clear()
            self._still_since, self.weight = None, 0.0
            return s

        # Velocidade da cabeça/ombros: deslocamento face à posição de há SPEED_WINDOW s,
        # relativo à largura dos ombros (o ruído de cada frame não se acumula).
        if self._hist and self._hist[-1][1].shape != anchor.shape:
            self._hist.clear()  # apareceu/desapareceu uma parte: recomeça a medir
        self._hist.append((t, anchor))
        while len(self._hist) > 1 and t - self._hist[1][0] >= SPEED_WINDOW:
            self._hist.popleft()
        t_old, old = self._hist[0]
        if t - t_old >= 0.5 * SPEED_WINDOW:
            self._speed = float(np.max(np.linalg.norm(anchor - old, axis=1))) / (t - t_old) / scale

        if self._speed < STILL_SPEED:
            if self._still_since is None:
                self._still_since = t
        else:
            self._still_since = None
        still = self._still_since is not None and t - self._still_since >= STILL_DELAY
        if still:
            self.weight = min(1.0, self.weight + dt / FADE_IN_S)
        else:
            self.weight = max(0.0, self.weight - dt / FADE_OUT_S)
        if self.weight <= 0.0:
            return s

        rise = self.weight * BREATH_AMPLITUDE * scale * breath(t / BREATH_PERIOD)
        changes = {}
        if s.pose is not None:
            pose = s.pose.copy()
            pose[:, 1] -= rise * _POSE_WEIGHTS
            changes["pose"] = pose
        if s.face is not None:
            face = s.face.copy()
            face[:, 1] -= rise * HEAD_WEIGHT
            changes["face"] = face
        return dataclasses.replace(s, **changes)
