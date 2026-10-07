"""Olhar vivo: piscar automático e micro-movimentos dos olhos.

* Piscar automático: a deteção corre a ~10-15 FPS e um piscar real dura ~0,1-0,15 s, por
  isso muitos piscares escapam (e sem calibração não se deteta nenhum). Se não houver um
  piscar real recente, o avatar pisca sozinho a intervalos aleatórios (2,5-6 s), com a
  curva de um piscar real: fecha depressa, abre mais devagar.
* Micro-movimentos: de vez em quando a íris dá um pequeno salto rápido (sacada) e fica
  lá. É subtil e soma-se ao olhar real detetado, não o substitui.
Tudo depende só do tempo, por isso anima a ritmo de desenho (30 FPS) e não de deteção.
"""
from __future__ import annotations

import math
import random

BLINK_INTERVAL = (2.5, 6.0)    # s entre piscares automáticos
BLINK_AFTER_REAL = 1.5         # s mínimos depois de um piscar real antes de um automático
BLINK_CLOSE_S = 0.07           # tempo a fechar
BLINK_OPEN_S = 0.11            # tempo a abrir
SACCADE_INTERVAL = (0.8, 2.5)  # s entre micro-movimentos
SACCADE_S = 0.04               # duração do salto
SACCADE_RADIUS = 0.2           # amplitude máxima (x raio da íris)


class EyeLife:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._t0: float | None = None
        self._next_blink = 0.0
        self._blink_start: float | None = None
        self._last_real_blink = -1e9
        self._next_saccade = 0.0
        self._sacc_from = (0.0, 0.0)
        self._sacc_to = (0.0, 0.0)
        self._sacc_start = -1e9

    def update(self, t: float, real_blink: bool) -> tuple[float, tuple[float, float]]:
        """Devolve (fecho do piscar automático 0..1, desvio da íris em raios da íris)."""
        if self._t0 is None:
            self._t0 = t
            self._next_blink = t + self._rng.uniform(*BLINK_INTERVAL)
            self._next_saccade = t + self._rng.uniform(*SACCADE_INTERVAL)

        # --- piscar ---
        if real_blink:
            # Piscar real detetado: cancela o automático e volta a contar a partir daqui.
            self._last_real_blink = t
            self._blink_start = None
            self._next_blink = t + self._rng.uniform(*BLINK_INTERVAL)
        elif self._blink_start is None and t >= self._next_blink:
            if t - self._last_real_blink >= BLINK_AFTER_REAL:
                self._blink_start = t
            self._next_blink = t + self._rng.uniform(*BLINK_INTERVAL)

        blink = 0.0
        if self._blink_start is not None:
            e = t - self._blink_start
            if e < BLINK_CLOSE_S:
                blink = math.sin(0.5 * math.pi * e / BLINK_CLOSE_S)
            elif e < BLINK_CLOSE_S + BLINK_OPEN_S:
                blink = math.cos(0.5 * math.pi * (e - BLINK_CLOSE_S) / BLINK_OPEN_S)
            else:
                self._blink_start = None

        # --- micro-movimentos (sacadas) ---
        if t >= self._next_saccade:
            self._sacc_from = self._saccade_pos(t)
            ang = self._rng.uniform(0, 2 * math.pi)
            r = SACCADE_RADIUS * math.sqrt(self._rng.random())
            # De vez em quando volta ao centro, para o olhar não "derivar".
            self._sacc_to = (0.0, 0.0) if self._rng.random() < 0.3 else (r * math.cos(ang), r * math.sin(ang))
            self._sacc_start = t
            self._next_saccade = t + self._rng.uniform(*SACCADE_INTERVAL)
        return blink, self._saccade_pos(t)

    def _saccade_pos(self, t: float) -> tuple[float, float]:
        e = min(1.0, max(0.0, (t - self._sacc_start) / SACCADE_S))
        k = e * e * (3 - 2 * e)  # arranque e paragem suaves
        return (self._sacc_from[0] + (self._sacc_to[0] - self._sacc_from[0]) * k,
                self._sacc_from[1] + (self._sacc_to[1] - self._sacc_from[1]) * k)
