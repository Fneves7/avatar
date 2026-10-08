"""Movimento secundário: cabelo e mangas seguem a cabeça/braços com atraso (molas).

Cada elemento tem uma mola amortecida que persegue a posição da parte a que está preso.
O desvio da mola em relação à parte é o "atraso": quando a cabeça acelera o cabelo fica
ligeiramente para trás e, quando pára, passa um pouco do ponto e assenta.
  * cabelo: posição e inclinação (roll) da cabeça; o renderer aplica o desvio sobretudo no
    topo do cabelo (junto à cara quase nada, para não abrir buracos);
  * mangas: a bainha da manga, no cotovelo, segue o braço com uma mola mais rígida;
  * bainha do tronco: o fundo da camisola segue o tronco com um pequeno atraso (quase sempre
    fora de imagem, por isso é subtil).
Os desvios têm limites e as molas recomeçam se a parte "teleportar" (perdida e reencontrada).
"""
from __future__ import annotations

import math

import numpy as np

from .tracker import BodyState

HAIR_FREQ, HAIR_DAMPING = 2.2, 0.35       # Hz, razão de amortecimento (< 1 = passa um pouco do ponto)
HAIR_MAX_LAG = 0.12                       # desvio máximo do cabelo (* largura da cara)
HAIR_MAX_ROT = math.radians(8)            # inclinação máxima do cabelo face à cabeça
SLEEVE_FREQ, SLEEVE_DAMPING = 4.0, 0.4
SLEEVE_MAX_LAG = 0.06                     # desvio máximo da bainha da manga (* largura dos ombros)
HEM_FREQ, HEM_DAMPING = 3.0, 0.4
HEM_MAX_LAG = 0.05                        # desvio máximo da bainha do tronco (* largura dos ombros)
SUBSTEP = 1 / 240                         # passo máximo da simulação (s)
ELBOWS = {15: 13, 16: 14}                 # pulso -> cotovelo (as mãos/braços usam o pulso como chave)


class Spring:
    """Mola amortecida que persegue um alvo; devolve o desvio (posição - alvo)."""

    def __init__(self, freq: float, damping: float):
        self.w = 2 * math.pi * freq
        self.z = damping
        self.p: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        self.p = self.v = None

    def update(self, target, dt: float, max_lag: float) -> np.ndarray:
        target = np.atleast_1d(np.asarray(target, dtype=np.float64))
        if self.p is None or self.p.shape != target.shape or dt <= 0 or dt > 0.25:
            self.p, self.v = target.copy(), np.zeros_like(target)
            return np.zeros_like(target)
        n = max(1, math.ceil(dt / SUBSTEP))
        h = dt / n
        for _ in range(n):
            a = self.w * self.w * (target - self.p) - 2 * self.z * self.w * self.v
            self.v = self.v + a * h
            self.p = self.p + self.v * h
        lag = self.p - target
        size = float(np.linalg.norm(lag))
        if size > 3 * max_lag:  # a parte "teleportou": recomeça sem atraso
            self.p, self.v = target.copy(), np.zeros_like(target)
            return np.zeros_like(target)
        if size > 1e-9:
            # Limite suave: aproxima-se do máximo sem bater nele (um corte seco parece rígido).
            lag = lag * (max_lag * math.tanh(size / max_lag) / size)
        return lag


class SecondaryMotion:
    def __init__(self):
        self._hair_pos = Spring(HAIR_FREQ, HAIR_DAMPING)
        self._hair_rot = Spring(HAIR_FREQ, HAIR_DAMPING)
        self._sleeves = {side: Spring(SLEEVE_FREQ, SLEEVE_DAMPING) for side in ELBOWS}
        self._hem = Spring(HEM_FREQ, HEM_DAMPING)
        self._t: float | None = None
        self.hair_offset = np.zeros(2)     # px
        self.hair_rot = 0.0                # rad
        self.sleeve_offset: dict[int, np.ndarray] = {}
        self.hem_offset = np.zeros(2)      # px

    def clear(self) -> None:
        self._hair_pos.reset()
        self._hair_rot.reset()
        for sp in self._sleeves.values():
            sp.reset()
        self._hem.reset()
        self._t = None
        self.hair_offset, self.hair_rot, self.sleeve_offset = np.zeros(2), 0.0, {}
        self.hem_offset = np.zeros(2)

    def update(self, s: BodyState, t: float) -> None:
        dt = 0.0 if self._t is None else t - self._t
        self._t = t

        if s.face is not None:
            f = s.face[:, :2]
            fw = float(np.linalg.norm(f[234] - f[454])) or 1.0
            eye = f[263] - f[33]
            if eye[0] < 0:
                eye = -eye
            roll = math.atan2(eye[1], eye[0])
            self.hair_offset = self._hair_pos.update(f[:468].mean(axis=0), dt, HAIR_MAX_LAG * fw)
            self.hair_rot = float(self._hair_rot.update(roll, dt, HAIR_MAX_ROT)[0])
        else:
            self._hair_pos.reset()
            self._hair_rot.reset()
            self.hair_offset, self.hair_rot = np.zeros(2), 0.0

        self.sleeve_offset = {}
        if s.pose is not None:
            sw = float(np.linalg.norm(s.pose[11, :2] - s.pose[12, :2])) or 1.0
            for side, elbow in ELBOWS.items():
                self.sleeve_offset[side] = self._sleeves[side].update(s.pose[elbow, :2], dt, SLEEVE_MAX_LAG * sw)
            # O fundo do tronco fica ~1,4x a largura dos ombros abaixo deles (como as ancas estimadas).
            ls, rs = s.pose[11, :2], s.pose[12, :2]
            down = np.array([-(rs - ls)[1], (rs - ls)[0]]) / sw
            if down[1] < 0:
                down = -down
            self.hem_offset = self._hem.update((ls + rs) / 2 + down * 1.4 * sw, dt, HEM_MAX_LAG * sw)
        else:
            for sp in self._sleeves.values():
                sp.reset()
            self._hem.reset()
            self.hem_offset = np.zeros(2)
