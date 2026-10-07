"""Camada de animação partilhada por todos os estilos de avatar.

Junta, frame a frame, tudo o que não depende de como o avatar é desenhado:
olhos fechados pelo piscar calibrado (com histerese), olhar vivo (piscar automático e
micro-movimentos), idle (respiração), movimento secundário (molas do cabelo e das mangas)
e transições (fade-in/fade-out das mãos e da cara). O resultado é um AnimFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .eyes import EyeLife
from .idle import BREATH_PERIOD, IdleAnimator, breath
from .secondary import SecondaryMotion
from .tracker import L_WRIST, R_WRIST, BodyState
from .transitions import Fade

BLINK_CLOSE, BLINK_OPEN = 0.6, 0.35  # piscar calibrado: fecha acima de, reabre abaixo de


@dataclass
class AnimFrame:
    state: BodyState                       # estado já com o idle aplicado
    t: float
    eyes_closed: list[bool]                # [EYE_A, EYE_B] fechados pelo piscar real (calibrado)
    auto_blink: float                      # fecho do piscar automático (0..1)
    saccade: tuple[float, float]           # micro-movimento da íris (x raio da íris)
    hair_offset: np.ndarray                # px
    hair_rot: float                        # rad
    sleeve_offset: dict[int, np.ndarray] = field(default_factory=dict)
    hand_fades: dict[int, Fade] = field(default_factory=dict)
    face_fade: Fade | None = None
    idle_weight: float = 0.0
    breath: float = 0.0                    # respiração atual (0..1, já multiplicada pelo idle)


class Animator:
    def __init__(self):
        self.lively_eyes = True     # piscar automático + micro-movimentos (tecla l)
        self.eye_life = EyeLife()
        self.transitions = True     # mãos/cara entram e saem aos poucos (tecla t)
        self.idle_enabled = True    # respiração quando parado (tecla i)
        self.idle = IdleAnimator()
        self.secondary_enabled = True  # cabelo e mangas seguem com atraso (tecla m)
        self.secondary = SecondaryMotion()
        self.hand_fades = {L_WRIST: Fade(), R_WRIST: Fade()}
        self.face_fade = Fade()
        self._eyes_closed = [False, False]

    def update(self, s: BodyState, now: float) -> AnimFrame:
        # Idle: respiração subtil quando a pessoa está parada (só altera o que se desenha).
        if self.idle_enabled:
            s = self.idle.apply(s, now)
        else:
            self.idle.weight = 0.0
        # Movimento secundário: molas do cabelo e das mangas.
        if self.secondary_enabled:
            self.secondary.update(s, now)
        else:
            self.secondary.clear()
        # Transições: opacidade de cada mão e da cara (e o último valor visto, para desvanecer).
        for side, fade in self.hand_fades.items():
            hand = s.hands.get(side)
            fade.update(now, None if hand is None else (hand, s.hand_meshes.get(side)), self.transitions)
        self.face_fade.update(now, None if s.face is None else (s.face, s.blendshapes), self.transitions)
        # Olhos fechados pelo piscar real e olhar vivo.
        self._update_eyes_closed(s)
        if self.lively_eyes:
            auto_blink, saccade = self.eye_life.update(now, any(self._eyes_closed))
        else:
            auto_blink, saccade = 0.0, (0.0, 0.0)
        sec = self.secondary
        return AnimFrame(
            state=s, t=now, eyes_closed=list(self._eyes_closed), auto_blink=auto_blink, saccade=saccade,
            hair_offset=sec.hair_offset, hair_rot=sec.hair_rot, sleeve_offset=dict(sec.sleeve_offset),
            hand_fades=self.hand_fades, face_fade=self.face_fade, idle_weight=self.idle.weight,
            breath=self.idle.weight * breath(now / BREATH_PERIOD))

    def _update_eyes_closed(self, s: BodyState) -> None:
        """Olhos fechados pelo piscar calibrado. Sem calibração não se fecha nada: os valores
        em bruto variam muito de pessoa para pessoa (óculos, formato dos olhos)."""
        if not s.calibrated or not s.blendshapes:
            self._eyes_closed = [False, False]
            return
        # EYE_A (landmark 33) é o olho direito da pessoa; EYE_B (263) o esquerdo.
        for i, key in enumerate(("eyeBlinkRight", "eyeBlinkLeft")):
            v = s.blendshapes.get(key, 0.0)
            if self._eyes_closed[i]:
                self._eyes_closed[i] = v > BLINK_OPEN
            else:
                self._eyes_closed[i] = v > BLINK_CLOSE
