"""Desenha o avatar: fundo + animação (partilhada) + rig + estilo escolhido.

  AnimFrame (animation.py)  estado + olhar vivo, idle, molas, transições
  Rig (rig.py)              parâmetros semânticos (cabeça, olhos, boca, esqueleto, ...)
  Estilo (styles/)          como o avatar é desenhado: "cartoon" (original) ou "robo"
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .animation import Animator
from .drawing import PALETTES, Palette
from .rig import build_rig
from .styles import STYLES
from .tracker import BodyState


class AvatarRenderer:
    def __init__(self, palette_index: int = 0, style: str = "cartoon"):
        self.palette_index = palette_index
        self.animator = Animator()
        self.styles = [cls() for cls in STYLES]
        self.cartoon = self.styles[0]
        self.style_index = next((i for i, st in enumerate(self.styles) if st.name == style), 0)
        self._bg_cache: dict = {}
        self.background: tuple[int, int, int] | None = None  # cor sólida (chroma key) ou None = gradiente

    # ------------------------------------------------------------- estilo e paleta
    @property
    def palette(self) -> Palette:
        return PALETTES[self.palette_index % len(PALETTES)]

    def next_palette(self) -> None:
        self.palette_index = (self.palette_index + 1) % len(PALETTES)

    @property
    def style(self):
        return self.styles[self.style_index]

    def next_style(self) -> str:
        self.style_index = (self.style_index + 1) % len(self.styles)
        return self.style.name

    # Opções do estilo cartoon (teclas h, e, +/-).
    head_3d = property(lambda self: self.cartoon.head_3d,
                       lambda self, v: setattr(self.cartoon, "head_3d", v))
    exaggerate = property(lambda self: self.cartoon.exaggerate,
                          lambda self, v: setattr(self.cartoon, "exaggerate", v))
    exaggeration = property(lambda self: self.cartoon.exaggeration,
                            lambda self, v: setattr(self.cartoon, "exaggeration", v))
    # Opções da animação (teclas l, t, i, m).
    lively_eyes = property(lambda self: self.animator.lively_eyes,
                           lambda self, v: setattr(self.animator, "lively_eyes", v))
    transitions = property(lambda self: self.animator.transitions,
                           lambda self, v: setattr(self.animator, "transitions", v))
    idle_enabled = property(lambda self: self.animator.idle_enabled,
                            lambda self, v: setattr(self.animator, "idle_enabled", v))
    secondary_enabled = property(lambda self: self.animator.secondary_enabled,
                                 lambda self, v: setattr(self.animator, "secondary_enabled", v))
    idle = property(lambda self: self.animator.idle)
    eye_life = property(lambda self: self.animator.eye_life)
    secondary = property(lambda self: self.animator.secondary)

    # ------------------------------------------------------------------ fundo
    def _background(self, w: int, h: int) -> np.ndarray:
        if self.background is not None:
            return np.full((h, w, 3), self.background, np.uint8)
        key = (w, h, self.palette_index)
        if key not in self._bg_cache:
            p = self.palette
            t = np.linspace(0, 1, h)[:, None, None]
            bg = (np.array(p.bg_top) * (1 - t) + np.array(p.bg_bottom) * t)
            bg = np.repeat(bg, w, axis=1)
            yy, xx = np.mgrid[0:h, 0:w]
            spot = np.exp(-(((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h * 0.45) / (h * 0.55)) ** 2))
            bg = bg * (0.75 + 0.35 * spot[..., None])
            self._bg_cache = {key: np.clip(bg, 0, 255).astype(np.uint8)}
        return self._bg_cache[key].copy()

    # ----------------------------------------------------------------- render
    def _animate(self, s: BodyState, t: float | None):
        """Avança a animação UMA vez por frame (molas, piscar, transições) e monta o rig."""
        frame = self.animator.update(s, time.monotonic() if t is None else t)
        return frame, build_rig(frame)

    def render(self, s: BodyState, t: float | None = None) -> np.ndarray:
        """t: instante (s) para as animações (olhos, transições, ...); por defeito o relógio atual."""
        img = self._background(s.width, s.height)
        frame, rig = self._animate(s, t)
        self.style.draw(img, frame, rig, self.palette)
        return img

    def render_with_alpha(self, s: BodyState, t: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Como render(), mas devolve também o avatar com transparência (BGRA).

        Desenha o mesmo frame sobre preto e sobre branco: a diferença dá a opacidade exata de
        cada píxel (incluindo as bordas suavizadas), para qualquer estilo. A imagem normal
        (sobre o fundo escolhido) obtém-se por composição, sem terceiro desenho."""
        frame, rig = self._animate(s, t)
        black = np.zeros((s.height, s.width, 3), np.uint8)
        white = np.full((s.height, s.width, 3), 255, np.uint8)
        self.style.draw(black, frame, rig, self.palette)
        self.style.draw(white, frame, rig, self.palette)
        img = self._background(s.width, s.height)
        bgra = np.zeros((s.height, s.width, 4), np.uint8)
        # Só a zona onde há avatar (no resto: transparente / fundo). Os estilos nunca usam preto
        # puro (os contornos são cinzento-escuros), por isso basta olhar para o desenho sobre preto.
        drawn = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        rows, cols = np.flatnonzero(drawn.any(axis=1)), np.flatnonzero(drawn.any(axis=0))
        if len(rows) == 0:
            return img, bgra
        roi = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        b, w, bg = black[roi], white[roi], img[roi]
        # alpha = 1 - (branco - preto); cor = preto / alpha (o desenho sobre preto é "pré-multiplicado").
        alpha = cv2.bitwise_not(cv2.cvtColor(cv2.subtract(w, b), cv2.COLOR_BGR2GRAY))
        a3 = cv2.merge([alpha, alpha, alpha])
        bgra[roi][..., :3] = cv2.divide(b, a3, scale=255)
        bgra[roi][..., 3] = alpha
        img[roi] = cv2.add(b, cv2.multiply(bg, cv2.bitwise_not(a3), scale=1 / 255))
        return img, bgra
