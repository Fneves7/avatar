"""Utilitários de desenho partilhados pelos estilos: paletas, conversões e formas com contorno."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

AA = cv2.LINE_AA


@dataclass(frozen=True)
class Palette:
    name: str
    bg_top: tuple
    bg_bottom: tuple
    skin: tuple
    skin_light: tuple
    hair: tuple
    shirt: tuple
    shirt_dark: tuple
    iris: tuple
    lips: tuple
    mouth: tuple = (40, 30, 70)
    outline: tuple = (40, 30, 35)
    eye_white: tuple = (250, 250, 250)
    blush: tuple = (150, 140, 255)


# Cores em BGR.
PALETTES = [
    Palette("Clássico", (70, 45, 35), (140, 100, 60), (150, 190, 235), (175, 210, 245), (35, 50, 90),
            (180, 110, 40), (130, 70, 25), (120, 80, 40), (110, 110, 200)),
    Palette("Neon", (60, 20, 40), (120, 40, 90), (200, 220, 160), (220, 235, 190), (200, 60, 230),
            (60, 200, 255), (30, 140, 200), (200, 120, 0), (180, 80, 230)),
    Palette("Floresta", (40, 60, 30), (90, 130, 70), (110, 150, 200), (140, 175, 220), (25, 30, 40),
            (70, 140, 60), (40, 90, 35), (40, 90, 50), (90, 90, 170)),
    Palette("Robô", (50, 50, 50), (110, 110, 110), (205, 205, 200), (230, 230, 225), (90, 90, 95),
            (60, 60, 200), (30, 30, 140), (220, 200, 0), (150, 150, 160)),
]


def _ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _poly(pts) -> np.ndarray:
    return np.round(np.asarray(pts)[:, :2]).astype(np.int32)


def _bezier(p0, p1, p2, n: int = 8) -> list[np.ndarray]:
    t = np.linspace(0, 1, n)[:, None]
    return list((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)


def _scale_about(pts: np.ndarray, center: np.ndarray, k: float) -> np.ndarray:
    return center + (pts - center) * k


def faded(img: np.ndarray, alpha: float, draw) -> None:
    """Desenha com opacidade alpha (só copia a imagem quando é mesmo preciso)."""
    if alpha >= 0.999:
        draw(img)
        return
    if alpha <= 0.001:
        return
    layer = img.copy()
    draw(layer)
    cv2.addWeighted(layer, alpha, img, 1.0 - alpha, 0, dst=img)


class Group:
    """Conjunto de formas com o mesmo preenchimento; o contorno de todas é desenhado
    primeiro para que as junções (ex.: cotovelo) fiquem sem linhas internas."""

    def __init__(self):
        self.shapes: list[tuple] = []

    def poly(self, pts):
        self.shapes.append(("poly", _poly(pts)))
        return self

    def circle(self, c, r):
        self.shapes.append(("circle", _ip(c), max(1, int(r))))
        return self

    def line(self, a, b, w):
        self.shapes.append(("line", _ip(a), _ip(b), max(1, int(w))))
        return self

    def draw(self, img, color, outline, ow: int):
        for s in self.shapes:  # contorno
            if s[0] == "poly":
                cv2.fillPoly(img, [s[1]], outline, AA)
                cv2.polylines(img, [s[1]], True, outline, ow * 2, AA)
            elif s[0] == "circle":
                cv2.circle(img, s[1], s[2] + ow, outline, -1, AA)
            else:
                cv2.line(img, s[1], s[2], outline, s[3] + ow * 2, AA)
        for s in self.shapes:  # preenchimento
            if s[0] == "poly":
                cv2.fillPoly(img, [s[1]], color, AA)
            elif s[0] == "circle":
                cv2.circle(img, s[1], s[2], color, -1, AA)
            else:
                cv2.line(img, s[1], s[2], color, s[3], AA)
