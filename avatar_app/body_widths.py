"""Afinação conservadora das larguras dos braços e do tronco pela silhueta (opção --body-widths).

Uma primeira versão redesenhava braços, ombros e tronco a partir da silhueta da pessoa e ficou
mal com a webcam real: perto da câmara os ombros reais ficam muito acima das articulações, e um
braço à frente do tronco dá larguras falsas. Esta versão mantém o desenho e só ajusta a largura
de cada parte por um fator pequeno:

  * mede-se a largura real na perpendicular ao osso (braço, antebraço) ou à linha do peito
    (tronco), em três pontos, e usa-se a mediana;
  * fator = largura medida / largura esperada (a proporção do desenho), limitado a
    [0,85; 1,15] nos braços e [0,9; 1,1] no tronco, e suavizado ao longo do tempo;
  * a medição é rejeitada (fica o fator anterior) quando o osso está pouco visível, quando
    passa por cima do tronco, ou quando o varrimento não sai da silhueta (braço encostado ao
    corpo ou ao fundo).
"""
from __future__ import annotations

import numpy as np

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24

# Largura esperada de cada parte (* largura entre ombros): a mesma proporção do desenho cartoon.
EXPECTED = {"upper": 0.28, "fore": 0.2, "torso": 1.2}
LIMITS = {"upper": (0.85, 1.15), "fore": (0.85, 1.15), "torso": (0.9, 1.1)}
SMOOTH = 0.15      # peso de cada medição nova (média exponencial)
MIN_VIS = 0.6      # visibilidade mínima das duas articulações do osso
MASK_THR = 0.5


def _run(mask: np.ndarray, p: np.ndarray, d: np.ndarray, max_len: float) -> float | None:
    """Comprimento da silhueta a partir de p na direção d (px), ou None se não sair dela."""
    h, w = mask.shape
    step = 1.0
    n = int(max_len / step)
    for i in range(1, n + 1):
        q = p + d * (i * step)
        x, y = int(round(q[0])), int(round(q[1]))
        if not (0 <= x < w and 0 <= y < h):
            return None  # saiu da imagem ainda dentro da silhueta: não se sabe a largura
        if mask[y, x] < MASK_THR:
            return i * step
    return None


def _inside(poly: np.ndarray, p: np.ndarray) -> bool:
    """Ponto dentro de um polígono convexo (ordem qualquer, sentido consistente)."""
    sign = 0.0
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if c != 0:
            if sign == 0:
                sign = np.sign(c)
            elif np.sign(c) != sign:
                return False
    return True


def measure_width(mask: np.ndarray, a: np.ndarray, b: np.ndarray, max_half: float) -> float | None:
    """Largura da silhueta na perpendicular ao segmento a->b (mediana de 3 pontos)."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 4:
        return None
    u = d / length
    n = np.array([-u[1], u[0]])
    widths = []
    for t in (0.35, 0.5, 0.65):
        p = a + d * t
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or mask[y, x] < MASK_THR:
            continue
        r1, r2 = _run(mask, p, n, max_half), _run(mask, p, -n, max_half)
        if r1 is not None and r2 is not None:
            widths.append(r1 + r2)
    return float(np.median(widths)) if len(widths) >= 2 else None


class BodyWidths:
    """Fatores de largura (1.0 = desenho normal), por parte: upper_15, fore_15, upper_16, fore_16, torso."""

    def __init__(self):
        self.factors: dict[str, float] = {}
        self.last_rejected: list[str] = []

    def reset(self) -> None:
        self.factors = {}

    def update(self, mask: np.ndarray | None, pose: np.ndarray | None, vis: np.ndarray | None) -> dict[str, float]:
        self.last_rejected = []
        if mask is None or pose is None or vis is None:
            return dict(self.factors)
        p = pose[:, :2]
        sw = float(np.linalg.norm(p[L_SHOULDER] - p[R_SHOULDER]))
        if sw < 10:
            return dict(self.factors)
        ls, rs = p[L_SHOULDER], p[R_SHOULDER]
        down = np.array([-(rs - ls)[1], (rs - ls)[0]]) / sw
        if down[1] < 0:
            down = -down
        lh, rh = ls + down * 1.4 * sw, rs + down * 1.4 * sw  # ancas estimadas (como no desenho)
        torso = np.array([ls, rs, rh, lh])

        for side, (sh, el) in ((L_WRIST, (L_SHOULDER, L_ELBOW)), (R_WRIST, (R_SHOULDER, R_ELBOW))):
            for part, (i, j) in (("upper", (sh, el)), ("fore", (el, side))):
                key = f"{part}_{side}"
                a, b = p[i], p[j]
                mid = (a + b) / 2
                if min(vis[i], vis[j]) < MIN_VIS or _inside(torso, mid):
                    self.last_rejected.append(key)
                    continue
                w = measure_width(mask, a, b, 0.6 * sw)
                self._add(key, part, w, sw)

        # Tronco: largura ao nível do peito, ao longo da linha dos ombros. Não se mede com um
        # pulso ou cotovelo perto dessa linha (o braço encostado alarga a silhueta).
        chest = (ls + rs) / 2 + down * 0.35 * sw
        near = [k for k in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
                if vis[k] > 0.3 and abs((p[k] - chest) @ down) < 0.25 * sw]
        if near:
            self.last_rejected.append("torso")
        else:
            axis = (rs - ls) / sw
            r1 = _run(mask, chest, axis, 1.0 * sw)
            r2 = _run(mask, chest, -axis, 1.0 * sw)
            self._add("torso", "torso", None if r1 is None or r2 is None else r1 + r2, sw)
        return dict(self.factors)

    def _add(self, key: str, part: str, width: float | None, sw: float) -> None:
        if width is None:
            self.last_rejected.append(key)
            return
        lo, hi = LIMITS[part]
        f = float(np.clip(width / (EXPECTED[part] * sw), lo, hi))
        old = self.factors.get(key, 1.0)
        self.factors[key] = old + (f - old) * SMOOTH
