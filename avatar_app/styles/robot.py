"""Estilo "robô": desenhado só a partir do Rig (não lê a malha do MediaPipe).

Serve também de exemplo de como fazer um avatar novo: cabeça em caixa com visor, olhos
LED (abertura e olhar do rig), sobrancelhas LED (levantar/franzir), grelha na boca (abre
e sorri), antena que balança com o movimento secundário, luz no peito que pulsa com a
respiração e garras que seguem os dedos.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..animation import AnimFrame
from ..drawing import AA, Group, Palette, _ip, _poly, faded
from ..landmarks import FINGERS
from ..rig import Rig, RigHead

METAL = (200, 200, 206)
METAL_DARK = (130, 130, 140)
VISOR = (45, 35, 30)


def _bright(color, k=1.8, add=60):
    return tuple(int(min(255, c * k + add)) for c in color)


def _rounded_rect(w: float, h: float, r: float, n: int = 6) -> np.ndarray:
    """Retângulo de cantos redondos centrado em (0, 0), em coordenadas locais."""
    r = min(r, w / 2, h / 2)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                       (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(pts)


class RobotStyle:
    name = "robo"

    def draw(self, img: np.ndarray, frame: AnimFrame, rig: Rig, palette: Palette) -> None:
        ow = max(2, rig.width // 320)
        self.p = palette
        self.led = _bright(palette.iris)
        if rig.shoulders is not None:
            self._draw_body(img, rig, ow)
        if rig.head is not None:
            faded(img, rig.head.alpha, lambda im: self._draw_head(im, rig, rig.head, ow))
        for side, (pts, alpha) in rig.hands.items():
            faded(img, alpha, lambda im, pts=pts: self._draw_claw(im, pts, ow))

    # ------------------------------------------------------------------ corpo
    def _draw_body(self, img, rig: Rig, ow: int):
        p = self.p
        sw = rig.shoulder_width
        ls, rs = rig.shoulders
        lh, rh = rig.hips
        axis = (rs - ls) / sw
        # Pescoço em segmentos.
        if rig.head is not None:
            top = rig.head.center + np.array([0.0, 0.45 * rig.head.height])
            base = rig.neck_base
            for k in range(3):
                c = base + (top - base) * (k + 0.5) / 3
                Group().line(c - axis * 0.12 * sw, c + axis * 0.12 * sw, 0.07 * sw).draw(img, METAL_DARK, p.outline, ow)
        # Tronco: placa com ombros redondos.
        body = [ls - axis * 0.12 * sw, rs + axis * 0.12 * sw, rh + axis * 0.04 * sw, lh - axis * 0.04 * sw]
        g = Group().poly(body).circle(ls - axis * 0.02 * sw, 0.16 * sw).circle(rs + axis * 0.02 * sw, 0.16 * sw)
        g.draw(img, p.shirt, p.outline, ow)
        # Painel no peito com uma luz que pulsa com a respiração.
        chest = (ls + rs) / 2 + (lh + rh - ls - rs) / 2 * 0.28
        panel = _rounded_rect(0.5 * sw, 0.28 * sw, 0.05 * sw)
        Group().poly(chest + panel).draw(img, p.shirt_dark, p.outline, max(1, ow // 2))
        glow = 0.55 + 0.45 * rig.breath
        light = tuple(int(c * glow) for c in self.led)
        cv2.circle(img, _ip(chest + axis * 0.12 * sw), max(2, int(0.06 * sw)), light, -1, AA)
        for k in range(3):
            y = chest + np.array([-0.17 * sw + k * 0.08 * sw, 0.06 * sw])
            cv2.line(img, _ip(y), _ip(y + np.array([0.0, -0.12 * sw])), METAL, max(1, ow), AA)
        # Braços por cima do tronco (tubos com articulações).
        for side, arm in rig.arms.items():
            g = Group().line(arm.shoulder, arm.elbow, 0.17 * sw)
            if arm.forearm_visible:
                g.line(arm.elbow, arm.wrist, 0.14 * sw)
            g.draw(img, METAL, p.outline, ow)
            Group().circle(arm.elbow, 0.09 * sw).draw(img, METAL_DARK, p.outline, ow)
            if arm.forearm_visible and side not in rig.hands:
                Group().circle(arm.wrist, 0.1 * sw).draw(img, METAL_DARK, p.outline, ow)  # punho fechado

    def _draw_claw(self, img, hand: np.ndarray, ow: int):
        p = self.p
        hs = float(np.linalg.norm(hand[0, :2] - hand[9, :2])) or 1.0
        palm = hand[[0, 5, 9, 13, 17], :2].mean(axis=0)
        Group().circle(palm, 0.45 * hs).draw(img, METAL_DARK, p.outline, ow)
        g = Group()
        for chain in FINGERS:
            for a, b in zip(chain[:-1], chain[1:]):
                g.line(hand[a, :2], hand[b, :2], 0.14 * hs)
        g.draw(img, METAL, p.outline, ow)
        for chain in FINGERS:
            for j in chain[1:]:
                cv2.circle(img, _ip(hand[j, :2]), max(1, int(0.05 * hs)), METAL_DARK, -1, AA)

    # ------------------------------------------------------------------ cabeça
    def _draw_head(self, img, rig: Rig, h: RigHead, ow: int):
        p = self.p
        w, ht = 1.3 * h.size, 1.15 * h.height
        c, s = math.cos(h.roll), math.sin(h.roll)
        rot = np.array([[c, -s], [s, c]])
        center = h.center + rot @ np.array([0.0, -0.06 * ht])

        def tr(local) -> np.ndarray:
            return center + np.atleast_2d(local) @ rot.T

        # Antena: a bola balança com a mola do cabelo (movimento secundário).
        base = tr([0.0, -ht / 2])[0]
        tip = tr([0.0, -ht / 2 - 0.32 * ht])[0] + rig.hair_offset * 1.5
        ang = rig.hair_rot * 2
        tip = base + np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) @ (tip - base)
        Group().line(base, tip, 0.05 * w).draw(img, METAL_DARK, p.outline, ow)
        Group().circle(tip, 0.08 * w).draw(img, self.led, p.outline, ow)
        # Orelhas (parafusos) e caixa da cabeça.
        ears = Group().circle(tr([-w / 2, 0.05 * ht])[0], 0.12 * w).circle(tr([w / 2, 0.05 * ht])[0], 0.12 * w)
        ears.draw(img, METAL_DARK, p.outline, ow)
        Group().poly(tr(_rounded_rect(w, ht, 0.18 * w))).draw(img, METAL, p.outline, ow)
        # Visor (a rodar para o lado com o yaw dá sensação de 3D).
        shift = math.sin(h.yaw) * 0.12 * w
        visor = _rounded_rect(0.86 * w, 0.42 * ht, 0.1 * w) + np.array([shift, -0.12 * ht])
        cv2.fillPoly(img, [_poly(tr(visor))], VISOR, AA)

        # Olhos LED: a altura segue a abertura do olho; deslocam-se com o olhar.
        signs = [1.0 if (ec - h.center) @ rot[:, 0] > 0 else -1.0 for ec in h.eye_centers]
        look = np.clip(h.look, -1, 1) * np.array([0.06 * w, 0.04 * ht])
        for i, sx in enumerate(signs):
            ex = sx * 0.2 * w + shift + look[0]
            ey = -0.12 * ht + look[1]
            openness = float(np.clip(h.eye_open[i], 0.0, 1.4))
            eh = max(0.012 * ht, 0.14 * ht * openness)
            eye = _rounded_rect(0.18 * w, eh, eh / 2) + np.array([ex, ey])
            cv2.fillPoly(img, [_poly(tr(eye))], self.led, AA)
            # Sobrancelha LED: sobe ao levantar; ao franzir a ponta de dentro desce.
            lift = -0.06 * ht * max(h.brow, 0.0)
            frown = 0.05 * ht * max(-h.brow, 0.0)
            inner = np.array([sx * 0.1 * w + shift, -0.26 * ht + lift + frown])
            outer = np.array([sx * 0.3 * w + shift, -0.26 * ht + lift])
            cv2.line(img, _ip(tr(inner)[0]), _ip(tr(outer)[0]), self.led, max(2, int(0.025 * ht)), AA)

        # Boca em grelha: abre com a boca real e curva para cima ao sorrir.
        mw = 0.42 * w * (1 + 0.25 * h.mouth_smile)
        mh = 0.035 * ht + 0.16 * ht * h.mouth_open
        xs = np.linspace(-mw / 2, mw / 2, 9)
        curve = -0.07 * ht * h.mouth_smile * (2 * xs / mw) ** 2
        y0 = 0.25 * ht
        top = np.stack([xs + shift, y0 + curve - mh / 2], axis=1)
        bottom = np.stack([xs[::-1] + shift, (y0 + curve + mh / 2)[::-1]], axis=1)
        mouth = _poly(tr(np.vstack([top, bottom])))
        cv2.fillPoly(img, [mouth], VISOR, AA)
        for k in range(1, 8):
            a = tr([xs[k] + shift, y0 + curve[k] - mh / 2])[0]
            b = tr([xs[k] + shift, y0 + curve[k] + mh / 2])[0]
            cv2.line(img, _ip(a), _ip(b), self.led if h.mouth_open > 0.15 else METAL_DARK, max(1, ow // 2), AA)
        cv2.polylines(img, [mouth], True, p.outline, max(1, ow // 2), AA)
