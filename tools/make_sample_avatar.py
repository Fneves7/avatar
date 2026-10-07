"""Gera um avatar de exemplo em PNG por camadas ("gato" estilo VTuber) para o estilo `png`.

As imagens são desenhadas por código só para testar a técnica; podem ser substituídas por
desenhos a sério com o mesmo nome e os mesmos pontos de encaixe (ver avatar.json).

    .venv\\Scripts\\python.exe tools\\make_sample_avatar.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "avatars" / "gato"
SS = 3  # supersampling: desenha a 3x e reduz (contornos suaves)

# Cores (BGRA).
FUR = (60, 140, 235, 255)
FUR_DARK = (40, 100, 190, 255)
CREAM = (215, 235, 250, 255)
PINK = (170, 150, 245, 255)
LINE = (45, 35, 40, 255)
WHITE = (252, 252, 252, 255)
IRIS = (90, 170, 60, 255)
PUPIL = (30, 30, 30, 255)
HOODIE = (150, 95, 60, 255)
HOODIE_DARK = (115, 70, 45, 255)
MOUTH_IN = (60, 40, 110, 255)
TONGUE = (140, 120, 235, 255)
BLUSH = (160, 150, 255, 255)
OW = 5  # espessura do contorno (px finais)


class Canvas:
    """Desenho RGBA a SS x a resolução final; coordenadas sempre na resolução final."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.img = np.zeros((h * SS, w * SS, 4), np.uint8)

    def _p(self, pts):
        return np.round(np.asarray(pts, float) * SS).astype(np.int32)

    def ellipse(self, c, axes, color, angle=0, outline=True, start=0, end=360):
        c = tuple(int(v * SS) for v in c)
        ax = tuple(int(v * SS) for v in axes)
        if outline:
            ax_o = (ax[0] + OW * SS, ax[1] + OW * SS)
            cv2.ellipse(self.img, c, ax_o, angle, start, end, LINE, -1, cv2.LINE_AA)
        cv2.ellipse(self.img, c, ax, angle, start, end, color, -1, cv2.LINE_AA)

    def poly(self, pts, color, outline=True):
        p = self._p(pts)
        if outline:
            cv2.polylines(self.img, [p], True, LINE, OW * SS * 2, cv2.LINE_AA)
        cv2.fillPoly(self.img, [p], color, cv2.LINE_AA)

    def line(self, pts, color, width, closed=False):
        cv2.polylines(self.img, [self._p(pts)], closed, color, max(1, int(width * SS)), cv2.LINE_AA)

    def circle(self, c, r, color, outline=False):
        if outline:
            cv2.circle(self.img, tuple(int(v * SS) for v in c), int((r + OW) * SS), LINE, -1, cv2.LINE_AA)
        cv2.circle(self.img, tuple(int(v * SS) for v in c), int(r * SS), color, -1, cv2.LINE_AA)

    def save(self, name: str):
        small = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / name), small)


def arc(c, rx, ry, a0, a1, n=24):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)


# ------------------------------------------------------------------ cabeça (512x512)
HW, HH = 512, 512
CENTER = (256, 290)               # centro da cara (= centro do rig.head)
EYES = {"l": (196, 272), "r": (316, 272)}
EYE_AX = (40, 44)
BROWS = {"l": (196, 210), "r": (316, 210)}
MOUTH = (256, 368)


def head_layers():
    # Pelo de trás (bochechas fofas): balança com o movimento secundário.
    c = Canvas(HW, HH)
    fluff = []
    for k, a in enumerate(np.linspace(0, 360, 28, endpoint=False)):
        r = 178 if k % 2 == 0 else 160
        fluff.append((CENTER[0] + r * math.cos(math.radians(a)), CENTER[1] + 0.86 * r * math.sin(math.radians(a))))
    c.poly(fluff, FUR_DARK)
    c.save("hair_back.png")

    # Cabeça: orelhas, cara, focinho, nariz e bigodes.
    c = Canvas(HW, HH)
    for sx in (-1, 1):
        ear = [(256 + sx * 60, 190), (256 + sx * 150, 60), (256 + sx * 165, 230)]
        c.poly(ear, FUR)
        inner = [(256 + sx * 82, 185), (256 + sx * 143, 92), (256 + sx * 150, 205)]
        c.poly(inner, PINK, outline=False)
    c.ellipse(CENTER, (158, 140), FUR)
    c.ellipse((256, 352), (78, 52), CREAM, outline=False)
    c.poly([(240, 330), (272, 330), (256, 346)], PINK, outline=False)
    c.line([(240, 330), (272, 330), (256, 346)], LINE, 3, closed=True)
    for sx in (-1, 1):
        for dy, ang in ((-6, -8), (8, 6)):
            x0 = 256 + sx * 72
            c.line([(x0, 350 + dy), (x0 + sx * 70, 350 + dy + ang)], LINE, 2.5)
    c.save("head.png")

    # Franja (tufo de pelo na testa): balança menos que o pelo de trás.
    c = Canvas(HW, HH)
    tuft = [(196, 168), (222, 128), (240, 162), (258, 112), (276, 160), (296, 126), (318, 170), (290, 182), (256, 176), (222, 182)]
    c.poly(tuft, FUR)
    c.save("hair_front.png")

    # Olhos (cada um em separado: branco, íris, meio fechado, fechado).
    for side, (ex, ey) in EYES.items():
        c = Canvas(HW, HH)
        c.ellipse((ex, ey), EYE_AX, WHITE)
        c.save(f"eye_white_{side}.png")
        c = Canvas(HW, HH)
        c.circle((ex, ey + 4), 28, IRIS)
        c.circle((ex, ey + 4), 14, PUPIL)
        c.circle((ex - 10, ey - 8), 8, WHITE)
        c.save(f"iris_{side}.png")
        c = Canvas(HW, HH)  # meio fechado: pálpebra de pelo sobre a metade de cima
        c.ellipse((ex, ey), EYE_AX, WHITE)
        c.circle((ex, ey + 10), 24, IRIS)
        c.circle((ex, ey + 10), 12, PUPIL)
        c.ellipse((ex, ey), (EYE_AX[0] + 2, EYE_AX[1] + 2), FUR, outline=False, start=180, end=360)
        c.line(arc((ex, ey), EYE_AX[0] + 2, 4, 180, 360), LINE, 6)
        c.save(f"eye_half_{side}.png")
        c = Canvas(HW, HH)  # fechado: "^" feliz
        c.line(arc((ex, ey + 10), EYE_AX[0], 22, 200, 340), LINE, 7)
        c.save(f"eye_closed_{side}.png")

    # Sobrancelhas (rodam à volta do seu centro ao franzir).
    for side, (bx, by) in BROWS.items():
        c = Canvas(HW, HH)
        c.line([(bx - 26, by + 4), (bx, by - 4), (bx + 26, by + 4)], FUR_DARK, 9)
        c.save(f"brow_{side}.png")

    # Bocas.
    mx, my = MOUTH
    c = Canvas(HW, HH)  # fechada ":3"
    c.line(np.vstack([arc((mx - 14, my), 14, 10, 0, 180), arc((mx + 14, my), 14, 10, 0, 180)]), LINE, 5)
    c.save("mouth_closed.png")
    c = Canvas(HW, HH)  # sorriso
    c.line(arc((mx, my - 14), 40, 30, 20, 160), LINE, 6)
    c.save("mouth_smile.png")
    c = Canvas(HW, HH)  # entreaberta
    c.ellipse((mx, my + 6), (20, 16), MOUTH_IN)
    c.ellipse((mx, my + 15), (12, 6), TONGUE, outline=False)
    c.save("mouth_open_small.png")
    c = Canvas(HW, HH)  # aberta
    c.ellipse((mx, my + 14), (34, 30), MOUTH_IN)
    c.ellipse((mx, my + 30), (20, 11), TONGUE, outline=False)
    c.save("mouth_open.png")

    # Rubor (opacidade segue o sorriso).
    c = Canvas(HW, HH)
    for sx in (-1, 1):
        c.ellipse((256 + sx * 112, 330), (30, 16), BLUSH, outline=False)
    c.save("blush.png")


# ------------------------------------------------------------------ corpo
def body_layers():
    # Tronco (camisola com capuz) 512x512: ombros em (96,120) e (416,120), ancas (centro) em (256,500).
    c = Canvas(512, 512)
    c.poly([(70, 150), (110, 92), (402, 92), (442, 150), (430, 510), (82, 510)], HOODIE)
    c.ellipse((256, 104), (104, 40), HOODIE_DARK)                     # capuz enrolado
    c.ellipse((256, 100), (46, 18), FUR_DARK, outline=False)          # gola (pelo do pescoço)
    c.line([(256, 150), (256, 500)], HOODIE_DARK, 6)                  # fecho
    c.poly([(160, 380), (352, 380), (372, 470), (140, 470)], HOODIE_DARK)  # bolso
    for dx, dy, r in ((0, 0, 22), (-30, -24, 10), (-10, -36, 10), (12, -36, 10), (32, -24, 10)):
        c.circle((330 + dx, 250 + dy), r, PINK)                      # pata no peito
    c.save("torso.png")

    # Pescoço 80x120.
    c = Canvas(80, 120)
    c.poly([(12, 6), (68, 6), (66, 114), (14, 114)], FUR)
    c.save("neck.png")

    # Braço (manga) 96x220 e antebraço (manga + punho) 80x200, verticais.
    c = Canvas(96, 220)
    c.poly([(10, 14), (86, 14), (80, 206), (16, 206)], HOODIE)
    c.save("upper_arm.png")
    c = Canvas(80, 200)
    c.poly([(10, 10), (70, 10), (66, 168), (14, 168)], HOODIE)
    c.poly([(12, 160), (68, 160), (68, 192), (12, 192)], HOODIE_DARK)
    c.save("forearm.png")

    # Patas 200x220: pulso em (100,200), base do dedo do meio em (100,110).
    for name, fist in (("hand_open.png", False), ("hand_fist.png", True)):
        c = Canvas(200, 220)
        c.ellipse((100, 140), (62, 66), FUR)
        if not fist:
            for ang in (-42, -15, 15, 42):
                a = math.radians(ang - 90)
                c.ellipse((100 + 70 * math.cos(a), 140 + 74 * math.sin(a)), (22, 26), FUR, angle=ang)
            for ang in (-42, -15, 15, 42):
                a = math.radians(ang - 90)
                c.ellipse((100 + 70 * math.cos(a), 140 + 74 * math.sin(a)), (11, 13), PINK, angle=ang, outline=False)
            c.ellipse((100, 152), (28, 22), PINK, outline=False)
        else:
            for k in range(3):
                c.line([(62 + k * 26, 96), (66 + k * 26, 120)], LINE, 4)
        c.save(name)


def manifest():
    return {
        "name": "Gato",
        "notas": "Coordenadas em píxeis de cada imagem. Podes trocar qualquer PNG por outro desenho "
                 "com o mesmo tamanho e os mesmos pontos de encaixe.",
        "head": {
            "canvas": [HW, HH], "center": list(CENTER), "face_width": 250,
            "eyes": {"l": list(EYES["l"]), "r": list(EYES["r"])}, "eye_radius": EYE_AX[1],
            "brows": {"l": list(BROWS["l"]), "r": list(BROWS["r"])}, "mouth": list(MOUTH),
            "parallax": {"hair_back": -0.06, "head": 0.0, "eyes": 0.14, "brows": 0.15, "mouth": 0.12,
                         "hair_front": 0.08},
        },
        "torso": {"file": "torso.png", "shoulders": [[96, 120], [416, 120]], "hips": [256, 500]},
        "neck": {"file": "neck.png", "start": [40, 6], "end": [40, 114], "width": 56, "thickness": 0.42},
        "upper_arm": {"file": "upper_arm.png", "start": [48, 14], "end": [48, 206], "width": 76, "thickness": 0.26},
        "forearm": {"file": "forearm.png", "start": [40, 10], "end": [40, 190], "width": 60, "thickness": 0.2},
        "hand": {"open": "hand_open.png", "fist": "hand_fist.png", "wrist": [100, 200], "knuckle": [100, 110]},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head_layers()
    body_layers()
    (OUT / "avatar.json").write_text(json.dumps(manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Avatar de exemplo gerado em {OUT} ({len(list(OUT.glob('*.png')))} imagens)")


if __name__ == "__main__":
    main()
