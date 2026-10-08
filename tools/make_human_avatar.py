"""Gera o avatar PNG "pessoa": personagem humana estilo anime/VTuber, por camadas, para o estilo png.

Mais trabalhado do que o gato de exemplo: sombreado cel (sombras recortadas à forma de baixo),
cabelo com madeixas e reflexo, olhos com pestanas e brilhos, rubor suave, camisola com gola
e mãos com dedos. Usa os mesmos pontos de encaixe do formato avatar.json (ver avatars/gato).
As imagens podem ser substituídas por desenhos feitos à mão com o mesmo nome e tamanho.

    .venv\\Scripts\\python.exe tools\\make_human_avatar.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "avatars" / "pessoa"
SS = 4  # supersampling: desenha a 4x e reduz (contornos suaves)

# Cores (BGRA).
SKIN = (196, 214, 247, 255)
SKIN_SHADE = (160, 178, 226, 255)
SKIN_DEEP = (130, 145, 200, 255)
HAIR = (39, 47, 74, 255)
HAIR_SHADE = (26, 30, 50, 255)
HAIR_LIGHT = (72, 92, 140, 255)
LINE = (38, 30, 42, 255)
WHITE = (250, 248, 248, 255)
EYE_SHADE = (222, 210, 206, 255)
IRIS = (70, 105, 160, 255)
IRIS_DARK = (40, 60, 100, 255)
IRIS_LIGHT = (120, 170, 220, 255)
PUPIL = (30, 28, 36, 255)
LIP = (110, 110, 200, 255)
MOUTH_IN = (60, 45, 120, 255)
TONGUE = (125, 125, 230, 255)
TEETH = (245, 245, 250, 255)
BLUSH = (150, 140, 250, 255)
SWEATER = (138, 92, 62, 255)
SWEATER_SHADE = (104, 66, 44, 255)
SWEATER_LIGHT = (165, 122, 92, 255)
SHIRT = (240, 238, 236, 255)
SHIRT_SHADE = (205, 200, 198, 255)
OW = 4  # espessura do contorno (px finais)


def bez(*pts, n: int = 24) -> np.ndarray:
    """Curva de Bézier (qualquer grau) pelos pontos de controlo."""
    p = np.asarray(pts, float)
    out = []
    for t in np.linspace(0, 1, n):
        q = p.copy()
        while len(q) > 1:
            q = q[:-1] * (1 - t) + q[1:] * t
        out.append(q[0])
    return np.array(out)


def mirror(pts, cx: float = 256) -> np.ndarray:
    p = np.asarray(pts, float).copy()
    p[:, 0] = 2 * cx - p[:, 0]
    return p[::-1]


class Canvas:
    """Desenho RGBA a SS x a resolução final; coordenadas sempre na resolução final."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.img = np.zeros((h * SS, w * SS, 4), np.uint8)

    def _p(self, pts):
        return np.round(np.asarray(pts, float) * SS).astype(np.int32)

    def poly(self, pts, color, outline: bool = True, width: float = OW):
        p = self._p(pts)
        if outline:
            cv2.polylines(self.img, [p], True, LINE, int(width * SS * 2), cv2.LINE_AA)
        cv2.fillPoly(self.img, [p], color, cv2.LINE_AA)

    def ellipse(self, c, axes, color, outline: bool = True, angle: float = 0, start=0, end=360):
        cc = tuple(int(v * SS) for v in c)
        ax = tuple(max(1, int(v * SS)) for v in axes)
        if outline:
            cv2.ellipse(self.img, cc, (ax[0] + OW * SS, ax[1] + OW * SS), angle, start, end, LINE, -1, cv2.LINE_AA)
        cv2.ellipse(self.img, cc, ax, angle, start, end, color, -1, cv2.LINE_AA)

    def line(self, pts, color, width: float, closed: bool = False):
        cv2.polylines(self.img, [self._p(pts)], closed, color, max(1, int(width * SS)), cv2.LINE_AA)

    def taper(self, pts, w0: float, w1: float, color):
        """Traço que afina de w0 para w1 (sobrancelhas, pestanas, madeixas)."""
        p = np.asarray(pts, float)
        d = np.gradient(p, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        w = np.linspace(w0, w1, len(p))[:, None] / 2
        self.poly(np.vstack([p + n * w, (p - n * w)[::-1]]), color, outline=False)

    def atop(self, draw, blur: float = 0.0):
        """Desenha só por cima do que já existe (sombras e brilhos recortados à forma)."""
        layer = Canvas(self.w, self.h)
        draw(layer)
        top = layer.img.astype(np.float32)
        if blur:
            k = int(blur * SS) * 2 + 1
            top = cv2.GaussianBlur(top, (k, k), 0)
        a = top[..., 3:4] / 255.0 * (self.img[..., 3:4] / 255.0)
        rgb = np.where(top[..., 3:4] > 0, top[..., :3] / np.maximum(top[..., 3:4] / 255.0, 1e-6), 0)
        self.img[..., :3] = np.clip(self.img[..., :3] * (1 - a) + rgb * a, 0, 255).astype(np.uint8)

    def save(self, name: str):
        small = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / name), small)


# ------------------------------------------------------------------ cabeça (512x512)
HW, HH = 512, 512
CENTER = (256, 290)           # centro da cara (= centro do rig.head)
EYES = {"l": (198, 294), "r": (314, 294)}
EYE_R = 30
BROWS = {"l": (196, 246), "r": (316, 246)}
MOUTH = (256, 392)

# Contorno da cara: têmporas -> bochechas -> queixo (estilo anime, queixo afilado).
FACE_L = bez((134, 220), (128, 330), (186, 414), (256, 442), n=30)
FACE = np.vstack([FACE_L, mirror(FACE_L)[1:], bez((378, 220), (372, 110), (140, 110), (134, 220), n=30)[1:-1]])

# Franja: pontas das madeixas por cima da testa (as do meio mais acima, para ver as sobrancelhas).
FRINGE_TIPS = [(390, 262), (372, 214), (356, 250), (334, 196), (310, 232), (290, 186), (266, 222),
               (244, 182), (222, 226), (200, 190), (178, 238), (158, 204), (140, 256), (122, 262)]


def head_layers():
    # Cabelo de trás: comprido até aos ombros, com pontas. Balança com o movimento secundário.
    c = Canvas(HW, HH)
    back = np.vstack([
        bez((112, 300), (76, 20), (436, 20), (400, 300), n=40),
        [(410, 420), (396, 470), (380, 430), (360, 490), (346, 440)],
        [(166, 440), (152, 490), (132, 430), (116, 470), (102, 420)],
    ])
    c.poly(back, HAIR)
    c.atop(lambda l: l.poly([(256, 60), (420, 200), (420, 512), (330, 512), (300, 200)], HAIR_SHADE, outline=False))
    for x0 in (150, 200, 312, 362):  # madeixas (linhas finas)
        c.atop(lambda l, x0=x0: l.line(bez((x0, 260), (x0 + (x0 - 256) * 0.15, 380), (x0 + (x0 - 256) * 0.1, 470)),
                                     HAIR_SHADE, 3))
    c.save("hair_back.png")

    # Cabeça: orelhas, cara, sombra da franja, nariz.
    c = Canvas(HW, HH)
    for sx in (-1, 1):
        c.ellipse((256 + sx * 126, 304), (20, 34), SKIN, angle=sx * 8)
        c.atop(lambda l, sx=sx: l.ellipse((256 + sx * 128, 306), (9, 20), SKIN_SHADE, outline=False, angle=sx * 8))
    c.poly(FACE, SKIN)
    # Sombra da franja na testa e sombra suave nas laterais do queixo.
    shade = [(p[0], p[1] + 16) for p in FRINGE_TIPS] + [(390, 120), (122, 120)]
    c.atop(lambda l: l.poly(shade, SKIN_SHADE, outline=False), blur=1.5)
    c.atop(lambda l: l.poly(np.vstack([bez((140, 330), (170, 410), (256, 446), n=20), [(256, 470), (120, 470)]]),
                            SKIN_SHADE, outline=False), blur=3)
    # Nariz: pequena sombra e um ponto de luz.
    c.line(bez((262, 336), (266, 352), (256, 358), n=10), SKIN_DEEP, 3)
    c.ellipse((250, 340), (3, 5), (225, 235, 252, 255), outline=False)
    c.save("head.png")

    # Franja e madeixas laterais (por cima de tudo, balançam metade do cabelo de trás).
    c = Canvas(HW, HH)
    cap = np.vstack([bez((122, 262), (88, 40), (424, 40), (390, 262), n=40),
                     FRINGE_TIPS[1:-1]])
    c.poly(cap, HAIR)
    for sx in (-1, 1):  # madeixas que descem pelas bochechas
        lock = [(256 + sx * 136, 230), (256 + sx * 112, 250), (256 + sx * 116, 340), (256 + sx * 128, 420),
                (256 + sx * 146, 360), (256 + sx * 150, 260)]
        c.poly(lock, HAIR)
    c.atop(lambda l: l.poly([(122, 262), (140, 170), (200, 120), (190, 240)], HAIR_SHADE, outline=False))
    # Reflexo ("anel de anjo"): traços claros em arco.
    for a0 in range(200, 340, 22):
        seg = [(256 + 118 * math.cos(math.radians(a)), 168 + 52 * math.sin(math.radians(a))) for a in (a0, a0 + 12)]
        c.atop(lambda l, seg=seg: l.taper(bez(seg[0], seg[1], n=8), 9, 3, HAIR_LIGHT))
    for tip in FRINGE_TIPS[2:-2:2]:  # linhas de separação das madeixas
        c.atop(lambda l, tip=tip: l.line(bez((tip[0] + (256 - tip[0]) * 0.2, 120), tip, n=10), HAIR_SHADE, 2.5))
    c.save("hair_front.png")

    # Olhos: branco (com pestanas), íris, meio fechado e fechado.
    for side, (ex, ey) in EYES.items():
        out = 1 if side == "r" else -1  # lado de fora do olho

        def lashes(cv, ex=ex, ey=ey, out=out, k=1.0):
            top = [(ex + EYE_R * 1.05 * math.cos(math.radians(a)), ey - EYE_R * 0.95 * k * math.sin(math.radians(a)))
                   for a in np.linspace(170, 10, 20)]
            if out < 0:
                top = top[::-1]
            cv.taper(top, 4, 9, LINE)
            tip = top[-1]
            cv.taper([tip, (tip[0] + out * 9, tip[1] - 5)], 7, 1, LINE)  # risco de fora

        c = Canvas(HW, HH)
        c.ellipse((ex, ey), (EYE_R, EYE_R * 0.92), WHITE, outline=False)
        c.atop(lambda l, ex=ex, ey=ey: l.ellipse((ex, ey - 22), (EYE_R + 4, 14), EYE_SHADE, outline=False))
        lashes(c)
        c.line(bez((ex - 14, ey + 27), (ex, ey + 30), (ex + 14, ey + 27), n=8), LINE, 2)
        c.save(f"eye_white_{side}.png")

        c = Canvas(HW, HH)
        c.ellipse((ex, ey + 3), (19, 24), IRIS, outline=False)
        c.atop(lambda l, ex=ex, ey=ey: l.ellipse((ex, ey - 8), (22, 14), IRIS_DARK, outline=False))
        c.atop(lambda l, ex=ex, ey=ey: l.ellipse((ex, ey + 16), (13, 8), IRIS_LIGHT, outline=False), blur=2)
        c.ellipse((ex, ey + 3), (8, 11), PUPIL, outline=False)
        c.ellipse((ex - 7, ey - 7), (6, 7), WHITE, outline=False)
        c.ellipse((ex + 8, ey + 12), (3, 3), WHITE, outline=False)
        c.save(f"iris_{side}.png")

        c = Canvas(HW, HH)  # meio fechado: pálpebra por cima, pestanas a meio
        c.ellipse((ex, ey + 8), (EYE_R, EYE_R * 0.55), WHITE, outline=False)
        c.ellipse((ex, ey + 14), (15, 13), IRIS, outline=False)
        c.ellipse((ex, ey + 14), (7, 7), PUPIL, outline=False)
        lashes(c, ey=ey + 12, k=0.45)
        c.save(f"eye_half_{side}.png")

        c = Canvas(HW, HH)  # fechado: arco das pestanas virado para baixo
        arc = [(ex + EYE_R * math.cos(math.radians(a)), ey + 8 + 9 * math.sin(math.radians(a)))
               for a in np.linspace(180, 0, 18)]
        c.taper(arc if out > 0 else arc[::-1], 4, 8, LINE)
        c.save(f"eye_closed_{side}.png")

    # Sobrancelhas: traço que afina para fora.
    for side, (bx, by) in BROWS.items():
        out = 1 if side == "r" else -1
        c = Canvas(HW, HH)
        c.taper(bez((bx - out * 30, by + 4), (bx, by - 8), (bx + out * 34, by), n=16), 8, 3, HAIR_SHADE)
        c.save(f"brow_{side}.png")

    # Bocas.
    mx, my = MOUTH
    c = Canvas(HW, HH)
    c.taper(bez((mx - 14, my), (mx, my + 4), (mx + 14, my), n=12), 3, 3, LINE)
    c.save("mouth_closed.png")
    c = Canvas(HW, HH)
    c.taper(bez((mx - 26, my - 8), (mx, my + 14), (mx + 26, my - 8), n=16), 3.5, 3.5, LINE)
    c.save("mouth_smile.png")
    c = Canvas(HW, HH)
    shape = np.vstack([bez((mx - 16, my - 4), (mx, my), (mx + 16, my - 4), n=10),
                       bez((mx + 16, my - 4), (mx + 12, my + 18), (mx - 12, my + 18), (mx - 16, my - 4), n=14)[1:]])
    c.poly(shape, MOUTH_IN, width=2.5)
    c.atop(lambda l: l.ellipse((mx, my + 14), (10, 6), TONGUE, outline=False))
    c.save("mouth_open_small.png")
    c = Canvas(HW, HH)
    shape = np.vstack([bez((mx - 24, my - 8), (mx, my - 2), (mx + 24, my - 8), n=12),
                       bez((mx + 24, my - 8), (mx + 18, my + 32), (mx - 18, my + 32), (mx - 24, my - 8), n=16)[1:]])
    c.poly(shape, MOUTH_IN, width=2.5)
    c.atop(lambda l: l.poly([(mx - 24, my - 12), (mx + 24, my - 12), (mx + 20, my - 1), (mx - 20, my - 1)], TEETH,
                            outline=False))
    c.atop(lambda l: l.ellipse((mx, my + 24), (14, 8), TONGUE, outline=False))
    c.save("mouth_open.png")

    # Rubor suave com três riscos (opacidade segue o sorriso).
    c = Canvas(HW, HH)
    for sx in (-1, 1):
        c.ellipse((256 + sx * 84, 350), (30, 13), BLUSH, outline=False)
    c.img = cv2.GaussianBlur(c.img, (6 * SS + 1, 6 * SS + 1), 0)
    for sx in (-1, 1):
        for k in (-1, 0, 1):
            x = 256 + sx * 84 + k * 12
            c.line([(x + 4, 343), (x - 4, 357)], (120, 110, 225, 255), 2)
    c.save("blush.png")


# ------------------------------------------------------------------ corpo
def body_layers():
    # Tronco 512x512: ombros em (96,120) e (416,120), ancas (centro) em (256,500).
    c = Canvas(512, 512)
    left = np.vstack([bez((256, 92), (150, 92), (92, 100), (66, 150), n=16), [(78, 330), (88, 512)]])
    c.poly(np.vstack([left, mirror(left)]), SWEATER)
    c.atop(lambda l: l.poly([(330, 92), (450, 140), (440, 512), (380, 512), (360, 200)], SWEATER_SHADE, outline=False))
    c.atop(lambda l: l.poly([(70, 150), (120, 120), (110, 512), (80, 512)], SWEATER_SHADE, outline=False))
    c.atop(lambda l: l.line(bez((140, 112), (200, 100), (240, 104), n=10), SWEATER_LIGHT, 6))
    # Gola da camisa em V e decote da camisola.
    c.poly([(206, 92), (256, 170), (306, 92)], SHIRT, width=3)
    c.atop(lambda l: l.poly([(256, 120), (306, 92), (256, 170)], SHIRT_SHADE, outline=False))
    for sx in (-1, 1):
        c.poly([(256, 108), (256 + sx * 52, 90), (256 + sx * 40, 140)], SHIRT, width=3)  # pontas da gola
    c.line([(196, 92), (256, 182), (316, 92)], SWEATER_SHADE, 9)          # canelado do decote
    c.line([(88, 470), (424, 470)], SWEATER_SHADE, 4)                      # bainha
    c.save("torso.png")

    # Pescoço 80x120, com a sombra do queixo em cima.
    c = Canvas(80, 120)
    c.poly([(14, 4), (66, 4), (64, 116), (16, 116)], SKIN)
    c.atop(lambda l: l.poly([(0, 0), (80, 0), (80, 40), (40, 52), (0, 40)], SKIN_SHADE, outline=False), blur=2)
    c.save("neck.png")

    # Braço (manga) 96x220 e antebraço 80x200 (manga, punho canelado e pulso).
    c = Canvas(96, 220)
    c.poly([(10, 14), (86, 14), (80, 210), (16, 210)], SWEATER)
    c.atop(lambda l: l.poly([(60, 0), (96, 0), (96, 220), (64, 220)], SWEATER_SHADE, outline=False))
    c.save("upper_arm.png")
    c = Canvas(80, 200)
    c.poly([(26, 150), (54, 150), (52, 198), (28, 198)], SKIN)
    c.poly([(8, 6), (72, 6), (66, 156), (14, 156)], SWEATER)
    c.atop(lambda l: l.poly([(48, 0), (80, 0), (80, 200), (50, 200)], SWEATER_SHADE, outline=False))
    c.poly([(12, 140), (68, 140), (66, 170), (14, 170)], SWEATER_SHADE, width=3)
    for x in range(20, 64, 8):
        c.line([(x, 144), (x, 166)], SWEATER, 2)
    c.save("forearm.png")

    # Mãos 200x220: pulso em (100,200), base do dedo do meio em (100,110). Polegar à esquerda.
    for name, fist in (("hand_open.png", False), ("hand_fist.png", True)):
        c = Canvas(200, 220)
        if not fist:
            for x, top, w in ((70, 40, 17), (95, 24, 18), (120, 32, 17), (143, 56, 15)):
                c.poly(_capsule((x, 120), (x + (x - 100) * 0.12, top), w), SKIN)
            c.poly(_capsule((64, 176), (24, 120), 19), SKIN)                     # polegar
            c.poly(np.vstack([bez((58, 112), (100, 100), (152, 116), n=12), bez((150, 116), (150, 206), (100, 214), n=10),
                              bez((100, 214), (52, 206), (56, 112), n=10)]), SKIN)
            c.atop(lambda l: l.poly([(128, 112), (160, 112), (160, 220), (124, 220)], SKIN_SHADE, outline=False), blur=2)
            c.line(bez((76, 150), (100, 170), (128, 150), n=10), SKIN_SHADE, 2.5)  # linha da palma
        else:
            c.poly(np.vstack([bez((52, 120), (100, 70), (152, 120), n=14), bez((152, 120), (156, 206), (100, 214), n=10),
                              bez((100, 214), (46, 206), (52, 120), n=10)]), SKIN)
            for k in range(3):
                x = 74 + k * 26
                c.line(bez((x, 98 + abs(k - 1) * 4), (x + 4, 130), n=6), SKIN_DEEP, 3)
            c.poly(_capsule((60, 176), (112, 140), 20), SKIN)                    # polegar dobrado
            c.atop(lambda l: l.poly([(130, 100), (170, 100), (170, 220), (126, 220)], SKIN_SHADE, outline=False), blur=2)
        c.save(name)


def _capsule(a, b, w, n: int = 12) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / (np.linalg.norm(b - a) or 1.0)
    ang = math.atan2(d[1], d[0])
    r = w / 2
    cap_b = [b + r * np.array([math.cos(ang + t), math.sin(ang + t)]) for t in np.linspace(-math.pi / 2, math.pi / 2, n)]
    cap_a = [a + r * np.array([math.cos(ang + t), math.sin(ang + t)]) for t in np.linspace(math.pi / 2, 3 * math.pi / 2, n)]
    return np.array(cap_b + cap_a)


def manifest():
    return {
        "name": "Pessoa",
        "notas": "Coordenadas em píxeis de cada imagem. Gerado por tools/make_human_avatar.py; podes trocar "
                 "qualquer PNG por um desenho teu com o mesmo tamanho e os mesmos pontos de encaixe.",
        "head": {
            "canvas": [HW, HH], "center": list(CENTER), "face_width": 246,
            "eyes": {"l": list(EYES["l"]), "r": list(EYES["r"])}, "eye_radius": EYE_R,
            "brows": {"l": list(BROWS["l"]), "r": list(BROWS["r"])}, "mouth": list(MOUTH),
            "parallax": {"hair_back": -0.06, "head": 0.0, "eyes": 0.12, "brows": 0.13, "mouth": 0.11,
                         "hair_front": 0.07},
        },
        "torso": {"file": "torso.png", "shoulders": [[96, 120], [416, 120]], "hips": [256, 500]},
        "neck": {"file": "neck.png", "start": [40, 6], "end": [40, 114], "width": 52, "thickness": 0.36},
        "upper_arm": {"file": "upper_arm.png", "start": [48, 14], "end": [48, 206], "width": 76, "thickness": 0.24},
        "forearm": {"file": "forearm.png", "start": [40, 10], "end": [40, 190], "width": 60, "thickness": 0.19},
        # thumb: lado do polegar na imagem (olhando para a palma, dedos para cima); o estilo
        # espelha a mão quando o polegar detetado está do outro lado.
        "hand": {"open": "hand_open.png", "fist": "hand_fist.png", "wrist": [100, 200], "knuckle": [100, 110],
                 "thumb": "left"},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head_layers()
    body_layers()
    (OUT / "avatar.json").write_text(json.dumps(manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Avatar 'pessoa' gerado em {OUT} ({len(list(OUT.glob('*.png')))} imagens)")


if __name__ == "__main__":
    main()
