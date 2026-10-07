"""Desenha um avatar 2D em estilo cartoon a partir do BodyState."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .hand_mesh import HandMesh
from .head import NECK_TOP, Ear, estimate_head
from .tracker import (L_EAR, L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_EAR, R_ELBOW, R_HIP,
                      R_SHOULDER, R_WRIST, BodyState)

AA = cv2.LINE_AA

# --- Índices do FaceLandmarker (478 pontos) ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
IRIS_A, IRIS_B = (468, 469), (473, 474)  # (centro, ponto do contorno)
BROW_A = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_B = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
BLINK_CLOSE, BLINK_OPEN = 0.6, 0.35  # piscar calibrado: fecha acima de, reabre abaixo de

LIPS_SEAM =[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
NOSE_TIP = 1
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_LOWER = [4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275]
NOSE_ALA_A = [115, 48, 64, 98]          # asa de um lado (de cima para a base)
NOSE_ALA_B = [344, 278, 294, 327]       # asa do outro lado
NOSE_BASE = [98, 97, 2, 326, 327]
CHEEK_A, CHEEK_B = 50, 280
FACE_LEFT, FACE_RIGHT, CHIN, FOREHEAD = 234, 454, 152, 10

# --- Índices do HandLandmarker ---
PALM = [0, 1, 2, 5, 9, 13, 17]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]

# Pontos da mão na pose (fallback quando o HandLandmarker não deteta a mão).
POSE_HAND = {L_WRIST: (17, 19, 21), R_WRIST: (18, 20, 22)}  # pinky, index, thumb


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


class AvatarRenderer:
    def __init__(self, palette_index: int = 0):
        self.palette_index = palette_index
        self.head_3d = True  # crânio/nuca/orelhas estimados em 3D (tecla h alterna)
        self._nose_side = 1.0
        self._geo = None  # HeadGeometry do frame atual
        self._eyes_closed = [False, False]  # EYE_A, EYE_B (com histerese)
        self._bg_cache: dict = {}

    @property
    def palette(self) -> Palette:
        return PALETTES[self.palette_index % len(PALETTES)]

    def next_palette(self) -> None:
        self.palette_index = (self.palette_index + 1) % len(PALETTES)

    # ------------------------------------------------------------------ fundo
    def _background(self, w: int, h: int) -> np.ndarray:
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
    def render(self, s: BodyState) -> np.ndarray:
        img = self._background(s.width, s.height)
        p = self.palette
        ow = max(2, s.width // 320)  # espessura do contorno

        sw = self._shoulder_width(s)
        self._geo = estimate_head(s.face, FACE_OVAL) if (s.face is not None and self.head_3d) else None
        self._update_eyes_closed(s)
        arms_behind, arms_front = self._split_arms_by_depth(s, sw)

        for side in arms_behind:
            self._draw_arm(img, s, side, sw, ow)
        self._draw_neck(img, s, sw, ow)
        if s.pose is not None:
            self._draw_torso(img, s, sw, ow)
            self._draw_collar(img, s, sw, ow)
        if s.face is not None:
            self._draw_face(img, s, ow)
        elif s.pose is not None:
            self._draw_fallback_head(img, s, ow)
        for side in arms_front:
            self._draw_arm(img, s, side, sw, ow)
        # Mãos detetadas sem pose (ex.: só as mãos em frente à câmara).
        if s.pose is None:
            for side, hand in s.hands.items():
                self._draw_hand(img, hand, ow, s.hand_meshes.get(side))
        return img

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def _shoulder_width(s: BodyState) -> float:
        if s.pose is not None:
            return float(np.linalg.norm(s.pose[L_SHOULDER, :2] - s.pose[R_SHOULDER, :2])) or 1.0
        if s.face is not None:
            return 2.2 * float(np.linalg.norm(s.face[FACE_LEFT, :2] - s.face[FACE_RIGHT, :2]))
        return s.width * 0.3

    @staticmethod
    def _split_arms_by_depth(s: BodyState, sw: float):
        """Braços com a mão claramente atrás da cabeça desenham-se antes do corpo."""
        behind, front = [], []
        if s.pose is None:
            return behind, front
        for side in (L_WRIST, R_WRIST):
            (behind if s.pose[side, 2] > s.pose[NOSE, 2] + 0.35 * sw else front).append(side)
        return behind, front

    @staticmethod
    def _shoulder_frame(s: BodyState, sw: float):
        """Ombros, eixo ombro-a-ombro, vetor 'para cima' do tronco e base do pescoço.

        O landmark do ombro é a articulação; a base do pescoço fica ~20% da largura
        dos ombros acima da linha entre articulações (trapézios)."""
        ls, rs = s.pose[L_SHOULDER, :2], s.pose[R_SHOULDER, :2]
        axis = (rs - ls) / sw
        up = np.array([axis[1], -axis[0]])
        if up[1] > 0:
            up = -up
        neck_base = (ls + rs) / 2 + up * 0.2 * sw
        return ls, rs, axis, up, neck_base

    def _hips(self, s: BodyState, sw: float) -> tuple[np.ndarray, np.ndarray]:
        if s.visible(L_HIP, 0.3) and s.visible(R_HIP, 0.3):
            return s.pose[L_HIP, :2], s.pose[R_HIP, :2]
        # Ancas fora de imagem: estima-as perpendicularmente aos ombros.
        ls, rs, _, up, _ = self._shoulder_frame(s, sw)
        return ls - up * 1.4 * sw, rs - up * 1.4 * sw

    # ------------------------------------------------------------------ corpo
    def _draw_torso(self, img, s: BodyState, sw: float, ow: int):
        p = self.palette
        ls, rs, axis, up, neck_base = self._shoulder_frame(s, sw)
        lh, rh = self._hips(s, sw)
        out_h = 0.02 * sw
        mid_l, mid_r = (ls + lh) / 2 - axis * 0.04 * sw, (rs + rh) / 2 + axis * 0.04 * sw
        # Linha dos trapézios: do topo do ombro sobe em curva até ao lado do pescoço.
        neck_l, neck_r = neck_base - axis * 0.13 * sw, neck_base + axis * 0.13 * sw
        top_l, top_r = ls + up * 0.12 * sw, rs + up * 0.12 * sw
        trap_l = _bezier(top_l, (top_l + neck_l) / 2 + up * 0.07 * sw, neck_l)
        trap_r = _bezier(neck_r, (top_r + neck_r) / 2 + up * 0.07 * sw, top_r)
        body = [ls - axis * 0.12 * sw, *trap_l, *trap_r, rs + axis * 0.12 * sw,
                mid_r, rh + axis * out_h, lh - axis * out_h, mid_l]
        r = 0.15 * sw  # deltoides
        g = Group().poly(body).circle(ls - axis * 0.03 * sw, r).circle(rs + axis * 0.03 * sw, r)
        g.draw(img, p.shirt, p.outline, ow)
        # Linha central para dar volume.
        bottom = (lh + rh) / 2
        cv2.line(img, _ip(neck_base + (bottom - neck_base) * 0.3), _ip(bottom), p.shirt_dark, max(1, ow), AA)

    def _neck_points(self, s: BodyState, sw: float):
        if s.face is not None:
            f = s.face
            fw = float(np.linalg.norm(f[FACE_LEFT, :2] - f[FACE_RIGHT, :2]))
            if self._geo is not None:
                # Pescoço sai de baixo do crânio (não do queixo): segue a rotação da cabeça.
                g = self._geo
                top = g.to_world(np.array([0.0, NECK_TOP[0] * g.fh, NECK_TOP[1] * g.fw]))[0, :2]
            else:
                top = (f[CHIN, :2] * 0.6 + f[NOSE_TIP, :2] * 0.4)
        elif s.pose is not None:
            fw = float(np.linalg.norm(s.pose[L_EAR, :2] - s.pose[R_EAR, :2])) * 1.1
            top = s.pose[NOSE, :2] + np.array([0, 0.3 * fw])
        else:
            return None
        if s.pose is not None:
            bottom = self._shoulder_frame(s, sw)[4]
        else:
            bottom = top + np.array([0, 0.9 * fw])
        return top, bottom, fw

    def _draw_neck(self, img, s: BodyState, sw: float, ow: int):
        np_ = self._neck_points(s, sw)
        if np_ is None:
            return
        top, bottom, fw = np_
        p = self.palette
        Group().line(top, bottom, 0.42 * fw).draw(img, p.skin, p.outline, ow)
        # Sombra do queixo sobre o pescoço.
        cv2.ellipse(img, _ip(top + (bottom - top) * 0.25), (int(0.2 * fw), int(0.06 * fw)), 0, 0, 180,
                    tuple(int(c * 0.8) for c in p.skin), -1, AA)

    def _draw_collar(self, img, s: BodyState, sw: float, ow: int):
        """Decote em U por cima do tronco, com a pele do pescoço visível."""
        np_ = self._neck_points(s, sw)
        if np_ is None:
            return
        top, bottom, fw = np_
        p = self.palette
        axes = (max(1, int(0.24 * fw)), max(1, int(0.16 * fw)))
        cv2.ellipse(img, _ip(bottom), axes, 0, 0, 180, p.skin, -1, AA)
        cv2.ellipse(img, _ip(bottom), axes, 0, 0, 180, p.shirt_dark, max(2, ow * 2), AA)

    def _draw_arm(self, img, s: BodyState, side: int, sw: float, ow: int):
        p = self.palette
        shoulder_i, elbow_i = (L_SHOULDER, L_ELBOW) if side == L_WRIST else (R_SHOULDER, R_ELBOW)
        hand = s.hands.get(side)
        if not s.visible(elbow_i, 0.25) and hand is None:
            return
        sh, el = s.pose[shoulder_i, :2], s.pose[elbow_i, :2]
        wr = hand[0, :2] if hand is not None else s.pose[side, :2]

        Group().line(sh, el, 0.28 * sw).circle(el, 0.13 * sw).draw(img, p.shirt, p.outline, ow)
        if hand is not None or s.visible(side, 0.25):
            Group().line(el, wr, 0.2 * sw).circle(el, 0.1 * sw).draw(img, p.skin, p.outline, ow)
            # Punho da manga por cima do cotovelo.
            Group().circle(el, 0.13 * sw).draw(img, p.shirt, p.outline, ow)
            cv2.line(img, _ip(sh), _ip(el), p.shirt, max(1, int(0.28 * sw)), AA)
            if hand is not None:
                self._draw_hand(img, hand, ow, s.hand_meshes.get(side))
            elif s.visible(side, 0.5):
                self._draw_mitten(img, s, side, sw, ow)

    def _draw_hand(self, img, hand: np.ndarray, ow: int, mesh: HandMesh | None = None):
        if mesh is not None:
            self._draw_hand_mesh(img, hand, mesh, ow)
            return
        p = self.palette
        hs = float(np.linalg.norm(hand[0, :2] - hand[9, :2])) or 1.0
        palm = cv2.convexHull(_poly(hand[PALM]))[:, 0, :]
        g = Group().poly(palm)
        for chain in FINGERS:
            w = (0.28 if chain[0] == 1 else 0.22) * hs
            for a, b in zip(chain[:-1], chain[1:]):
                g.line(hand[a], hand[b], w)
            g.circle(hand[chain[-1]], w / 2)
        g.draw(img, p.skin, p.outline, ow)
        # Unhas e nós dos dedos.
        for tip, prev in zip(TIPS, [3, 7, 11, 15, 19]):
            d = hand[tip, :2] - hand[prev, :2]
            c = hand[tip, :2] - d * 0.15
            cv2.circle(img, _ip(c), max(1, int(0.07 * hs)), p.skin_light, -1, AA)
        for k in (5, 9, 13, 17):
            cv2.circle(img, _ip(hand[k]), max(1, int(0.03 * hs)), tuple(int(c * 0.8) for c in p.skin), -1, AA)

    def _draw_hand_mesh(self, img, hand: np.ndarray, mesh: HandMesh, ow: int):
        """Mão com o contorno real dos dedos (larguras medidas na imagem)."""
        p = self.palette
        shade = tuple(int(c * 0.8) for c in p.skin)
        # Contorno exterior da mão inteira, depois a palma por cima.
        Group().poly(mesh.palm).draw(img, p.skin, p.outline, ow)
        g = Group()
        for poly in mesh.fingers:
            g.poly(poly)
        g.draw(img, p.skin, p.outline, ow)
        cv2.fillPoly(img, [_poly(mesh.palm)], p.skin, AA)
        # Dedos do mais distante para o mais próximo (z), cada um com as suas bordas,
        # para se distinguirem quando se sobrepõem. A base fica aberta para fundir com a palma.
        order = sorted(range(5), key=lambda i: -hand[FINGERS[i], 2].mean())
        for i in order:
            poly = _poly(mesh.fingers[i])
            cv2.fillPoly(img, [poly], p.skin, AA)
            # Aberto = sem a base. O polegar nasce dentro da palma: as bordas só a partir do MCP.
            edge = poly[1:-1] if i == 0 else poly
            cv2.polylines(img, [edge], False, p.outline, max(1, ow - 1), AA)
            left, right = mesh.rails[i]
            w = mesh.widths[i]
            joints = hand[FINGERS[i], :2]
            # Pregas nas articulações (PIP e DIP).
            for j in (1, 2):
                a, b = left[j], right[j]
                c = (a + b) / 2
                cv2.line(img, _ip(c + (a - c) * 0.45), _ip(c + (b - c) * 0.45), shade, max(1, ow // 2), AA)
            # Unha: elipse alinhada com a falange distal.
            d = joints[3] - joints[2]
            ang = float(np.degrees(np.arctan2(d[1], d[0])))
            c = joints[2] + d * 0.72
            axes = (max(1, int(np.linalg.norm(d) * 0.3)), max(1, int(w[2] * 0.3)))
            cv2.ellipse(img, _ip(c), axes, ang, 0, 360, p.skin_light, -1, AA)

    def _draw_mitten(self, img, s: BodyState, side: int, sw: float, ow: int):
        p = self.palette
        pinky, index, thumb = (s.pose[i, :2] for i in POSE_HAND[side])
        wr = s.pose[side, :2]
        tip = (pinky + index) / 2
        g = Group().poly([wr, pinky, index]).circle(tip, 0.1 * sw).line(wr, thumb, 0.08 * sw)
        g.draw(img, p.skin, p.outline, ow)

    # ------------------------------------------------------------------ cabeça
    def _draw_face(self, img, s: BodyState, ow: int):
        p = self.palette
        f = s.face[:, :2]
        oval = f[FACE_OVAL]
        center = oval.mean(axis=0)
        fw = float(np.linalg.norm(f[FACE_LEFT] - f[FACE_RIGHT])) or 1.0
        up = (center - f[CHIN]) / (np.linalg.norm(center - f[CHIN]) or 1.0)
        geo = self._geo

        if geo is not None:
            # Cabeça completa: orelhas escondidas -> crânio/cabelo -> orelhas visíveis -> cara.
            for ear in geo.ears:
                if ear.hidden:
                    self._draw_ear(img, ear, ow)
            Group().poly(geo.skull).draw(img, p.hair, p.outline, ow)
            for ear in geo.ears:
                if not ear.hidden:
                    self._draw_ear(img, ear, ow)
            # Silhueta real da cara = envolvente de toda a malha. De perfil inclui
            # testa, nariz, lábios e queixo (o contorno do maxilar já não serve).
            silhouette = cv2.convexHull(_poly(_scale_about(f[:468], center, 1.02)))[:, 0, :]
            # Só a zona da cabeça é guardada/recortada (copiar o frame inteiro é lento).
            m = int(0.4 * fw)
            x, y, w, h = cv2.boundingRect(silhouette)
            x0, y0 = max(x - m, 0), max(y - m, 0)
            x1, y1 = min(x + w + m, img.shape[1]), min(y + h + m, img.shape[0])
            before = img[y0:y1, x0:x1].copy()
            cv2.fillPoly(img, [silhouette], p.skin, AA)
            # Franja sem contorno exterior para fundir com o cabelo do crânio.
            # Só os pontos do contorno que estão na silhueta: de perfil, o lado de trás
            # da linha do cabelo fica escondido atrás da testa e não deve ter franja.
            hairline = _scale_about(oval, center, 1.02)
            sil_f = silhouette.astype(np.float32).reshape(-1, 1, 2)
            on_edge = [cv2.pointPolygonTest(sil_f, (float(px), float(py)), True) < 0.06 * fw
                       for px, py in hairline]
            hair = self._hair_crescent(hairline[on_edge], center, up, fw, outer_k=1.0)
            if hair is not None:
                outer, fringe = hair
                cv2.fillPoly(img, [_poly(np.vstack([outer, fringe[::-1]]))], p.hair, AA)
                cv2.polylines(img, [_poly(fringe)], False, p.outline, ow, AA)
            self._draw_features(img, f, fw, ow, s.blendshapes)
            # Tudo o que saia da silhueta (ex.: o olho do lado de trás) é recortado.
            if x1 > x0 and y1 > y0:
                mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(mask, [silhouette - np.array([x0, y0])], 255)
                roi = img[y0:y1, x0:x1]
                np.copyto(roi, before, where=(mask == 0)[..., None])
            cv2.polylines(img, [silhouette], True, p.outline, ow, AA)
            return
        else:
            head = _scale_about(oval, center, 1.06)
            ears = Group()
            for idx in (FACE_LEFT, FACE_RIGHT):
                ears.circle(_scale_about(f[idx], center, 1.08), 0.13 * fw)
            ears.draw(img, p.skin, p.outline, ow)
            Group().poly(head).draw(img, p.skin, p.outline, ow)
            hair = self._hair_crescent(head, center, up, fw, outer_k=1.14)
            if hair is not None:
                outer, fringe = hair
                Group().poly(np.vstack([outer + up * 0.04 * fw, fringe[::-1]])).draw(img, p.hair, p.outline, ow)
        self._draw_features(img, f, fw, ow, s.blendshapes)

    def _draw_features(self, img, f, fw, ow, blendshapes):
        self._draw_eyes(img, f, fw, ow)
        self._draw_brows(img, f, ow)
        self._draw_nose(img, f, fw, ow)
        self._draw_mouth(img, f, fw, ow)
        self._draw_blush(img, f, fw, blendshapes)

    @staticmethod
    def _hair_crescent(head, center, up, fw, outer_k: float):
        """Arco de cabelo sobre a metade superior do contorno da cara, com franja."""
        top = np.array([pt for pt in head if np.dot(pt - center, up) > -0.05 * fw])
        if len(top) < 3:
            return None
        # Ordena pelo ângulo para obter um arco contínuo.
        ang = np.arctan2(*(top - center).T[::-1])
        ref = np.arctan2(up[1], up[0])
        arc = top[np.argsort((ang - ref + np.pi) % (2 * np.pi))]
        outer = _scale_about(arc, center, outer_k)
        inner = _scale_about(arc, center, 0.94) + up * 0.02 * fw
        # Franja: o arco interior desce um pouco no meio da testa.
        fringe = inner - up * 0.10 * fw * np.sin(np.linspace(0, np.pi, len(inner)))[:, None]
        return outer, fringe

    def _draw_ear(self, img, ear: Ear, ow: int):
        p = self.palette
        Group().poly(ear.polygon).draw(img, p.skin, p.outline, ow)
        cv2.fillPoly(img, [ear.inner], tuple(int(c * 0.82) for c in p.skin), AA)

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

    def _draw_eyes(self, img, f, fw, ow):
        p = self.palette
        # Largura de cada olho (canto a canto): de perfil o olho de trás fica muito estreito.
        widths = [float(np.linalg.norm(f[c[0]] - f[c[8]])) for c in (EYE_A, EYE_B)]
        for i, ((contour, (ci, ri)), width) in enumerate(zip(((EYE_A, IRIS_A), (EYE_B, IRIS_B)), widths)):
            eye = f[contour]
            ec = eye.mean(axis=0)
            if self._eyes_closed[i]:
                # Olho fechado: só a curva da pálpebra inferior (canto a canto, por baixo).
                lower = _poly(_scale_about(eye[:9], ec, 1.3))
                cv2.polylines(img, [lower], False, p.outline, ow + 1, AA)
                continue
            if width < 0.45 * max(widths):
                # Olho de trás muito encolhido: só a linha da pálpebra superior.
                upper = _poly(np.vstack([eye[8:], eye[:1]]))
                cv2.polylines(img, [upper], False, p.outline, max(1, ow), AA)
                continue
            k = 1.3  # olhos maiores, estilo cartoon
            eye_big = _scale_about(eye, ec, k)
            poly = _poly(eye_big)
            iris_c = ec + (f[ci] - ec) * k
            iris_r = float(np.linalg.norm(f[ci] - f[ri])) * k * 1.1

            cv2.fillPoly(img, [poly], p.eye_white, AA)
            # Íris recortada pelo contorno do olho (pestanejar fecha-a naturalmente).
            x, y, w, h = cv2.boundingRect(poly)
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
            if x1 > x0 and y1 > y0:
                roi = img[y0:y1, x0:x1]
                layer = roi.copy()
                off = np.array([x0, y0])
                c = _ip(iris_c - off)
                cv2.circle(layer, c, max(1, int(iris_r)), p.iris, -1, AA)
                cv2.circle(layer, c, max(1, int(iris_r * 0.5)), (20, 20, 20), -1, AA)
                cv2.circle(layer, _ip(iris_c - off + np.array([-iris_r * 0.35, -iris_r * 0.35])),
                           max(1, int(iris_r * 0.25)), (255, 255, 255), -1, AA)
                mask = np.zeros(roi.shape[:2], np.uint8)
                cv2.fillPoly(mask, [poly - off], 255, AA)
                roi[mask > 127] = layer[mask > 127]
            cv2.polylines(img, [poly], True, p.outline, ow + 1, AA)
            # Pestanas na pálpebra superior (canto interno -> pontos superiores -> canto externo).
            upper = np.vstack([poly[8:], poly[:1]])
            cv2.polylines(img, [upper], False, p.outline, ow + 2, AA)

    def _draw_brows(self, img, f, ow):
        for brow in (BROW_A, BROW_B):
            Group().poly(f[brow]).draw(img, self.palette.hair, self.palette.outline, max(1, ow // 2))

    def _draw_nose(self, img, f, fw, ow):
        """Nariz completo a partir dos 26 landmarks: cana, ponta, asas e narinas."""
        p = self.palette
        shade = tuple(int(c * 0.82) for c in p.skin)

        # Lado da sombra da cana: o lado virado para o centro da cara (o que se vê quando
        # a cabeça roda). Com histerese para não saltar de lado quando se está de frente.
        mid_x = (f[FACE_LEFT, 0] + f[FACE_RIGHT, 0]) / 2
        delta = mid_x - f[NOSE_TIP, 0]
        if abs(delta) > 0.03 * fw:
            self._nose_side = 1.0 if delta > 0 else -1.0
        # Eixo horizontal da cara apontado sempre para +x da imagem (vale com ou sem espelho).
        across = f[FACE_RIGHT] - f[FACE_LEFT]
        across = across / (np.linalg.norm(across) or 1.0)
        if across[0] < 0:
            across = -across
        offset = across * self._nose_side * 0.035 * fw
        bridge = _poly(f[NOSE_BRIDGE] + offset)
        # De frente a cana é quase invisível; ganha força à medida que a cabeça roda.
        alpha = float(np.clip(abs(delta) / (0.12 * fw), 0.25, 1.0))
        thick = max(2, int(0.035 * fw))
        x, y, w, h = cv2.boundingRect(bridge)
        x0, y0 = max(x - thick, 0), max(y - thick, 0)
        x1, y1 = min(x + w + thick, img.shape[1]), min(y + h + thick, img.shape[0])
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            layer = roi.copy()
            cv2.polylines(layer, [bridge - np.array([x0, y0])], False, shade, thick, AA)
            cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)

        # Ponta + asas: forma preenchida com sombra lateral e brilho na ponta.
        lower = _poly(f[NOSE_LOWER])
        cv2.fillPoly(img, [lower], p.skin, AA)
        ala_a_right = f[NOSE_ALA_A, 0].mean() > f[NOSE_ALA_B, 0].mean()
        shade_a = ala_a_right == (self._nose_side > 0)
        side_pts = f[NOSE_ALA_A if shade_a else NOSE_ALA_B]
        cv2.fillPoly(img, [_poly(np.vstack([side_pts, f[[NOSE_TIP]]]))], shade, AA)
        tip = f[NOSE_TIP] + (f[4] - f[NOSE_TIP]) * 0.6
        cv2.circle(img, _ip(tip), max(1, int(0.035 * fw)), p.skin_light, -1, AA)

        # Narinas: entre a base da asa e o subnasal, puxadas para a ponta.
        for ala, inner in ((98, 97), (327, 326)):
            c = f[ala] * 0.45 + f[inner] * 0.35 + f[NOSE_TIP] * 0.20
            d = f[inner] - f[ala]
            ang = float(np.degrees(np.arctan2(d[1], d[0])))
            axes = (max(1, int(np.linalg.norm(d) * 0.55)), max(1, int(0.022 * fw)))
            cv2.ellipse(img, _ip(c), axes, ang, 0, 360, p.mouth, -1, AA)

        # Contorno só nas asas e na base (um nariz todo contornado parece uma bola).
        for chain in (NOSE_ALA_A, NOSE_ALA_B, NOSE_BASE):
            cv2.polylines(img, [_poly(f[chain])], False, p.outline, max(1, ow - 1), AA)

    def _draw_mouth(self, img, f, fw, ow):
        p = self.palette
        outer, inner = _poly(f[LIPS_OUTER]), _poly(f[LIPS_INNER])
        cv2.fillPoly(img, [outer], p.lips, AA)
        open_h = float(np.linalg.norm(f[13] - f[14]))
        if open_h <= 0.02 * fw:
            # Boca fechada: só a linha entre os lábios.
            cv2.polylines(img, [_poly(f[LIPS_SEAM])], False, p.outline, ow, AA)
            cv2.polylines(img, [outer], True, p.outline, max(1, ow // 2), AA)
            return
        cv2.fillPoly(img, [inner], p.mouth, AA)
        # Dentes superiores + língua, recortados pela boca.
        x, y, w, h = cv2.boundingRect(inner)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [inner], 255, AA)
        layer = img.copy()
        cv2.rectangle(layer, (x, y), (x + w, y + max(1, int(min(h * 0.3, 0.06 * fw)))), (245, 245, 245), -1)
        cv2.ellipse(layer, (x + w // 2, y + h), (max(1, w // 3), max(1, h // 3)), 0, 180, 360,
                    (110, 100, 220), -1, AA)
        sel = mask > 127
        img[sel] = layer[sel]
        cv2.polylines(img, [outer], True, p.outline, ow, AA)

    def _draw_blush(self, img, f, fw, bs: dict[str, float]):
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        if smile < 0.3:
            return
        alpha = min(1.0, (smile - 0.3) / 0.4) * 0.55
        r = int(0.11 * fw)
        for idx in (CHEEK_A, CHEEK_B):
            c = _ip(f[idx])
            x0, y0 = max(c[0] - r, 0), max(c[1] - r, 0)
            x1, y1 = min(c[0] + r, img.shape[1]), min(c[1] + r, img.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            layer = roi.copy()
            cv2.ellipse(layer, (c[0] - x0, c[1] - y0), (r, int(r * 0.6)), 0, 0, 360, self.palette.blush, -1, AA)
            cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)

    def _draw_fallback_head(self, img, s: BodyState, ow: int):
        """Cabeça simples a partir da pose quando o FaceLandmarker perde o rosto."""
        p = self.palette
        le, re, nose = s.pose[L_EAR, :2], s.pose[R_EAR, :2], s.pose[NOSE, :2]
        fw = float(np.linalg.norm(le - re)) * 1.2 or 40.0
        c = (le + re) / 2 * 0.5 + nose * 0.5
        Group().circle(c, fw * 0.62).draw(img, p.skin, p.outline, ow)
        cv2.ellipse(img, _ip(c - np.array([0, fw * 0.1])), (int(fw * 0.66), int(fw * 0.6)), 0, 180, 360,
                    p.hair, -1, AA)
        for i in (2, 5):
            cv2.circle(img, _ip(s.pose[i, :2]), max(2, int(fw * 0.06)), p.outline, -1, AA)
        cv2.line(img, _ip(s.pose[9, :2]), _ip(s.pose[10, :2]), p.outline, ow, AA)
