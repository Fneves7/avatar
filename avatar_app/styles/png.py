"""Estilo "png": avatar 2D feito de imagens PNG por camadas (estilo VTuber), animado pelo Rig.

As imagens e os pontos de encaixe estão numa pasta com um avatar.json (ver
avatars/gato, gerado por tools/make_sample_avatar.py). Cada imagem é colocada com uma
transformação afim:
  * cabeça: todas as camadas partilham a mesma tela; a tela é posta no centro da cara,
    à escala da largura da cara e com a inclinação da cabeça. Cada camada tem ainda o seu
    desvio local: paralaxe ao rodar (as feições deslizam mais do que a cabeça), olhar
    (íris recortada pelo branco do olho), sobrancelhas (sobem/rodam), pelo/cabelo (molas);
  * olhos e boca: escolhe-se a imagem pelo estado do rig (aberto/meio/fechado; fechada,
    sorriso, entreaberta, aberta);
  * tronco: 3 pontos (ombros e ancas); pescoço, braço e antebraço: esticados entre as
    articulações; patas: pulso e base do dedo do meio, aberta ou fechada conforme os dedos.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..animation import AnimFrame
from ..drawing import Palette
from ..landmarks import FINGERS
from ..rig import Rig, RigHead

AVATARS = Path(__file__).resolve().parents[2] / "avatars"
DEFAULT_AVATAR = AVATARS / "gato"
EYE_OPEN, EYE_HALF = 0.6, 0.2       # abertura do rig a partir da qual o olho está aberto / meio
MOUTH_OPEN, MOUTH_SMALL, SMILE = 0.45, 0.15, 0.35
FIST_RATIO = 1.3                    # dedos (ponta->pulso / nó->pulso) abaixo disto = pata fechada
LOOK_RANGE = 0.45                   # deslocamento máximo da íris (* raio do olho)


def _m3(m23: np.ndarray) -> np.ndarray:
    return np.vstack([m23, [0.0, 0.0, 1.0]])


def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])


def _rotate_about(c, deg: float) -> np.ndarray:
    return _m3(cv2.getRotationMatrix2D((float(c[0]), float(c[1])), deg, 1.0))


def _reflect_about(p, d) -> np.ndarray:
    """Reflexão (3x3) na reta que passa por p com direção d."""
    u = _unit(np.asarray(d, float))
    r = 2 * np.outer(u, u) - np.eye(2)
    out = np.eye(3)
    out[:2, :2] = r
    out[:2, 2] = np.asarray(p, float) - r @ np.asarray(p, float)
    return out


def _perp(d: np.ndarray) -> np.ndarray:
    return np.array([d[1], -d[0]])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 1.0])


class Sprite:
    """Imagem RGBA recortada à zona não transparente; `offset` é a posição do recorte na
    imagem original (as coordenadas do avatar.json referem-se sempre à imagem original)."""

    def __init__(self, rgba: np.ndarray, margin: int = 2):
        ys, xs = np.nonzero(rgba[..., 3])
        if len(xs) == 0:
            self.data, self.offset = rgba[:1, :1], (0, 0)
            return
        x0, y0 = max(int(xs.min()) - margin, 0), max(int(ys.min()) - margin, 0)
        x1, y1 = min(int(xs.max()) + margin + 1, rgba.shape[1]), min(int(ys.max()) + margin + 1, rgba.shape[0])
        self.data = np.ascontiguousarray(rgba[y0:y1, x0:x1])
        self.offset = (x0, y0)

    def matrix(self, m: np.ndarray) -> np.ndarray:
        """Transformação da imagem original -> transformação do recorte."""
        return np.asarray(_m3(np.asarray(m)[:2]) @ _translate(*self.offset))


def blit(img: np.ndarray, sprite: Sprite, m: np.ndarray, alpha: float = 1.0,
         clip: tuple[Sprite, np.ndarray] | None = None) -> None:
    """Desenha um sprite com a transformação afim m (2x3 ou 3x3, da imagem original para a
    imagem final), só na zona que ocupa. clip=(sprite, m): só fica visível onde esse outro
    sprite é opaco (ex.: íris dentro do branco do olho)."""
    if alpha <= 0.001:
        return
    m = sprite.matrix(m)[:2]
    sprite_data = sprite.data
    h, w = sprite_data.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ m.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m_roi = m.copy()
    m_roi[:, 2] -= (x0, y0)
    layer = cv2.warpAffine(sprite_data, m_roi, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # Mistura em inteiros com operações do OpenCV (bem mais rápido do que em float com numpy).
    a = layer[..., 3]
    if alpha < 0.999:
        a = cv2.convertScaleAbs(a, alpha=alpha)
    if clip is not None:
        cm = clip[0].matrix(clip[1])[:2].copy()
        cm[:, 2] -= (x0, y0)
        mask = cv2.warpAffine(clip[0].data[..., 3], cm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        a = cv2.multiply(a, mask, scale=1 / 255)
    a3 = cv2.merge([a, a, a])
    roi = img[y0:y1, x0:x1]
    fg = cv2.multiply(layer[..., :3], a3, scale=1 / 255)
    bg = cv2.multiply(roi, cv2.bitwise_not(a3), scale=1 / 255)
    roi[:] = cv2.add(fg, bg)


class PngStyle:
    name = "png"

    def __init__(self, folder: Path | str = DEFAULT_AVATAR):
        self.load(folder)

    def load(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        self.error: str | None = None
        self.images: dict[str, Sprite] = {}
        try:
            self.cfg = json.loads((self.folder / "avatar.json").read_text(encoding="utf-8"))
            for f in self.folder.glob("*.png"):
                im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if im is not None and im.ndim == 3 and im.shape[2] == 4:
                    self.images[f.stem] = Sprite(im)  # recortada: as camadas da cabeça são quase vazias
        except (OSError, ValueError) as exc:
            self.cfg = None
            self.error = f"avatar PNG nao encontrado em {self.folder} ({exc.__class__.__name__}): " \
                         f"corre tools/make_sample_avatar.py"

    def _img(self, name: str) -> Sprite | None:
        return self.images.get(Path(name).stem)

    # ------------------------------------------------------------------ desenho
    def draw(self, img: np.ndarray, frame: AnimFrame, rig: Rig, palette: Palette) -> None:
        if self.cfg is None:
            cv2.putText(img, self.error or "avatar PNG em falta", (20, img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            return
        if rig.shoulders is not None:
            self._draw_neck(img, rig)
            self._draw_torso(img, rig)
        if rig.head is not None:
            self._draw_head(img, rig, rig.head)
        for side, arm in rig.arms.items():
            self._draw_limb(img, "upper_arm", arm.shoulder, arm.elbow, rig.shoulder_width)
            if arm.forearm_visible:
                self._draw_limb(img, "forearm", arm.elbow, arm.wrist, rig.shoulder_width)
                if side not in rig.hands:  # sem mão detetada: pata fechada na ponta do antebraço
                    d = _unit(arm.wrist - arm.elbow)
                    self._draw_hand(img, arm.wrist, arm.wrist + d * 0.3 * rig.shoulder_width, fist=True)
        for side, (pts, alpha) in rig.hands.items():
            p = pts[:, :2]
            span = np.linalg.norm(p[9] - p[0]) or 1.0
            reach = np.mean([np.linalg.norm(p[c[-1]] - p[0]) for c in FINGERS[1:]]) / span
            # Lado do polegar: (pulso->nó) x (pulso->polegar) < 0 = polegar à esquerda do eixo da mão
            # (com os dedos para cima na imagem). Quando não bate com o desenho, a mão é espelhada.
            d, t = p[9] - p[0], p[4] - p[0]
            thumb_left = d[0] * t[1] - d[1] * t[0] < 0
            self._draw_hand(img, p[0], p[9], fist=reach < FIST_RATIO, alpha=alpha, thumb_left=thumb_left)

    def _limb_matrix(self, cfg, a, b, thickness):
        """Afim que leva start->a, end->b e a largura do sprite à espessura pedida."""
        s0, s1 = np.array(cfg["start"], float), np.array(cfg["end"], float)
        d_src, d_dst = _unit(s1 - s0), _unit(b - a)
        src = np.float32([s0, s1, s0 + _perp(d_src) * cfg["width"] / 2])
        dst = np.float32([a, b, a + _perp(d_dst) * thickness / 2])
        return cv2.getAffineTransform(src, dst)

    def _draw_limb(self, img, key, a, b, sw):
        cfg = self.cfg[key]
        sprite = self._img(cfg["file"])
        if sprite is None or np.linalg.norm(b - a) < 2:
            return
        blit(img, sprite, self._limb_matrix(cfg, np.asarray(a, float), np.asarray(b, float), cfg["thickness"] * sw))

    def _draw_neck(self, img, rig: Rig):
        if rig.head is None or rig.neck_base is None:
            return
        h = rig.head
        down = np.array([-math.sin(h.roll), math.cos(h.roll)])
        top = h.center + down * 0.3 * h.height
        cfg = self.cfg["neck"]
        sprite = self._img(cfg["file"])
        if sprite is not None:
            blit(img, sprite, self._limb_matrix(cfg, top, rig.neck_base, cfg["thickness"] * h.size), h.alpha)

    def _draw_torso(self, img, rig: Rig):
        cfg = self.cfg["torso"]
        sprite = self._img(cfg["file"])
        if sprite is None:
            return
        src_sh = sorted((np.array(p, float) for p in cfg["shoulders"]), key=lambda p: p[0])
        dst_sh = sorted(rig.shoulders, key=lambda p: p[0])
        src = np.float32([src_sh[0], src_sh[1], cfg["hips"]])
        dst = np.float32([dst_sh[0], dst_sh[1], rig.hips.mean(axis=0)])
        blit(img, sprite, cv2.getAffineTransform(src, dst))

    def _draw_hand(self, img, wrist, knuckle, fist: bool, alpha: float = 1.0, thumb_left: bool | None = None):
        cfg = self.cfg["hand"]
        sprite = self._img(cfg["fist" if fist else "open"])
        if sprite is None:
            return
        w_src, k_src = np.array(cfg["wrist"], float), np.array(cfg["knuckle"], float)
        d_src, d_dst = k_src - w_src, np.asarray(knuckle, float) - np.asarray(wrist, float)
        k = (np.linalg.norm(d_dst) or 1.0) / (np.linalg.norm(d_src) or 1.0)
        ang = math.degrees(math.atan2(d_dst[1], d_dst[0]) - math.atan2(d_src[1], d_src[0]))
        m = cv2.getRotationMatrix2D((float(w_src[0]), float(w_src[1])), -ang, k)
        m[:, 2] += np.asarray(wrist, float) - w_src
        if "thumb" in cfg and thumb_left is not None and thumb_left != (cfg["thumb"] == "left"):
            m = (_m3(m) @ _reflect_about(w_src, d_src))[:2]  # espelha à volta do eixo pulso->nó
        blit(img, sprite, m, alpha)

    # ------------------------------------------------------------------ cabeça
    def _draw_head(self, img, rig: Rig, h: RigHead):
        cfg = self.cfg["head"]
        c = np.array(cfg["center"], float)
        k = h.size / cfg["face_width"]
        # Tela da cabeça -> imagem: escala da cara, inclinação e centro da cara.
        head_m = _translate(h.center[0] - c[0], h.center[1] - c[1]) @ \
            _m3(cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -math.degrees(h.roll), k))
        par = cfg.get("parallax", {})
        turn = h.turn * cfg["face_width"]

        def layer_m(name_par: str, extra: np.ndarray | None = None) -> np.ndarray:
            m = head_m @ _translate(par.get(name_par, 0.0) * turn, 0.0)
            return m @ extra if extra is not None else m

        # Pelo/cabelo: molas do movimento secundário (desvio em px da imagem -> tela).
        hair_shift = np.array(rig.hair_offset) / max(k, 1e-6)
        hair_rot = -math.degrees(rig.hair_rot)

        def hair_m(name: str, amount: float) -> np.ndarray:
            return layer_m(name, _translate(*(hair_shift * amount)) @ _rotate_about(c, hair_rot * amount))

        a = h.alpha
        self._blit_named(img, "hair_back", hair_m("hair_back", 1.0), a)
        self._blit_named(img, "head", layer_m("head"), a)
        blush = self._img("blush")
        if blush is not None and h.mouth_smile > 0.3:
            blit(img, blush, layer_m("mouth"), a * min(1.0, (h.mouth_smile - 0.3) / 0.4))

        # Olhos: o olho do rig mais à direita da cara usa as imagens "_r".
        x_axis = np.array([math.cos(h.roll), math.sin(h.roll)])
        radius = cfg["eye_radius"]
        look = np.clip(h.look, -1, 1) * LOOK_RANGE * radius
        for i, ec in enumerate(h.eye_centers):
            side = "r" if (ec - h.center) @ x_axis > 0 else "l"
            openness = h.eye_open[i]
            m = layer_m("eyes")
            if openness >= EYE_OPEN:
                white = self._img(f"eye_white_{side}")
                iris = self._img(f"iris_{side}")
                if white is not None:
                    blit(img, white, m, a)
                    if iris is not None:
                        blit(img, iris, m @ _translate(look[0], look[1]), a, clip=(white, m))
            elif openness >= EYE_HALF:
                self._blit_named(img, f"eye_half_{side}", m, a)
            else:
                self._blit_named(img, f"eye_closed_{side}", m, a)

        # Sobrancelhas: sobem ao levantar; ao franzir a ponta de dentro desce (rodam).
        for side, pivot in cfg["brows"].items():
            lift = -14.0 * max(h.brow, 0.0)
            frown = 16.0 * max(-h.brow, 0.0) * (-1.0 if side == "l" else 1.0)
            self._blit_named(img, f"brow_{side}", layer_m("brows", _translate(0, lift) @ _rotate_about(pivot, frown)), a)

        # Boca pelo estado do rig.
        if h.mouth_open >= MOUTH_OPEN:
            mouth = "mouth_open"
        elif h.mouth_open >= MOUTH_SMALL:
            mouth = "mouth_open_small"
        elif h.mouth_smile >= SMILE:
            mouth = "mouth_smile"
        else:
            mouth = "mouth_closed"
        self._blit_named(img, mouth, layer_m("mouth"), a)
        self._blit_named(img, "hair_front", hair_m("hair_front", 0.5), a)

    def _blit_named(self, img, name, m, alpha):
        sprite = self._img(name)
        if sprite is not None:
            blit(img, sprite, m, alpha)


class PersonStyle(PngStyle):
    """O mesmo estilo png com o avatar "pessoa" (tools/make_human_avatar.py)."""
    name = "pessoa"

    def __init__(self, folder: Path | str = AVATARS / "pessoa"):
        super().__init__(folder)
