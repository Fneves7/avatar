"""Malha da mão: mede a largura real de cada falange na imagem e constrói o contorno dos dedos.

O HandLandmarker só dá o esqueleto (21 pontos). Para obter os contornos:
  1. aprende-se a cor da pele (Cr/Cb) na palma e ao longo dos ossos dos dedos;
  2. segmenta-se a pele num "tubo" à volta do esqueleto (exclui cara, fundo, etc.);
  3. em cada falange procura-se, perpendicularmente ao osso, onde a pele acaba;
  4. com as larguras constrói-se um polígono por dedo + palma (a malha).
Se a medição falhar usam-se as últimas larguras medidas ou proporções anatómicas.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
HAND_BONES = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)] + [
    (c[i], c[i + 1]) for c in FINGERS for i in range(3)]

# Largura por defeito de cada falange, relativa ao tamanho da mão (pulso -> nó do dedo médio).
DEFAULT_W = (np.array([0.26, 0.21, 0.21, 0.20, 0.17])[:, None]
             * np.array([1.0, 0.92, 0.85])[None, :])
WRIST_HALF = 0.36          # meia-largura do pulso (* tamanho da mão)
SKIN_SIGMA2 = 9.0          # limite da distância de Mahalanobis² na cor da pele (~3σ)
MAX_BG_SKIN = 0.35         # fração máxima de "pele" no fundo à volta da mão
WORK_HS = 70.0             # tamanho da mão (px) na resolução de trabalho
N_MEASURABLE = 14         # 5 dedos x 3 falanges, exceto a base do polegar (dentro da palma)


@dataclass
class HandMesh:
    fingers: list[np.ndarray]                     # 5 polígonos (N, 2)
    palm: np.ndarray                              # (M, 2)
    rails: list[tuple[np.ndarray, np.ndarray]]    # por dedo: bordas esquerda/direita (4, 2)
    widths: np.ndarray                            # (5, 3) em píxeis
    measured: int                                 # falanges medidas neste frame
    contour: np.ndarray | None                    # contorno real segmentado (para debug)


class HandMeshEstimator:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha                            # suavização temporal das larguras
        self._widths: dict[int, np.ndarray] = {}      # larguras relativas por mão

    def reset(self, side: int) -> None:
        self._widths.pop(side, None)

    def estimate(self, frame: np.ndarray, measure_pts: np.ndarray | None,
                 draw_pts: np.ndarray, side: int) -> HandMesh:
        """measure_pts: landmarks em bruto (alinhados com o frame); draw_pts: suavizados."""
        rel = DEFAULT_W.copy()
        got = np.zeros_like(rel, dtype=bool)
        contour = None
        if measure_pts is not None:
            p = measure_pts[:, :2]
            hs = float(np.linalg.norm(p[0] - p[9])) or 1.0
            mask, off, sc = self._skin_mask(frame, p, hs)
            if mask is not None:
                q = (p - off) * sc  # coordenadas na máscara (reduzida)
                measured = _measure_all(mask, q, hs * sc) / (hs * sc)
                got = ~np.isnan(measured)
                rel[got] = np.clip(measured[got], DEFAULT_W[got] * 0.6, DEFAULT_W[got] * 1.6)
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    contour = max(cs, key=cv2.contourArea)[:, 0, :] / sc + off

        prev = self._widths.get(side)
        if prev is not None:
            rel = np.where(got, prev + self.alpha * (rel - prev), prev)
        self._widths[side] = rel

        p = draw_pts[:, :2]
        hs = float(np.linalg.norm(p[0] - p[9])) or 1.0
        widths = rel * hs
        fingers, rails = [], []
        for fi, chain in enumerate(FINGERS):
            poly, left, right = _finger_outline(p[chain], widths[fi])
            fingers.append(poly)
            rails.append((left, right))
        palm = _palm(p, rails, hs)
        return HandMesh(fingers, palm, rails, widths, int(got.sum()), contour)

    @staticmethod
    def _skin_mask(frame: np.ndarray, p: np.ndarray, hs: float):
        """Máscara de pele na região da mão (ROI), o deslocamento da ROI e a escala usada.

        A ROI é reduzida para que a mão tenha ~WORK_HS píxeis (rápido e precisão suficiente)."""
        fail = (None, None, 1.0)
        m = 0.4 * hs
        x0, y0 = np.maximum(np.floor(p.min(0) - m), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(p.max(0) + m), [frame.shape[1], frame.shape[0]]).astype(int)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return fail
        off = np.array([x0, y0], dtype=np.float64)
        sc = min(1.0, WORK_HS / hs)
        roi = frame[y0:y1, x0:x1]
        if sc < 1.0:
            roi = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        hs = hs * sc
        q = np.round((p - off) * sc).astype(np.int32)
        ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        h, w = ycc.shape[:2]

        # Amostras de pele: palma encolhida + linhas finas sobre os ossos dos dedos.
        sample = np.zeros((h, w), np.uint8)
        palm = q[[0, 5, 9, 13, 17]]
        cv2.fillPoly(sample, [np.round(palm.mean(0) + (palm - palm.mean(0)) * 0.6).astype(np.int32)], 255)
        for a, b in HAND_BONES[6:]:
            cv2.line(sample, tuple(q[a]), tuple(q[b]), 255, max(1, int(0.04 * hs)))
        px = ycc[sample > 0][:, 1:3].astype(np.float64)
        if len(px) < 30:
            return fail
        mean = px.mean(0)
        (ia, ib), (_, ic) = np.linalg.inv(np.cov(px.T) + np.eye(2) * 4.0)
        cr = ycc[..., 1].astype(np.float32) - np.float32(mean[0])
        cb = ycc[..., 2].astype(np.float32) - np.float32(mean[1])
        d2 = ia * cr * cr + 2 * ib * cr * cb + ic * cb * cb  # distância de Mahalanobis²
        skin = (d2 < SKIN_SIGMA2).astype(np.uint8) * 255

        # Tubo à volta do esqueleto: impede que a cara ou o tronco entrem na máscara.
        tube = np.zeros((h, w), np.uint8)
        for a, b in HAND_BONES:
            cv2.line(tube, tuple(q[a]), tuple(q[b]), 255, max(2, int(0.55 * hs)))
        cv2.fillPoly(tube, [cv2.convexHull(q[[0, 1, 5, 9, 13, 17]])], 255)
        # Validação 1: a cor da pele tem de se distinguir do fundo à volta da mão
        # (fundo bege, ruído ou a cara atrás da mão tornam a medição pouco fiável).
        ring = (tube == 0)
        if ring.any() and (skin[ring] > 0).mean() > MAX_BG_SKIN:
            return fail
        skin &= tube
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Validação 2: os ossos devem cair maioritariamente em pele, senão o modelo de cor falhou.
        bones = np.zeros((h, w), np.uint8)
        for a, b in HAND_BONES[6:]:
            cv2.line(bones, tuple(q[a]), tuple(q[b]), 255, 1)
        if (skin[bones > 0] > 0).mean() < 0.5:
            return fail
        return skin, off, sc


_SEGMENTS = [(fi, si, c[si], c[si + 1]) for fi, c in enumerate(FINGERS) for si in range(3)
             if not (fi == 0 and si == 0)]
_FRACS = np.array([0.35, 0.5, 0.65])


def _measure_all(mask: np.ndarray, q: np.ndarray, hs: float) -> np.ndarray:
    """Largura de cada falange (5, 3), NaN se não medida.

    Em 3 pontos de cada osso procura-se, para cada lado, o primeiro píxel fora da pele.
    Se um lado nunca sai da pele (dedo encostado ao vizinho), usa-se o dobro do outro lado.
    Tudo vetorizado: segmentos x amostras x lados x passos."""
    h, w = mask.shape
    a = q[[s[2] for s in _SEGMENTS]]
    d = q[[s[3] for s in _SEGMENTS]] - a                                # (S, 2)
    length = np.linalg.norm(d, axis=1)
    t = d / np.maximum(length, 1e-6)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)                           # (S, 2)
    centers = a[:, None, :] + d[:, None, :] * _FRACS[None, :, None]      # (S, F, 2)
    steps = np.arange(0, int(0.35 * hs) + 2, dtype=np.float64)          # 0 = o próprio centro
    sgn = np.array([1.0, -1.0])
    pts = (centers[:, :, None, None, :]
           + sgn[None, None, :, None, None] * n[:, None, None, None, :] * steps[None, None, None, :, None])
    pi = np.round(pts).astype(np.int32)                                 # (S, F, 2, K, 2)
    ok = (pi[..., 0] >= 0) & (pi[..., 0] < w) & (pi[..., 1] >= 0) & (pi[..., 1] < h)
    vals = np.zeros(ok.shape, bool)
    vals[ok] = mask[pi[..., 1][ok], pi[..., 0][ok]] > 0
    center_in = vals[:, :, 0, 0]                                        # (S, F)
    exits = ~vals[..., 1:]
    has_exit = exits.any(-1)                                            # (S, F, 2)
    dist = np.where(has_exit, exits.argmax(-1) + 1.0, np.nan)
    both = dist.sum(-1)                                                 # NaN se algum lado falhou
    one = 2 * np.nanmax(np.where(has_exit, dist, -np.inf), axis=-1)
    width = np.where(np.isnan(both), np.where(has_exit.any(-1), one, np.nan), both)
    width[~center_in | (length[:, None] < 3)] = np.nan
    med = np.full(len(_SEGMENTS), np.nan)
    valid = ~np.isnan(width).all(1)
    med[valid] = np.nanmedian(width[valid], axis=1)
    out = np.full((5, 3), np.nan)
    for k, (fi, si, _, _) in enumerate(_SEGMENTS):
        out[fi, si] = med[k]
    return out


def _finger_outline(joints: np.ndarray, w3: np.ndarray):
    """Polígono do dedo: bordas esquerda/direita em cada articulação + ponta arredondada."""
    segs = np.diff(joints, axis=0)
    dirs = segs / np.maximum(np.linalg.norm(segs, axis=1, keepdims=True), 1e-6)
    jd = np.array([dirs[0], dirs[0] + dirs[1], dirs[1] + dirs[2], dirs[2]])
    jd /= np.maximum(np.linalg.norm(jd, axis=1, keepdims=True), 1e-6)
    normals = np.stack([-jd[:, 1], jd[:, 0]], axis=1)
    wj = np.array([w3[0], (w3[0] + w3[1]) / 2, (w3[1] + w3[2]) / 2, w3[2] * 0.9])
    left = joints + normals * wj[:, None] / 2
    right = joints - normals * wj[:, None] / 2
    r = wj[-1] / 2
    ang = np.linspace(0, np.pi, 9)[1:-1]
    cap = joints[-1] + np.outer(np.cos(ang), normals[-1] * r) + np.outer(np.sin(ang), dirs[-1] * r)
    return np.vstack([left, cap, right[::-1]]), left, right


def _palm(p: np.ndarray, rails, hs: float) -> np.ndarray:
    up = p[9] - p[0]
    up = up / (np.linalg.norm(up) or 1.0)
    n = np.array([-up[1], up[0]])
    pts = [p[0] + n * WRIST_HALF * hs, p[0] - n * WRIST_HALF * hs]
    # Polegar: as duas primeiras articulações fazem parte da palma (eminência tenar).
    thumb_l, thumb_r = rails[0]
    pts += [thumb_l[0], thumb_r[0], thumb_l[1], thumb_r[1]]
    for left, right in rails[1:]:
        pts += [left[0], right[0]]
    pts = np.array(pts)
    return cv2.convexHull(pts.astype(np.float32))[:, 0, :].astype(np.float64)
