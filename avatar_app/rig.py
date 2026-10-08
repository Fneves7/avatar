"""Rig: parâmetros do avatar independentes de como é desenhado.

O estilo cartoon desenha diretamente a partir da malha, mas um avatar de outro estilo
(robô, PNG por camadas, 3D, ...) só precisa de saber, em cada frame, coisas como:
onde está a cabeça e como está rodada, quanto está aberto cada olho e para onde olha,
quanto está aberta a boca e se sorri, a posição das sobrancelhas, a respiração e o
esqueleto do tronco, braços e mãos. É isso que o Rig contém (parecido com os parâmetros
de um modelo VTuber). Tudo em píxeis da imagem, ângulos em radianos, expressões em 0..1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .animation import AnimFrame
from .landmarks import (BROW_A, BROW_B, CHIN, EYE_A, EYE_B, FACE_LEFT, FACE_OVAL, FACE_RIGHT, FOREHEAD,
                        IRIS_A, IRIS_B, LIP_BOTTOM, LIP_TOP, MOUTH_CORNERS)
from .tracker import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, BodyState

AUTO_BLINK_SHUT = 0.85    # piscar automático: a partir daqui o olho está fechado


@dataclass
class RigHead:
    center: np.ndarray          # centro da cara (px)
    size: float                 # largura da cara (px)
    height: float               # altura da cara, testa ao queixo (px)
    roll: float                 # inclinação (rad, positivo = sentido dos ponteiros do relógio na imagem)
    yaw: float                  # rotação para os lados (rad, aproximada)
    pitch: float                # rotação para cima/baixo (rad, aproximada)
    eye_centers: np.ndarray     # (2, 2): olho A (landmark 33) e olho B (263)
    eye_open: np.ndarray        # (2,) 0 = fechado, 1 = aberto normal, >1 = arregalado
    look: np.ndarray            # (2,) direção do olhar (-1..1 em x e y, no referencial da cara)
    brow: float                 # -1 = franzir .. 0 neutro .. +1 levantadas
    mouth_center: np.ndarray    # (2,)
    mouth_width: float          # px
    mouth_open: float           # 0..1
    mouth_smile: float          # 0..1
    turn: float = 0.0           # para onde aponta o nariz: -1 (esquerda da imagem) .. 1 (direita)
    alpha: float = 1.0          # opacidade (transições)


@dataclass
class RigArm:
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    forearm_visible: bool


@dataclass
class Rig:
    width: int
    height: int
    t: float
    head: RigHead | None = None
    shoulders: np.ndarray | None = None      # (2, 2) [L_SHOULDER, R_SHOULDER]
    hips: np.ndarray | None = None           # (2, 2) (estimadas se fora de imagem)
    neck_base: np.ndarray | None = None
    shoulder_width: float = 0.0
    arms: dict[int, RigArm] = field(default_factory=dict)          # chave: pulso (15/16)
    hands: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)  # (21x3 pontos, opacidade)
    breath: float = 0.0                       # 0..1
    hair_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hair_rot: float = 0.0
    sleeve_offset: dict[int, np.ndarray] = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _head(s: BodyState, face: np.ndarray, bs: dict[str, float], frame: AnimFrame, alpha: float) -> RigHead:
    f = face[:, :2]
    center = f[FACE_OVAL].mean(axis=0)
    fw = float(np.linalg.norm(f[FACE_RIGHT] - f[FACE_LEFT])) or 1.0
    fh = float(np.linalg.norm(f[CHIN] - f[FOREHEAD])) or 1.0
    eye_line = f[EYE_B[0]] - f[EYE_A[0]]
    if eye_line[0] < 0:
        eye_line = -eye_line  # vale com a imagem espelhada ou não
    roll = math.atan2(eye_line[1], eye_line[0])
    if s.head_angles is not None:
        yaw, pitch = math.radians(s.head_angles[0]), math.radians(s.head_angles[1])
    else:
        # Aproximação pela posição do nariz em relação ao centro da cara.
        side = _unit(eye_line)
        yaw = math.asin(float(np.clip((f[1] - center) @ side / (0.5 * fw), -1, 1)))
        pitch = 0.0

    calibrated = s.calibrated and bool(bs)
    eye_centers = np.array([f[EYE_A].mean(axis=0), f[EYE_B].mean(axis=0)])
    eye_open = np.ones(2)
    look = np.zeros(2)
    # Eixos da cara (iguais para os dois olhos, para o olhar ter o mesmo sentido em ambos):
    # u = para a direita da imagem ao longo da linha dos olhos, n = para baixo.
    u = _unit(eye_line)
    n = np.array([-u[1], u[0]])
    for i, (contour, (ci, _), blink_key, wide_key) in enumerate(
            ((EYE_A, IRIS_A, "eyeBlinkRight", "eyeWideRight"), (EYE_B, IRIS_B, "eyeBlinkLeft", "eyeWideLeft"))):
        eye = f[contour]
        half_w = 0.5 * float(np.linalg.norm(eye[8] - eye[0])) or 1.0
        half_h = 0.5 * float(np.ptp((eye - eye_centers[i]) @ n)) or 1.0
        if calibrated:
            value = 1.0 - bs.get(blink_key, 0.0) + 0.5 * bs.get(wide_key, 0.0)
        else:
            value = float(np.clip((half_h / half_w) / 0.32, 0.0, 1.5))  # abertura geométrica
        if frame.eyes_closed[i]:
            value = 0.0
        value *= 1.0 - frame.auto_blink
        if frame.auto_blink > AUTO_BLINK_SHUT:
            value = 0.0
        eye_open[i] = value
        if len(face) > ci:
            rel = f[ci] - eye_centers[i]
            look += np.array([rel @ u / half_w, rel @ n / max(half_h, 0.35 * half_w)])
    look = np.clip(look / 2 + np.array(frame.saccade), -1.0, 1.0)

    brow = 0.0
    if calibrated:
        brow = float(np.clip(bs.get("browInnerUp", 0.0)
                             - (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2, -1, 1))
    mouth_center = (f[LIP_TOP] + f[LIP_BOTTOM]) / 2
    mouth_width = float(np.linalg.norm(f[MOUTH_CORNERS[0]] - f[MOUTH_CORNERS[1]]))
    gap = float(np.linalg.norm(f[LIP_TOP] - f[LIP_BOTTOM]))
    if calibrated:
        mouth_open = float(np.clip(max(bs.get("jawOpen", 0.0), gap / (0.25 * fh)), 0, 1))
        smile = float(np.clip((bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2, 0, 1))
    else:
        mouth_open = float(np.clip(gap / (0.25 * fh), 0, 1))
        smile = 0.0
    turn = float(np.clip((f[1] - center) @ _unit(eye_line) / (0.35 * fw), -1, 1))
    return RigHead(center=center, size=fw, height=fh, roll=roll, yaw=yaw, pitch=pitch,
                   eye_centers=eye_centers, eye_open=eye_open, look=look, brow=brow,
                   mouth_center=mouth_center, mouth_width=mouth_width, mouth_open=mouth_open,
                   mouth_smile=smile, turn=turn, alpha=alpha)


def build_rig(frame: AnimFrame) -> Rig:
    s = frame.state
    rig = Rig(width=s.width, height=s.height, t=frame.t, breath=frame.breath,
              hair_offset=frame.hair_offset, hair_rot=frame.hair_rot, sleeve_offset=frame.sleeve_offset)

    # Cabeça (a atual, ou a última vista enquanto desvanece).
    fade = frame.face_fade
    if s.face is not None:
        rig.head = _head(s, s.face, s.blendshapes, frame, fade.alpha if fade else 1.0)
    elif fade is not None and fade.visible:
        last_face, last_bs = fade.data
        rig.head = _head(s, last_face, last_bs, frame, fade.alpha)

    # Tronco e braços.
    if s.pose is not None:
        ls, rs = s.pose[L_SHOULDER, :2], s.pose[R_SHOULDER, :2]
        sw = float(np.linalg.norm(rs - ls)) or 1.0
        axis = (rs - ls) / sw
        up = np.array([axis[1], -axis[0]])
        if up[1] > 0:
            up = -up
        rig.shoulders = np.array([ls, rs])
        rig.shoulder_width = sw
        rig.neck_base = (ls + rs) / 2 + up * 0.2 * sw
        if s.visible(L_HIP, 0.3) and s.visible(R_HIP, 0.3):
            rig.hips = np.array([s.pose[L_HIP, :2], s.pose[R_HIP, :2]])
        else:
            rig.hips = np.array([ls - up * 1.4 * sw, rs - up * 1.4 * sw])
        for wrist, (shoulder, elbow) in ((L_WRIST, (L_SHOULDER, L_ELBOW)), (R_WRIST, (R_SHOULDER, R_ELBOW))):
            hand = s.hands.get(wrist)
            if not s.visible(elbow, 0.25) and hand is None:
                continue
            wr = hand[0, :2] if hand is not None else s.pose[wrist, :2]
            rig.arms[wrist] = RigArm(shoulder=s.pose[shoulder, :2], elbow=s.pose[elbow, :2], wrist=wr,
                                     forearm_visible=hand is not None or s.visible(wrist, 0.25))
    elif rig.head is not None:
        rig.shoulder_width = 2.2 * rig.head.size

    # Mãos (com a opacidade da transição, incluindo as que estão a desaparecer).
    for side, hf in frame.hand_fades.items():
        if hf.visible:
            rig.hands[side] = (hf.data[0], hf.alpha)
    return rig


def rig_to_dict(rig: Rig) -> dict:
    """O rig em JSON para o avatar 3D (página /3d da Fonte de Browser).

    Coordenadas normalizadas pela altura da imagem, com a origem no centro e y para cima
    (como num referencial 3D): x em [-aspeto, aspeto], y em [-1, 1]. Tamanhos na mesma escala.
    "olho esquerdo/direito" é o da imagem (esquerda/direita de quem vê)."""
    w, h = rig.width, rig.height

    def pt(p) -> list[float]:
        return [round((float(p[0]) - w / 2) / h * 2, 4), round((h / 2 - float(p[1])) / h * 2, 4)]

    def size(v: float) -> float:
        return round(float(v) / h * 2, 4)

    out: dict = {"aspect": round(w / h, 4), "breath": round(rig.breath, 3),
                 "hair": [size(rig.hair_offset[0]), -size(rig.hair_offset[1])], "hair_rot": round(-rig.hair_rot, 4)}
    hd = rig.head
    if hd is not None:
        left = 0 if hd.eye_centers[0][0] <= hd.eye_centers[1][0] else 1
        out["head"] = {
            "center": pt(hd.center), "size": size(hd.size), "height": size(hd.height),
            # Ângulos no referencial 3D (y para cima): roll positivo = sentido anti-horário.
            "roll": round(-hd.roll, 4), "yaw": round(hd.yaw, 4), "pitch": round(hd.pitch, 4),
            "turn": round(hd.turn, 4),
            "eye_open": [round(float(hd.eye_open[left]), 3), round(float(hd.eye_open[1 - left]), 3)],
            "look": [round(float(hd.look[0]), 3), round(-float(hd.look[1]), 3)],
            "brow": round(hd.brow, 3), "mouth_open": round(hd.mouth_open, 3),
            "mouth_smile": round(hd.mouth_smile, 3), "alpha": round(hd.alpha, 3),
        }
    if rig.shoulders is not None:
        out["body"] = {"shoulders": [pt(p) for p in rig.shoulders], "hips": [pt(p) for p in rig.hips],
                       "neck_base": pt(rig.neck_base), "width": size(rig.shoulder_width)}
    out["arms"] = {str(side): {"shoulder": pt(a.shoulder), "elbow": pt(a.elbow), "wrist": pt(a.wrist),
                               "sleeve": [size(v) * s for v, s in zip(rig.sleeve_offset.get(side, (0, 0)), (1, -1))]}
                   for side, a in rig.arms.items()}
    out["hands"] = {str(side): {"points": [pt(p) for p in pts[:21]], "alpha": round(alpha, 3)}
                    for side, (pts, alpha) in rig.hands.items()}
    return out
