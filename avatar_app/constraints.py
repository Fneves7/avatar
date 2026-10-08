"""Limites anatómicos da pose: corrige só o que é claramente impossível.

A pose do MediaPipe inventa articulações quando estão fora de imagem ou pouco visíveis
(por exemplo, um cotovelo em cima do tronco quando o braço real está por baixo da imagem).
Regras, todas conservadoras:
  * saltos: um cotovelo/pulso pouco visível que salta mais de meia largura de ombros entre
    duas deteções fica onde estava;
  * cotovelo vs. mão: se o antebraço fizer um ângulo impossível com o eixo da mão detetada
    (e o cotovelo não estiver claramente visível), o cotovelo passa para o prolongamento
    da mão; o pulso da pose passa a ser o pulso da mão;
  * braços esticados demais: braço/antebraço com mais de MAX_SEGMENT x a largura dos
    ombros é encurtado;
  * ancas acima dos ombros: as ancas são ignoradas (o tronco é estimado a partir dos ombros).
Cada correção fica registada para o HUD.
"""
from __future__ import annotations

import numpy as np

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
ARMS = {L_WRIST: (L_SHOULDER, L_ELBOW, "left"), R_WRIST: (R_SHOULDER, R_ELBOW, "right")}  # nomes para o HUD

GLITCH_JUMP = 0.5          # salto máximo (* largura dos ombros) de uma articulação pouco visível
GLITCH_VISIBILITY = 0.5
GLITCH_MAX_HOLD = 3        # deteções seguidas em que se ignora o salto antes de o aceitar
WRIST_BEND_FIX = 60.0      # graus entre antebraço e mão a partir dos quais se corrige (cotovelo pouco visível)
WRIST_BEND_ALWAYS = 110.0  # graus a partir dos quais se corrige sempre
ELBOW_TRUSTED = 0.85       # visibilidade a partir da qual o cotovelo é "claramente visível"
FOREARM_LENGTH = (0.6, 1.1)  # comprimento do antebraço ao recolocar o cotovelo (* largura dos ombros)
MAX_SEGMENT = 1.5          # braço/antebraço mais comprido do que isto (* largura dos ombros) é encurtado


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def shoulder_width(pose: np.ndarray) -> float:
    return float(np.linalg.norm(pose[L_SHOULDER, :2] - pose[R_SHOULDER, :2]))


def reject_glitches(raw: np.ndarray, prev: np.ndarray | None, vis: np.ndarray,
                    fixes: list[str], held: dict[int, int]) -> np.ndarray:
    """Antes de suavizar: articulações pouco visíveis que saltam demasiado ficam onde estavam.
    held conta as deteções seguidas em que cada articulação foi retida: ao fim de
    GLITCH_MAX_HOLD aceita-se a nova posição (o braço pode ter mesmo mudado de sítio)."""
    if prev is None or prev.shape != raw.shape:
        held.clear()
        return raw
    sw = shoulder_width(prev)
    if sw < 1:
        return raw
    out = raw.copy()
    for wrist, (_, elbow, name) in ARMS.items():
        for j, label in ((elbow, "elbow"), (wrist, "wrist")):
            jump = np.linalg.norm(raw[j, :2] - prev[j, :2]) > GLITCH_JUMP * sw
            if jump and vis[j] < GLITCH_VISIBILITY and held.get(j, 0) < GLITCH_MAX_HOLD:
                out[j] = prev[j]
                held[j] = held.get(j, 0) + 1
                fixes.append(f"{name} {label}: jump ignored")
            else:
                held[j] = 0
    return out


def apply_constraints(pose: np.ndarray, vis: np.ndarray, hands: dict[int, np.ndarray],
                      fixes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Depois de suavizar: corrige cotovelos, braços esticados e ancas. Devolve (pose, visibilidade)."""
    pose = pose.copy()
    vis = vis.copy()
    sw = shoulder_width(pose)
    if sw < 1:
        return pose, vis

    for wrist, (shoulder, elbow, name) in ARMS.items():
        hand = hands.get(wrist)
        if hand is not None:
            pose[wrist, :2] = hand[0, :2]  # o pulso da mão é mais preciso do que o da pose
            axis = hand[9, :2] - hand[0, :2]
            fore = pose[wrist, :2] - pose[elbow, :2]
            if np.linalg.norm(axis) > 1e-6 and np.linalg.norm(fore) > 1e-6:
                cos = float(np.clip(np.dot(_unit(axis), _unit(fore)), -1, 1))
                bend = float(np.degrees(np.arccos(cos)))
                if bend > WRIST_BEND_ALWAYS or (bend > WRIST_BEND_FIX and vis[elbow] < ELBOW_TRUSTED):
                    length = float(np.clip(np.linalg.norm(fore), FOREARM_LENGTH[0] * sw, FOREARM_LENGTH[1] * sw))
                    pose[elbow, :2] = pose[wrist, :2] - _unit(axis) * length
                    fixes.append(f"{name} elbow: moved to follow the hand ({bend:.0f} deg)")

        # Braços esticados demais.
        upper = pose[elbow, :2] - pose[shoulder, :2]
        if np.linalg.norm(upper) > MAX_SEGMENT * sw:
            pose[elbow, :2] = pose[shoulder, :2] + _unit(upper) * MAX_SEGMENT * sw
            fixes.append(f"{name} upper arm: shortened")
        fore = pose[wrist, :2] - pose[elbow, :2]
        if np.linalg.norm(fore) > MAX_SEGMENT * sw and hand is None:
            pose[wrist, :2] = pose[elbow, :2] + _unit(fore) * MAX_SEGMENT * sw
            fixes.append(f"{name} forearm: shortened")

    # Ancas acima dos ombros (pessoa de pé/sentada): pose impossível, ignora as ancas.
    if (vis[L_HIP] > 0 or vis[R_HIP] > 0) and \
            (pose[L_HIP, 1] + pose[R_HIP, 1]) / 2 < (pose[L_SHOULDER, 1] + pose[R_SHOULDER, 1]) / 2:
        vis[[L_HIP, R_HIP]] = 0.0
        fixes.append("hips above shoulders: ignored")
    return pose, vis
