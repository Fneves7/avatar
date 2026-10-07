"""Índices dos landmarks do MediaPipe usados pelo rig e pelos estilos."""
from __future__ import annotations

from .tracker import L_WRIST, R_WRIST

# --- FaceLandmarker / Face Mesh (478 pontos) ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
IRIS_A, IRIS_B = (468, 469), (473, 474)  # (centro, ponto do contorno)
BROW_A = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_B = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS_SEAM = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 95, 88, 178, 87, 14, 317, 402, 318, 324]
MOUTH_PTS = sorted(set(LIPS_OUTER + LIPS_INNER + LIPS_SEAM))
MOUTH_CORNERS = (61, 291)
LIP_TOP, LIP_BOTTOM = 13, 14
NOSE_TIP = 1
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_LOWER = [4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275]
NOSE_ALA_A = [115, 48, 64, 98]          # asa de um lado (de cima para a base)
NOSE_ALA_B = [344, 278, 294, 327]       # asa do outro lado
NOSE_BASE = [98, 97, 2, 326, 327]
CHEEK_A, CHEEK_B = 50, 280
FACE_LEFT, FACE_RIGHT, CHIN, FOREHEAD = 234, 454, 152, 10

# --- HandLandmarker ---
PALM = [0, 1, 2, 5, 9, 13, 17]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]

# Pontos da mão na pose (fallback quando o HandLandmarker não deteta a mão).
POSE_HAND = {L_WRIST: (17, 19, 21), R_WRIST: (18, 20, 22)}  # pinky, index, thumb
