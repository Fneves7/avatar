"""Desenha os landmarks detetados por cima da imagem da webcam."""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from .hand_mesh import HandMesh
from .head import estimate_head
from .landmarks import FACE_OVAL
from .tracker import BodyState

_FACE = [(c.start, c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]
_IRIS = [(c.start, c.end) for c in (vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
                                     + vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)]
# Contorno do nariz (não existe nas ligações do mediapipe 0.10.21).
_NOSE_CHAIN = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
_NOSE_LOOP = [4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4]
_NOSE = list(zip(_NOSE_CHAIN, _NOSE_CHAIN[1:])) + list(zip(_NOSE_LOOP, _NOSE_LOOP[1:]))
_POSE =[(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
_HAND = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

_FACE_COLOR = (200, 230, 120)
_POSE_COLOR = (80, 200, 255)
_HAND_COLOR = (255, 120, 200)
_HEAD_COLOR = (0, 200, 255)
_HAND_MESH_COLOR = (255, 230, 80)
_HAND_CONTOUR_COLOR = (0, 255, 255)


def _lines(img, pts, conns, color, thickness=1, mask=None):
    p = pts[:, :2].astype(np.int32)
    for a, b in conns:
        if mask is not None and not (mask[a] and mask[b]):
            continue
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)


def _draw_hand_mesh(img: np.ndarray, mesh: HandMesh) -> None:
    """Contorno real segmentado (amarelo) + malha dos dedos (triângulos, ciano)."""
    if mesh.contour is not None and len(mesh.contour) >= 3:
        cv2.polylines(img, [mesh.contour.astype(np.int32)], True, _HAND_CONTOUR_COLOR, 1, cv2.LINE_AA)
    for left, right in mesh.rails:
        l, r = left.astype(np.int32), right.astype(np.int32)
        for i in range(len(l)):
            cv2.line(img, tuple(l[i]), tuple(r[i]), _HAND_MESH_COLOR, 1, cv2.LINE_AA)
            if i + 1 < len(l):
                cv2.line(img, tuple(l[i]), tuple(r[i + 1]), _HAND_MESH_COLOR, 1, cv2.LINE_AA)
    for poly in mesh.fingers:
        cv2.polylines(img, [poly.astype(np.int32)], True, _HAND_MESH_COLOR, 1, cv2.LINE_AA)
    cv2.polylines(img, [mesh.palm.astype(np.int32)], True, _HAND_MESH_COLOR, 1, cv2.LINE_AA)


def draw_landmarks(img: np.ndarray, s: BodyState) -> None:
    if s.pose is not None:
        vis = s.pose_visibility >= 0.5
        # Ignora os pontos da cara da pose (0-10): o FaceLandmarker é muito mais preciso.
        conns = [(a, b) for a, b in _POSE if a > 10 and b > 10]
        _lines(img, s.pose, conns, _POSE_COLOR, 2, vis)
        for i in range(11, 33):
            if vis[i]:
                cv2.circle(img, tuple(s.pose[i, :2].astype(int)), 4, _POSE_COLOR, -1, cv2.LINE_AA)
    if s.face is not None:
        geo = estimate_head(s.face, FACE_OVAL)
        if geo is not None:  # cabeça completa estimada (crânio + orelhas)
            cv2.polylines(img, [geo.skull], True, _HEAD_COLOR, 1, cv2.LINE_AA)
            for ear in geo.ears:
                if not ear.hidden:
                    cv2.polylines(img, [ear.polygon], True, _HEAD_COLOR, 1, cv2.LINE_AA)
        _lines(img, s.face, _FACE, _FACE_COLOR, 1)
        _lines(img, s.face, _NOSE, _FACE_COLOR, 1)
        _lines(img, s.face, _IRIS, (255, 255, 255), 1)
    for mesh in s.hand_meshes.values():
        _draw_hand_mesh(img, mesh)
    for hand in s.hands.values():
        _lines(img, hand, _HAND, _HAND_COLOR, 2)
        for p in hand[:, :2].astype(int):
            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
