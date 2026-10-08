"""Deteção com MediaPipe: rosto (468/478 pts), corpo (33 pts) e mãos (21 pts cada).

Dois motores:
  * "holistic" (mediapipe <= 0.10.21): modelos incluídos no pacote, não precisa de downloads.
  * "tasks" (FaceLandmarker + PoseLandmarker + HandLandmarker): precisa dos ficheiros .task
    em models/, mas fornece blendshapes e a matriz de rotação da cabeça.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from .body_widths import BodyWidths
from .constraints import apply_constraints, reject_glitches
from .face_filter import RigidFaceFilter
from .hand_mesh import HandMesh, HandMeshEstimator
from .smoothing import OneEuroFilter

# Índices do PoseLandmarker usados pelo avatar.
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

HOLD_SECONDS = 0.25  # mantém a última pose de uma parte que deixou de ser detetada

# Perfis de suavização: multiplicadores de (min_cutoff, beta) dos filtros One Euro.
# Menos min_cutoff = mais suave/estável; mais beta = menos atraso em movimentos rápidos.
SMOOTHING_PRESETS = {"leve": (2.0, 1.5), "normal": (1.0, 1.0), "forte": (0.5, 0.7)}

HAS_HOLISTIC = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")


@dataclass
class BodyState:
    """Landmarks em píxeis (x, y, z) do frame já espelhado."""

    width: int
    height: int
    face: np.ndarray | None = None              # (468 ou 478, 3)
    blendshapes: dict[str, float] = field(default_factory=dict)
    head_angles: tuple[float, float, float] | None = None  # yaw, pitch, roll (graus)
    pose: np.ndarray | None = None              # (33, 3)
    pose_visibility: np.ndarray | None = None   # (33,)
    hands: dict[int, np.ndarray] = field(default_factory=dict)  # chave = índice do pulso na pose (15/16)
    hand_meshes: dict[int, HandMesh] = field(default_factory=dict)  # contorno dos dedos por mão
    calibrated: bool = False  # head_angles/blendshapes já relativos à pose neutra
    pose_fixes: list[str] = field(default_factory=list)  # correções anatómicas feitas neste frame
    body_widths: dict[str, float] = field(default_factory=dict)  # fatores de largura (--body-widths)

    def visible(self, idx: int, thr: float = 0.5) -> bool:
        return self.pose is not None and self.pose_visibility is not None and self.pose_visibility[idx] >= thr


@dataclass
class _Raw:
    face: np.ndarray | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    head_angles: np.ndarray | None = None
    pose: np.ndarray | None = None
    visibility: np.ndarray | None = None
    hands: dict[int, np.ndarray] = field(default_factory=dict)
    mask: np.ndarray | None = None  # silhueta da pessoa (0..1), só com segmentação


class _Part:
    """Suavização + retenção curta de uma parte do corpo."""

    def __init__(self, min_cutoff: float, beta: float):
        self.filter = OneEuroFilter(min_cutoff, beta)
        self.value: np.ndarray | None = None
        self.last_seen = -1e9

    def update(self, pts: np.ndarray | None, t: float) -> np.ndarray | None:
        if pts is not None:
            self.value = self.filter(pts, t)
            self.last_seen = t
        elif t - self.last_seen > HOLD_SECONDS:
            self.value = None
            self.filter.reset()
        return self.value


def _to_array(landmarks, scale: np.ndarray) -> np.ndarray:
    return np.array([(p.x, p.y, p.z) for p in landmarks]) * scale


def _matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(m)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def _geometry_expressions(f: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Aproxima blendshapes e rotação da cabeça a partir da malha (motor holistic)."""
    fw = float(np.linalg.norm(f[234, :2] - f[454, :2])) or 1.0
    fh = float(np.linalg.norm(f[10, :2] - f[152, :2])) or 1.0

    def eye_open(top, bottom, a, b):
        return np.linalg.norm(f[top, :2] - f[bottom, :2]) / (np.linalg.norm(f[a, :2] - f[b, :2]) or 1.0)

    def blink(ratio):
        return float(np.clip(1.0 - (ratio - 0.12) / 0.15, 0, 1))

    # Roll pela linha dos olhos; o sorriso mede-se no referencial da cara (sem inclinação).
    eye_vec = f[263, :2] - f[33, :2]
    roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))
    c, s = math.cos(-math.radians(roll)), math.sin(-math.radians(roll))
    rot = np.array([[c, -s], [s, c]])
    lip_mid = rot @ ((f[13, :2] + f[14, :2]) / 2)
    corners = (rot @ f[61, :2] + rot @ f[291, :2]) / 2
    # Sorriso = cantos a subir + boca a alargar. Escalas centradas de forma a que a cara
    # neutra fique a meio (a calibração remove depois o valor de repouso de cada pessoa).
    lift = np.clip(((lip_mid[1] - corners[1]) / fw + 0.03) / 0.08, 0, 1)
    widen = np.clip((np.linalg.norm(f[61, :2] - f[291, :2]) / fw - 0.30) / 0.25, 0, 1)
    smile = float(0.5 * lift + 0.5 * widen)
    jaw = float(np.clip(np.linalg.norm(f[13, :2] - f[14, :2]) / fh / 0.25, 0, 1))
    # Sobrancelhas: distância sobrancelha-pálpebra superior relativa à altura da cara.
    brow = (np.linalg.norm(f[105, :2] - f[159, :2]) + np.linalg.norm(f[334, :2] - f[386, :2])) / 2 / fh
    brow_up = float(np.clip((brow - 0.06) / 0.10, 0, 1))
    brow_down = float(np.clip((0.16 - brow) / 0.10, 0, 1))

    bs = {
        "eyeBlinkLeft": blink(eye_open(386, 374, 263, 362)),
        "eyeBlinkRight": blink(eye_open(159, 145, 33, 133)),
        "mouthSmileLeft": smile, "mouthSmileRight": smile, "jawOpen": jaw,
        "browInnerUp": brow_up, "browDownLeft": brow_down, "browDownRight": brow_down,
    }
    # Yaw e pitch pela profundidade (z) da malha: lado-a-lado da cara e testa-queixo.
    side = f[454] - f[234]
    yaw = math.degrees(math.atan2(side[2], side[0]))
    vert = f[152] - f[10]
    pitch = math.degrees(math.atan2(vert[2], vert[1]))
    return bs, np.array([yaw, pitch, roll])


class _HolisticBackend:
    name = "holistic"

    def __init__(self, pose_model: str, segmentation: bool = False):
        self.segmentation = segmentation
        complexity = {"lite": 0, "full": 1, "heavy": 2}[pose_model]
        # 0.10.21 só inclui o modelo "full"; os outros seriam descarregados da Google.
        if complexity != 1:
            print("[holistic] só o modelo de pose 'full' vem incluído; a usar 'full'.")
            complexity = 1
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=complexity, smooth_landmarks=True,
            refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,
            enable_segmentation=segmentation, smooth_segmentation=segmentation,
        )

    def close(self) -> None:
        self.holistic.close()

    def detect(self, rgb: np.ndarray, scale: np.ndarray) -> _Raw:
        raw = _Raw()
        rgb.flags.writeable = False
        r = self.holistic.process(rgb)
        if r.face_landmarks:
            raw.face = _to_array(r.face_landmarks.landmark, scale)
            raw.blendshapes, raw.head_angles = _geometry_expressions(raw.face)
        if r.pose_landmarks:
            lms = r.pose_landmarks.landmark
            raw.pose = _to_array(lms, scale)
            raw.visibility = np.array([p.visibility for p in lms])
        if self.segmentation and r.segmentation_mask is not None:
            raw.mask = r.segmentation_mask
        # As mãos do holistic são recortadas a partir dos pulsos da pose, por isso
        # "left_hand" corresponde sempre ao pulso 15 e "right_hand" ao 16.
        if r.left_hand_landmarks:
            raw.hands[L_WRIST] = _to_array(r.left_hand_landmarks.landmark, scale)
        if r.right_hand_landmarks:
            raw.hands[R_WRIST] = _to_array(r.right_hand_landmarks.landmark, scale)
        return raw


class _TasksBackend:
    name = "tasks"

    def __init__(self, pose_model: str, num_hands: int = 2):
        from mediapipe.tasks.python import BaseOptions, vision

        from .models import get_model

        mode = vision.RunningMode.VIDEO
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(get_model("face"))),
            running_mode=mode, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        ))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(get_model(f"pose_{pose_model}"))),
            running_mode=mode, num_poses=1,
        ))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(get_model("hand"))),
            running_mode=mode, num_hands=num_hands,
        ))
        self._last_ts = 0

    def close(self) -> None:
        for task in (self.face, self.pose, self.hands):
            task.close()

    def _timestamp_ms(self) -> int:
        ts = int(time.monotonic() * 1000)
        ts = max(ts, self._last_ts + 1)  # o modo VIDEO exige timestamps estritamente crescentes
        self._last_ts = ts
        return ts

    def detect(self, rgb: np.ndarray, scale: np.ndarray) -> _Raw:
        raw = _Raw()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()

        fr = self.face.detect_for_video(image, ts)
        if fr.face_landmarks:
            raw.face = _to_array(fr.face_landmarks[0], scale)
            if fr.face_blendshapes:
                raw.blendshapes = {c.category_name: c.score for c in fr.face_blendshapes[0]}
            if fr.facial_transformation_matrixes:
                raw.head_angles = np.array(_matrix_to_euler(fr.facial_transformation_matrixes[0]))

        pr = self.pose.detect_for_video(image, ts)
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            raw.pose = _to_array(lms, scale)
            raw.visibility = np.array([p.visibility if p.visibility is not None else 1.0 for p in lms])

        hr = self.hands.detect_for_video(image, ts)
        for i, lms in enumerate(hr.hand_landmarks):
            pts = _to_array(lms, scale)
            label = hr.handedness[i][0].category_name if hr.handedness else None
            side = self._assign_hand(pts, raw.pose, label)
            if side in raw.hands:  # duas mãos para o mesmo pulso: fica a outra
                side = R_WRIST if side == L_WRIST else L_WRIST
            raw.hands[side] = pts
        return raw

    @staticmethod
    def _assign_hand(pts: np.ndarray, pose: np.ndarray | None, label: str | None) -> int:
        if pose is not None:
            wrist = pts[0, :2]
            dl = np.linalg.norm(pose[L_WRIST, :2] - wrist)
            dr = np.linalg.norm(pose[R_WRIST, :2] - wrist)
            return L_WRIST if dl < dr else R_WRIST
        # Sem pose: a imagem está espelhada, por isso o rótulo do MediaPipe fica trocado.
        return L_WRIST if label == "Right" else R_WRIST


class Tracker:
    def __init__(self, pose_model: str = "full", backend: str = "auto", smoothing: bool = True,
                 hand_mesh: bool = True, body_widths: bool = False):
        if backend == "auto":
            backend = "holistic" if HAS_HOLISTIC else "tasks"
        if backend == "holistic" and not HAS_HOLISTIC:
            raise SystemExit("O motor 'holistic' precisa de mediapipe<=0.10.21 (pip install mediapipe==0.10.21).")
        # A silhueta (para afinar as larguras do corpo) só existe no motor holistic e custa algum tempo.
        self.backend = _HolisticBackend(pose_model, segmentation=body_widths) if backend == "holistic"             else _TasksBackend(pose_model)
        self.body_widths = BodyWidths() if body_widths and backend == "holistic" else None
        print(f"[tracker] motor: {self.backend.name}")

        self.smoothing = smoothing
        # O rosto precisa de pouco atraso (expressões); o corpo pode ser mais suave.
        self._face = _Part(min_cutoff=2.0, beta=0.08)
        self._face_point_filter = self._face.filter
        self._face_rigid_filter = RigidFaceFilter(min_cutoff=2.0, beta=0.08)
        self._pose = _Part(min_cutoff=1.2, beta=0.04)
        self._hands = {L_WRIST: _Part(1.5, 0.06), R_WRIST: _Part(1.5, 0.06)}
        self._angles_filter = OneEuroFilter(1.0, 0.02)
        self._filter_base = {f: (f.min_cutoff, f.beta) for f in self._filters()}
        self.smoothing_preset = "normal"
        self.hand_mesh = HandMeshEstimator() if hand_mesh else None
        self.constraints = True   # limites anatómicos da pose (tecla a)
        self._glitch_held: dict[int, int] = {}
        self.rigid_face = True    # cara suavizada como um todo + forma local (tecla r)

    @property
    def rigid_face(self) -> bool:
        return self._face.filter is self._face_rigid_filter

    @rigid_face.setter
    def rigid_face(self, on: bool) -> None:
        new = self._face_rigid_filter if on else self._face_point_filter
        if new is not self._face.filter:
            new.reset()
            self._face.filter = new

    def _filters(self) -> list[OneEuroFilter]:
        return [self._face_point_filter, *self._face_rigid_filter.filters(), self._pose.filter,
                self._angles_filter] + [p.filter for p in self._hands.values()]

    def set_smoothing_preset(self, name: str) -> None:
        """leve = mais rápido (menos atraso, mais tremor); forte = mais estável (mais atraso)."""
        k_cut, k_beta = SMOOTHING_PRESETS[name]
        for f, (cut, beta) in self._filter_base.items():
            f.min_cutoff, f.beta = cut * k_cut, beta * k_beta
        self.smoothing_preset = name

    def next_smoothing_preset(self) -> str:
        names = list(SMOOTHING_PRESETS)
        self.set_smoothing_preset(names[(names.index(self.smoothing_preset) + 1) % len(names)])
        return self.smoothing_preset

    def close(self) -> None:
        self.backend.close()

    def process(self, frame_bgr: np.ndarray) -> BodyState:
        h, w = frame_bgr.shape[:2]
        scale = np.array([w, h, w], dtype=np.float64)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = time.monotonic()
        raw = self.backend.detect(rgb, scale)

        state = BodyState(width=w, height=h, blendshapes=raw.blendshapes)
        if self.constraints and raw.pose is not None and raw.visibility is not None:
            raw.pose = reject_glitches(raw.pose, self._pose.value, raw.visibility, state.pose_fixes,
                                       self._glitch_held)
        if raw.head_angles is not None:
            ang = self._angles_filter(raw.head_angles, t) if self.smoothing else raw.head_angles
            state.head_angles = tuple(float(a) for a in ang)
        state.face = self._smooth(self._face, raw.face, t)
        state.pose = self._smooth(self._pose, raw.pose, t)
        if state.pose is not None:
            state.pose_visibility = raw.visibility if raw.visibility is not None else np.ones(33)
        for side, part in self._hands.items():
            val = self._smooth(part, raw.hands.get(side), t)
            if val is not None:
                state.hands[side] = val
                if self.hand_mesh is not None:
                    # Mede com os pontos em bruto (alinhados com o frame), desenha com os suavizados.
                    state.hand_meshes[side] = self.hand_mesh.estimate(
                        frame_bgr, raw.hands.get(side), val, side)
            elif self.hand_mesh is not None:
                self.hand_mesh.reset(side)
        if self.constraints and state.pose is not None:
            # Corrige só o que é impossível (não mexe no estado dos filtros).
            state.pose, state.pose_visibility = apply_constraints(
                state.pose, state.pose_visibility, state.hands, state.pose_fixes)
        if self.body_widths is not None:
            state.body_widths = self.body_widths.update(raw.mask, state.pose, state.pose_visibility)
        return state

    def _smooth(self, part: _Part, pts: np.ndarray | None, t: float) -> np.ndarray | None:
        if not self.smoothing:
            part.value = pts
            return pts
        return part.update(pts, t)
