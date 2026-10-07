"""Deteção numa thread separada + interpolação para desenhar o avatar a ritmo constante.

O Holistic demora ~55-70 ms por frame (≈ 15 FPS no máximo). Para o stream não ficar aos
saltos, a captura e a deteção correm numa thread própria e o avatar é desenhado a um
ritmo fixo (ex.: 30 FPS). Entre deteções, o estado mostrado desliza suavemente para a
última deteção (aproximação exponencial). Isto custa ~50 ms de atraso extra, em troca de
movimento fluido.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time

import cv2
import numpy as np

from .tracker import BodyState, Tracker

BLEND_TAU = 0.05   # constante de tempo da aproximação à última deteção (s)


class DetectionWorker(threading.Thread):
    """Lê a webcam e corre o tracker em contínuo; guarda só o resultado mais recente."""

    def __init__(self, cap: cv2.VideoCapture, tracker: Tracker, mirror: bool = True):
        super().__init__(name="deteccao", daemon=True)
        self.cap = cap
        self.tracker = tracker
        self.mirror = mirror
        self.failed = False          # a webcam deixou de dar frames
        self.fps = 0.0               # ritmo da deteção
        self._lock = threading.Lock()
        self._stop_event = threading.Event()  # (Thread já tem um método interno _stop)
        self._latest: tuple[int, np.ndarray, BodyState] | None = None
        self._seq = 0

    def run(self) -> None:
        last = time.perf_counter()
        while not self._stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            state = self.tracker.process(frame)
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, frame, state)

    def latest(self) -> tuple[int, np.ndarray, BodyState] | None:
        """(número da deteção, frame, estado) mais recentes, ou None se ainda não há."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)


def _lerp(prev, new, a: float):
    """Interpola recursivamente valores numéricos com a mesma estrutura; caso contrário
    (partes que apareceram/desapareceram, inteiros, booleanos) fica o valor novo."""
    if prev is None or new is None:
        return new
    if isinstance(new, np.ndarray):
        if (isinstance(prev, np.ndarray) and prev.shape == new.shape
                and np.issubdtype(new.dtype, np.floating)):
            return prev + (new - prev) * a
        return new
    if isinstance(new, bool) or isinstance(new, int):
        return new
    if isinstance(new, float):
        return prev + (new - prev) * a if isinstance(prev, (int, float)) else new
    if isinstance(new, dict):
        return {k: _lerp(prev.get(k) if isinstance(prev, dict) else None, v, a) for k, v in new.items()}
    if isinstance(new, (list, tuple)):
        if isinstance(prev, (list, tuple)) and len(prev) == len(new):
            return type(new)(_lerp(p, n, a) for p, n in zip(prev, new))
        return new
    if dataclasses.is_dataclass(new) and type(prev) is type(new):
        return dataclasses.replace(new, **{f.name: _lerp(getattr(prev, f.name), getattr(new, f.name), a)
                                           for f in dataclasses.fields(new)})
    return new


class StateBlender:
    """Estado mostrado: aproxima-se da última deteção a cada frame desenhado."""

    def __init__(self, tau: float = BLEND_TAU):
        self.tau = tau
        self._shown: BodyState | None = None
        self._t: float | None = None

    def update(self, target: BodyState) -> BodyState:
        now = time.perf_counter()
        if self._shown is None or self._t is None:
            self._shown = target
        else:
            a = 1.0 - math.exp(-(now - self._t) / self.tau)
            self._shown = _lerp(self._shown, target, a)
        self._t = now
        return self._shown
