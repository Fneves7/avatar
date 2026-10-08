"""Saída para streaming: câmara virtual e fundos para chroma key.

A câmara virtual usa o pyvirtualcam. No Windows precisa da câmara virtual do OBS Studio
(vem com o OBS 26+). O avatar aparece como uma webcam ("OBS Virtual Camera") no OBS,
Teams, Zoom, Discord, etc.
"""
from __future__ import annotations

import cv2
import numpy as np

# Fundos do avatar (BGR). None = gradiente da paleta. As cores sólidas servem para chroma key.
BACKGROUNDS: list[tuple[str, tuple[int, int, int] | None]] = [
    ("gradiente", None),
    ("verde", (0, 255, 0)),
    ("azul", (255, 0, 0)),
    ("magenta", (255, 0, 255)),
]
BACKGROUND_NAMES = [name for name, _ in BACKGROUNDS]


def background_color(name: str) -> tuple[int, int, int] | None:
    return dict(BACKGROUNDS)[name]


class VirtualCamera:
    """Envia frames BGR para uma câmara virtual. Falhas não interrompem a aplicação."""

    def __init__(self, width: int, height: int, fps: int = 30):
        self.size = (width, height)
        self.fps = fps
        self._cam = None
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._cam is not None

    @property
    def device(self) -> str | None:
        return self._cam.device if self._cam is not None else None

    def start(self) -> bool:
        if self._cam is not None:
            return True
        try:
            import pyvirtualcam
            self._cam = pyvirtualcam.Camera(width=self.size[0], height=self.size[1], fps=self.fps,
                                            fmt=pyvirtualcam.PixelFormat.BGR)
            self.error = None
            print(f"[virtual camera] on: {self._cam.device} ({self.size[0]}x{self.size[1]})")
            return True
        except ImportError:
            self.error = "pyvirtualcam not installed (pip install pyvirtualcam)"
        except Exception as exc:  # driver em falta, câmara ocupada, ...
            self.error = f"{exc}"
        print(f"[virtual camera] could not start: {self.error}")
        return False

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._cam is None:
            return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        try:
            self._cam.send(np.ascontiguousarray(frame_bgr))
        except Exception as exc:
            self.error = f"{exc}"
            print(f"[virtual camera] send error, turning off: {exc}")
            self.stop()

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            finally:
                self._cam = None
                print("[virtual camera] off")
