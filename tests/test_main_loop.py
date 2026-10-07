"""Ciclo completo do main.py com uma câmara simulada (usa o MediaPipe a sério: mais lento).

    .venv\\Scripts\\python.exe -m pytest -m mediapipe
"""
import itertools
import sys
import time

import cv2
import numpy as np
import pytest

import main

pytestmark = pytest.mark.mediapipe


class FakeCamera:
    """Webcam simulada a ~30 FPS com uma imagem fixa; acaba ao fim de n frames."""

    def __init__(self, n=45):
        self.n, self.read_count = n, 0
        self.img = np.full((720, 1280, 3), 200, np.uint8)

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}.get(prop, 0)

    def read(self):
        time.sleep(1 / 30)
        self.read_count += 1
        return self.read_count <= self.n, self.img.copy()

    def release(self):
        pass


@pytest.fixture
def run_main(monkeypatch):
    """Corre main.main() sem janelas nem webcam; devolve o que foi mostrado em cada janela."""

    def _run(args, keys=()):
        cam = FakeCamera()
        shown = {}
        # As teclas só começam depois de a deteção arrancar (no arranque só q/Esc contam).
        pending = itertools.chain(keys, itertools.repeat(-1))

        def wait_key(ms=1):
            time.sleep(ms / 1000)
            return (next(pending) if cam.read_count > 10 else -1) & 0xFF

        monkeypatch.setattr(main, "open_camera", lambda *a: cam)
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "resizeWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.__setitem__(name, img.copy()))
        monkeypatch.setattr(cv2, "waitKey", wait_key)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(main, "CALIBRATION_FILE", main.Path("calibracao_que_nao_existe.json"))
        monkeypatch.setattr(sys, "argv", ["main.py", *args])
        main.main()
        return shown

    return _run


def test_stream_window_is_clean_and_resized(run_main):
    shown = run_main(["--stream-window", "--background", "verde", "--output", "960x540"])
    stream = shown[main.STREAM_WINDOW]
    assert stream.shape == (540, 960, 3)
    assert stream[5, 5].tolist() == [0, 255, 0], "sem HUD no canto da saída de stream"
    assert shown[main.WINDOW].shape == (720, 2560, 3), "janela de controlo: webcam + avatar"


def test_keys_change_background_and_style(run_main):
    shown = run_main(["--stream-window", "--background", "verde"], keys=[ord("b"), -1, ord("y"), -1, ord("y")])
    assert shown[main.STREAM_WINDOW][5, 5].tolist() == [255, 0, 0], "[b] verde -> azul"


@pytest.mark.parametrize("style", ["cartoon", "robo", "png"])
def test_runs_with_every_style(run_main, style):
    shown = run_main(["--style", style])
    assert main.WINDOW in shown
