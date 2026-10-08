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
def run_main(monkeypatch, tmp_path):
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
        monkeypatch.setattr(main, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(sys, "argv", ["main.py", *args])
        main.main()
        return shown

    _run.settings_file = tmp_path / "settings.json"
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


@pytest.mark.parametrize("style", ["cartoon", "robo", "png", "pessoa"])
def test_runs_with_every_style(run_main, style):
    shown = run_main(["--style", style])
    assert main.WINDOW in shown


def test_3d_page_gets_the_rig(run_main):
    """Com --browser-source, quem pede /rig (a página /3d) recebe o rig em JSON."""
    import json
    import socket
    import threading
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    got = []

    def poll():
        deadline = time.monotonic() + 30
        while not got and time.monotonic() < deadline:
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/rig?after=-1", timeout=3)
                if r.status == 200:
                    got.append(json.loads(r.read()))
            except OSError:
                time.sleep(0.1)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    run_main(["--browser-source", "--browser-port", str(port)])
    th.join(5)
    assert got and got[0]["aspect"] == pytest.approx(1280 / 720, abs=1e-3)


def test_settings_are_saved_and_restored(run_main):
    """O que se muda com as teclas fica guardado ao sair e volta no arranque seguinte."""
    import json
    run_main([], keys=[ord("l"), -1, ord("S")])  # olhar vivo OFF, perfil normal -> forte
    saved = json.loads(run_main.settings_file.read_text(encoding="utf-8"))
    assert saved["lively_eyes"] is False and saved["smoothing_preset"] == "forte"
    run_main([])  # arranca com as preferências guardadas e volta a gravá-las iguais
    again = json.loads(run_main.settings_file.read_text(encoding="utf-8"))
    assert again["lively_eyes"] is False and again["smoothing_preset"] == "forte"
