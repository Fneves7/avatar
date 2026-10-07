"""Fundo transparente (desenho sobre preto e branco) e servidor da Fonte de Browser do OBS."""
import socket
import time
import urllib.request

import cv2
import numpy as np
import pytest

from avatar_app.browser_source import BrowserSource
from avatar_app.renderer import AvatarRenderer

from conftest import make_face, make_state


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("style", ["cartoon", "robo", "png"])
def test_alpha_is_exact(style):
    a, b = AvatarRenderer(style=style), AvatarRenderer(style=style)
    a.lively_eyes = b.lively_eyes = False
    s = make_state(face=make_face(yaw=15))
    for i in range(6):
        normal = a.render(s, t=i / 30)
        comp, bgra = b.render_with_alpha(s, t=i / 30)
    diff = np.abs(normal.astype(int) - comp.astype(int)).max(axis=2)
    # Igual ao desenho direto, a menos de arredondamentos de 8 bits (mais visíveis em píxeis de
    # borda quase transparentes: alguns isolados podem passar dos 6 níveis).
    assert diff.max() <= 12 and (diff > 6).sum() <= 20, "composição = desenho direto"
    alpha = bgra[..., 3]
    assert (alpha == 0).mean() > 0.5 and (alpha == 255).mean() > 0.1
    assert 0 < ((alpha > 0) & (alpha < 255)).mean() < 0.05, "bordas suavizadas com opacidade parcial"


def test_alpha_of_empty_frame_is_transparent():
    _, bgra = AvatarRenderer().render_with_alpha(make_state(face=None, pose=False, hand=None))
    assert not bgra[..., 3].any()


@pytest.fixture
def source():
    src = BrowserSource(_free_port())
    yield src
    src.close()


def test_page_is_transparent_and_does_not_use_raf(source):
    html = urllib.request.urlopen(source.url + "/").read().decode()
    assert "transparent" in html and "<canvas" in html
    assert "requestAnimationFrame(loop)" not in html, "rAF pára com a fonte escondida no OBS"


def test_frame_roundtrip(source):
    _, bgra = AvatarRenderer(style="png").render_with_alpha(make_state())
    source.publish(bgra)
    r = urllib.request.urlopen(source.url + "/frame?after=-1")
    x, y, w, h = map(int, r.headers["X-Rect"].split(","))
    png = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert png.shape == (h, w, 4)
    assert np.array_equal(png, bgra[y:y + h, x:x + w]), "PNG recortado sem perdas"
    assert r.headers["X-Size"] == "1280,720"
    assert source.active


def test_no_new_frame_returns_204_after_timeout(source):
    source.publish(np.zeros((10, 10, 4), np.uint8))
    seq = urllib.request.urlopen(source.url + "/frame?after=-1").headers["X-Seq"]
    t0 = time.perf_counter()
    r = urllib.request.urlopen(source.url + f"/frame?after={seq}")
    assert r.status == 204 and 0.8 < time.perf_counter() - t0 < 3


def test_busy_port_reports_error():
    port = _free_port()
    a, b = BrowserSource(port), BrowserSource(port)
    try:
        assert a.error is None and b.error is not None, "no Windows o SO_REUSEADDR deixava abrir duas vezes"
    finally:
        a.close()
        b.close()
