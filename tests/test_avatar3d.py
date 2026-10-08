"""Avatar 3D: rig em JSON (rig_to_dict) e a página /3d servida pela Fonte de Browser."""
import json
import socket
import threading
import time
import urllib.request

import numpy as np
import pytest

from avatar_app.browser_source import WEB_DIR, BrowserSource
from avatar_app.renderer import AvatarRenderer
from avatar_app.rig import rig_to_dict

from conftest import H, W, make_face, make_state


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rig(**kw):
    r = AvatarRenderer()
    r.lively_eyes = False
    for i in range(10):
        r.render(make_state(**kw), t=i / 30)
    return r.last_rig


def test_rig_dict_is_normalized_json():
    rig = _rig()
    d = json.loads(json.dumps(rig_to_dict(rig)))
    assert d["aspect"] == pytest.approx(W / H, abs=1e-3)
    cx, cy = d["head"]["center"]
    # Centro da cara em (640, ~250) px: x perto de 0, acima do meio da imagem (y > 0, y para cima).
    assert abs(cx) < 0.05 and 0 < cy < 1
    assert d["head"]["size"] == pytest.approx(rig.head.size / H * 2, abs=1e-3)
    assert set(d["arms"]) == {"15", "16"} and len(d["hands"]["15"]["points"]) == 21
    ls, rs = d["body"]["shoulders"]
    assert ls[1] < cy and rs[1] < cy, "ombros abaixo da cara"


def test_rig_dict_eyes_are_ordered_by_image_side():
    d = rig_to_dict(_rig(calibrated=True, eyeBlinkLeft=1.0))
    eo = d["head"]["eye_open"]
    # eyeBlinkLeft é o olho esquerdo da pessoa (landmark 263): na imagem espelhada (como num
    # espelho) fica à esquerda de quem vê.
    assert eo[0] < 0.1 < eo[1]


def test_rig_dict_roll_follows_3d_convention():
    # roll > 0 na imagem (sentido dos ponteiros) = roll < 0 no referencial 3D (y para cima).
    assert rig_to_dict(_rig(face=make_face(roll=15)))["head"]["roll"] * \
        rig_to_dict(_rig(face=make_face(roll=-15)))["head"]["roll"] < 0


def test_rig_dict_without_body_or_face():
    d = rig_to_dict(_rig(face=None, pose=False, hand=None))
    assert "head" not in d and "body" not in d and d["arms"] == {} and d["hands"] == {}


@pytest.fixture
def source():
    src = BrowserSource(_free_port())
    yield src
    src.close()


def test_3d_page_and_three_js_are_served_locally(source):
    html = urllib.request.urlopen(source.url + "/3d").read().decode()
    assert "/avatar3d.js" in html and "transparent" in html
    js = urllib.request.urlopen(source.url + "/avatar3d.js").read().decode()
    assert "from '/three.module.min.js'" in js and "requestAnimationFrame(loop)" not in js
    three = urllib.request.urlopen(source.url + "/three.module.min.js").read()
    assert three == (WEB_DIR / "three.module.min.js").read_bytes() and b"Three.js Authors" in three[:200]
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(source.url + "/../browser_source.py")


def test_rig_long_poll(source):
    assert not source.active_3d
    t0 = time.perf_counter()
    r = urllib.request.urlopen(source.url + "/rig?after=-1")
    assert r.status == 204 and time.perf_counter() - t0 < 1.8
    assert source.active_3d, "a página a pedir o rig conta como ligada"
    threading.Timer(0.2, lambda: source.publish_rig({"aspect": 1.5})).start()
    r = urllib.request.urlopen(source.url + "/rig?after=0")
    assert r.status == 200 and json.loads(r.read()) == {"aspect": 1.5} and r.headers["X-Seq"] == "1"
