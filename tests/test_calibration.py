"""Calibração da pose neutra (com relógio simulado)."""
import json

import numpy as np
import pytest

import avatar_app.calibration as cal
from avatar_app.tracker import BodyState

from conftest import H, W, make_face


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(cal.time, "monotonic", lambda: now[0])
    return now


def _state(rng, blink=0.7, smile=0.05, head=(3.0, 25.0, -2.0)):
    s = BodyState(W, H, face=make_face(), head_angles=tuple(np.array(head) + rng.normal(0, 0.5, 3)))
    s.blendshapes = {"eyeBlinkLeft": blink + rng.normal(0, 0.02), "eyeBlinkRight": blink + rng.normal(0, 0.02),
                     "mouthSmileLeft": smile, "mouthSmileRight": smile, "jawOpen": 0.02,
                     "browInnerUp": 0.5, "browDownLeft": 0.3, "browDownRight": 0.3}
    return s


def _calibrate(path, clock, rng, frames=100):
    c = cal.Calibrator(path)
    c.start()
    for i in range(frames):  # ~3,3 s a 30 FPS, com um piscar a meio
        clock[0] += 1 / 30
        c.process(_state(rng, blink=1.0 if 40 <= i < 44 else 0.7))
    return c


def test_calibration_makes_neutral_zero_and_saves(tmp_path, clock):
    rng = np.random.default_rng(0)
    path = tmp_path / "calibration.json"
    c = _calibrate(path, clock, rng)
    assert c.calibration is not None and path.exists()
    s = _state(rng)
    c.process(s)
    assert s.calibrated
    assert np.allclose(s.head_angles, 0, atol=2.0)
    assert s.blendshapes["eyeBlinkLeft"] < 0.1, "olhos abertos em repouso -> 0, mesmo com óculos (0,7 em bruto)"
    closed = _state(rng, blink=1.0)
    c.process(closed)
    assert closed.blendshapes["eyeBlinkLeft"] > 0.9


def test_calibration_reloads_from_file(tmp_path, clock):
    rng = np.random.default_rng(1)
    path = tmp_path / "calibration.json"
    c = _calibrate(path, clock, rng)
    assert cal.Calibrator(path).calibration == c.calibration


def test_calibration_fails_without_face(tmp_path, clock):
    c = cal.Calibrator(tmp_path / "c.json")
    c.start()
    for _ in range(100):
        clock[0] += 1 / 30
        c.process(BodyState(W, H))
    assert c.calibration is None
    assert "failed" in (c.status() or "")


def test_old_calibration_asks_to_recalibrate(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"head": [0, 0, 0], "base": {"eyeBlinkLeft": 0.5}}), encoding="utf-8")
    c = cal.Calibrator(path)
    assert c.calibration.outdated
    assert "[k]" in c.status()
