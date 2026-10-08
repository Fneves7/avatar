"""Suavização rígida da cara (tecla r): menos tremor e menos deformação do que ponto a ponto."""
import math

import numpy as np

from avatar_app.face_filter import RigidFaceFilter
from avatar_app.smoothing import OneEuroFilter

from conftest import make_face

FPS = 15  # ritmo típico da deteção


def _truth(t):
    roll = 15 * math.sin(2 * math.pi * 0.6 * t) if t > 2 else 0.0
    dx = 80 * math.sin(2 * math.pi * 0.5 * t) if t > 2 else 0.0
    return make_face(roll=roll, center=(640 + dx, 300), scale=14)


def _noisy(g, rng):
    """Ruído como o da deteção: a malha toda treme junta (posição e inclinação) e cada ponto um pouco."""
    c = g[:468, :2].mean(axis=0)
    th = rng.normal(0, math.radians(0.6))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = g.copy()
    out[:, :2] = (g[:, :2] - c) @ rot.T + c + rng.normal(0, 1.5, 2)
    return out + rng.normal(0, 0.6, g.shape)


def _shape_error(a, b):
    """Diferença de forma depois de alinhar (só deformação, não atraso)."""
    a, b = a[:468, :2] - a[:468, :2].mean(axis=0), b[:468, :2] - b[:468, :2].mean(axis=0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return float(np.sqrt(((a - b @ (u @ vt)) ** 2).sum(axis=1).mean()))


def _run(f):
    rng = np.random.default_rng(0)
    jitter, deform, lag, prev = [], [], [], None
    for i in range(6 * FPS):
        t = i / FPS
        g = _truth(t)
        out = f(_noisy(g, rng), t)
        top = out[10, :2]
        if 1 < t < 2 and prev is not None:
            jitter.append(np.linalg.norm(top - prev))
        if t > 3:
            deform.append(_shape_error(out, g))
            lag.append(np.linalg.norm(out[:468, :2].mean(axis=0) - g[:468, :2].mean(axis=0)))
        prev = top
    return np.mean(jitter), np.mean(deform), np.mean(lag)


def test_rigid_filter_reduces_tremor_and_deformation():
    p_jit, p_def, p_lag = _run(OneEuroFilter(2.0, 0.08))
    r_jit, r_def, r_lag = _run(RigidFaceFilter(2.0, 0.08))
    assert r_jit < 0.8 * p_jit, "menos tremor parado"
    assert r_def < 0.8 * p_def, "menos deformação a mexer"
    assert r_lag < p_lag + 1.0, "atraso parecido"


def test_static_face_is_reproduced():
    f = RigidFaceFilter()
    g = make_face(yaw=20, roll=10)
    for i in range(5):
        out = f(g, i / FPS)
    assert np.abs(out - g).max() < 1e-6


def test_follows_a_big_roll_without_jumps():
    f = RigidFaceFilter()
    for i in range(60):
        out = f(make_face(roll=min(170, i * 6)), i / FPS)
    assert _shape_error(out, make_face(roll=170)) < 1.0
    assert np.linalg.norm(out[:468, :2].mean(axis=0) - make_face(roll=170)[:468, :2].mean(axis=0)) < 2.0


def test_tracker_toggle_and_presets(monkeypatch):
    from avatar_app import tracker as tr
    monkeypatch.setattr(tr, "_HolisticBackend", lambda *a, **k: type("B", (), {"name": "fake"})())
    t = tr.Tracker(backend="holistic") if tr.HAS_HOLISTIC else None
    if t is None:
        return
    assert t.rigid_face and isinstance(t._face.filter, RigidFaceFilter)
    t.rigid_face = False
    assert isinstance(t._face.filter, OneEuroFilter) and not t.rigid_face
    t.rigid_face = True
    t.set_smoothing_preset("forte")
    assert t._face_rigid_filter.motion.min_cutoff < 0.5
