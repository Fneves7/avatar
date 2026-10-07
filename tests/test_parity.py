"""Paridade pixel a pixel do estilo cartoon com um commit (para refatorizações que não devem
mudar o aspeto). Por defeito compara com HEAD; outro commit com AVATAR_PARITY_REF=<ref>.

    .venv\\Scripts\\python.exe -m pytest -m parity
"""
import importlib
import math
import os
import random
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from avatar_app.hand_mesh import HandMeshEstimator
from avatar_app.renderer import AvatarRenderer
from avatar_app.tracker import BodyState

from conftest import H, ROOT, W, make_face, make_hand, make_pose

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def old_app(tmp_path_factory):
    """O pacote avatar_app do commit de referência, importado como `old_app`."""
    ref = os.environ.get("AVATAR_PARITY_REF", "HEAD")
    out = tmp_path_factory.mktemp("parity")
    archive = out / "old.tar"
    try:
        subprocess.run(["git", "archive", ref, "avatar_app", "-o", str(archive)], cwd=ROOT, check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git archive {ref} falhou: {exc}")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    (out / "avatar_app").rename(out / "old_app")
    sys.path.insert(0, str(out))
    try:
        yield importlib.import_module("old_app.renderer"), importlib.import_module("old_app.tracker"), \
            importlib.import_module("old_app.hand_mesh")
    finally:
        sys.path.remove(str(out))


def _sequence():
    """150 frames: a cara mexe e roda, expressões, mãos com malha a entrar/sair, a cara perde-se."""
    pose0, vis = make_pose()
    hand = make_hand(wrist=(880, 420), spacing=18, base=25)
    mesh = HandMeshEstimator().estimate(np.zeros((H, W, 3), np.uint8), None, hand, 15)
    for i in range(150):
        t = i / 30
        dx = 120 * math.sin(t * 2.0) if 30 < i < 90 else 0.0
        yaw = 35 * math.sin(t * 1.3) if 60 < i < 120 else 0.0
        s = BodyState(W, H)
        if not 95 <= i < 105:
            s.face = make_face(yaw=yaw, center=(640 + dx, 250))
        s.pose = pose0.copy()
        s.pose[:, 0] += dx * 0.5
        s.pose_visibility = vis.copy()
        if 20 <= i < 70 or i == 110:
            s.hands, s.hand_meshes = {15: hand.copy()}, {15: mesh}
        s.calibrated = i >= 40
        s.blendshapes = {"eyeBlinkLeft": 0.9 if 50 <= i < 53 else 0.0, "eyeBlinkRight": 0.9 if 50 <= i < 53 else 0.0,
                         "mouthSmileLeft": max(0.0, math.sin(t)), "mouthSmileRight": max(0.0, math.sin(t)),
                         "jawOpen": 0.6 if 70 <= i < 80 else 0.0, "browInnerUp": 0.5 if 80 <= i < 90 else 0.0,
                         "browDownLeft": 0.0, "browDownRight": 0.0, "eyeWideLeft": 0.0, "eyeWideRight": 0.0}
        s.head_angles = (yaw, 0.0, 0.0)
        yield t, s


def _to_old(s, old_tracker, old_hand_mesh):
    o = old_tracker.BodyState(s.width, s.height)
    for k in ("face", "blendshapes", "head_angles", "pose", "pose_visibility", "hands", "calibrated"):
        if hasattr(o, k):
            setattr(o, k, getattr(s, k))
    o.hand_meshes = {k: old_hand_mesh.HandMesh(m.fingers, m.palm, m.rails, m.widths, m.measured, m.contour)
                     for k, m in s.hand_meshes.items()}
    return o


@pytest.mark.parametrize("config", ["tudo ligado", "olhar vivo desligado", "cabeca simples sem exagero"])
def test_cartoon_matches_reference(old_app, config):
    old_renderer, old_tracker, old_hand_mesh = old_app
    old, new = old_renderer.AvatarRenderer(), AvatarRenderer()
    old.eye_life._rng = random.Random(7)
    new.eye_life._rng = random.Random(7)
    if config == "olhar vivo desligado":
        old.lively_eyes = new.lively_eyes = False
    if config == "cabeca simples sem exagero":
        old.head_3d = new.head_3d = False
        old.exaggerate = new.exaggerate = False
    different = []
    for t, s in _sequence():
        a = old.render(_to_old(s, old_tracker, old_hand_mesh), t=t)
        b = new.render(s, t=t)
        if not np.array_equal(a, b):
            different.append(round(t * 30))
    assert not different, f"frames diferentes do commit de referência: {different[:10]}"
