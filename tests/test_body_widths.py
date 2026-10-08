"""Larguras dos braços e do tronco afinadas pela silhueta (--body-widths), com limites apertados."""
import cv2
import numpy as np

from avatar_app.body_widths import EXPECTED, LIMITS, BodyWidths, measure_width
from avatar_app.renderer import AvatarRenderer

from conftest import H, W, make_pose, make_state


def _silhouette(pose, arm_k=1.0, torso_k=1.0):
    """Silhueta sintética: tronco e braços com as larguras do desenho vezes um fator."""
    p = pose[:, :2]
    sw = float(np.linalg.norm(p[11] - p[12]))
    m = np.zeros((H, W), np.uint8)
    mid = (p[11] + p[12]) / 2
    half = EXPECTED["torso"] * sw * torso_k / 2
    cv2.rectangle(m, (int(mid[0] - half), int(mid[1] - 0.1 * sw)), (int(mid[0] + half), H - 1), 255, -1)
    for a, b, part in ((11, 13, "upper"), (13, 15, "fore"), (12, 14, "upper"), (14, 16, "fore")):
        cv2.line(m, tuple(map(int, p[a])), tuple(map(int, p[b])), 255, int(EXPECTED[part] * sw * arm_k))
    return m.astype(np.float32) / 255


def _factors(arm_k=1.0, torso_k=1.0, n=60, pose=None):
    pose, vis = make_pose() if pose is None else pose
    bw = BodyWidths()
    for _ in range(n):
        out = bw.update(_silhouette(pose, arm_k, torso_k), pose, vis)
    return out, bw


def test_measure_width_of_a_bar():
    m = np.zeros((200, 200), np.float32)
    m[:, 80:120] = 1.0
    assert abs(measure_width(m, np.array([100.0, 20]), np.array([100.0, 180]), 100) - 40) <= 2


def test_matching_silhouette_keeps_widths():
    f, _ = _factors()
    assert f, "deve haver medições"
    for k, v in f.items():
        assert abs(v - 1.0) < 0.08, (k, v)


def test_wider_arms_are_clamped():
    f, _ = _factors(arm_k=1.6)
    up = [v for k, v in f.items() if k.startswith(("upper", "fore"))]
    assert up and all(1.08 < v <= LIMITS["upper"][1] + 1e-9 for v in up)


def test_arm_over_the_torso_is_rejected():
    pose, vis = make_pose()
    pose[13, :2] = (640, 520)   # cotovelo à frente do peito
    pose[15, :2] = (600, 470)
    f, bw = _factors(arm_k=1.6, pose=(pose, vis))
    assert "upper_15" in bw.last_rejected and "upper_15" not in f


def test_low_visibility_is_rejected():
    pose, vis = make_pose()
    vis[[14, 16]] = 0.2
    f, bw = _factors(pose=(pose, vis))
    assert "fore_16" not in f and "fore_16" in bw.last_rejected


def test_without_mask_nothing_changes():
    bw = BodyWidths()
    pose, vis = make_pose()
    assert bw.update(None, pose, vis) == {}


def test_cartoon_uses_the_factors_only_when_enabled():
    def render(factors, enabled=True):
        r = AvatarRenderer()
        r.lively_eyes = False
        r.cartoon.body_widths = enabled
        s = make_state(face=None, hand=None)
        s.body_widths = factors
        return r.render(s, t=0)

    base = render({})
    wide = render({"upper_15": 1.15, "upper_16": 1.15, "torso": 1.1})
    assert (np.abs(wide.astype(int) - base.astype(int)).max(axis=2) > 30).sum() > 200
    assert np.array_equal(render({"upper_15": 1.15}, enabled=False), base), "tecla n desligada = desenho normal"
