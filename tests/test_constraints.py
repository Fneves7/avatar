"""Limites anatómicos da pose."""
import math

import numpy as np

from avatar_app.constraints import GLITCH_MAX_HOLD, MAX_SEGMENT, apply_constraints, reject_glitches


def _close_up_pose():
    """Pessoa perto da câmara (caso do print): cotovelo 14 inventado em cima do tronco."""
    pose = np.zeros((33, 3))
    vis = np.ones(33)
    for i, p in {11: (215, 720), 12: (600, 730), 13: (150, 1000), 14: (535, 845), 15: (140, 1100),
                 16: (785, 740), 23: (250, 1300), 24: (560, 1300)}.items():
        pose[i, :2] = p
    vis[[13, 15, 23, 24]] = 0.1
    return pose, vis


def _vertical_hand(wrist, tilt_deg=0.0):
    a = math.radians(tilt_deg)
    hand = np.zeros((21, 3))
    hand[:, :2] = wrist
    hand[9, :2] = np.asarray(wrist) + 120 * np.array([math.sin(a), -math.cos(a)])
    return hand


def test_invented_elbow_is_moved_below_raised_hand():
    pose, vis = _close_up_pose()
    vis[14] = 0.7
    fixes = []
    out, _ = apply_constraints(pose, vis, {16: _vertical_hand(np.array([785.0, 740.0]))}, fixes)
    assert abs(out[14, 0] - 785) < 1 and out[14, 1] > 740, "cotovelo no prolongamento da mão, por baixo"
    assert any("elbow" in f for f in fixes)


def test_clearly_visible_elbow_is_kept():
    pose, vis = _close_up_pose()
    vis[14] = 0.95
    out, _ = apply_constraints(pose, vis, {16: _vertical_hand(np.array([785.0, 740.0]))}, [])
    assert np.allclose(out[14], pose[14])


def test_natural_wrist_bend_is_kept():
    pose, vis = _close_up_pose()
    pose[14, :2] = (785, 1000)
    out, _ = apply_constraints(pose, vis, {16: _vertical_hand(np.array([785.0, 740.0]), 30)}, [])
    assert np.allclose(out[14], pose[14])


def test_overlong_arm_is_shortened():
    pose, vis = _close_up_pose()
    sw = np.linalg.norm(pose[11, :2] - pose[12, :2])
    pose[13, :2] = pose[11, :2] + [0, 2.2 * sw]
    out, _ = apply_constraints(pose, vis, {}, [])
    assert np.linalg.norm(out[13, :2] - out[11, :2]) / sw <= MAX_SEGMENT + 1e-6


def test_hips_above_shoulders_are_ignored():
    pose, vis = _close_up_pose()
    vis[[23, 24]] = 0.9
    pose[[23, 24], 1] = 500
    _, v2 = apply_constraints(pose, vis, {}, [])
    assert (v2[[23, 24]] == 0).all()


def test_glitch_is_held_a_few_detections_then_accepted():
    pose, vis = _close_up_pose()
    sw = np.linalg.norm(pose[11, :2] - pose[12, :2])
    jumped = pose.copy()
    jumped[13, :2] += [0.8 * sw, 0]
    held = {}
    kept = [np.allclose(reject_glitches(jumped, pose, vis, [], held)[13], pose[13]) for _ in range(GLITCH_MAX_HOLD + 1)]
    assert kept == [True] * GLITCH_MAX_HOLD + [False]


def test_visible_joint_jump_is_accepted():
    pose, vis = _close_up_pose()
    vis[13] = 0.9
    jumped = pose.copy()
    jumped[13, :2] += [300, 0]
    assert not np.allclose(reject_glitches(jumped, pose, vis, [], {})[13], pose[13])
