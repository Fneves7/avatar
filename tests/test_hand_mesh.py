"""Malha das mãos: larguras dos dedos medidas numa mão sintética desenhada na imagem."""
import cv2
import numpy as np

from avatar_app.hand_mesh import FINGERS, HandMeshEstimator

from conftest import H, W

TRUE_WIDTHS = [34, 26, 28, 25, 20]  # polegar .. mindinho (px)


def _synthetic_hand():
    rng = np.random.default_rng(1)
    hand = np.zeros((21, 3))
    wrist = np.array([640.0, 560.0])
    hand[0, :2] = wrist
    for f, ang in enumerate(np.linspace(-1.1, 0.55, 5)):
        segs = [45, 38, 30] if f == 0 else [70, 45, 35]
        pts = [wrist + (40 if f == 0 else 120) * np.array([np.sin(ang), -np.cos(ang)])]
        for s in segs:
            pts.append(pts[-1] + s * np.array([np.sin(ang * 1.1), -np.cos(ang * 1.1)]))
        for j in range(4):
            hand[FINGERS[f][j], :2] = pts[j]
    img = np.full((H, W, 3), (120, 140, 150), np.uint8)
    img = cv2.add(img, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))
    skin = (120, 160, 210)
    cv2.fillPoly(img, [cv2.convexHull(np.round(hand[[0, 1, 5, 9, 13, 17], :2]).astype(np.int32))], skin)
    cv2.circle(img, tuple(wrist.astype(int)), 50, skin, -1)
    for f, chain in enumerate(FINGERS):
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(img, tuple(np.round(hand[a, :2]).astype(int)), tuple(np.round(hand[b, :2]).astype(int)),
                     skin, TRUE_WIDTHS[f])
        cv2.circle(img, tuple(np.round(hand[chain[-1], :2]).astype(int)), TRUE_WIDTHS[f] // 2, skin, -1)
    return cv2.GaussianBlur(img, (3, 3), 0), hand


def test_finger_widths_are_measured():
    img, hand = _synthetic_hand()
    est = HandMeshEstimator()
    for _ in range(10):
        m = est.estimate(img, hand, hand, 16)
    assert m.measured == 14
    for f in range(1, 5):  # a base do polegar não se mede (está dentro da palma)
        assert np.allclose(m.widths[f], TRUE_WIDTHS[f], atol=3.5), f"dedo {f}: {m.widths[f]}"


def test_noise_is_rejected_and_widths_are_kept():
    img, hand = _synthetic_hand()
    est = HandMeshEstimator()
    good = est.estimate(img, hand, hand, 16)
    noise = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    m = est.estimate(noise, hand, hand, 16)
    assert m.measured == 0
    assert np.allclose(m.widths, good.widths, atol=3)


def test_without_points_nothing_is_measured():
    img, hand = _synthetic_hand()
    assert HandMeshEstimator().estimate(img, None, hand, 16).measured == 0
