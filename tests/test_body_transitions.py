"""Transições do corpo inteiro: o tronco e os braços entram e saem aos poucos (tecla t)."""
import numpy as np
import pytest

from avatar_app.renderer import AvatarRenderer
from avatar_app.transitions import FADE_IN_S, FADE_OUT_S

from conftest import make_face, make_state

STYLES = ["cartoon", "robo", "png"]


def _renderer(style, transitions=True):
    r = AvatarRenderer(style=style)
    r.lively_eyes = False
    r.idle_enabled = False
    r.transitions = transitions
    return r


def _body_amount(style, pose_frames, n=20, transitions=True, face=None):
    """Quanto do corpo se vê em cada frame (0 = só o resto, 1 = corpo inteiro), a 30 FPS."""
    r = _renderer(style, transitions)
    with_body = make_state(face=face, pose=True, hand=None)
    bare = make_state(face=face, pose=False, hand=None)
    ref_on = _renderer(style).render(with_body, t=0).astype(float)
    ref_off = _renderer(style).render(bare, t=0).astype(float)
    full = np.abs(ref_on - ref_off).sum()
    out = []
    for i in range(n):
        img = r.render(with_body if i in pose_frames else bare, t=i / 30).astype(float)
        out.append(1.0 - np.abs(img - ref_on).sum() / full if full else 0.0)
    return np.array(out)


@pytest.mark.parametrize("style", STYLES)
def test_body_fades_out_instead_of_cutting(style):
    amount = _body_amount(style, pose_frames=range(10))
    assert amount[9] > 0.98
    lost = amount[10:]
    assert 0.5 < lost[0] < 0.98, "no 1.º frame sem pose o corpo ainda se vê"
    assert np.all(np.diff(lost) <= 0.02), "desvanece sempre"
    gone = 10 + int(np.ceil(FADE_OUT_S * 30))
    assert np.all(amount[gone:] < 0.02), f"desaparece em ~{FADE_OUT_S}s"


@pytest.mark.parametrize("style", STYLES)
def test_body_fades_in(style):
    amount = _body_amount(style, pose_frames=range(5, 20))
    assert np.all(amount[:5] < 0.02)
    assert 0.1 < amount[5] < 0.9, "aparece aos poucos"
    assert np.all(amount[5 + int(np.ceil(FADE_IN_S * 30)):] > 0.98)


def test_single_frame_pose_is_faint():
    amount = _body_amount("cartoon", pose_frames={5})
    assert amount.max() < 0.4


@pytest.mark.parametrize("style", STYLES)
def test_disabled_transitions_cut(style):
    amount = _body_amount(style, pose_frames=range(10), transitions=False)
    assert amount[9] > 0.98 and np.all(amount[10:] < 0.02)


def test_face_stays_while_body_fades():
    """Com a cara detetada, só o corpo desvanece: a cara fica igual."""
    r = _renderer("cartoon")
    face = make_face()
    for i in range(10):
        r.render(make_state(face=face, hand=None), t=i / 30)
    fading = r.render(make_state(face=face, pose=False, hand=None), t=10 / 30)
    ref = _renderer("cartoon")
    for i in range(11):  # a cara também entra aos poucos: deixar assentar
        bare = ref.render(make_state(face=face, pose=False, hand=None), t=i / 30)
    x0, x1, y0, y1 = 540, 740, 120, 330  # zona da cara (centro em 640, 250)
    face_diff = np.abs(fading[y0:y1, x0:x1].astype(int) - bare[y0:y1, x0:x1].astype(int))
    body_diff = np.abs(fading[400:, :].astype(int) - bare[400:, :].astype(int))
    assert body_diff.max() > 50, "o corpo ainda se vê"
    assert face_diff.max() <= 2, "a cara não desvanece com o corpo"


@pytest.mark.parametrize("style", STYLES)
def test_alpha_stays_exact_while_fading(style):
    a, b = _renderer(style), _renderer(style)
    for i in range(12):
        s = make_state(face=make_face(), hand=None, pose=i < 10)
        normal = a.render(s, t=i / 30)
        comp, _ = b.render_with_alpha(s, t=i / 30)
    diff = np.abs(normal.astype(int) - comp.astype(int)).max(axis=2)
    assert diff.max() <= 12 and (diff > 6).sum() <= 40
