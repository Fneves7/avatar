"""Rig (parâmetros semânticos) e os estilos de avatar."""
import numpy as np
import pytest

from avatar_app.renderer import AvatarRenderer
from avatar_app.rig import build_rig
from avatar_app.styles import STYLES
from avatar_app.styles.png import PngStyle
from avatar_app.tracker import BodyState

from conftest import H, W, make_face, make_hand, make_pose, make_state

STYLE_NAMES = [cls.name for cls in STYLES]


def _rig(**blendshapes):
    r = AvatarRenderer()
    r.lively_eyes = False
    frame = r.animator.update(make_state(calibrated=True, **blendshapes), 0.0)
    return build_rig(frame)


def test_rig_neutral_face():
    rig = _rig()
    h = rig.head
    assert np.allclose(h.eye_open, 1.0)
    assert np.allclose(h.look, 0.0, atol=0.05)
    assert h.mouth_open < 0.2 and h.mouth_smile == 0.0 and h.brow == 0.0
    assert abs(h.turn) < 0.05


def test_rig_follows_expressions():
    assert np.allclose(_rig(eyeBlinkLeft=1, eyeBlinkRight=1).head.eye_open, 0.0)
    assert _rig(jawOpen=0.8).head.mouth_open >= 0.8
    assert _rig(mouthSmileLeft=1, mouthSmileRight=1).head.mouth_smile == 1.0
    assert _rig(browInnerUp=1).head.brow == 1.0
    assert _rig(browDownLeft=1, browDownRight=1).head.brow == -1.0


def test_rig_look_has_same_sign_for_both_eyes():
    """A íris desviada para a direita da imagem nos dois olhos dá olhar positivo (não se anula)."""
    face = make_face()
    face[468:478, 0] += 6.0
    r = AvatarRenderer()
    r.lively_eyes = False
    rig = build_rig(r.animator.update(BodyState(W, H, face=face), 0.0))
    assert rig.head.look[0] > 0.2


def test_rig_turn_follows_head_rotation():
    turns = []
    for yaw in (-30, 0, 30):
        r = AvatarRenderer()
        turns.append(build_rig(r.animator.update(BodyState(W, H, face=make_face(yaw=yaw)), 0.0)).head.turn)
    assert turns[0] * turns[2] < 0 and abs(turns[1]) < 0.05


def test_rig_skeleton():
    rig = _rig()
    assert rig.shoulders.shape == (2, 2) and rig.hips.shape == (2, 2)
    assert 15 in rig.arms and 15 in rig.hands or rig.hands == {}


@pytest.mark.parametrize("style", STYLE_NAMES)
@pytest.mark.parametrize("case", ["tudo", "vazio", "so_pose", "so_maos", "so_cara"])
def test_every_style_renders_every_case(style, case):
    states = {
        "tudo": make_state(calibrated=True, mouthSmileLeft=0.8, mouthSmileRight=0.8, jawOpen=0.5),
        "vazio": BodyState(W, H),
        "so_pose": BodyState(W, H, pose=make_pose()[0], pose_visibility=make_pose()[1]),
        "so_maos": BodyState(W, H, hands={15: make_hand()}),
        "so_cara": BodyState(W, H, face=make_face(yaw=40)),
    }
    r = AvatarRenderer(style=style)
    for i in range(5):
        img = r.render(states[case], t=i / 30)
    assert img.shape == (H, W, 3)


@pytest.mark.parametrize("style", ["cartoon", "robo", "png"])
def test_style_draws_something(style):
    r = AvatarRenderer(style=style)
    r.background = (0, 255, 0)
    for i in range(6):
        img = r.render(make_state(), t=i / 30)
    drawn = np.any(img != (0, 255, 0), axis=2).mean()
    assert 0.05 < drawn < 0.9


def test_next_style_cycles():
    r = AvatarRenderer()
    seen = [r.style.name] + [r.next_style() for _ in STYLES]
    assert seen[0] == seen[-1] and set(seen) == set(STYLE_NAMES)


def test_png_missing_folder_reports_error():
    st = PngStyle("pasta_que_nao_existe")
    assert st.cfg is None and "make_sample_avatar" in st.error
    img = np.zeros((H, W, 3), np.uint8)
    r = AvatarRenderer()
    frame = r.animator.update(make_state(), 0.0)
    st.draw(img, frame, build_rig(frame), r.palette)  # escreve o aviso, não rebenta
    assert img.any()
