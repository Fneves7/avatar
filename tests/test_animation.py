"""Animação: olhar vivo, idle (respiração), movimento secundário e transições."""
import numpy as np

from avatar_app.eyes import SACCADE_RADIUS, EyeLife
from avatar_app.idle import BREATH_AMPLITUDE, IdleAnimator
from avatar_app.renderer import AvatarRenderer
from avatar_app.secondary import HAIR_MAX_LAG, SecondaryMotion
from avatar_app.tracker import BodyState
from avatar_app.transitions import FADE_IN_S, FADE_OUT_S, Fade

from conftest import H, W, make_face, make_hand, make_pose

FPS = 30


# ------------------------------------------------------------------ olhar vivo
def _simulate_eyes(real_every=None, seconds=60, seed=1):
    e = EyeLife(seed=seed)
    blinks, saccades = [], []
    for t in np.arange(0, seconds, 1 / FPS):
        real = real_every is not None and (t % real_every) < 0.1
        b, s = e.update(t, real)
        blinks.append(b)
        saccades.append(s)
    blinks = np.array(blinks)
    closed_starts = np.flatnonzero(np.diff((blinks > 0.85).astype(int)) == 1)
    return len(closed_starts), blinks.max(), np.linalg.norm(saccades, axis=1)


def test_auto_blink_rate_and_full_close():
    n, max_close, _ = _simulate_eyes()
    assert 9 <= n <= 24, "piscar automático deve ser ~1 a cada 2,5-6 s"
    assert max_close > 0.99, "o olho fecha por completo (a 30 FPS o pico pode cair entre frames)"


def test_real_blinks_suppress_auto_blinks():
    n, _, _ = _simulate_eyes(real_every=3.0)
    assert n <= 4


def test_saccades_stay_small():
    _, _, amp = _simulate_eyes()
    assert amp.max() <= SACCADE_RADIUS + 1e-9


def test_lively_eyes_off_means_no_animation():
    r = AvatarRenderer()
    r.lively_eyes = False
    frame = r.animator.update(BodyState(W, H, face=make_face()), 0.0)
    assert frame.auto_blink == 0.0 and frame.saccade == (0.0, 0.0)


# ------------------------------------------------------------------ idle
def _run_idle(seconds=10.0, move_from=6.0, move_until=7.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    face0 = make_face(scale=11, center=(640, 230))
    pose0, vis = make_pose()
    idle = IdleAnimator()
    log = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        shift = 150 * min(1.0, max(0.0, t - move_from)) if t >= move_from else 0.0
        n = rng.normal(0, noise, 2)
        pose = pose0.copy(); pose[:, 0] += shift + n[0]; pose[:, 1] += n[1]
        face = face0.copy(); face[:, 0] += shift + n[0]; face[:, 1] += n[1]
        out = idle.apply(BodyState(W, H, face=face, pose=pose, pose_visibility=vis), t)
        log.append((t, idle.weight, pose[11, 1] - out.pose[11, 1], abs(pose[15, 1] - out.pose[15, 1])))
    return np.array(log), float(np.linalg.norm(pose0[11, :2] - pose0[12, :2]))


def test_idle_starts_when_still_despite_noise():
    log, _ = _run_idle()
    weight = {round(t, 2): w for t, w, _, _ in log}
    assert weight[0.5] == 0.0
    assert weight[2.5] == 1.0 and weight[5.9] == 1.0, "o ruído da deteção não pode desligar o idle"


def test_breathing_moves_shoulders_not_hands():
    log, sw = _run_idle()
    still = log[(log[:, 0] > 2.2) & (log[:, 0] < 6.0)]
    assert 0.7 * BREATH_AMPLITUDE * sw <= still[:, 2].max() <= 1.05 * BREATH_AMPLITUDE * sw
    assert still[:, 3].max() == 0.0


def test_idle_stops_quickly_when_moving():
    log, _ = _run_idle()
    moving = log[(log[:, 0] >= 6.0) & (log[:, 0] < 7.0)]
    assert (moving[:, 1] == 0).any() and moving[moving[:, 1] == 0][0, 0] < 6.6


# ------------------------------------------------------------------ movimento secundário
def _face_state(dx):
    f = make_face(scale=14, center=(500, 300))
    f[:, 0] += dx
    return BodyState(W, H, face=f)


def test_hair_lags_overshoots_and_settles():
    sec = SecondaryMotion()
    fw = float(np.linalg.norm(make_face(scale=14)[234, :2] - make_face(scale=14)[454, :2]))
    lag = []
    for i in range(int(2.5 * FPS)):
        t = i / FPS
        dx = 200 * min(1.0, max(0.0, (t - 0.5) / 0.3))
        sec.update(_face_state(dx), t)
        lag.append((t, sec.hair_offset[0]))
    lag = np.array(lag)
    moving = lag[(lag[:, 0] >= 0.5) & (lag[:, 0] <= 0.8), 1]
    after = lag[lag[:, 0] > 0.8, 1]
    assert moving.min() < -0.3 * HAIR_MAX_LAG * fw, "o cabelo deve ficar para trás"
    assert abs(moving).max() <= HAIR_MAX_LAG * fw + 1e-6
    assert after.max() > 0, "deve passar um pouco do ponto"
    assert abs(lag[int(2.0 * FPS), 1]) < 0.5, "e assentar"


def test_hair_spring_resets_on_teleport():
    sec = SecondaryMotion()
    for i in range(10):
        sec.update(_face_state(0), i / FPS)
    sec.update(_face_state(400), 10 / FPS)
    assert np.linalg.norm(sec.hair_offset) < 1e-6


# ------------------------------------------------------------------ transições
def test_fade_in_and_out_times():
    f = Fade()
    f.update(0.0, "x")
    f.update(FADE_IN_S / 2, "x")
    assert 0.4 < f.alpha < 0.6
    f.update(FADE_IN_S * 2, "x")
    assert f.alpha == 1.0
    f.update(FADE_IN_S * 2 + FADE_OUT_S / 2, None)
    assert 0.4 < f.alpha < 0.6 and f.data == "x", "a desaparecer guarda o último valor"
    f.update(FADE_IN_S * 2 + FADE_OUT_S * 2, None)
    assert f.alpha == 0.0 and f.data is None


def test_disabled_transitions_are_instant():
    f = Fade()
    assert f.update(0.0, "x", enabled=False) == 1.0
    assert f.update(0.01, None, enabled=False) == 0.0


def _hand_visibility(transitions, hand_frames, n=40):
    pose, vis = make_pose()
    hand = make_hand(wrist=(880, 420), spacing=18, base=25)
    roi = (slice(300, 420), slice(840, 980))
    r = AvatarRenderer()
    r.transitions = transitions
    r.lively_eyes = False
    face = make_face(scale=11, center=(640, 230))
    base = r.render(BodyState(W, H, pose=pose, pose_visibility=vis, face=face), t=-1.0)
    diffs = []
    for i in range(n):
        s = BodyState(W, H, pose=pose, pose_visibility=vis, face=face,
                      hands={15: hand} if i in hand_frames else {})
        img = r.render(s, t=i / FPS)
        diffs.append(float(np.abs(img[roi].astype(int) - base[roi].astype(int)).mean()))
    return np.array(diffs)


def test_hand_fades_out_instead_of_cutting():
    on = _hand_visibility(True, set(range(20)))
    off = _hand_visibility(False, set(range(20)))
    assert off[20] < 0.05 * off[15], "sem transições corta logo"
    assert 0.3 * on[15] < on[21] < 0.9 * on[15], "com transições desvanece"
    assert on[30] < 0.05 * on[15]


def test_single_frame_hand_flicker_is_faint():
    flicker = _hand_visibility(True, {10}).max()
    full = _hand_visibility(True, set(range(40)))[15]
    assert flicker < 0.35 * full
