"""Preferências guardadas entre sessões (settings.json)."""
import json

from avatar_app import settings
from avatar_app.renderer import AvatarRenderer


class FakeTracker:
    def __init__(self):
        self.smoothing, self.rigid_face, self.constraints = True, True, True
        self.smoothing_preset = "normal"

    def set_smoothing_preset(self, name):
        self.smoothing_preset = name


def test_roundtrip(tmp_path):
    t, r = FakeTracker(), AvatarRenderer()
    t.rigid_face, r.lively_eyes, r.exaggeration = False, False, 1.6
    t.set_smoothing_preset("leve")
    path = tmp_path / "settings.json"
    settings.save(path, t, r)
    t2, r2 = FakeTracker(), AvatarRenderer()
    applied = settings.load(path, t2, r2)
    assert "rigid_face" in applied and not t2.rigid_face and not r2.lively_eyes
    assert r2.exaggeration == 1.6 and t2.smoothing_preset == "leve"


def test_bad_or_missing_file_changes_nothing(tmp_path):
    t, r = FakeTracker(), AvatarRenderer()
    assert settings.load(tmp_path / "nao_existe.json", t, r) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{nao e json", encoding="utf-8")
    assert settings.load(bad, t, r) == []
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"version": 1, "lively_eyes": "sim", "exaggeration": 99,
                               "smoothing_preset": "turbo", "desconhecido": True}), encoding="utf-8")
    assert settings.load(odd, t, r) == [] and r.lively_eyes and r.exaggeration == 1.0
