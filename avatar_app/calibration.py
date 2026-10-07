"""Calibração da pose neutra: tornar a cabeça e as expressões relativas à cara de cada pessoa.

Cada cara (e cada câmara, óculos, iluminação) dá valores diferentes em repouso: há quem
tenha "piscar 0.7" com os olhos abertos ou "pitch +25" a olhar em frente. Com a tecla de
calibração regista-se a cara neutra durante uns segundos e, a partir daí:
  * a rotação da cabeça é relativa à pose neutra;
  * piscar, sorriso e boca passam a 0 em repouso e chegam a 1 no máximo.
A calibração é guardada em JSON e carregada automaticamente no arranque.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tracker import BodyState

CALIBRATED_KEYS = ("eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight", "jawOpen",
                   "browInnerUp", "browDownLeft", "browDownRight")
# Olhos mais abertos que o neutro = "arregalar" (derivado do piscar, abaixo do repouso).
WIDE_FROM_BLINK = {"eyeWideLeft": "eyeBlinkLeft", "eyeWideRight": "eyeBlinkRight"}
WARMUP_S = 0.7       # tempo para a pessoa se pôr em posição depois de carregar na tecla
DURATION_S = 2.0     # tempo de recolha da pose neutra
MIN_SAMPLES = 10
MESSAGE_S = 3.0      # quanto tempo a mensagem final fica no HUD


@dataclass
class Calibration:
    head: tuple[float, float, float] = (0.0, 0.0, 0.0)   # yaw, pitch, roll neutros
    base: dict[str, float] = field(default_factory=dict)  # blendshapes em repouso

    def apply(self, s: BodyState) -> None:
        if s.head_angles is not None:
            s.head_angles = tuple(a - n for a, n in zip(s.head_angles, self.head))
        if s.blendshapes:
            raw = s.blendshapes
            bs = dict(raw)
            for k in CALIBRATED_KEYS:
                if k not in bs:
                    continue
                if k not in self.base:
                    bs[k] = 0.0  # calibração antiga sem esta chave: fica neutra até recalibrar
                    continue
                # Repouso -> 0, máximo -> 1.
                b = self.base[k]
                bs[k] = float(np.clip((raw[k] - b) / max(1.0 - b, 0.05), 0.0, 1.0))
            for wide, blink in WIDE_FROM_BLINK.items():
                if blink in raw and blink in self.base and wide not in raw:
                    b = self.base[blink]
                    bs[wide] = float(np.clip((b - raw[blink]) / max(b, 0.05), 0.0, 1.0))
            s.blendshapes = bs
        s.calibrated = True

    @property
    def outdated(self) -> bool:
        """Calibração feita antes de existirem algumas expressões (ex.: sobrancelhas)."""
        return any(k not in self.base for k in CALIBRATED_KEYS)

    def to_dict(self) -> dict:
        return {"head": list(self.head), "base": self.base}

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(head=tuple(float(x) for x in d.get("head", (0, 0, 0))),
                   base={k: float(v) for k, v in d.get("base", {}).items()})


class Calibrator:
    """Recolhe a pose neutra quando pedido e aplica a calibração a cada frame."""

    def __init__(self, path: Path):
        self.path = path
        self.calibration: Calibration | None = None
        self._t0: float | None = None
        self._samples: list[tuple[np.ndarray, dict[str, float]]] = []
        self._message: tuple[str, float] | None = None
        if path.exists():
            try:
                self.calibration = Calibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
                print(f"[calibração] carregada de {path}")
            except (OSError, ValueError) as exc:
                print(f"[calibração] ficheiro inválido ({exc}); ignorado")

    @property
    def active(self) -> bool:
        return self._t0 is not None

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._samples = []
        self._message = None

    def reset(self) -> None:
        """Apaga a calibração (volta aos valores em bruto)."""
        self.calibration = None
        self._t0 = None
        self.path.unlink(missing_ok=True)
        self._message = ("calibracao apagada", time.monotonic())

    def process(self, s: BodyState) -> None:
        """Chamado a cada frame com os valores em bruto do tracker."""
        if self.active:
            self._collect(s)
        elif self.calibration is not None:
            self.calibration.apply(s)

    def _collect(self, s: BodyState) -> None:
        elapsed = time.monotonic() - self._t0
        if elapsed >= WARMUP_S and s.face is not None and s.head_angles is not None:
            self._samples.append((np.array(s.head_angles, dtype=np.float64), dict(s.blendshapes)))
        if elapsed < WARMUP_S + DURATION_S:
            return
        self._t0 = None
        if len(self._samples) < MIN_SAMPLES:
            self._message = ("calibracao falhou: rosto nao detetado", time.monotonic())
            return
        # Mediana: um piscar ou um movimento a meio da recolha não estraga a calibração.
        head = np.median(np.array([h for h, _ in self._samples]), axis=0)
        base = {}
        for k in CALIBRATED_KEYS:
            vals = [bs[k] for _, bs in self._samples if k in bs]
            if vals:
                base[k] = float(np.median(vals))
        self.calibration = Calibration(head=tuple(float(x) for x in head), base=base)
        try:
            self.path.write_text(json.dumps(self.calibration.to_dict(), indent=2), encoding="utf-8")
            self._message = ("calibracao guardada", time.monotonic())
        except OSError as exc:
            self._message = (f"calibracao feita (nao guardada: {exc})", time.monotonic())

    def status(self) -> str | None:
        """Texto para o HUD (durante e logo após a calibração)."""
        if self.active:
            elapsed = time.monotonic() - self._t0
            if elapsed < WARMUP_S:
                return "CALIBRAR: olha em frente, cara neutra, olhos abertos..."
            pct = min(100, int((elapsed - WARMUP_S) / DURATION_S * 100))
            return f"CALIBRAR: nao te mexas... {pct}%"
        if self._message and time.monotonic() - self._message[1] < MESSAGE_S:
            return self._message[0]
        if self.calibration is not None and self.calibration.outdated:
            return "calibracao antiga: carrega em [k] para ativar todas as expressoes"
        return None
