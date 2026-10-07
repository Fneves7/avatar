"""Transições suaves: partes que aparecem/desaparecem entram e saem aos poucos.

Quando uma mão sai da imagem (ou a deteção falha) a última mão conhecida desvanece em vez
de cortar de repente; quando aparece, entra em fade-in, por isso deteções esporádicas de
um só frame já não "piscam". O mesmo para a cara (por cima da cabeça simples da pose).
"""
from __future__ import annotations

FADE_IN_S = 0.12    # tempo a aparecer
FADE_OUT_S = 0.25   # tempo a desaparecer (depois da retenção curta do tracker)


class Fade:
    """Opacidade 0..1 de uma parte e o último valor visto (para a desenhar a desaparecer)."""

    def __init__(self):
        self.alpha = 0.0
        self.data = None
        self._t: float | None = None

    def update(self, t: float, data, enabled: bool = True) -> float:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        if not enabled:
            self.data = data
            self.alpha = 1.0 if data is not None else 0.0
            return self.alpha
        if data is not None:
            self.data = data
            self.alpha = min(1.0, self.alpha + dt / FADE_IN_S)
        else:
            self.alpha = max(0.0, self.alpha - dt / FADE_OUT_S)
            if self.alpha == 0.0:
                self.data = None
        return self.alpha

    @property
    def visible(self) -> bool:
        return self.data is not None and self.alpha > 0.0
