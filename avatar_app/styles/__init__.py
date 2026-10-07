"""Estilos de avatar. Cada estilo desenha a partir do AnimFrame (estado + animação) e do Rig.

Para criar um avatar novo: uma classe com `name` e `draw(img, frame, rig, palette)`,
adicionada a STYLES. O estilo "robô" é o exemplo que só usa o Rig.
"""
from .cartoon import CartoonStyle
from .robot import RobotStyle

STYLES = [CartoonStyle, RobotStyle]
