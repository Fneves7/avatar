"""Estilos de avatar. Cada estilo desenha a partir do AnimFrame (estado + animação) e do Rig.

Para criar um avatar novo: uma classe com `name` e `draw(img, frame, rig, palette)`,
adicionada a STYLES. O estilo "robô" é o exemplo que só usa o Rig; o estilo "png" desenha
um avatar feito de imagens por camadas (pasta avatars/<nome> com avatar.json).
"""
from .cartoon import CartoonStyle
from .png import PngStyle
from .robot import RobotStyle

STYLES = [CartoonStyle, RobotStyle, PngStyle]
