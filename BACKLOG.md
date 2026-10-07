# Backlog

Objetivo do projeto: avatar em tempo real para **streaming**.

## Feito

- [x] **Calibração da pose neutra** (tecla `k`, `Shift+K` apaga): a cabeça e as expressões passam a ser relativas à cara de cada pessoa. É guardada em `calibration.json`.

## Próximo

- [ ] **Expressões exageradas** (usa a calibração): olhos mais expressivos, sobrancelhas a subir mais, boca em "D" ao sorrir, boca aberta mais marcada.

## Animação

- [ ] **Movimento secundário:** o cabelo, as mangas e a bainha seguem com atraso, como uma mola.
- [ ] **Olhar vivo:** micro-movimentos dos olhos e piscar automático quando o rosto não é detetado.
- [ ] **Idle:** respiração (o tronco e a cabeça sobem e descem ligeiramente) quando não há movimento.
- [ ] **Transições suaves:** quando uma parte deixa de ser detetada, desaparece ou recolhe aos poucos em vez de cortar.

## Estabilidade

- [ ] **Limites anatómicos da pose:** corrigir só o que é claramente impossível (cotovelo a dobrar para trás, ombro abaixo da anca, cotovelo inventado fora de imagem).
- [ ] **Afinar a suavização por parte** com a webcam real: dedos com menos atraso, tronco com mais.
- [ ] **Braços e ombros (abordagem conservadora):** manter o desenho atual e usar as medições da silhueta só para afinar larguras, com limites apertados. Uma primeira versão (medir tudo na segmentação e redesenhar o tronco a partir disso) foi revertida em 2026-10-07 porque ficou mal com a webcam real.

## Streaming

- [ ] **Câmara virtual:** enviar o avatar como webcam para OBS, Teams, Zoom e Discord (`pyvirtualcam`).
- [ ] **Fundo para OBS:** cor sólida para chroma key, ou fundo transparente.
- [ ] **Janela só com o avatar:** sem HUD nem webcam, pronta para captura.
- [ ] **Modo performance:** resolução de deteção mais baixa e menos medições, para dar mais FPS durante o stream.

## Avatares

- [ ] **Separar o rig do estilo:** o rig são as posições calculadas; o estilo é a forma de as desenhar. Permite vários avatares sem mexer na deteção.
- [ ] **Avatares PNG por camadas** (estilo VTuber 2D): cabeça, olhos, várias bocas, braços e mãos desenhados num editor.
- [ ] **Avatar 3D** (VRM, Blender ou three.js).

## Descartado

- Gravar e reproduzir sessões: não faz sentido num projeto de streaming em direto.
