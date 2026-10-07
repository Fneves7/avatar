# Backlog

Objetivo do projeto: avatar em tempo real para **streaming**.

## Feito

- [x] **Calibração da pose neutra** (tecla `k`, `Shift+K` apaga): a cabeça e as expressões passam a ser relativas à cara de cada pessoa. É guardada em `calibration.json`.
- [x] **Expressões exageradas** (tecla `e`, intensidade com `+` e `-`; precisa de calibração): arregalar e semicerrar os olhos, sobrancelhas a subir e a franzir, sorriso curvado, boca aberta mais marcada. Os ganhos estão no topo do `renderer.py`. Falta afiná-los com a webcam real.

## Próximo

- [ ] Escolher o próximo item das listas abaixo.

## Animação

- [ ] **Movimento secundário:** o cabelo, as mangas e a bainha seguem com atraso, como uma mola.
- [x] **Olhar vivo** (tecla `l`): o avatar pisca sozinho (de 2,5 a 6 s, ~170 ms) quando não há um piscar real recente, porque a deteção a ~10–15 FPS perde muitos piscares. A íris faz micro-movimentos subtis (≤20% do raio), somados ao olhar real. Também funciona na cabeça simples, quando o rosto não é detetado.
- [ ] **Idle:** respiração (o tronco e a cabeça sobem e descem ligeiramente) quando não há movimento.
- [ ] **Transições suaves:** quando uma parte deixa de ser detetada, desaparece ou recolhe aos poucos em vez de cortar.

## Estabilidade

- [ ] **Limites anatómicos da pose:** corrigir só o que é claramente impossível (cotovelo a dobrar para trás, ombro abaixo da anca, cotovelo inventado fora de imagem).
- [ ] **Afinar a suavização por parte** com a webcam real: dedos com menos atraso, tronco com mais.
- [ ] **Braços e ombros (abordagem conservadora):** manter o desenho atual e usar as medições da silhueta só para afinar larguras, com limites apertados. Uma primeira versão (medir tudo na segmentação e redesenhar o tronco a partir disso) foi revertida em 2026-10-07 porque ficou mal com a webcam real.

## Streaming

- [x] **Câmara virtual:** o avatar como webcam ("OBS Virtual Camera") para OBS, Teams, Zoom e Discord. Usa `pyvirtualcam`, liga e desliga com a tecla `v` ou com `--virtual-cam`, e `--output` define a resolução.
- [x] **Fundo para chroma key:** verde, azul ou magenta (`--background`, ou a tecla `b` ao vivo).
- [x] **Janela só com o avatar** (`--stream-window`): sem HUD nem webcam, para "Captura de janela" no OBS.
- [ ] **Fundo transparente:** a câmara virtual não suporta transparência. Seria preciso outra via, por exemplo uma fonte de browser no OBS.
- [x] **Fluidez:** a deteção corre numa thread separada e o avatar é desenhado a ritmo fixo (`--fps`, 30 por defeito), deslizando para a última deteção. No teste, o avatar passou de 11 para 26 FPS, em passos ~2× mais pequenos. `--sync` volta ao modo antigo. (Reduzir a imagem de deteção ou desligar a íris não acelera o Holistic, que custa ~55 ms por frame.)

## Avatares

- [ ] **Separar o rig do estilo:** o rig são as posições calculadas; o estilo é a forma de as desenhar. Permite vários avatares sem mexer na deteção.
- [ ] **Avatares PNG por camadas** (estilo VTuber 2D): cabeça, olhos, várias bocas, braços e mãos desenhados num editor.
- [ ] **Avatar 3D** (VRM, Blender ou three.js).

## Descartado

- Gravar e reproduzir sessões: não faz sentido num projeto de streaming em direto.
