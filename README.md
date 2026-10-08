# Avatar MediaPipe

Avatar 2D em estilo cartoon que imita, em tempo real e através da webcam, o rosto, o pescoço, o tronco, os ombros, os braços, as mãos e os dedos.

O programa tem dois motores de deteção:

| Motor | Requisitos | Deteta |
|---|---|---|
| **holistic** (por omissão) | `mediapipe==0.10.21`, Python 3.9–3.12; os modelos vêm incluídos no pacote | rosto com 478 pontos (inclui íris), corpo com 33 pontos, 2 mãos com 21 pontos cada; expressões e rotação da cabeça calculadas a partir da geometria |
| **tasks** | `mediapipe>=0.10.14` e os ficheiros `.task` em `models/` | o mesmo, mais 52 blendshapes e a matriz de rotação da cabeça dados pelo próprio modelo |

## Instalação (Windows)

O projeto precisa de Python 3.12, porque o `mediapipe 0.10.21` não tem versão para 3.13:

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe main.py
```

### Motor `tasks` (opcional)

Na primeira execução com `--backend tasks`, os modelos `.task` são descarregados para `models/`.

> Se a rede bloquear `storage.googleapis.com` (acontece em algumas redes empresariais), descarrega os ficheiros noutra rede e coloca-os em `models/`:
> - `face_landmarker.task`: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
> - `hand_landmarker.task`: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
> - `pose_landmarker_full.task`: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

## Execução

```bash
.venv\Scripts\python.exe main.py
```

Opções:

- `--backend auto|holistic|tasks`: escolhe o motor de deteção
- `--camera 1`: escolhe outra webcam
- `--width 1280 --height 720`: define a resolução de captura
- `--pose-model lite|full|heavy`: troca velocidade por precisão no corpo
- `--no-mirror`: não espelha a imagem
- `--palette N`: define a paleta inicial
- `--body-widths`: afina as larguras dos braços e do tronco do cartoon pela silhueta da pessoa (±15%, tecla `n`); custa ~11 ms por deteção

### Teclas

| Tecla | Ação |
|---|---|
| `c` | muda a paleta (Clássico, Neon, Floresta, Robô) |
| `y` | muda o estilo do avatar (cartoon, robô, png/gato ou pessoa); também `--style pessoa` no arranque |
| `d` | mostra ou esconde os landmarks sobre a webcam |
| `w` | mostra só o avatar ou o avatar ao lado da webcam |
| `k` | calibra a pose neutra: olha em frente com a cara neutra durante ~2 s; fica guardada em `calibration.json` |
| `Shift+K` | apaga a calibração |
| `e` | liga ou desliga as expressões exageradas (só com calibração) |
| `+` / `-` | aumenta ou diminui a intensidade do exagero |
| `l` | liga ou desliga o olhar vivo (piscar automático e micro-movimentos dos olhos) |
| `t` | liga ou desliga as transições suaves (as mãos, a cara e o corpo entram e saem aos poucos) |
| `i` | liga ou desliga o idle (respiração subtil quando estás parado) |
| `m` | liga ou desliga o movimento secundário (o cabelo, as mangas e a bainha seguem com atraso) |
| `s` | liga ou desliga a suavização |
| `Shift+S` | muda o perfil de suavização: leve (mais rápido), normal ou forte (mais estável) |
| `r` | liga ou desliga a suavização rígida da cara (menos tremor e deformação ao rodar a cabeça) |
| `n` | liga ou desliga as larguras medidas na silhueta (só com `--body-widths`) |
| `a` | liga ou desliga os limites anatómicos da pose (corrige cotovelos inventados, saltos, etc.) |
| `v` | liga ou desliga a câmara virtual |
| `b` | muda o fundo do avatar (gradiente, verde, azul ou magenta) |
| `p` | guarda uma captura em `screenshots/` |
| `q` / `Esc` | sai |

**As tuas escolhas ficam guardadas.** Ao sair, o perfil de suavização, a intensidade do exagero e as partes ligadas ou desligadas com as teclas (`s`, `r`, `a`, `l`, `t`, `i`, `m`, `h`, `e`, `n`) ficam em `settings.json`, e voltam no arranque seguinte. Afina com a webcam e fica assim. As opções da linha de comandos (estilo, paleta, fundo, ...) não são guardadas. Para voltar ao normal, apaga o `settings.json`.

## Streaming

O avatar sai limpo para o stream, sem HUD nem webcam. Há duas formas de o usar:

- **Câmara virtual (recomendado).** O avatar aparece como uma webcam chamada "OBS Virtual Camera" no OBS, Teams, Zoom, Discord, etc. Precisa do OBS Studio instalado, porque é o driver dele que é usado. Liga e desliga com `v`, ou já ligada no arranque:

  ```bash
  .venv\Scripts\python.exe main.py --virtual-cam
  ```

  No OBS, junta uma fonte "Dispositivo de captura de vídeo" com a "OBS Virtual Camera". Se o próprio OBS tiver a câmara virtual ligada, desliga-a primeiro: só um programa a pode usar de cada vez.

- **Janela só com o avatar.** Abre uma janela "Avatar (stream)" para usar com "Captura de janela" no OBS:

  ```bash
  .venv\Scripts\python.exe main.py --stream-window
  ```

**Fundo para chroma key:** `--background verde|azul|magenta`, ou a tecla `b` ao vivo. No OBS, junta à fonte o filtro "Chroma Key" com a mesma cor. Escolhe uma cor que não exista no avatar: com a paleta "Clássico" a camisola é azul, por isso aí usa verde ou magenta.

**Resolução da saída:** `--output 1920x1080`. Por defeito é a resolução da webcam.

**Fundo transparente (Fonte de Browser do OBS).** A câmara virtual não tem transparência. Para ter o avatar por cima de tudo, sem chroma key nem franjas nas bordas:

```bash
.venv\Scripts\python.exe main.py --browser-source
```

No OBS, junta uma fonte **Browser** com o URL `http://127.0.0.1:8765` e o tamanho do teu stream. Usa `127.0.0.1` e não `localhost`: no Windows, `localhost` tenta primeiro o IPv6 e alguns programas perdem ~2 s em cada pedido.

- **Como funciona:** o programa serve, só neste computador (127.0.0.1), uma página transparente com o avatar. A transparência é exata: o mesmo frame é desenhado sobre preto e sobre branco, e a diferença dá a opacidade de cada píxel, incluindo as bordas suavizadas.
- **Custo:** cerca de 3× o desenho normal, e só enquanto o OBS está a mostrar a página. `--browser-fps 15` baixa o ritmo desta saída, se for preciso.
- **Porta:** `--browser-port` muda a porta.

**Avatar 3D (Fonte de Browser do OBS).** Com `--browser-source`, o mesmo servidor tem também um avatar 3D em `http://127.0.0.1:8765/3d`, igualmente com fundo transparente. É desenhado pelo próprio OBS (three.js, em WebGL) a partir do rig que o programa envia a cada frame (um JSON de ~1 KB), por isso quase não custa nada ao programa.

- **Aspeto:** cel shading com contorno. A cabeça roda em 3D; os olhos piscam e seguem o olhar; as sobrancelhas, a boca e o rubor seguem as expressões; o tronco, os braços, as mãos e os dedos seguem o esqueleto.
- **Sem internet:** o three.js (licença MIT) vem dentro do projeto, em `avatar_app/web/`, porque a rede bloqueia os CDNs.
- **Para mudar o avatar:** as cores e as formas estão no topo de [avatar3d.js](avatar_app/web/avatar3d.js).

**Fluidez:** a deteção (MediaPipe, ~55–70 ms por frame) corre numa thread própria, e o avatar é desenhado e enviado a ritmo fixo (`--fps 30`, por defeito). Entre deteções, os pontos deslizam para a última posição detetada, o que custa ~50 ms de atraso em troca de movimento fluido. `--sync` volta ao modo antigo, com um desenho por deteção. O HUD mostra os dois ritmos (`Avatar FPS` e `Detection FPS`).

## Testes

Os testes não precisam de webcam: usam caras, poses e mãos sintéticas (a cara vem do modelo canónico do MediaPipe, em `tests/data/`).

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```bash
.venv\Scripts\python.exe -m pytest
```

Este comando corre os testes rápidos (~100, ~45 s): animação, calibração, limites anatómicos, malha das mãos, filtros, larguras pela silhueta, rig e estilos, preferências, transparência e avatar 3D com a Fonte de Browser. Há mais dois grupos, que só correm quando os pedes:

- `-m mediapipe`: o ciclo completo do `main.py` com uma câmara simulada e o MediaPipe a sério.
- `-m parity`: o estilo cartoon tem de ficar pixel a pixel igual ao último commit. Serve para refatorizações que não devem mudar o aspeto. `AVATAR_PARITY_REF=<commit>` compara com outro commit.

## Estrutura

```
main.py                  loop da webcam, HUD (em inglês: diagnóstico em cima, dicas das teclas em baixo) e teclas
avatar_app/
  models.py              download e cache dos modelos
  tracker.py             motores holistic/tasks, associação mão↔pulso, suavização e retenção
  smoothing.py           filtro One Euro (reduz o tremor sem acrescentar atraso)
  face_filter.py         suavização rígida da cara: movimento de conjunto + forma local
  head.py                crânio, nuca e orelhas estimados em 3D a partir da malha da cara
  hand_mesh.py           largura real dos dedos medida na imagem + malha/contorno da mão
  streaming.py           câmara virtual (pyvirtualcam) e fundos para chroma key
  browser_source.py      fundo transparente: página local para a Fonte de Browser do OBS (2D e /3d)
  web/avatar3d.*         avatar 3D (three.js) desenhado a partir do rig; three.js incluído (MIT)
  pipeline.py            thread de deteção + interpolação para desenhar a ritmo fixo
  eyes.py                olhar vivo: piscar automático e micro-movimentos da íris
  transitions.py         transições suaves (fade-in/fade-out) das mãos, da cara e do corpo
  idle.py                idle: respiração subtil quando a pessoa está parada
  secondary.py           movimento secundário: molas do cabelo, das mangas e da bainha
  settings.py            preferências guardadas entre sessões (settings.json)
  constraints.py         limites anatómicos da pose (cotovelos, saltos, braços, ancas)
  body_widths.py         larguras dos braços/tronco medidas na silhueta, com limites apertados
  animation.py           camada de animação partilhada (olhar vivo, idle, molas, transições) -> AnimFrame
  rig.py                 parâmetros do avatar independentes do desenho (cabeça, olhos, boca, esqueleto)
  renderer.py            fachada: fundo + animação + rig + estilo escolhido
  styles/cartoon.py      estilo "cartoon" (o avatar original, desenhado a partir da malha)
  styles/robot.py        estilo "robô" (exemplo de estilo feito só a partir do rig)
  styles/png.py          estilo "png": avatar de imagens PNG por camadas (estilo VTuber 2D)
tools/make_sample_avatar.py  gera o avatar PNG de exemplo (avatars/gato)
tools/make_human_avatar.py   gera o avatar PNG "pessoa" (avatars/pessoa), estilo anime/VTuber
tests/                   testes pytest sem webcam (ver "Testes")
avatars/gato/            imagens PNG + avatar.json do avatar de exemplo
avatars/pessoa/          imagens PNG + avatar.json da personagem humana (estilo `pessoa`)
  landmarks.py           índices dos landmarks do MediaPipe usados pelo rig e pelos estilos
  drawing.py             paletas e utilitários de desenho partilhados
  debug_draw.py          desenha os landmarks sobre a webcam
```

## Estilos de avatar (rig + estilo)

O desenho está separado em três camadas:

1. **Animação** ([animation.py](avatar_app/animation.py)): o olhar vivo, o idle, o movimento secundário, as transições e os olhos fechados, iguais para todos os avatares.
2. **Rig** ([rig.py](avatar_app/rig.py)): parâmetros independentes de como o avatar é desenhado, parecidos com os de um modelo VTuber.
   - **Cabeça:** centro, tamanho, inclinação e rotação.
   - **Olhos:** abertura de cada olho e direção do olhar.
   - **Expressão:** sobrancelhas, abertura da boca e sorriso.
   - **Corpo:** respiração, esqueleto do tronco, braços e mãos, e as molas do cabelo e das mangas.

   O rig também sai em JSON (`rig_to_dict`) para o avatar 3D da página `/3d`.
3. **Estilo** ([styles/](avatar_app/styles)): como o avatar é desenhado. A tecla `y` alterna entre estilos.

**Para criar um avatar novo:** faz uma classe com `name` e `draw(img, frame, rig, palette)` e junta-a a `STYLES` em [styles/\_\_init\_\_.py](avatar_app/styles/__init__.py). O estilo [robô](avatar_app/styles/robot.py) serve de exemplo, porque usa só o rig. O cartoon também lê a malha da cara diretamente, porque desenha a partir dos 478 pontos.

### Avatar PNG por camadas (estilo VTuber 2D)

O estilo `png` desenha um avatar feito de imagens PNG com transparência. Vêm de uma pasta com um `avatar.json`; por defeito é `avatars/gato`, um gato de exemplo gerado por código com `tools/make_sample_avatar.py`.

```bash
.venv\Scripts\python.exe main.py --style png
```

Há também uma personagem humana mais trabalhada, `avatars/pessoa` (estilo `pessoa` na tecla `y`), em estilo anime/VTuber: sombreado cel, cabelo com madeixas e reflexo, olhos com pestanas e brilhos, rubor, camisola com gola e mãos com dedos. É gerada por `tools/make_human_avatar.py`; as cores e as formas estão no topo desse ficheiro.

```bash
.venv\Scripts\python.exe main.py --style pessoa
```

**As imagens** (os nomes dos ficheiros são fixos):

| Parte | Ficheiros |
|---|---|
| Cabeça (todas com o mesmo tamanho de tela, ex.: 512×512) | `hair_back`, `head`, `hair_front`, `blush` |
| Olhos (`_l`/`_r` = olho à esquerda/direita na imagem) | `eye_white_*`, `iris_*` (recortada pelo branco), `eye_half_*`, `eye_closed_*` |
| Expressão | `brow_l`, `brow_r`; `mouth_closed`, `mouth_smile`, `mouth_open_small`, `mouth_open` |
| Corpo | `torso`, `neck`, `upper_arm`, `forearm`, `hand_open`, `hand_fist` |

**O `avatar.json`** diz onde encaixa cada imagem:
- **cabeça:** o centro da cara e a largura da cara na tela; o centro de cada olho e de cada sobrancelha; a paralaxe de cada camada.
- **tronco:** os ombros e o centro das ancas.
- **membros:** os pontos de início e fim, a largura do desenho e a espessura relativa aos ombros.
- **mãos:** o pulso e a base do dedo do meio. Opcional: `"thumb": "left"` (ou `"right"`), o lado do polegar no desenho; a mão é então espelhada quando o polegar detetado está do outro lado, para servir às duas mãos.

**Para usar o teu próprio avatar:** desenha as imagens com os mesmos nomes, ajusta os pontos no `avatar.json` e corre com `--png-avatar avatars/<pasta>`.

## Malha das mãos

O MediaPipe só dá o esqueleto da mão (21 pontos). A espessura e o contorno de cada dedo são medidos na imagem, em [avatar_app/hand_mesh.py](avatar_app/hand_mesh.py):

1. Em cada frame, aprende-se a cor da pele (Cr/Cb) na palma e ao longo dos ossos dos dedos.
2. A pele é segmentada num "tubo" à volta do esqueleto, para não apanhar a cara nem o fundo.
3. Em cada uma das 14 falanges, procura-se perpendicularmente ao osso onde a pele acaba.
4. Com essas larguras, cada dedo ganha um contorno (bordas e ponta arredondada) e a palma junta-se a eles.

Se a medição não for fiável (fundo com cor de pele, pouca luz, mão de lado), mantêm-se as últimas larguras medidas ou proporções anatómicas. Com `d`, a webcam mostra o contorno real (amarelo) e a malha (ciano), e o HUD indica quantas falanges foram medidas. `--no-hand-mesh` desliga a medição.

## Notas

- Quando o rosto não é detetado mas o corpo é, o programa desenha uma cabeça simplificada a partir da pose. Quando o `HandLandmarker` perde uma mão, desenha uma "luva" com os pontos da mão da pose.
- Se as ancas estiverem fora de imagem, o tronco é estimado a partir dos ombros.
- Uma mão claramente atrás da cabeça, pela profundidade z, faz com que o braço seja desenhado atrás do corpo.
