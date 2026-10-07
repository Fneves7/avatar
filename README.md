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

### Teclas

| Tecla | Ação |
|---|---|
| `c` | muda a paleta (Clássico, Neon, Floresta, Robô) |
| `y` | muda o estilo do avatar (cartoon ou robô); também `--style robo` no arranque |
| `d` | mostra ou esconde os landmarks sobre a webcam |
| `w` | mostra só o avatar ou o avatar ao lado da webcam |
| `k` | calibra a pose neutra: olha em frente com a cara neutra durante ~2 s; fica guardada em `calibration.json` |
| `Shift+K` | apaga a calibração |
| `e` | liga ou desliga as expressões exageradas (só com calibração) |
| `+` / `-` | aumenta ou diminui a intensidade do exagero |
| `l` | liga ou desliga o olhar vivo (piscar automático e micro-movimentos dos olhos) |
| `t` | liga ou desliga as transições suaves (as mãos e a cara entram e saem aos poucos) |
| `i` | liga ou desliga o idle (respiração subtil quando estás parado) |
| `m` | liga ou desliga o movimento secundário (o cabelo e as mangas seguem com atraso) |
| `s` | liga ou desliga a suavização |
| `Shift+S` | muda o perfil de suavização: leve (mais rápido), normal ou forte (mais estável) |
| `a` | liga ou desliga os limites anatómicos da pose (corrige cotovelos inventados, saltos, etc.) |
| `v` | liga ou desliga a câmara virtual |
| `b` | muda o fundo do avatar (gradiente, verde, azul ou magenta) |
| `p` | guarda uma captura em `screenshots/` |
| `q` / `Esc` | sai |

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

**Fluidez:** a deteção (MediaPipe, ~55–70 ms por frame) corre numa thread própria, e o avatar é desenhado e enviado a ritmo fixo (`--fps 30`, por defeito). Entre deteções, os pontos deslizam para a última posição detetada, o que custa ~50 ms de atraso em troca de movimento fluido. `--sync` volta ao modo antigo, com um desenho por deteção. O HUD mostra os dois ritmos (`FPS avatar` e `detecao`).

## Estrutura

```
main.py                  loop da webcam, HUD e teclas
avatar_app/
  models.py              download e cache dos modelos
  tracker.py             motores holistic/tasks, associação mão↔pulso, suavização e retenção
  smoothing.py           filtro One Euro (reduz o tremor sem acrescentar atraso)
  head.py                crânio, nuca e orelhas estimados em 3D a partir da malha da cara
  hand_mesh.py           largura real dos dedos medida na imagem + malha/contorno da mão
  streaming.py           câmara virtual (pyvirtualcam) e fundos para chroma key
  pipeline.py            thread de deteção + interpolação para desenhar a ritmo fixo
  eyes.py                olhar vivo: piscar automático e micro-movimentos da íris
  transitions.py         transições suaves (fade-in/fade-out) das mãos e da cara
  idle.py                idle: respiração subtil quando a pessoa está parada
  secondary.py           movimento secundário: molas do cabelo e das mangas
  constraints.py         limites anatómicos da pose (cotovelos, saltos, braços, ancas)
  animation.py           camada de animação partilhada (olhar vivo, idle, molas, transições) -> AnimFrame
  rig.py                 parâmetros do avatar independentes do desenho (cabeça, olhos, boca, esqueleto)
  renderer.py            fachada: fundo + animação + rig + estilo escolhido
  styles/cartoon.py      estilo "cartoon" (o avatar original, desenhado a partir da malha)
  styles/robot.py        estilo "robô" (exemplo de estilo feito só a partir do rig)
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
3. **Estilo** ([styles/](avatar_app/styles)): como o avatar é desenhado. A tecla `y` alterna entre estilos.

**Para criar um avatar novo:** faz uma classe com `name` e `draw(img, frame, rig, palette)` e junta-a a `STYLES` em [styles/\_\_init\_\_.py](avatar_app/styles/__init__.py). O estilo [robô](avatar_app/styles/robot.py) serve de exemplo, porque usa só o rig. O cartoon também lê a malha da cara diretamente, porque desenha a partir dos 478 pontos.

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
