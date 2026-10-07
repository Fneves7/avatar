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
| `d` | mostra ou esconde os landmarks sobre a webcam |
| `w` | mostra só o avatar ou o avatar ao lado da webcam |
| `s` | liga ou desliga a suavização |
| `p` | guarda uma captura em `screenshots/` |
| `q` / `Esc` | sai |

## Estrutura

```
main.py                  loop da webcam, HUD e teclas
avatar_app/
  models.py              download e cache dos modelos
  tracker.py             motores holistic/tasks, associação mão↔pulso, suavização e retenção
  smoothing.py           filtro One Euro (reduz o tremor sem acrescentar atraso)
  renderer.py            desenha o avatar (cabeça, pescoço, tronco, braços, mãos e dedos)
  debug_draw.py          desenha os landmarks sobre a webcam
```

## Notas

- Quando o rosto não é detetado mas o corpo é, o programa desenha uma cabeça simplificada a partir da pose. Quando o `HandLandmarker` perde uma mão, desenha uma "luva" com os pontos da mão da pose.
- Se as ancas estiverem fora de imagem, o tronco é estimado a partir dos ombros.
- Uma mão claramente atrás da cabeça, pela profundidade z, faz com que o braço seja desenhado atrás do corpo.
