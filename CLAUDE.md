# CLAUDE.md — memória do projeto

Contexto para o Claude retomar o trabalho (é carregado no início de cada sessão e depois de
comprimir o contexto). O README explica o projeto ao utilizador; este ficheiro explica-o ao Claude.

## O projeto

Avatar 2D em tempo real controlado pela webcam (MediaPipe), para **streaming em direto**
(OBS/Teams/Zoom/Discord). O utilizador fala **português de Portugal**: responder, comentar o código
e escrever documentação em PT-PT. O roteiro está em [BACKLOG.md](BACKLOG.md); antes de propor
trabalho novo, ver o que lá está e mantê-lo atualizado.

## Ambiente (Windows)

- Usar **sempre** o `.venv` (Python 3.12 + `mediapipe==0.10.21`, que traz os modelos Holistic
  dentro do pacote): `.venv\Scripts\python.exe main.py`. O `python` do sistema é 3.13 com
  mediapipe 1.1.0 (sem `solutions`, tenta descarregar modelos) — não serve.
- A rede bloqueia `storage.googleapis.com` (modelos `.task` do MediaPipe). PyPI, GitHub e
  HuggingFace funcionam. O motor `tasks` só funciona se os `.task` forem postos à mão em `models/`.
- OBS Studio 32 instalado → câmara virtual "OBS Virtual Camera" via `pyvirtualcam`.
- `calibration.json` e `settings.json` (preferências das teclas, `avatar_app/settings.py`) são
  pessoais (estão no `.gitignore`).
- O utilizador faz os commits e os pushes. Não fazer commit sem pedido.

## Arquitetura (caminho de um frame)

```
main.py: webcam -> DetectionWorker (thread, pipeline.py) -> Tracker (tracker.py)
  Tracker: Holistic -> One Euro (perfis leve/normal/forte) -> limites anatómicos (constraints.py)
           -> malha das mãos (hand_mesh.py)
-> StateBlender (interpola a 30 FPS) -> Calibrator (calibration.py, pose neutra)
-> AvatarRenderer (renderer.py):
     Animator (animation.py: olhar vivo eyes.py, idle.py, molas secondary.py, transições transitions.py)
     -> Rig (rig.py: parâmetros semânticos estilo VTuber)
     -> estilo (styles/: cartoon, robo, png) desenha sobre o fundo
     -> rig_to_dict (JSON) -> página /3d (web/avatar3d.js, three.js) desenha o avatar 3D no browser
-> saídas: janela (HUD + webcam com debug_draw.py), câmara virtual (streaming.py),
   janela só-avatar, Fonte de Browser com fundo transparente (browser_source.py)
```

- O estilo `cartoon` desenha a partir da malha de 478 pontos; `robo` só a partir do Rig;
  `png` com imagens por camadas de `avatars/<nome>/` + `avatar.json` (gato gerado por
  `tools/make_sample_avatar.py`; `pessoa` é o mesmo estilo com `avatars/pessoa`, gerado por
  `tools/make_human_avatar.py`). Índices dos landmarks em `landmarks.py`, paletas em `drawing.py`.
- `render_with_alpha` desenha o mesmo frame sobre preto e branco para tirar a transparência exata;
  a animação avança **uma vez** por frame (`_animate`) — avançar duas vezes estraga as molas.
- Quase tudo tem tecla para ligar/desligar (lista no docstring de `main.py` e no README).

## Regras de trabalho (combinadas com o utilizador)

- O utilizador julga pelo resultado **real na webcam** (manda prints). Os testes do Claude são
  sintéticos: dizer sempre que não foram feitos com a webcam real.
- Mudanças visuais **conservadoras** e atrás de uma tecla/opção, para comparar e voltar atrás.
- Um passo de cada vez; sugerir commit quando o utilizador aprova um passo.
- Reverter: ver `git status`/`git log`, perguntar o âmbito se for ambíguo, e repor a partir do
  último commit (`git restore`), não à mão.
- Refatorizações que não devem mudar o aspeto: provar com teste de paridade pixel a pixel contra
  o último commit (ver "Testes" abaixo).
- **UI em inglês** (HUD, mensagens no ecrã e na consola, `--help`); código, comentários e
  documentação em PT-PT. As opções da linha de comandos mantêm os valores internos
  (`--background verde`, `--style pessoa`, perfis `leve/forte`); o HUD mostra-os traduzidos
  (`DISPLAY_NAMES` no `main.py`). Texto do HUD **sem acentos** (fontes Hershey do OpenCV).
- HUD: dados de diagnóstico em cima (FPS, deteção, ângulos, expressões, correções); dicas e
  controlos com tecla no fundo da janela; mensagem da calibração ao centro, em cima.

## Já tentado e revertido (não repetir da mesma forma)

1. **Malha dos braços/ombros/pescoço pela segmentação da pessoa** (redesenhar o tronco a partir
   da silhueta) — ficou mal na webcam real (perto da câmara os ombros reais estão muito acima das
   articulações; braço à frente do tronco dá larguras falsas). Feito depois dessa forma:
   `body_widths.py` (`--body-widths`, tecla `n`, desligado por defeito) só afina larguras, ±15%.
2. **Cabeça**: pose por ajuste Kabsch do modelo canónico (468 pontos) + forma do crânio medida na
   segmentação (elipsoide com 3 proporções → "bolota" de perfil) + contorno de 64 raios medidos.
   O utilizador reverteu tudo. Depois disso, o tremor passou a ser atacado só no filtro
   (`face_filter.py`, tecla `r`), sem mexer no desenho — ainda por validar na webcam real.
3. **Modo performance** (detetar em meia resolução, desligar íris): medido, não acelera o Holistic.

## Números medidos (para não voltar a medir)

- Holistic: ~55–70 ms por frame, custo praticamente fixo (≈ 15 FPS de deteção).
- Segmentação do Holistic (`--body-widths`): +~11 ms por deteção (foto real, 1280 px).
- Avatar 3D (/3d): o programa só envia ~1 KB de JSON por frame; o desenho é do browser (WebGL).
- Suavização rígida da cara (sintético): −29% tremor parado, −36% deformação a mexer, +0,5 px atraso.
- Desenho por frame (máquina sem carga): cartoon ~10 ms, robô ~7 ms, png ~19 ms
  (o png chegou a 244 ms antes de recortar os sprites e misturar em inteiros com OpenCV).
- Fundo transparente: ~3× o desenho normal; PNG recortado à zona do avatar ~18 ms, ~17 KB.
- A máquina do utilizador anda muitas vezes carregada: comparar tempos sempre na mesma execução.

## Testes (sem webcam)

- Pytest em `tests/` (instalar com `requirements-dev.txt`). Por defeito, `.venv\Scripts\python.exe -m pytest`
  corre os rápidos (~100, ~45 s). `-m mediapipe` corre o ciclo completo do `main.py` com câmara
  simulada; `-m parity` compara o cartoon pixel a pixel com HEAD (ou `AVATAR_PARITY_REF`).
  Correr os rápidos depois de cada mudança; a paridade em refatorizações. Acrescentar testes ao
  que se fizer de novo.
- `tests/conftest.py` tem `make_face` (modelo canónico em `tests/data/`, com yaw/roll e as
  íris 468–477), `make_pose`, `make_hand` e `make_state`.
- Cara sintética: modelo canónico do MediaPipe
  `https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj`
  (468 vértices; y para cima, z para a câmara; espelhar x para imitar a webcam).
- Deteção real sem webcam: `lena.jpg`/`messi5.jpg` de `opencv/opencv/4.x/samples/data` (GitHub).
  O MediaPipe não deteta o avatar cartoon como cara.
- Ciclo completo do `main.py` (`tests/test_main_loop.py`): câmara falsa em vez de
  `main.open_camera` e funções vazias em vez das janelas do OpenCV. No arranque (antes da
  1.ª deteção) só `q`/Esc contam, por isso as teclas simuladas só começam depois.
- A Fonte de Browser anuncia `http://127.0.0.1:<porta>`: no Windows `localhost` tenta primeiro
  o IPv6 e alguns clientes (urllib) perdem ~2 s por pedido.
- PowerShell `Set-Content -Encoding utf8` grava BOM e estraga acentos: ler com `utf-8-sig`, ou
  escrever ficheiros com a ferramenta Write.
- Browser integrado da app consegue abrir `http://localhost:<porta>` (serve para testar a Fonte
  de Browser); usar uma porta diferente da do utilizador (8765).
