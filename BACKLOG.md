# Backlog

Objetivo do projeto: avatar em tempo real para **streaming**.

## Feito

- [x] **Calibração da pose neutra** (tecla `k`, `Shift+K` apaga): a cabeça e as expressões passam a ser relativas à cara de cada pessoa. É guardada em `calibration.json`.
- [x] **Expressões exageradas** (tecla `e`, intensidade com `+` e `-`; precisa de calibração): arregalar e semicerrar os olhos, sobrancelhas a subir e a franzir, sorriso curvado, boca aberta mais marcada. Os ganhos estão no topo de `styles/cartoon.py`; a intensidade escolhida com `+`/`-` fica guardada (ver Preferências).

## Estado

Todos os itens planeados estão feitos. O que falta é só validação com a webcam real (cada parte
tem uma tecla para comparar), e o que se escolher fica guardado (ver Preferências):

- confirmar a suavização rígida da cara (`r`) e as larguras pela silhueta (`--body-widths`, `n`);
- escolher o perfil de suavização (`Shift+S`) e a intensidade do exagero (`+`/`-`).

## Animação

- [x] **Movimento secundário** (tecla `m`): molas amortecidas.
  - **Cabelo:** o topo segue a posição e a inclinação da cabeça com atraso, passa um pouco do ponto e assenta (até ~12% da largura da cara e ~8°, com limite suave). A franja fica presa à testa.
  - **Mangas:** a bainha, no cotovelo, segue o braço (até ~6% da largura dos ombros).
  - **Bainha do tronco** (estilo cartoon): o fundo da camisola segue o tronco com atraso (até ~5% da largura dos ombros); a cintura mexe metade. É subtil, porque o fundo quase sempre fica fora de imagem. No robô (metal) e no png (sprite rígido) não se aplica.
- [x] **Olhar vivo** (tecla `l`): o avatar pisca sozinho (de 2,5 a 6 s, ~170 ms) quando não há um piscar real recente, porque a deteção a ~10–15 FPS perde muitos piscares. A íris faz micro-movimentos subtis (≤20% do raio), somados ao olhar real. Também funciona na cabeça simples, quando o rosto não é detetado.
- [x] **Idle** (tecla `i`): depois de ~1 s parado, o avatar respira (ciclo de ~4,5 s, ombros a subir ~1,5% da largura dos ombros; cotovelos e cabeça acompanham em parte, mãos e ancas não). Entra aos poucos e sai em ~0,4 s ao primeiro movimento. O movimento é medido numa janela de 0,4 s, para o ruído da deteção não o desligar.
- [x] **Transições suaves** (tecla `t`): as mãos e a cara aparecem em ~0,12 s e desaparecem em ~0,25 s, em vez de cortar. A cara faz uma transição cruzada com a cabeça simples da pose, e a mão com a luva. Uma mão detetada só num frame fica quase invisível (~20%).
- [x] **Transições do corpo inteiro** (também na tecla `t`): quando a pose se perde, o tronco e os braços desvanecem em ~0,25 s a partir da última pose vista; quando volta, entram em ~0,12 s. O renderer mistura o desenho com e sem a pose, por isso funciona nos três estilos e a transparência continua exata. Custa ~3–4 ms a mais por frame, só durante a transição. No arranque o corpo aparece logo.

## Estabilidade

- [x] **Limites anatómicos da pose** (tecla `a`): só corrige o que é claramente impossível, e o HUD diz o que foi corrigido.
  - **Cotovelo vs. mão:** com um ângulo impossível entre o antebraço e a mão (>60° com o cotovelo pouco visível, ou >110° sempre), o cotovelo passa para o prolongamento da mão.
  - **Saltos:** uma articulação pouco visível que salta mais de 0,5× a largura dos ombros fica retida até 3 deteções.
  - **Braços esticados:** braço ou antebraço com mais de 1,5× a largura dos ombros é encurtado.
  - **Ancas acima dos ombros:** são ignoradas.
- [x] **Suavização rígida da cara** (tecla `r`, ligada por defeito): o filtro ponto a ponto dava a cada ponto um atraso diferente quando a cabeça rodava, e a malha tremia e deformava-se. Agora a posição, a escala e a inclinação da cara (ajuste 2D sobre pontos estáveis) são filtradas à parte, mais forte, e as expressões como antes. Num teste sintético com ruído parecido com o da deteção: −29% de tremor parado e −36% de deformação a mexer, com +0,5 px de atraso. Só muda o filtro, não o desenho.
- [x] **Perfis de suavização** (`Shift+S`): leve (menos atraso, mais tremor), normal e forte (mais estável). O perfil escolhido fica guardado.
- [x] **Preferências guardadas** (`settings.json`, pessoal): ao sair guarda o perfil de suavização, a intensidade do exagero e as partes ligadas/desligadas com as teclas; repõe-nos no arranque. As opções da linha de comandos mandam sempre.
- [x] **Braços e ombros (abordagem conservadora)** (`--body-widths`, tecla `n`; desligado por defeito): o desenho é o mesmo; a silhueta do Holistic só afina a largura de cada braço, antebraço e do tronco por um fator limitado (±15% nos braços, ±10% no tronco) e suavizado. A medição é rejeitada com a articulação pouco visível, com o braço por cima do tronco, ou quando não se encontra a borda da silhueta. Numa foto real: fatores dos braços entre 0,88 e 1,13, tronco rejeitado (braços junto ao peito). A segmentação custa ~11 ms por deteção. Uma primeira versão (medir tudo na segmentação e redesenhar o tronco a partir disso) tinha sido revertida em 2026-10-07.

## Streaming

- [x] **Câmara virtual:** o avatar como webcam ("OBS Virtual Camera") para OBS, Teams, Zoom e Discord. Usa `pyvirtualcam`, liga e desliga com a tecla `v` ou com `--virtual-cam`, e `--output` define a resolução.
- [x] **Fundo para chroma key:** verde, azul ou magenta (`--background`, ou a tecla `b` ao vivo).
- [x] **Janela só com o avatar** (`--stream-window`): sem HUD nem webcam, para "Captura de janela" no OBS.
- [x] **Fundo transparente** (`--browser-source`): página local para a Fonte de Browser do OBS. A transparência é exata (desenho sobre preto e sobre branco), funciona com os três estilos e é enviada em PNG recortado à zona do avatar. Custa ~3× o desenho normal, só enquanto o OBS está ligado. Testado num browser Chromium: animado, com fundo transparente.
- [x] **Fluidez:** a deteção corre numa thread separada e o avatar é desenhado a ritmo fixo (`--fps`, 30 por defeito), deslizando para a última deteção. No teste, o avatar passou de 11 para 26 FPS, em passos ~2× mais pequenos. `--sync` volta ao modo antigo. (Reduzir a imagem de deteção ou desligar a íris não acelera o Holistic, que custa ~55 ms por frame.)

## Avatares

- [x] **Separar o rig do estilo:** `animation.py` (animação partilhada), `rig.py` (parâmetros estilo VTuber) e `styles/` (cartoon e robô; a tecla `y` alterna). O cartoon ficou pixel a pixel igual ao do commit anterior (450/450 frames de teste). O robô é o exemplo de um estilo feito só a partir do rig.
- [x] **Avatares PNG por camadas** (estilo `png`): imagens com transparência e um `avatar.json` com os pontos de encaixe, animadas pelo rig. Os olhos (aberto, meio e fechado, um de cada vez, com a íris a seguir o olhar), as sobrancelhas, a boca (4 estados) e o rubor seguem as expressões. Há paralaxe ao rodar a cabeça, molas no pelo, membros esticados pelos ossos e patas abertas ou fechadas conforme os dedos. Inclui o avatar de exemplo "gato", gerado por `tools/make_sample_avatar.py`, a ~19 ms por frame.
- [x] **Avatar PNG a sério** (estilo `pessoa`, `avatars/pessoa`): personagem humana estilo anime/VTuber, gerada por `tools/make_human_avatar.py` com sombreado cel, cabelo com madeixas e reflexo, olhos com pestanas e brilhos, rubor, camisola com gola e mãos com dedos. A mão é espelhada conforme o lado do polegar detetado (`"thumb"` no `avatar.json`), para a mesma imagem servir às duas mãos. As imagens podem ser trocadas por desenhos feitos à mão com os mesmos nomes e pontos de encaixe.
- [x] **Avatar 3D** (three.js): página `/3d` da Fonte de Browser (com `--browser-source`), com fundo transparente, desenhada pelo OBS a partir do rig em JSON (`rig_to_dict`, ~1 KB por frame). Cel shading com contorno; a cabeça roda em 3D; olhos, sobrancelhas, boca e rubor seguem as expressões; tronco, braços, mãos e dedos seguem o esqueleto. O three.js r169 vem no projeto (a rede bloqueia os CDNs). Testado num Chromium com um rig sintético animado.
  - Ideia para mais tarde: carregar um modelo VRM feito noutro programa (precisa do modelo e do `three-vrm`) em vez das formas simples.

## Descartado

- Gravar e reproduzir sessões: não faz sentido num projeto de streaming em direto.
