"""Avatar em tempo real com webcam + MediaPipe.

Teclas:
  q / ESC  sair
  c        mudar paleta de cores do avatar
  d        mostrar/esconder landmarks sobre a webcam
  w        mostrar/esconder a imagem da webcam
  h        alternar cabeça 3D (crânio, nuca, orelhas) / cabeça simples
  k        calibrar a pose neutra (olhar em frente, cara neutra ~2 s)
  K        apagar a calibração
  e        ligar/desligar expressões exageradas (precisa de calibração)
  + / -    aumentar/diminuir a intensidade do exagero
  s        ligar/desligar suavização
  v        ligar/desligar a câmara virtual (avatar como webcam no OBS/Teams/Zoom/Discord)
  b        mudar o fundo do avatar (gradiente / verde / azul / magenta para chroma key)
  p        guardar captura de ecrã em screenshots/
"""
from __future__ import annotations

import argparse
import os
import sys

# Silencia os logs internos do MediaPipe/TensorFlow (têm de ser definidos antes do import).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import time
from pathlib import Path

import cv2
import numpy as np

from avatar_app.calibration import Calibrator
from avatar_app.debug_draw import draw_landmarks
from avatar_app.hand_mesh import N_MEASURABLE
from avatar_app.pipeline import DetectionWorker, StateBlender
from avatar_app.renderer import AvatarRenderer
from avatar_app.streaming import BACKGROUND_NAMES, VirtualCamera, background_color
from avatar_app.tracker import BodyState, Tracker

WINDOW = "Avatar MediaPipe"
STREAM_WINDOW = "Avatar (stream)"
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Avatar controlado pela webcam (rosto, corpo, mãos).")
    ap.add_argument("--camera", type=int, default=0, help="índice da webcam (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full",
                    help="lite = mais rápido, heavy = mais preciso")
    ap.add_argument("--backend", choices=["auto", "holistic", "tasks"], default="auto",
                    help="holistic = modelos incluídos no mediapipe 0.10.21 (sem downloads); "
                         "tasks = FaceLandmarker/PoseLandmarker/HandLandmarker (precisa de models/*.task)")
    ap.add_argument("--no-hand-mesh", action="store_true",
                    help="não medir o contorno dos dedos (usa larguras por defeito)")
    ap.add_argument("--no-mirror", action="store_true", help="não espelhar a imagem")
    ap.add_argument("--palette", type=int, default=0)
    # Streaming.
    ap.add_argument("--virtual-cam", action="store_true",
                    help="ligar logo a câmara virtual (precisa do OBS Studio instalado no Windows)")
    ap.add_argument("--stream-window", action="store_true",
                    help="abrir uma janela só com o avatar (para 'Captura de janela' no OBS)")
    ap.add_argument("--output", default=None, metavar="LxA",
                    help="resolução da saída de stream, ex.: 1280x720 (por defeito = a da webcam)")
    ap.add_argument("--background", choices=BACKGROUND_NAMES, default="gradiente",
                    help="fundo do avatar; verde/azul/magenta para chroma key")
    ap.add_argument("--fps", type=int, default=30,
                    help="ritmo a que o avatar é desenhado/enviado (a deteção corre ao seu ritmo)")
    ap.add_argument("--sync", action="store_true",
                    help="modo antigo: um desenho por deteção, sem interpolação")
    return ap.parse_args()


def parse_size(text: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if not text:
        return default
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        return w, h
    except ValueError:
        sys.exit(f"--output inválido: {text!r} (usa por exemplo 1280x720)")


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # A deteção é mais lenta do que a webcam: fila de 1 frame para não processar imagens antigas.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_hud(img: np.ndarray, s: BodyState, fps: float, smoothing: bool,
             calib_status: str | None = None, exaggeration: float | None = None,
             stream_status: str | None = None, detect_fps: float | None = None) -> None:
    def status(label, ok):
        return f"{label}:{'OK' if ok else '--'}"

    fps_text = f"FPS avatar {fps:4.1f}" + (f"  detecao {detect_fps:4.1f}" if detect_fps is not None else "")
    lines = [
        f"{fps_text}   suavizacao {'ON' if smoothing else 'OFF'}   "
        f"{'calibrado' if s.calibrated else 'sem calibracao [k]'}",
        "  ".join([status("rosto", s.face is not None), status("corpo", s.pose is not None),
                   status("maos", len(s.hands))]).replace("maos:OK", f"maos:{len(s.hands)}"),
    ]
    if s.hand_meshes:
        # Quantas falanges (de 14 por mão) tiveram a largura medida na imagem neste frame.
        parts = [f"{m.measured}/{N_MEASURABLE}" for m in s.hand_meshes.values()]
        lines.append("malha dedos medida: " + "  ".join(parts))
    if s.head_angles:
        yaw, pitch, roll = s.head_angles
        lines.append(f"cabeca  yaw {yaw:+5.0f}  pitch {pitch:+5.0f}  roll {roll:+5.0f}")
    bs = s.blendshapes
    if bs:
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        brow = bs.get("browInnerUp", 0) - (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        lines.append(f"boca {bs.get('jawOpen', 0):.2f}  sorriso {smile:.2f}  piscar {blink:.2f}"
                     f"  sobrancelhas {brow:+.2f}")
    if s.calibrated:
        lines.append(f"exagero {'x%.2f' % exaggeration if exaggeration else 'OFF'}  [e] ligar/desligar  [+/-] intensidade")
    if stream_status:
        lines.append(stream_status)
    lines.append("[c] cores [d] landmarks [w] webcam [h] cabeca 3D [k] calibrar [s] suavizar [p] print [q] sair")

    y = 24
    for text in lines:
        cv2.putText(img, text, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    if calib_status:
        # Mensagem de calibração em destaque, ao centro em baixo.
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        (tw, th), _ = cv2.getTextSize(calib_status, font, scale, thick)
        x, yb = (img.shape[1] - tw) // 2, img.shape[0] - 40
        cv2.rectangle(img, (x - 14, yb - th - 14), (x + tw + 14, yb + 14), (0, 0, 0), -1)
        cv2.putText(img, calib_status, (x, yb), font, scale, (0, 230, 255), thick, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir a webcam {args.camera}.")

    tracker = Tracker(pose_model=args.pose_model, backend=args.backend, hand_mesh=not args.no_hand_mesh)
    renderer = AvatarRenderer(args.palette)
    renderer.background = background_color(args.background)
    background_name = args.background
    calibrator = Calibrator(CALIBRATION_FILE)
    show_landmarks, show_webcam = True, True
    fps, last = 0.0, time.perf_counter()

    # Saída de stream: tamanho por defeito = o da webcam (o OpenCV pode não dar o pedido).
    cam_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height)
    out_size = parse_size(args.output, cam_size)
    vcam = VirtualCamera(*out_size, fps=args.fps)
    if args.virtual_cam:
        vcam.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if args.stream_window:
        cv2.namedWindow(STREAM_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(STREAM_WINDOW, *out_size)
    # A deteção corre numa thread própria; aqui desenha-se a ritmo fixo (ou, com --sync,
    # um desenho por deteção, como antes).
    worker = DetectionWorker(cap, tracker, mirror=not args.no_mirror)
    worker.start()
    blender = StateBlender()
    frame_period = 1.0 / max(args.fps, 1)
    last_seq, state, frame = -1, None, None
    try:
        while True:
            t_frame = time.perf_counter()
            latest = worker.latest()
            if worker.failed and (latest is None or latest[0] == last_seq):
                print("Frame da webcam falhou; a terminar.")
                break
            if latest is None:  # ainda a arrancar
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            seq, frame, detected = latest
            new_detection = seq != last_seq
            if new_detection:
                last_seq = seq
                calibrator.process(detected)  # recolhe a pose neutra ou aplica a calibração (1x por deteção)
            elif args.sync:
                cv2.waitKey(1)
                continue
            state = detected if args.sync else blender.update(detected)
            avatar = renderer.render(state)

            # Saída limpa para o stream (sem HUD nem webcam), antes de desenhar o HUD.
            stream = avatar
            if (avatar.shape[1], avatar.shape[0]) != out_size:
                stream = cv2.resize(avatar, out_size, interpolation=cv2.INTER_AREA)
            vcam.send(stream)
            if args.stream_window:
                cv2.imshow(STREAM_WINDOW, stream)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if show_webcam:
                cam_view = frame.copy()  # o frame é partilhado com a thread de deteção
                if show_landmarks:
                    draw_landmarks(cam_view, detected)
                view = np.hstack([cam_view, avatar])
            else:
                view = avatar.copy()
            if vcam.active:
                cam_text = f"camara virtual ON ({vcam.device})"
            elif vcam.error:
                cam_text = f"camara virtual com erro: {vcam.error[:60]}"
            else:
                cam_text = "camara virtual OFF"
            stream_status = f"stream {out_size[0]}x{out_size[1]}: {cam_text} [v]   fundo {background_name} [b]"
            draw_hud(view, state, fps, tracker.smoothing, calibrator.status(),
                     renderer.exaggeration if renderer.exaggerate else None, stream_status, worker.fps)
            cv2.imshow(WINDOW, view)

            # Ritmo fixo: espera o que falta para completar o período do frame.
            wait_ms = 1 if args.sync else max(1, int((frame_period - (time.perf_counter() - t_frame)) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                renderer.next_palette()
            elif key == ord("d"):
                show_landmarks = not show_landmarks
            elif key == ord("w"):
                show_webcam = not show_webcam
            elif key == ord("h"):
                renderer.head_3d = not renderer.head_3d
            elif key == ord("k"):
                calibrator.start()
            elif key == ord("K"):
                calibrator.reset()
            elif key == ord("e"):
                renderer.exaggerate = not renderer.exaggerate
            elif key in (ord("+"), ord("=")):
                renderer.exaggeration = min(2.5, renderer.exaggeration + 0.25)
            elif key == ord("-"):
                renderer.exaggeration = max(0.25, renderer.exaggeration - 0.25)
            elif key == ord("s"):
                tracker.smoothing = not tracker.smoothing
            elif key == ord("v"):
                if vcam.active:
                    vcam.stop()
                else:
                    vcam.start()
            elif key == ord("b"):
                background_name = BACKGROUND_NAMES[(BACKGROUND_NAMES.index(background_name) + 1)
                                                   % len(BACKGROUND_NAMES)]
                renderer.background = background_color(background_name)
            elif key == ord("p"):
                out = Path("screenshots")
                out.mkdir(exist_ok=True)
                path = out / f"avatar_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), view)
                print(f"Guardado {path}")
    finally:
        worker.stop()
        vcam.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
