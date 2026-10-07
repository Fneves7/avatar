"""Avatar em tempo real com webcam + MediaPipe.

Teclas:
  q / ESC  sair
  c        mudar paleta de cores do avatar
  d        mostrar/esconder landmarks sobre a webcam
  w        mostrar/esconder a imagem da webcam
  h        alternar cabeça 3D (crânio, nuca, orelhas) / cabeça simples
  k        calibrar a pose neutra (olhar em frente, cara neutra ~2 s)
  K        apagar a calibração
  s        ligar/desligar suavização
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
from avatar_app.renderer import AvatarRenderer
from avatar_app.tracker import BodyState, Tracker

WINDOW = "Avatar MediaPipe"
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
    return ap.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_hud(img: np.ndarray, s: BodyState, fps: float, smoothing: bool,
             calib_status: str | None = None) -> None:
    def status(label, ok):
        return f"{label}:{'OK' if ok else '--'}"

    lines = [
        f"FPS {fps:4.1f}   suavizacao {'ON' if smoothing else 'OFF'}   "
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
        lines.append(f"boca {bs.get('jawOpen', 0):.2f}  sorriso {smile:.2f}  piscar {blink:.2f}")
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
    calibrator = Calibrator(CALIBRATION_FILE)
    show_landmarks, show_webcam = True, True
    fps, last = 0.0, time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame da webcam falhou; a terminar.")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            state = tracker.process(frame)
            calibrator.process(state)  # recolhe a pose neutra ou aplica a calibração
            avatar = renderer.render(state)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if show_webcam:
                if show_landmarks:
                    draw_landmarks(frame, state)
                view = np.hstack([frame, avatar])
            else:
                view = avatar
            draw_hud(view, state, fps, tracker.smoothing, calibrator.status())
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
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
            elif key == ord("s"):
                tracker.smoothing = not tracker.smoothing
            elif key == ord("p"):
                out = Path("screenshots")
                out.mkdir(exist_ok=True)
                path = out / f"avatar_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), view)
                print(f"Guardado {path}")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
