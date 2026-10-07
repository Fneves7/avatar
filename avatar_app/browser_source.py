"""Saída com fundo transparente para a "Fonte de Browser" do OBS.

A câmara virtual não tem transparência, mas a Fonte de Browser do OBS sim. Este módulo
corre um servidor HTTP local (só em 127.0.0.1) com:
  /        página com fundo transparente que mostra o avatar num <canvas>;
  /frame   o último frame em PNG com transparência, recortado à zona do avatar
           (?after=N espera até haver um frame mais recente do que N, até 1 s).
O PNG é codificado na thread do servidor, por isso não atrasa o desenho do avatar.
No OBS: Fontes -> + -> Browser -> URL http://localhost:8765 (largura/altura do stream).
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ACTIVE_FOR_S = 2.0  # a página conta como ligada até 2 s depois do último pedido

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Avatar</title>
<style>html,body{margin:0;height:100%;background:transparent;overflow:hidden}canvas{display:block}</style>
</head><body><canvas id="c"></canvas><script>
const c = document.getElementById('c'), ctx = c.getContext('2d');
let seq = -1;
function fit() { c.width = innerWidth; c.height = innerHeight; }
addEventListener('resize', fit); fit();
async function loop() {
  try {
    const r = await fetch('/frame?after=' + seq, {cache: 'no-store'});
    if (r.status === 200) {
      seq = +r.headers.get('X-Seq');
      const [x, y, w, h] = r.headers.get('X-Rect').split(',').map(Number);
      const [W, H] = r.headers.get('X-Size').split(',').map(Number);
      const im = await createImageBitmap(await r.blob());
      const k = Math.min(c.width / W, c.height / H), ox = (c.width - W * k) / 2, oy = (c.height - H * k) / 2;
      ctx.clearRect(0, 0, c.width, c.height);
      if (w > 0) ctx.drawImage(im, ox + x * k, oy + y * k, w * k, h * k);
      im.close();
    }
  } catch (e) { await new Promise(res => setTimeout(res, 500)); }
  // setTimeout e não requestAnimationFrame: este pára quando o browser acha que a página não
  // está visível (ex.: fonte escondida no OBS) e o avatar congelava. O ritmo vem do servidor
  // (/frame só responde quando há um frame novo).
  setTimeout(loop, 0);
}
loop();
</script></body></html>
"""


class _Server(ThreadingHTTPServer):
    # No Windows, SO_REUSEADDR deixa dois programas abrir a mesma porta sem erro.
    allow_reuse_address = False
    daemon_threads = True


class BrowserSource:
    def __init__(self, port: int = 8765, host: str = "127.0.0.1"):
        self.url = f"http://localhost:{port}"
        self._cond = threading.Condition()
        self._frame: tuple[np.ndarray | None, tuple[int, int, int, int], tuple[int, int]] | None = None
        self._seq = 0
        self._encoded: tuple[int, bytes] | None = None
        self._last_request = -1e9
        self.error: str | None = None
        self._server = None
        try:
            self._server = _Server((host, port), self._handler())
            threading.Thread(target=self._server.serve_forever, name="fonte-browser", daemon=True).start()
            print(f"[fonte browser] a servir em {self.url} (OBS: Fonte de Browser com este URL)")
        except OSError as exc:
            self.error = f"porta {port} indisponivel: {exc}"
            print(f"[fonte browser] {self.error}")

    @property
    def active(self) -> bool:
        """True se a página (OBS) pediu frames recentemente: só então vale a pena gerá-los."""
        return self._server is not None and time.monotonic() - self._last_request < ACTIVE_FOR_S

    def publish(self, bgra: np.ndarray) -> None:
        """Novo frame (BGRA). Guarda só a zona onde o avatar não é transparente."""
        h, w = bgra.shape[:2]
        alpha = bgra[..., 3]
        rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
        if len(rows) == 0:
            crop, rect = None, (0, 0, 0, 0)
        else:
            y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            crop, rect = bgra[y0:y1, x0:x1].copy(), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        with self._cond:
            self._frame = (crop, rect, (w, h))
            self._seq += 1
            self._cond.notify_all()

    def _wait_frame(self, after: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after and self._frame is not None, timeout)
            if self._seq <= after or self._frame is None:
                return None
            seq, (crop, rect, size) = self._seq, self._frame
            cached = self._encoded
        if cached is not None and cached[0] == seq:
            data = cached[1]
        else:
            img = crop if crop is not None else np.zeros((1, 1, 4), np.uint8)
            data = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
            self._encoded = (seq, data)
        return seq, data, rect, size

    def _handler(self):
        source = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # sem spam na consola
                pass

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/":
                    body = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if url.path != "/frame":
                    self.send_error(404)
                    return
                source._last_request = time.monotonic()
                after = int(parse_qs(url.query).get("after", ["-1"])[0])
                result = source._wait_frame(after)
                if result is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                seq, data, rect, size = result
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Seq", str(seq))
                self.send_header("X-Rect", ",".join(map(str, rect)))
                self.send_header("X-Size", ",".join(map(str, size)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
