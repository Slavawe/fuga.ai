
"""YouTube Stream Ingest: видео -> кадры 512x512 -> 32K VSA-состояния.

Трубопровод (без записи сырого видео на диск):
  yt-dlp -g            -> прямая ссылка потока
  ffmpeg image2pipe    -> RAW bgr24 кадры в RAM-буфер
  Temporal Delta       -> HV(кадр t) и HV_дельта(t -> t+1) для H-JEPA
"""

from __future__ import annotations

from __future__ import annotations

import subprocess
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astral.astral_env import ScaledAstralEnvironment
from antitf.rust_bridge import packed_to_torch


class YouTubeStreamIngest:
    def __init__(self, youtube_url: str, resolution=(512, 512), fps: int = 15,
                 max_height: int = 720):
        self.url = youtube_url
        self.res = resolution
        self.fps = fps
        self.max_height = max_height
        self.env = ScaledAstralEnvironment(vector_dim=32768)

    def resolve_stream(self) -> str:
        if os.path.exists(self.url):                 # локальный файл — тот же пайплайн
            return self.url
        cmd = ["yt-dlp", "-g", "-f",
               f"bestvideo[height<={self.max_height}]/best[height<={self.max_height}]",
               self.url]
        out = subprocess.check_output(cmd, timeout=120)
        return out.decode().strip().splitlines()[-1]

    def stream_frames(self):
        """Генератор сырых кадров BGR 512x512."""
        stream_url = self.resolve_stream()
        ffmpeg_cmd = [
            "ffmpeg", "-i", stream_url,
            "-vf", f"scale={self.res[0]}:{self.res[1]},fps={self.fps}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-loglevel", "error", "pipe:1",
        ]
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                                bufsize=10 ** 7)
        frame_size = self.res[0] * self.res[1] * 3

        def read_exact(stream, n):
            buf = bytearray()
            while len(buf) < n:
                chunk = stream.read(n - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            return bytes(buf)

        try:
            while True:
                raw = read_exact(pipe.stdout, frame_size)
                if raw is None:
                    break
                yield np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self.res[1], self.res[0], 3))
        finally:
            pipe.kill()

    @staticmethod
    def temporal_delta_hv(hv_t, hv_t1) -> np.ndarray:
        """Дельта-кристалл: что ИЗМЕНИЛОСЬ между кадрами."""
        d = hv_t1.astype(np.int16) - hv_t.astype(np.int16)
        d = np.sign(d).astype(np.uint64) > 0
        packed = np.zeros((1, hv_t.shape[-1] // 64), dtype=np.uint64)
        flat = d.flatten()
        for i in range(64):
            packed |= flat[:, i::64].astype(np.uint64) << np.uint64(i)
        return packed


def run_ingest(url: str, max_frames: int = 120, train_steps: int = 300):
    """Полный цикл: стрим -> HV состояния -> обучение H-JEPA на дельтах."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from astral.astral_runner import JepaPredictor

    ing = YouTubeStreamIngest(url)
    predictor = JepaPredictor(dim=32768)
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    prev_hv = None
    frames = ing.stream_frames()
    errs, base = [], []
    t0 = time.time()
    fi = 0
    while fi < max_frames:
        try:
            frame = next(frames)
        except (StopIteration, subprocess.CalledProcessError, Exception) as e:
            print(f"[ingest] поток завершён/ошибка: {type(e).__name__}: {e}"[:200])
            break
        tokens = ing.env._frame_tokens(frame)
        pk = np.asarray(ing.env.binder.bind_batch([tokens]))
        hv = packed_to_torch(pk)[0].flatten()

        if prev_hv is not None:
            h_prev = prev_hv.view(1, -1)
            pred = predictor(h_prev, torch.tensor([0]))
            real = hv.view(1, -1)
            err_p = float((pred - real).detach().norm() / (real.norm() + 1e-9))
            err_b = float((h_prev - real).norm() / (real.norm() + 1e-9))
            loss = ((pred - real) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            errs.append(err_p); base.append(err_b)

            if fi % 20 == 0:
                w = min(len(errs), 30)
                print(f"  frame {fi}: pred_err={np.mean(errs[-w:]):.4f} "
                      f"baseline={np.mean(base[-w:]):.4f}")
        prev_hv = hv
        fi += 1

    if errs:
        print(f"\n[youtube ingest] кадров={fi} за {time.time()-t0:.0f}s | "
              f"pred_err={np.mean(errs[-50:]):.4f} vs baseline={np.mean(base[-50:]):.4f} | "
              f"improvement={(np.mean(base[-50:])-np.mean(errs[-50:]))/max(np.mean(base[-50:]),1e-9)*100:.1f}%")
    return fi


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        print("usage: python astral/ingest_yt.py <youtube_url> [max_frames]")
        sys.exit(2)
    mf = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    run_ingest(url, max_frames=mf)
