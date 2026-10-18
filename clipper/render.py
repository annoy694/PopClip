"""Pass 4: cut the virtual camera out of the source video and encode it with the original audio."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera import CameraPath
from .video import VideoInfo, audio_codec, ffmpeg_exe

ProgressFn = Callable[[float, str], None]


def render(
    info: VideoInfo,
    path: CameraPath,
    out_size: tuple[int, int],
    out_file: str | Path,
    start: int = 0,
    end: int | None = None,
    crf: int = 18,
    preset: str = "fast",
    progress: ProgressFn | None = None,
) -> Path:
    out_w, out_h = out_size
    end = min(end if end is not None else len(path.cx), len(path.cx))
    rects = path.rects()
    out_file = Path(out_file)

    t0, dur = start / info.fps, (end - start) / info.fps
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{info.fps:.6f}", "-i", "-",
    ]  # fmt: skip
    if info.has_audio:
        cmd += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", info.path, "-map", "0:v", "-map", "1:a"]
        # Copy the original audio untouched when MP4 can hold it; otherwise encode high-bitrate AAC.
        cmd += ["-c:a", "copy"] if audio_codec(info.path) in ("aac", "mp3") else ["-c:a", "aac", "-b:a", "256k"]
    # Frames arrive as RGB: convert with the HD colour matrix and label the file, otherwise players read
    # the colours with a different matrix than they were written with and saturated colours drift.
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"]
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    cmd += ["-movflags", "+faststart", "-shortest", str(out_file)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(info.path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for f in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            x, y, w, h = rects[f]
            s = out_h / h
            # Crop + scale in one sub-pixel accurate warp, so slow pans glide instead of stepping pixel by pixel.
            M = np.array([[s, 0, -x * s], [0, s, -y * s]], np.float64)
            interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
            if interp == cv2.INTER_AREA:
                out = cv2.resize(frame[int(y) : int(y + h), int(x) : int(x + w)], (out_w, out_h), interpolation=interp)
            else:
                out = cv2.warpAffine(frame, M, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
            enc.stdin.write(out.tobytes())
            if progress and (f - start) % 15 == 0:
                progress((f - start) / max(end - start, 1), f"Rendering frame {f - start}/{end - start}")
    finally:
        cap.release()
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        code = enc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[-800:]}")
    if progress:
        progress(1.0, "Render done")
    return out_file
