"""Split Screen and Overlay rendering.

The live preview in the browser (web/studio.js) mirrors this module's maths exactly: `crop_rect`,
`clip_frames`, `output_frames`, `split_geometry`, `seam_alpha` and the blend modes in `_blend`.
Change them together.

Each video layer is decoded by its own ffmpeg process that seeks to the trim start and outputs
exactly one pass of the trimmed clip, already cropped and scaled to the layer's box at 60 fps.
Looping restarts (or replays from memory) that same pass, so every repeat is the trimmed section.
Frames are composited with numpy and piped to an H.264 encoder; sound is built with an ffmpeg
filter graph that trims, loops or pads each layer to the exact output length.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .media import MediaInfo, load_picture
from .video import ffmpeg_exe

FPS = 60
SAMPLE_RATE = 48000
SHAPES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "21:9": (2560, 1080),
}
SIZE_RANGE = (320, 3840)
DURATION_RANGE = (1.0, 120.0)
ZOOM_RANGE = (1.0, 4.0)
BLEND_MODES = ("normal", "screen", "multiply", "overlay", "lighten", "darken", "difference", "add")
FADES = ("both", "left", "right")
ON_END = ("loop", "freeze", "hide")
CACHE_LIMIT = 300 * 1024 * 1024  # replay looping clips from memory up to this many bytes

ProgressFn = Callable[[float, str], None]


def js_round(v: float) -> int:
    """Math.round from JavaScript (Python's round() rounds halves to even)."""
    return math.floor(v + 0.5)


@dataclass
class Layer:
    media: MediaInfo
    pos_x: float = 0.5  # 0..1, which part of the excess is cropped away
    pos_y: float = 0.5
    zoom: float = 1.0  # 1 = just fills its box
    trim_start: float = 0.0
    trim_end: float = 0.0  # seconds; 0 means the end of the video
    on_end: str = "freeze"  # what a video does after its trimmed clip: loop | freeze | hide
    volume: float = 1.0  # 0..1.5

    def __post_init__(self) -> None:
        if self.media.kind == "video":
            dur = self.media.duration
            end = self.trim_end if self.trim_end > 0 else dur
            self.trim_start = float(np.clip(self.trim_start, 0.0, max(dur - 0.1, 0.0)))
            self.trim_end = float(np.clip(end, self.trim_start + 0.1, max(dur, self.trim_start + 0.1)))
        self.zoom = float(np.clip(self.zoom, *ZOOM_RANGE))
        self.pos_x = float(np.clip(self.pos_x, 0, 1))
        self.pos_y = float(np.clip(self.pos_y, 0, 1))
        if self.on_end not in ON_END:
            raise ValueError(f"on_end must be one of {ON_END}")

    @property
    def is_video(self) -> bool:
        return self.media.kind == "video"

    @property
    def sound(self) -> bool:
        return self.is_video and self.media.has_audio and self.volume > 0


@dataclass
class Composition:
    layout: str  # "split" (layers = [left, right]) or "overlay" (layers = [bottom, top])
    width: int
    height: int
    layers: list[Layer]
    auto_duration: bool = True
    duration: float = 10.0  # used when auto_duration is off, or when no video decides it
    blend: float = 240.0  # split: width of the soft seam in output pixels
    fade: str = "both"  # split: both | left (left side fades over the right) | right
    opacity: float = 0.5  # overlay: top layer opacity
    mode: str = "normal"  # overlay: blend mode
    layer_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # filled in by __post_init__

    def __post_init__(self) -> None:
        if self.layout not in ("split", "overlay") or len(self.layers) != 2:
            raise ValueError("A composition needs a layout (split/overlay) and exactly two layers.")
        self.width = even(np.clip(self.width, *SIZE_RANGE))
        self.height = even(np.clip(self.height, *SIZE_RANGE))
        self.duration = float(np.clip(self.duration, *DURATION_RANGE))
        self.opacity = float(np.clip(self.opacity, 0, 1))
        if self.mode not in BLEND_MODES or self.fade not in FADES:
            raise ValueError("Unknown blend mode or fade direction.")
        if self.layout == "split":
            z0, z1 = split_geometry(self.width, self.blend, self.fade)
            self.layer_boxes = [(0, 0, z1, self.height), (z0, 0, self.width - z0, self.height)]
        else:
            self.layer_boxes = [(0, 0, self.width, self.height)] * 2


def even(v: float) -> int:
    return int(js_round(v / 2) * 2)


def crop_rect(src_w: int, src_h: int, box_w: int, box_h: int, pos_x: float, pos_y: float, zoom: float) -> tuple:
    """Part of the source (x, y, w, h in source pixels) shown in a box: cover the box, then zoom in."""
    s = max(box_w / src_w, box_h / src_h) * zoom
    w = min(src_w, max(1, js_round(box_w / s)))
    h = min(src_h, max(1, js_round(box_h / s)))
    return js_round((src_w - w) * pos_x), js_round((src_h - h) * pos_y), w, h


def clip_frames(layer: Layer) -> int:
    """Frames in one pass of a video's trimmed clip (its loop period)."""
    return max(1, js_round((layer.trim_end - layer.trim_start) * FPS))


def output_frames(comp: Composition) -> int:
    """Auto duration = the shortest video clip that does not loop; otherwise the chosen duration."""
    deciding = [clip_frames(ly) for ly in comp.layers if ly.is_video and ly.on_end != "loop"]
    if comp.auto_duration and deciding:
        return max(js_round(DURATION_RANGE[0] * FPS), min(deciding))
    return js_round(comp.duration * FPS)


def split_geometry(width: int, blend: float, fade: str) -> tuple[int, int]:
    """Seam zone [z0, z1): the left side covers [0, z1), the right side [z0, width)."""
    half = width / 2
    b = float(np.clip(blend, 0, 0.9 * half))
    if fade == "left":  # the left side reaches past the middle and fades out over the right side
        return js_round(half), js_round(half + b)
    if fade == "right":  # the right side reaches past the middle and fades out over the left side
        return js_round(half - b), js_round(half)
    return js_round(half - b / 2), js_round(half + b / 2)


def seam_alpha(width: int, z0: int, z1: int) -> np.ndarray:
    """(width,) uint8 opacity of the LEFT side per column: 255 before the seam, smoothstep down to 0 across it."""
    x = np.arange(width) + 0.5
    t = np.clip((x - z0) / max(z1 - z0, 1e-9), 0, 1) if z1 > z0 else (x >= z0).astype(float)
    right = t * t * (3 - 2 * t)
    return (255 - np.floor(right * 255 + 0.5)).astype(np.uint8)


# ----------------------------------------------------------------------------------------- sources


class _Source:
    """Frames of one layer for consecutive output frames, already cropped and scaled to its box."""

    def __init__(self, layer: Layer, box_w: int, box_h: int):
        self.layer = layer
        self.w, self.h = box_w, box_h
        m = layer.media
        self.rect = crop_rect(m.width, m.height, box_w, box_h, layer.pos_x, layer.pos_y, layer.zoom)
        self.still: np.ndarray | None = None
        self.proc: subprocess.Popen | None = None
        if not layer.is_video:
            x, y, w, h = self.rect
            img = load_picture(m.path).crop((x, y, x + w, y + h)).resize((box_w, box_h), Image.Resampling.BICUBIC)
            self.still = np.asarray(img)
            return
        self.n = clip_frames(layer)
        self.frame_bytes = box_w * box_h * 3
        self.cache: list[bytes] | None = [] if layer.on_end == "loop" and self.n * self.frame_bytes <= CACHE_LIMIT else None
        self.last: np.ndarray | None = None
        self.next_index = 0

    def _start(self) -> None:
        x, y, w, h = self.rect
        ly = self.layer
        vf = f"crop={w}:{h}:{x}:{y}:exact=1,scale={self.w}:{self.h}:flags=bicubic,fps={FPS},format=rgb24"
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-v", "error", "-ss", f"{ly.trim_start:.4f}", "-i", ly.media.path, "-an",
             "-t", f"{self.n / FPS:.4f}", "-vf", vf, "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )  # fmt: skip

    def _read(self, idx: int) -> np.ndarray:
        """Frame idx (0..n-1) of the trimmed clip; short decodes repeat their last frame."""
        if self.cache is not None and idx < len(self.cache):
            return np.frombuffer(self.cache[idx], np.uint8).reshape(self.h, self.w, 3)
        if idx == 0:
            self.close()
            self._start()
        data = b""
        if self.proc is not None and self.proc.stdout is not None:
            data = self.proc.stdout.read(self.frame_bytes)
        if len(data) == self.frame_bytes:
            frame = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)
        else:
            frame = self.last if self.last is not None else np.zeros((self.h, self.w, 3), np.uint8)
        if self.cache is not None:
            self.cache.append(frame.tobytes())
        if idx == self.n - 1:
            self.close()
        return frame

    def frame(self, k: int) -> np.ndarray | None:
        """Picture for output frame k (frames must be requested in order), or None when hidden."""
        if self.still is not None:
            return self.still
        if k >= self.n and self.layer.on_end != "loop":
            if self.layer.on_end == "hide":
                return None
            return self.last  # freeze on the last frame of the clip
        idx = k % self.n
        frame = self._read(idx)
        self.last = frame
        return frame

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


# ----------------------------------------------------------------------------------------- compositing


def _paste(dst: np.ndarray, src: np.ndarray, col_alpha: np.ndarray | None = None) -> None:
    """Draw src over dst (same size) with optional per-column opacity (uint8) and the picture's own alpha."""
    rgb = src[..., :3]
    alpha = None
    if src.shape[2] == 4:
        alpha = src[..., 3:4].astype(np.float32) / 255
    if col_alpha is not None and (col_alpha < 255).any():
        ca = (col_alpha.astype(np.float32) / 255)[None, :, None]
        alpha = ca if alpha is None else alpha * ca
    if alpha is None:
        dst[...] = rgb
        return
    out = rgb.astype(np.float32) * alpha + dst.astype(np.float32) * (1 - alpha)
    dst[...] = np.floor(out + 0.5).astype(np.uint8)


def _blend(b: np.ndarray, t: np.ndarray, mode: str) -> np.ndarray:
    """Separable blend modes as defined for HTML canvas / CSS, on uint8 pictures."""
    if mode == "screen":
        return 255 - cv2.multiply(255 - b, 255 - t, scale=1 / 255)
    if mode == "multiply":
        return cv2.multiply(b, t, scale=1 / 255)
    if mode == "overlay":
        low = cv2.multiply(b, t, scale=2 / 255)
        high = 255 - cv2.multiply(255 - b, 255 - t, scale=2 / 255)
        return np.where(b <= 127, low, high)
    if mode == "lighten":
        return cv2.max(b, t)
    if mode == "darken":
        return cv2.min(b, t)
    if mode == "difference":
        return cv2.absdiff(b, t)
    return t


def composite(comp: Composition, frames: list[np.ndarray | None]) -> np.ndarray:
    """One output frame (H, W, 3 uint8) from the two layers' box-sized pictures (None = hidden)."""
    out = np.zeros((comp.height, comp.width, 3), np.uint8)
    if comp.layout == "split":
        left, right = frames
        z0, z1 = split_geometry(comp.width, comp.blend, comp.fade)
        if right is not None:
            _paste(out[:, z0:], right)
        if left is not None:
            if left.shape[2] == 4:
                _paste(out[:, :z1], left, seam_alpha(comp.width, z0, z1)[:z1])
            else:  # opaque: copy up to the seam, blend only the seam columns
                out[:, :z0] = left[:, :z0]
                if z1 > z0:
                    a = (seam_alpha(comp.width, z0, z1)[z0:z1].astype(np.float32) / 255)[None, :, None]
                    zone = left[:, z0:z1].astype(np.float32) * a + out[:, z0:z1].astype(np.float32) * (1 - a)
                    out[:, z0:z1] = np.floor(zone + 0.5).astype(np.uint8)
        return out
    bottom, top = frames
    if bottom is not None:
        _paste(out, bottom)
    if top is None or comp.opacity <= 0:
        return out
    a = comp.opacity
    if top.shape[2] == 3:  # fast path: one opacity for the whole layer
        if comp.mode == "add":  # canvas "lighter": colours add up, scaled by the top layer's opacity
            return cv2.addWeighted(out, 1.0, top, a, 0)
        return cv2.addWeighted(_blend(out, top, comp.mode), a, out, 1 - a, 0)
    alpha = a * top[..., 3:4].astype(np.float32) / 255  # picture with transparency
    rgb = np.ascontiguousarray(top[..., :3])
    b = out.astype(np.float32)
    if comp.mode == "add":
        res = np.minimum(255, b + rgb.astype(np.float32) * alpha)
    else:
        res = b * (1 - alpha) + _blend(out, rgb, comp.mode).astype(np.float32) * alpha
    return np.floor(res + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------------------- rendering


def _audio_args(comp: Composition, total_frames: int) -> tuple[list[str], list[str]]:
    """Extra ffmpeg inputs and output options for the sound track (empty when silent)."""
    dur = total_frames / FPS
    inputs: list[str] = []
    chains: list[str] = []
    for ly in comp.layers:
        if not ly.sound:
            continue
        j = 1 + len(inputs) // 6  # input 0 is the video pipe; each audio input adds 6 arguments
        clip = clip_frames(ly) / FPS
        inputs += ["-ss", f"{ly.trim_start:.4f}", "-t", f"{clip:.4f}", "-i", ly.media.path]
        chain = (
            f"[{j}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad=whole_dur={clip:.4f},atrim=duration={clip:.4f},asetpts=N/SR/TB"
        )
        if ly.on_end == "loop":  # repeat exactly the trimmed clip, in step with the picture
            chain += f",aloop=loop=-1:size={js_round(clip * SAMPLE_RATE)}"
        chain += f",apad=whole_dur={dur:.4f},atrim=duration={dur:.4f},volume={ly.volume:.3f}[a{j}]"
        chains.append(chain)
    if not chains:
        return [], ["-an"]
    if len(chains) == 1:
        graph = chains[0].rsplit("[", 1)[0] + "[aout]"
    else:
        graph = ";".join(chains) + ";[a1][a2]amix=inputs=2:duration=longest:normalize=0[aout]"
    opts = ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "256k"]
    return inputs, opts


def render_composition(comp: Composition, out_file: str | Path, progress: ProgressFn | None = None) -> Path:
    out_file = Path(out_file)
    total = output_frames(comp)
    sources = [_Source(ly, w, h) for ly, (_, _, w, h) in zip(comp.layers, comp.layer_boxes)]
    audio_inputs, audio_opts = _audio_args(comp, total)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{comp.width}x{comp.height}", "-r", str(FPS), "-i", "-",
        *audio_inputs, *audio_opts,
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS),
        "-t", f"{total / FPS:.4f}", "-movflags", "+faststart", str(out_file),
    ]  # fmt: skip
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    prev_key, prev_bytes = None, b""
    try:
        for k in range(total):
            frames = [s.frame(k) for s in sources]
            key = tuple(id(f) for f in frames)  # pictures and frozen frames are the same objects each time
            if key != prev_key:
                prev_key, prev_bytes = key, composite(comp, frames).tobytes()
            enc.stdin.write(prev_bytes)
            if progress and k % 30 == 0:
                progress(k / total, f"Rendering frame {k}/{total}")
    except BrokenPipeError:
        pass  # the encoder stopped early; its error message is reported below
    finally:
        for s in sources:
            s.close()
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        err = enc.stderr.read().decode(errors="replace")
        code = enc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[-800:]}")
    if progress:
        progress(1.0, "Render done")
    return out_file
