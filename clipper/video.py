"""Small helpers for reading video metadata and locating ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass

import cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool
    codec: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ffmpeg_exe() -> str:
    """Return a usable ffmpeg binary: system ffmpeg first, then the one bundled by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from e


BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}


def _codec(path: str, cap: cv2.VideoCapture) -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
        )
        if out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    tag = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).lower()
    return {"avc1": "h264", "h264": "h264", "hev1": "hevc", "hvc1": "hevc", "vp09": "vp9", "av01": "av1"}.get(tag, tag)


def needs_browser_proxy(info: VideoInfo) -> bool:
    """Browsers can't play HEVC & co, and 4K sources make scrubbing sluggish."""
    return info.codec not in BROWSER_CODECS or info.height > 1440


def make_browser_proxy(src: str, dst: str) -> None:
    """720p H.264 copy for the web player, keeping every frame's timestamp so overlays stay in sync."""
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-vf", "scale=-2:'min(720,ih)'", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst,
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"Could not make a browser preview: {out.stderr.strip()[-500:]}")


def _has_audio(path: str) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", path],
            capture_output=True,
            text=True,
        )
        try:
            return bool(json.loads(out.stdout or "{}").get("streams"))
        except json.JSONDecodeError:
            pass
    # Fallback: ask ffmpeg to describe the file and look for an audio stream line.
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True)
    return "Audio:" in out.stderr


def audio_codec(path: str) -> str:
    """Codec name of the first audio stream ("" if none or unknown)."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True)
    for line in out.stderr.splitlines():
        if "Audio:" in line:
            return line.split("Audio:")[1].split()[0].strip(",")
    return ""


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = _codec(path, cap)
    cap.release()
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=float(fps),
        frame_count=count,
        duration=count / fps if fps else 0.0,
        has_audio=_has_audio(path),
        codec=codec,
    )
