"""Uploaded pictures and videos for the Split Screen and Overlay tools: probing and storage."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .video import BROWSER_CODECS, ffmpeg_exe

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


@dataclass
class MediaInfo:
    id: str
    name: str
    kind: str  # "image" | "video"
    width: int  # display size, i.e. after rotation metadata / EXIF orientation
    height: int
    duration: float = 0.0  # seconds, 0 for pictures
    fps: float = 0.0
    has_audio: bool = False  # the file really contains a sound track
    codec: str = ""
    has_alpha: bool = False  # pictures with transparency
    path: str = ""
    playback: str = ""  # file the browser plays (a light H.264 copy when the original is not browser friendly)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        d.pop("playback")
        return d


def _probe_video_ffprobe(path: str) -> dict | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path], capture_output=True, text=True
    )
    if out.returncode != 0:
        return None
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        return None
    rotation = 0.0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = float(sd["rotation"])
    if "rotate" in v.get("tags", {}):
        rotation = float(v["tags"]["rotate"])
    num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "rotation": rotation,
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration") or v.get("duration") or 0),
        "codec": v.get("codec_name", ""),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def _probe_video_ffmpeg(path: str) -> dict | None:
    """Same facts from `ffmpeg -i` output, for installs without ffprobe (the bundled ffmpeg has none)."""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    video = next((ln for ln in out.splitlines() if re.search(r"Stream #.*: Video:", ln)), None)
    if video is None:
        return None
    size = re.search(r", (\d{2,5})x(\d{2,5})[ ,]", video)
    fps = re.search(r", ([\d.]+) fps", video) or re.search(r", ([\d.]+) tbr", video)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    rot = re.search(r"rotation of (-?[\d.]+) degrees", out) or re.search(r"rotate\s*:\s*(-?[\d.]+)", out)
    return {
        "width": int(size.group(1)) if size else 0,
        "height": int(size.group(2)) if size else 0,
        "rotation": float(rot.group(1)) if rot else 0.0,
        "fps": float(fps.group(1)) if fps else 0.0,
        "duration": int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0.0,
        "codec": video.split("Video:")[1].split()[0].strip(","),
        "has_audio": bool(re.search(r"Stream #.*: Audio:", out)),
    }


def load_picture(path: str) -> Image.Image:
    """Open a picture the way browsers show it: EXIF orientation applied, first frame of animations."""
    img = Image.open(path)
    img.seek(0)
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    return img.convert("RGBA" if has_alpha else "RGB")


def probe_media(path: str, name: str, media_id: str) -> MediaInfo:
    if Path(name).suffix.lower() in IMAGE_EXTS:
        img = load_picture(path)
        return MediaInfo(
            media_id, name, "image", img.width, img.height, has_alpha=img.mode == "RGBA", path=path, playback=path
        )
    info = _probe_video_ffprobe(path) or _probe_video_ffmpeg(path)
    if info is None or not info["width"]:
        raise ValueError("This file does not look like a picture or a video the app can read.")
    w, h = info["width"], info["height"]
    if round(abs(info["rotation"])) % 180 == 90:  # phone videos filmed upright
        w, h = h, w
    return MediaInfo(
        media_id,
        name,
        "video",
        w,
        h,
        duration=info["duration"],
        fps=info["fps"] or 30.0,
        has_audio=info["has_audio"],
        codec=info["codec"],
        path=path,
        playback=path,
    )


class MediaStore:
    """Uploaded files live in workspace/media/<id>/ with an info.json beside them."""

    def __init__(self, root: Path):
        self.root = root

    def add(self, upload_path: Path, name: str) -> MediaInfo:
        media_id = uuid.uuid4().hex[:12]
        folder = self.root / media_id
        folder.mkdir(parents=True)
        suffix = Path(name).suffix.lower() or ".bin"
        dest = folder / f"original{suffix}"
        shutil.move(str(upload_path), dest)
        try:
            info = probe_media(str(dest), name, media_id)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        self._save(info)
        return info

    def get(self, media_id: str) -> MediaInfo:
        if not re.fullmatch(r"[0-9a-f]{12}", media_id):
            raise KeyError(media_id)
        f = self.root / media_id / "info.json"
        if not f.exists():
            raise KeyError(media_id)
        return MediaInfo(**json.loads(f.read_text()))

    def delete(self, media_id: str) -> None:
        self.get(media_id)
        shutil.rmtree(self.root / media_id, ignore_errors=True)

    def needs_proxy(self, info: MediaInfo) -> bool:
        return info.kind == "video" and (info.codec not in BROWSER_CODECS or max(info.width, info.height) > 2560)

    def set_playback(self, info: MediaInfo, path: str) -> None:
        info.playback = path
        self._save(info)

    def _save(self, info: MediaInfo) -> None:
        (self.root / info.id / "info.json").write_text(json.dumps(asdict(info)))
