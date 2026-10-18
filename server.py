"""PopClip web app: K-pop fancam clipper.

    python server.py            # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clipper.analyze import Analysis, analyze
from clipper.compose import BLEND_MODES, FADES, ON_END, Composition, Layer, output_frames, render_composition
from clipper.media import MediaStore
from clipper.camera import FORMATS, FRAMINGS, SIZE_RANGE, SMOOTHNESS, target_boxes
from clipper.project import MemberState, Project
from clipper.render import render
from clipper.video import make_browser_proxy, needs_browser_proxy

VERSION = "2.2"  # shown in the page header and the start-up message
ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("CLIPPER_WORKSPACE", ROOT / "workspace"))
MODELS = ROOT / "models"
DET_WEIGHTS = os.environ.get("CLIPPER_DETECTOR", str(MODELS / "yolo26s-pose.pt"))
REID_WEIGHTS = os.environ.get("CLIPPER_REID", str(MODELS / "yolo11n-cls.pt"))

app = FastAPI(title="PopClip")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.middleware("http")
async def always_fresh_ui(request: Request, call_next):
    """Make the browser re-check the page and its scripts, so an update shows up without a hard refresh."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

_projects: dict[str, Project] = {}
_projects_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_gpu_lock = threading.Lock()  # one heavy job (analysis or render) at a time


def get_project(pid: str) -> Project:
    if not re.fullmatch(r"[0-9a-f]{10}", pid):
        raise HTTPException(404, "No such project")
    with _projects_lock:
        if pid not in _projects:
            root = WORKSPACE / pid
            if not (root / "state.json").exists():
                raise HTTPException(404, "No such project")
            _projects[pid] = Project(root)
        return _projects[pid]


def start_job(kind: str, owner: str, fn, exclusive: bool = True) -> str:
    """Run fn(progress) in the background. Exclusive jobs (analysis, renders) run one at a time."""
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "kind": kind, "project": owner, "status": "queued", "progress": 0.0, "message": "Starting"}
    _jobs[jid] = job

    def progress(frac: float, msg: str) -> None:
        job["progress"], job["message"] = round(float(frac), 4), msg

    def work() -> None:
        job["status"] = "running"
        try:
            job["result"] = fn(progress)
            job["status"] = "done"
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            job["status"], job["message"] = "error", f"{type(e).__name__}: {e}"

    def run() -> None:
        if not exclusive:
            return work()
        if _gpu_lock.locked():
            job["message"] = "Waiting for another video to finish"
        with _gpu_lock:
            work()

    threading.Thread(target=run, daemon=True).start()
    return jid


def run_analysis(p: Project) -> str:
    def fn(progress):
        p.state["status"] = "analyzing"
        p.save()
        old = Analysis.load(p.analysis_file) if p.analysis_file.exists() else None
        try:
            a = analyze(str(p.source), p.analysis_file, DET_WEIGHTS, REID_WEIGHTS, progress=progress)
            if needs_browser_proxy(a.info) and not (p.root / "preview.mp4").exists():
                progress(1.0, "Making a browser-friendly preview copy")
                make_browser_proxy(str(p.source), str(p.root / "preview.mp4"))
        except Exception:
            p.state["status"] = "failed"
            p.save()
            raise
        p.reset_cache()
        if old is not None:  # re-analysis: keep the user's clicks
            p.remap_members(old, a)
        p.state["status"] = "ready"
        p.save()

    jid = start_job("analyze", p.id, fn)
    p.state["job"] = jid
    p.save()
    return jid


# ---------------------------------------------------------------- pages


@app.get("/")
def index():
    # Give every script and stylesheet a ?v=<last change> stamp: after an update the browser sees new
    # addresses and cannot keep running a cached older copy of the app.
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    def stamp(m: re.Match) -> str:
        f = ROOT / "web" / m.group(1)
        return f"/static/{m.group(1)}?v={int(f.stat().st_mtime) if f.exists() else VERSION}"

    html = re.sub(r"/static/([\w./-]+\.(?:js|css))", stamp, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- projects


@app.get("/api/projects")
def list_projects():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(WORKSPACE.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if (d / "state.json").exists():
            p = get_project(d.name)
            out.append({"id": p.id, "name": p.state.get("name"), "status": p.state.get("status")})
    return out


@app.post("/api/projects")
async def upload(file: UploadFile):
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=WORKSPACE, suffix=".upload")
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    p = Project.create(WORKSPACE, file.filename or "video.mp4", Path(tmp))
    with _projects_lock:
        _projects[p.id] = p
    jid = run_analysis(p)
    return {"id": p.id, "job": jid}


@app.get("/api/projects/{pid}")
def project_summary(pid: str):
    p = get_project(pid)
    s = p.summary()
    job = _jobs.get(p.state.get("job", ""))
    if p.state.get("status") in ("queued", "analyzing"):
        if job:
            s["job"] = job
        else:  # server restarted mid-analysis: start again
            s["job"] = _jobs[run_analysis(p)]
    return s


@app.post("/api/projects/{pid}/reanalyze")
def reanalyze(pid: str):
    p = get_project(pid)
    return {"job": run_analysis(p)}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    p = get_project(pid)
    with _projects_lock:
        _projects.pop(pid, None)
    shutil.rmtree(p.root, ignore_errors=True)
    return {"ok": True}


@app.get("/api/projects/{pid}/video")
def video(pid: str):
    p = get_project(pid)
    media = "video/webm" if p.playback.suffix == ".webm" else "video/mp4"
    return FileResponse(p.playback, media_type=media)


def require_ready(p: Project) -> None:
    if not p.ready:
        raise HTTPException(409, "Video is still being analyzed")


@app.get("/api/projects/{pid}/detections")
def detections(pid: str):
    p = get_project(pid)
    require_ready(p)
    return p.detections_json()


# ---------------------------------------------------------------- members


class NewMember(BaseModel):
    name: str
    det: int | None = None


class MemberPatch(BaseModel):
    name: str | None = None
    color: str | None = None


class Fix(BaseModel):
    action: str  # "is" | "is_not" | "absent" | "remove" | "reset"
    det: int | None = None
    frame: int | None = None
    kind: str | None = None  # for "remove": "seed" | "not" | "absent"


def track_json(p: Project, mid: str) -> dict:
    t0 = time.time()
    r = p.identity(mid)
    boxes = target_boxes(p.analysis(), r)  # short gaps filled in, so the on-screen box does not flicker
    return {
        "member": mid,
        "det": r.det.tolist(),
        "score": [round(float(x), 1) for x in r.score],
        "boxes": [None if np.isnan(b[0]) else [round(float(v)) for v in b] for b in boxes],
        "seconds": round(time.time() - t0, 2),
    }


def _assign_seed(p: Project, m: MemberState, det: int) -> None:
    a = p.analysis()
    frame = int(a.det_frame[det])
    # One answer per frame for this member, and a box can only be one member.
    m.seeds = [d for d in m.seeds if int(a.det_frame[d]) != frame] + [det]
    m.not_dets = [d for d in m.not_dets if d != det]
    m.absent_frames = [f for f in m.absent_frames if f != frame]
    for o in p.members():
        if o.id != m.id and det in o.seeds:
            o.seeds.remove(det)
            p.put_member(o)


@app.post("/api/projects/{pid}/members")
def add_member(pid: str, body: NewMember):
    p = get_project(pid)
    require_ready(p)
    m = p.new_member(body.name.strip() or f"Member {len(p.members()) + 1}")
    if body.det is not None:
        _assign_seed(p, m, body.det)
        p.put_member(m)
    return {"member": m.__dict__, "track": track_json(p, m.id) if m.seeds else None}


@app.patch("/api/projects/{pid}/members/{mid}")
def patch_member(pid: str, mid: str, body: MemberPatch):
    p = get_project(pid)
    m = p.member(mid)
    if body.name is not None:
        m.name = body.name.strip() or m.name
    if body.color is not None:
        m.color = body.color
    p.put_member(m, retrack=False)
    return {"member": m.__dict__}


@app.delete("/api/projects/{pid}/members/{mid}")
def delete_member(pid: str, mid: str):
    p = get_project(pid)
    p.delete_member(mid)
    return {"ok": True}


@app.post("/api/projects/{pid}/members/{mid}/fix")
def fix_member(pid: str, mid: str, body: Fix):
    p = get_project(pid)
    require_ready(p)
    m = p.member(mid)
    a = p.analysis()
    if body.action == "is" and body.det is not None:
        _assign_seed(p, m, body.det)
    elif body.action == "is_not" and body.det is not None:
        if body.det in m.seeds and len(m.seeds) > 1:
            m.seeds.remove(body.det)
        if body.det not in m.seeds:
            m.not_dets.append(body.det)
    elif body.action == "absent" and body.frame is not None:
        m.seeds = [d for d in m.seeds if int(a.det_frame[d]) != body.frame] or m.seeds
        if body.frame not in m.absent_frames:
            m.absent_frames.append(body.frame)
    elif body.action == "remove":
        if body.kind == "seed" and body.det in m.seeds and len(m.seeds) > 1:
            m.seeds.remove(body.det)
        elif body.kind == "not" and body.det in m.not_dets:
            m.not_dets.remove(body.det)
        elif body.kind == "absent" and body.frame in m.absent_frames:
            m.absent_frames.remove(body.frame)
    elif body.action == "reset":
        m.seeds, m.not_dets, m.absent_frames = m.seeds[:1], [], []
    else:
        raise HTTPException(400, "Unknown fix")
    p.put_member(m)
    return {"member": m.__dict__, "track": track_json(p, mid)}


@app.get("/api/projects/{pid}/members/{mid}/track")
def get_track(pid: str, mid: str):
    p = get_project(pid)
    require_ready(p)
    try:
        return track_json(p, mid)
    except KeyError:
        raise HTTPException(404, "No such member") from None


# ---------------------------------------------------------------- camera + render


def _check_opts(fmt: str, framing: str, smoothness: str, size: float) -> None:
    if fmt not in FORMATS or framing not in FRAMINGS or smoothness not in SMOOTHNESS:
        raise HTTPException(400, "Bad output options")
    if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise HTTPException(400, f"Box size must be between {SIZE_RANGE[0]} and {SIZE_RANGE[1]}")


@app.get("/api/projects/{pid}/camera")
def camera(
    pid: str,
    member: str,
    format: str = "vertical",
    framing: str = "full",
    smoothness: str = "medium",
    size: float = 1.0,
):
    p = get_project(pid)
    require_ready(p)
    _check_opts(format, framing, smoothness, size)
    t0 = time.time()
    cam = p.camera(member, format, framing, smoothness, size)
    seen = target_boxes(p.analysis(), p.identity(member))[:, 0]
    crop_h = cam.h[np.isfinite(seen)]
    return {
        "aspect": cam.aspect,
        "rects": [round(float(v), 1) for v in cam.rects().ravel()],
        # How much the source gets enlarged; high values mean a soft-looking clip.
        "upscale": round(FORMATS[format][1] / float(np.median(crop_h)), 1) if len(crop_h) else None,
        "has_pose": p.analysis().det_kpts is not None,
        "seconds": round(time.time() - t0, 2),
    }


class RenderRequest(BaseModel):
    member: str
    format: str = "vertical"
    framing: str = "full"
    smoothness: str = "medium"
    size: float = 1.0
    start: int = 0
    end: int | None = None


@app.post("/api/projects/{pid}/render")
def start_render(pid: str, body: RenderRequest):
    p = get_project(pid)
    require_ready(p)
    _check_opts(body.format, body.framing, body.smoothness, body.size)
    m = p.member(body.member)
    a = p.analysis()
    start = max(0, body.start)
    end = min(a.frame_count, body.end if body.end else a.frame_count)
    if end - start < 2:
        raise HTTPException(400, "Clip is too short")
    safe = re.sub(r"[^\w\-]+", "_", f"{p.state.get('name', 'clip')}_{m.name}_{body.format}").strip("_")
    out = p.renders_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.mp4"

    def fn(progress):
        progress(0.0, "Planning camera")
        cam = p.camera(body.member, body.format, body.framing, body.smoothness, body.size)
        render(a.info, cam, FORMATS[body.format], out, start, end, progress=progress)
        return {"file": out.name}

    return {"job": start_job("render", p.id, fn)}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in _jobs:
        raise HTTPException(404, "No such job")
    return _jobs[jid]


@app.get("/api/projects/{pid}/renders/{name}")
def get_render(pid: str, name: str, request: Request):
    p = get_project(pid)
    f = (p.renders_dir / name).resolve()
    if f.parent != p.renders_dir.resolve() or not f.exists():
        raise HTTPException(404, "No such file")
    download = request.query_params.get("download") == "1"
    return FileResponse(f, media_type="video/mp4", filename=name if download else None)


# ---------------------------------------------------------------- studio: Split Screen and Overlay

MEDIA = MediaStore(WORKSPACE / "media")
STUDIO_RENDERS = WORKSPACE / "studio"


async def _save_upload(file: UploadFile) -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=WORKSPACE, suffix=".upload")
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return Path(tmp)


def _media_json(info) -> dict:
    d = info.public()
    d["preview_ready"] = not MEDIA.needs_proxy(info) or info.playback != info.path
    d["url"] = f"/api/media/{info.id}/file"
    return d


@app.post("/api/media")
async def upload_media(file: UploadFile):
    tmp = await _save_upload(file)
    try:
        info = MEDIA.add(tmp, file.filename or "upload")
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e)) from None
    if MEDIA.needs_proxy(info):  # e.g. HEVC phone videos: make a copy the browser can preview

        def fn(progress):
            progress(0.0, "Preparing a preview copy")
            dst = str(Path(info.path).with_name("preview.mp4"))
            make_browser_proxy(info.path, dst)
            MEDIA.set_playback(info, dst)

        start_job("preview", info.id, fn, exclusive=False)
    return _media_json(info)


@app.get("/api/media/{mid}")
def media_info(mid: str):
    return _media_json(MEDIA.get(mid))


@app.get("/api/media/{mid}/file")
def media_file(mid: str):
    info = MEDIA.get(mid)
    return FileResponse(info.playback)


@app.delete("/api/media/{mid}")
def delete_media(mid: str):
    MEDIA.delete(mid)
    return {"ok": True}


class LayerIn(BaseModel):
    media: str
    pos_x: float = 0.5
    pos_y: float = 0.5
    zoom: float = 1.0
    trim_start: float = 0.0
    trim_end: float = 0.0
    on_end: str = "freeze"
    volume: float = 1.0


class ComposeIn(BaseModel):
    layout: str
    width: int = 1920
    height: int = 1080
    layers: list[LayerIn]
    auto_duration: bool = True
    duration: float = 10.0
    blend: float = 240.0
    fade: str = "both"
    opacity: float = 0.5
    mode: str = "normal"


def _composition(body: ComposeIn) -> Composition:
    try:
        layers = [
            Layer(
                MEDIA.get(ly.media),
                ly.pos_x,
                ly.pos_y,
                ly.zoom,
                ly.trim_start,
                ly.trim_end,
                ly.on_end,
                float(np.clip(ly.volume, 0, 1.5)),
            )
            for ly in body.layers
        ]
        return Composition(
            body.layout,
            body.width,
            body.height,
            layers,
            body.auto_duration,
            body.duration,
            body.blend,
            body.fade,
            body.opacity,
            body.mode,
        )
    except KeyError:
        raise HTTPException(404, "An uploaded file is missing; please upload it again.") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.get("/api/compose/options")
def compose_options():
    return {"blend_modes": BLEND_MODES, "fades": FADES, "on_end": ON_END}


@app.post("/api/compose/render")
def compose_render(body: ComposeIn):
    comp = _composition(body)
    STUDIO_RENDERS.mkdir(parents=True, exist_ok=True)
    out = STUDIO_RENDERS / f"{body.layout}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    seconds = output_frames(comp) / 60

    def fn(progress):
        render_composition(comp, out, progress=progress)
        return {"file": out.name, "url": f"/api/compose/renders/{out.name}", "seconds": seconds}

    return {"job": start_job("compose", body.layout, fn), "seconds": seconds}


@app.get("/api/compose/renders/{name}")
def compose_file(name: str, request: Request):
    f = (STUDIO_RENDERS / name).resolve()
    if f.parent != STUDIO_RENDERS.resolve() or not f.exists():
        raise HTTPException(404, "No such file")
    download = request.query_params.get("download") == "1"
    return FileResponse(f, media_type="video/mp4", filename=name if download else None)


@app.exception_handler(KeyError)
def key_error(_: Request, exc: KeyError):
    return JSONResponse({"detail": f"Not found: {exc}"}, status_code=404)


def _free_port(host: str, first: int) -> int:
    """First port from `first` that nothing is listening on (an older PopClip may still hold 8000)."""
    import socket

    for port in range(first, first + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    return first


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="PopClip web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the app in the browser once it is running")
    args = ap.parse_args()
    port = _free_port(args.host, args.port)
    if port != args.port:
        print(f"\n  Port {args.port} is busy (probably another PopClip window), using {port} instead.")
    args.port = port
    if args.open:
        import webbrowser

        threading.Timer(2.0, webbrowser.open, [f"http://{args.host}:{args.port}"]).start()
    print(f"\n  PopClip {VERSION}: Fancam Clipper, Split Screen, Overlay, Caption Maker")
    print(f"  Running from {ROOT}")
    print(f"  Open http://{args.host}:{args.port} in your browser (keep this window open)\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
