"""A project = one uploaded video plus everything derived from it, stored in its own folder."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .analyze import Analysis
from .camera import CameraPath, plan_camera
from .identity import IdentityResult, Member, box_iou, identify
from .identity import features_for

PALETTE = ["#ff4d6d", "#4dabf7", "#ffd43b", "#69db7c", "#da77f2", "#ff922b", "#38d9a9", "#f783ac", "#a9e34b", "#748ffc"]


@dataclass
class MemberState:
    id: str
    name: str
    color: str
    seeds: list[int] = field(default_factory=list)
    not_dets: list[int] = field(default_factory=list)
    absent_frames: list[int] = field(default_factory=list)

    def to_member(self) -> Member:
        return Member(self.name, list(self.seeds), list(self.not_dets), list(self.absent_frames))


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self._analysis: Analysis | None = None
        self._identity: dict[str, IdentityResult] = {}
        self.state = self._load_state()

    # ---------- storage
    @property
    def id(self) -> str:
        return self.root.name

    @property
    def source(self) -> Path:
        return self.root / self.state["source"]

    @property
    def playback(self) -> Path:
        """What the browser plays: a light H.264 copy when the source is not browser friendly."""
        proxy = self.root / "preview.mp4"
        return proxy if proxy.exists() else self.source

    @property
    def analysis_file(self) -> Path:
        return self.root / "analysis.npz"

    @property
    def renders_dir(self) -> Path:
        d = self.root / "renders"
        d.mkdir(exist_ok=True)
        return d

    def _load_state(self) -> dict:
        f = self.root / "state.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def save(self) -> None:
        with self.lock:
            tmp = self.root / "state.json.tmp"
            tmp.write_text(json.dumps(self.state, indent=1))
            tmp.replace(self.root / "state.json")

    @classmethod
    def create(cls, workspace: Path, filename: str, data_path: Path) -> "Project":
        root = workspace / uuid.uuid4().hex[:10]
        root.mkdir(parents=True)
        suffix = Path(filename).suffix.lower() or ".mp4"
        shutil.move(str(data_path), root / f"source{suffix}")
        p = cls(root)
        p.state = {"name": Path(filename).stem, "source": f"source{suffix}", "status": "queued", "members": []}
        p.save()
        return p

    # ---------- analysis
    @property
    def ready(self) -> bool:
        return self.state.get("status") == "ready" and self.analysis_file.exists()

    def reset_cache(self) -> None:
        """Forget the loaded analysis and all tracking (after the video was analysed again)."""
        with self.lock:
            self._analysis = None
            self._identity.clear()

    def remap_members(self, old: Analysis, new: Analysis) -> None:
        """Point every click at the matching box of a fresh analysis: detection ids change on re-analysis."""

        def remap(d: int) -> int | None:
            f, box = int(old.det_frame[d]), old.det_box[d]
            best, best_iou = None, 0.3
            for g in (f, f - 1, f + 1, f - 2, f + 2):  # boxes may now exist only on every other frame
                sl = new.frame_slice(g) if 0 <= g < new.frame_count else slice(0, 0)
                if sl.stop > sl.start:
                    ious = box_iou(box, new.det_box[sl])
                    k = int(np.argmax(ious))
                    if ious[k] > best_iou:
                        best, best_iou = sl.start + k, float(ious[k])
                if best is not None and g == f:
                    break
            return best

        for m in self.members():
            m.seeds = [x for x in map(remap, m.seeds) if x is not None]
            m.not_dets = [x for x in map(remap, m.not_dets) if x is not None]
            self.put_member(m)

    def analysis(self) -> Analysis:
        with self.lock:
            if self._analysis is None:
                self._analysis = Analysis.load(self.analysis_file)
                self._analysis.info.path = str(self.source)  # the project folder may have been moved
                # Build the appearance features in the background so the first click feels fast.
                threading.Thread(target=features_for, args=(self._analysis,), daemon=True).start()
            return self._analysis

    # ---------- members
    def members(self) -> list[MemberState]:
        return [MemberState(**m) for m in self.state.get("members", [])]

    def member(self, mid: str) -> MemberState:
        for m in self.members():
            if m.id == mid:
                return m
        raise KeyError(mid)

    def put_member(self, m: MemberState, retrack: bool = True) -> None:
        """Save a member; with retrack=False (rename, recolour) the cached tracking is kept."""
        with self.lock:
            ms = [x for x in self.state.get("members", []) if x["id"] != m.id]
            existing = [x["id"] for x in self.state.get("members", [])]
            entry = asdict(m)
            if m.id in existing:
                ms.insert(existing.index(m.id), entry)
            else:
                ms.append(entry)
            self.state["members"] = ms
            self.save()
            if not retrack:
                return
            self._identity.pop(m.id, None)
            # Another member whose path used one of these boxes must be re-solved too (a box is one person).
            for other, res in list(self._identity.items()):
                if np.isin(res.det, m.seeds).any():
                    self._identity.pop(other, None)

    def new_member(self, name: str) -> MemberState:
        used = {m.color for m in self.members()}
        color = next((c for c in PALETTE if c not in used), PALETTE[len(used) % len(PALETTE)])
        m = MemberState(id=uuid.uuid4().hex[:6], name=name, color=color)
        self.put_member(m)
        return m

    def delete_member(self, mid: str) -> None:
        with self.lock:
            self.state["members"] = [x for x in self.state.get("members", []) if x["id"] != mid]
            self.save()
            self._identity.clear()  # its seeds no longer count as negatives for the others

    def identity(self, mid: str) -> IdentityResult:
        with self.lock:
            if mid not in self._identity:
                a = self.analysis()
                m = self.member(mid)
                others = [o.to_member() for o in self.members() if o.id != mid]
                self._identity[mid] = identify(a, m.to_member(), others)
            return self._identity[mid]

    def camera(self, mid: str, fmt: str, framing: str, smoothness: str, size: float = 1.0) -> CameraPath:
        return plan_camera(self.analysis(), self.identity(mid), fmt, framing, smoothness, size)

    # ---------- JSON views for the web UI
    def detections_json(self) -> dict:
        a = self.analysis()
        return {
            "offsets": a.frame_offsets.tolist(),
            "boxes": np.round(a.det_box).astype(int).ravel().tolist(),
            "tids": a.det_tid.tolist(),
        }

    def summary(self) -> dict:
        s = dict(self.state)
        s["id"] = self.id
        if self.ready:
            a = self.analysis()
            s["video"] = {
                "width": a.info.width,
                "height": a.info.height,
                "fps": a.info.fps,
                "frames": a.frame_count,
                "duration": a.frame_count / a.info.fps,
                "cuts": a.cuts.tolist(),
            }
        s["renders"] = sorted(
            [{"file": f.name, "size": f.stat().st_size} for f in self.renders_dir.glob("*.mp4")],
            key=lambda r: r["file"],
            reverse=True,
        )
        return s
