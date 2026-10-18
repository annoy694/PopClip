"""Pass 1: find every person in every frame.

For each frame we store the person boxes, a short-term track id (ByteTrack), and two appearance
descriptors per box (deep features + clothing colour histogram) that the identity step uses to
re-find the same member after crossings, occlusions and camera cuts. Shot cuts are detected here too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .video import VideoInfo, probe

ProgressFn = Callable[[float, str], None]

DET_BATCH = 8
CUT_THRESHOLD = 0.42  # Bhattacharyya distance between consecutive frame histograms


@dataclass
class Analysis:
    info: VideoInfo
    frame_count: int
    det_frame: np.ndarray  # (D,) int32 frame index of each detection
    det_box: np.ndarray  # (D, 4) float32 x1 y1 x2 y2 in source pixels
    det_conf: np.ndarray  # (D,) float32
    det_tid: np.ndarray  # (D,) int32 short-term track id, -1 if unassigned
    det_deep: np.ndarray  # (D, F) float16 L2-normalised deep appearance feature
    det_color: np.ndarray  # (D, C) float16 L2-normalised colour histogram
    cuts: np.ndarray  # (K,) int32 frame indices that start a new shot (excluding 0)
    frame_offsets: np.ndarray  # (N+1,) int64; detections of frame f are [off[f], off[f+1])

    def frame_slice(self, f: int) -> slice:
        return slice(int(self.frame_offsets[f]), int(self.frame_offsets[f + 1]))

    def shots(self) -> list[tuple[int, int]]:
        """Shot ranges as [start, end) frame indices."""
        bounds = [0, *self.cuts.tolist(), self.frame_count]
        return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            info=np.array([self.info.to_dict()], dtype=object),
            frame_count=self.frame_count,
            det_frame=self.det_frame,
            det_box=self.det_box,
            det_conf=self.det_conf,
            det_tid=self.det_tid,
            det_deep=self.det_deep,
            det_color=self.det_color,
            cuts=self.cuts,
            frame_offsets=self.frame_offsets,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Analysis":
        z = np.load(path, allow_pickle=True)
        return cls(
            info=VideoInfo(**z["info"][0]),
            frame_count=int(z["frame_count"]),
            det_frame=z["det_frame"],
            det_box=z["det_box"],
            det_conf=z["det_conf"],
            det_tid=z["det_tid"],
            det_deep=z["det_deep"],
            det_color=z["det_color"],
            cuts=z["cuts"],
            frame_offsets=z["frame_offsets"],
        )


class AppearanceEncoder:
    """Per-person appearance descriptors.

    Deep part: a small YOLO classification backbone run on the person crop, pooled separately over the
    head / torso / legs stripes so layout survives. Colour part: HSV histograms of the same stripes,
    which is very discriminative for stage outfits.
    """

    CROP_H, CROP_W = 256, 128
    DIM = 1024  # stored size of the deep feature (fixed random projection keeps files small)

    def __init__(self, weights: str, device: str | None = None):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = YOLO(weights).model.float().eval().to(self.device)
        self.layers = list(net.model[:-1])  # backbone; the Classify head is dropped
        self.head_conv = net.model[-1].conv  # 1x1 conv that widens features before pooling

    def deep(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 0), np.float32)
        torch = self.torch
        batch = np.stack([cv2.resize(c, (self.CROP_W, self.CROP_H), interpolation=cv2.INTER_AREA) for c in crops])
        x = torch.from_numpy(batch[..., ::-1].copy()).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode():
            for layer in self.layers:
                x = layer(x)
            x = self.head_conv(x)  # (B, C, 8, 4)
            h = x.shape[2]
            stripes = [x[:, :, : h * 3 // 8], x[:, :, h * 3 // 8 : h * 5 // 8], x[:, :, h * 5 // 8 :]]
            feats = [torch.nn.functional.normalize(s.mean(dim=(2, 3)), dim=1) for s in stripes]
            out = torch.nn.functional.normalize(torch.cat(feats, dim=1), dim=1).cpu().numpy()
        return project(out, self.DIM)

    @staticmethod
    def color(crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            hsv = cv2.cvtColor(cv2.resize(c, (48, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
            inner = hsv[:, 10:38]  # drop the sides of the box, mostly background
            parts = [inner[0:22], inner[18:54], inner[50:96]]  # hair/face, torso, legs
            hist = []
            for p in parts:
                h = cv2.calcHist([p], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
                hist.append(np.sqrt(h / max(h.sum(), 1.0)))
            v = np.concatenate(hist)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.array(out, np.float32).reshape(len(crops), -1)


_PROJ: dict[tuple[int, int], np.ndarray] = {}


def project(x: np.ndarray, dim: int) -> np.ndarray:
    """Fixed Gaussian random projection (Johnson-Lindenstrauss): preserves similarities, 4x less storage."""
    if x.shape[1] <= dim:
        return x
    key = (x.shape[1], dim)
    if key not in _PROJ:
        _PROJ[key] = np.random.default_rng(1234).standard_normal(key).astype(np.float32) / np.sqrt(dim)
    y = x.astype(np.float32) @ _PROJ[key]
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)


def _frame_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def _crops(frame: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
    H, W = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(np.ceil(x2)), W), min(int(np.ceil(y2)), H)
        out.append(frame[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)])
    return out


def _make_tracker(fps: float):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    cfg.update(track_high_thresh=0.35, track_low_thresh=0.1, new_track_thresh=0.4, track_buffer=int(fps * 2))
    try:
        return BYTETracker(IterableSimpleNamespace(**cfg))
    except TypeError:  # ultralytics < 8.4 also wants the frame rate
        return BYTETracker(IterableSimpleNamespace(**cfg), frame_rate=30)


def analyze(
    video_path: str,
    out_path: str | Path | None = None,
    det_weights: str = "yolo26s.pt",
    reid_weights: str = "yolo11n-cls.pt",
    conf: float = 0.3,
    imgsz: int = 640,
    device: str | None = None,
    progress: ProgressFn | None = None,
) -> Analysis:
    from ultralytics import YOLO

    if progress:
        progress(0.0, "Loading the person detector")
    info = probe(video_path)
    detector = YOLO(det_weights)
    encoder = AppearanceEncoder(reid_weights, device)
    tracker = _make_tracker(info.fps)

    det_frame, det_box, det_conf, det_tid, det_deep, det_color = [], [], [], [], [], []
    cuts: list[int] = []
    prev_hist = None
    id_offset = max_tid = 0  # the tracker restarts ids at 1 after a reset; keep them unique across shots
    n_frames = 0
    t0 = time.time()

    cap = cv2.VideoCapture(video_path)
    total = max(info.frame_count, 1)
    done = False
    while not done:
        frames: list[np.ndarray] = []
        while len(frames) < DET_BATCH:
            ok, frame = cap.read()
            if not ok:
                done = True
                break
            frames.append(frame)
        if not frames:
            break
        results = detector.predict(frames, classes=[0], conf=0.1, imgsz=imgsz, device=device, verbose=False)
        for frame, res in zip(frames, results):
            f = n_frames
            n_frames += 1
            hist = _frame_hist(frame)
            if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_THRESHOLD:
                cuts.append(f)
                tracker.reset()
                id_offset = max_tid
            prev_hist = hist

            boxes = res.boxes.cpu().numpy()
            tids = np.full(len(boxes), -1, np.int32)
            if len(boxes):
                for row in tracker.update(boxes, frame):
                    tids[int(row[-1])] = int(row[4]) + id_offset
                max_tid = max(max_tid, int(tids.max()))
            else:
                tracker.update(boxes, frame)
            # Keep confident boxes plus any low-confidence box the tracker managed to associate.
            keep = (boxes.conf >= conf) | (tids >= 0)
            xyxy = boxes.xyxy[keep].astype(np.float32)
            crops = _crops(frame, xyxy)
            if crops:
                det_frame.append(np.full(len(xyxy), f, np.int32))
                det_box.append(xyxy)
                det_conf.append(boxes.conf[keep].astype(np.float32))
                det_tid.append(tids[keep])
                det_deep.append(encoder.deep(crops).astype(np.float16))
                det_color.append(encoder.color(crops).astype(np.float16))
        if progress:
            elapsed = time.time() - t0
            frac = min(n_frames / total, 1.0)
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            progress(frac, f"Tracking people: frame {n_frames}/{total} (about {int(eta)}s left)")
    cap.release()

    det_frame_a = np.concatenate(det_frame) if det_frame else np.zeros(0, np.int32)
    counts = np.bincount(det_frame_a, minlength=n_frames)
    analysis = Analysis(
        info=info,
        frame_count=n_frames,
        det_frame=det_frame_a,
        det_box=np.concatenate(det_box) if det_box else np.zeros((0, 4), np.float32),
        det_conf=np.concatenate(det_conf) if det_conf else np.zeros(0, np.float32),
        det_tid=np.concatenate(det_tid) if det_tid else np.zeros(0, np.int32),
        det_deep=np.concatenate(det_deep) if det_deep else np.zeros((0, 1), np.float16),
        det_color=np.concatenate(det_color) if det_color else np.zeros((0, 1), np.float16),
        cuts=np.array(cuts, np.int32),
        frame_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
    )
    if out_path:
        analysis.save(out_path)
    if progress:
        progress(1.0, "Tracking done")
    return analysis
