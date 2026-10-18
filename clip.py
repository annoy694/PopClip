"""Command-line fancam clipper.

  # 1) see who is where at 0:12 (writes people_12.0s.jpg with numbered boxes)
  python clip.py stage.mp4 --show 12

  # 2) follow person #3 (counted left to right at 0:12) and render a vertical fancam
  python clip.py stage.mp4 --at 12 --person 3 -o fancam.mp4

  # options: --format vertical|horizontal  --framing full|knees|waist  --smoothness low|medium|high
  #          --start 30 --end 95 (seconds)   --fix 48:2 (at 0:48 it is person #2; repeatable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from clipper.analyze import Analysis, analyze
from clipper.camera import FORMATS, FRAMINGS, SMOOTHNESS, plan_camera
from clipper.identity import Member, identify
from clipper.render import render

MODELS = Path(__file__).resolve().parent / "models"


def progress_printer(prefix: str):
    last = [-1]

    def fn(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r{prefix} {pct:3d}%  {msg[:70]:<70}", end="", file=sys.stderr, flush=True)
        if frac >= 1:
            print(file=sys.stderr)

    return fn


def load_or_analyze(video: str) -> Analysis:
    cache = Path(video).with_suffix(Path(video).suffix + ".clipper.npz")
    if cache.exists() and cache.stat().st_mtime >= Path(video).stat().st_mtime:
        a = Analysis.load(cache)
        a.info.path = str(video)
        return a
    return analyze(
        video, cache, str(MODELS / "yolo26s.pt"), str(MODELS / "yolo11n-cls.pt"), progress=progress_printer("analyze")
    )


def people_at(a: Analysis, t: float) -> tuple[int, list[int]]:
    """Frame index and its detections sorted left to right."""
    f = int(np.clip(round(t * a.info.fps), 0, a.frame_count - 1))
    dets = list(range(*a.frame_slice(f).indices(len(a.det_box))))
    dets.sort(key=lambda d: a.det_box[d, 0] + a.det_box[d, 2])
    return f, dets


def show(a: Analysis, t: float) -> Path:
    f, dets = people_at(a, t)
    cap = cv2.VideoCapture(a.info.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame at {t}s")
    thick = max(2, img.shape[0] // 300)
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = a.det_box[d].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 255), thick)
        cv2.putText(img, str(i), (x1 + 4, y1 + 12 * thick), cv2.FONT_HERSHEY_SIMPLEX, thick / 2, (80, 80, 255), thick)
    out = Path(f"people_{t:.1f}s.jpg")
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--show", type=float, metavar="SEC", help="save a picture of everyone at this time and exit")
    ap.add_argument("--at", type=float, metavar="SEC", help="time where you point out the member")
    ap.add_argument("--person", type=int, metavar="N", help="the member is the N-th person from the left at --at")
    ap.add_argument("--fix", action="append", default=[], metavar="SEC:N", help="extra hint: at SEC she is person N")
    ap.add_argument("--format", choices=FORMATS, default="vertical")
    ap.add_argument("--framing", choices=FRAMINGS, default="full")
    ap.add_argument("--smoothness", choices=SMOOTHNESS, default="medium")
    ap.add_argument("--start", type=float, default=0.0, metavar="SEC")
    ap.add_argument("--end", type=float, default=None, metavar="SEC")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    a = load_or_analyze(args.video)
    if args.show is not None:
        print(f"Saved {show(a, args.show)}")
        return
    if args.at is None or args.person is None:
        ap.error("pick the member with --at SEC --person N (use --show SEC to see the numbers)")

    seeds = []
    for t, n in [(args.at, args.person)] + [tuple(map(float, s.split(":"))) for s in args.fix]:
        f, dets = people_at(a, t)
        if not 1 <= int(n) <= len(dets):
            raise SystemExit(f"At {t}s there are {len(dets)} people; person {int(n)} does not exist")
        seeds.append(dets[int(n) - 1])

    print("Tracking the member through the video…", file=sys.stderr)
    res = identify(a, Member("member", seeds=seeds))
    print(f"Visible in {100 * (res.det >= 0).mean():.0f}% of frames", file=sys.stderr)
    cam = plan_camera(a, res, args.format, args.framing, args.smoothness)
    out = args.output or f"{Path(args.video).stem}_fancam_{args.format}.mp4"
    start = int(args.start * a.info.fps)
    end = int(args.end * a.info.fps) if args.end else None
    render(a.info, cam, FORMATS[args.format], out, start, end, progress=progress_printer("render "))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
