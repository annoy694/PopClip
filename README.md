# PopClip

Clips K-pop fancams and focuses on one member.

Turn a group performance video into a **focus cam** of one member. Upload the video, click the member
you want, and get a smooth 9:16 (1080×1920) or 16:9 (1920×1080) clip that follows only them, with the
original audio.

- Every person is detected and tracked, and named members get labelled boxes, like other fancam editors.
- The member is re-found after crossings, formation changes, occlusions (camera operators walking in front)
  and camera cuts, using per-video appearance learning instead of fragile track IDs.
- The virtual camera moves like a human operator: it holds still while the member dances in place, then
  pans with eased starts and stops when she moves. Zoom stays steady, so jumps and crouches don't make it
  pump in and out. Hard cuts happen only where the source video cuts.
- A timeline shows where the tracking is confident, unsure, or the member isn't visible. One click on the
  right person fixes a mistake, and everything is re-solved around that click.

## Quick start

Requirements: Python 3.10+ (3.11–3.13 recommended). ffmpeg is optional because a bundled copy is used if
it isn't installed. The two model files (about 25 MB) download into `models/` automatically on first use.

**Windows:** double-click `start.bat`. **macOS / Linux:** run `./start.sh`.

The first launch creates a virtual environment and installs the dependencies (a few minutes). After that
it opens http://127.0.0.1:8000.

Manual setup:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip ...
.venv/bin/python server.py                     # then open http://127.0.0.1:8000
```

**NVIDIA GPU:** install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/).
Analysis then runs many times faster. On a 4-core CPU without a GPU, a 4-minute video takes about 12–15
minutes to analyse. That happens once per video, and everything after it is interactive.

## Using it

1. **Drop a video** on the start page. Each person in each frame is detected and tracked, and a progress
   bar shows how far it has got.
2. **Click the member** you want to follow and give them a name. They're tracked through the whole video
   in a few seconds. You can name other members too, which also helps tell similar outfits apart.
3. **Check the timeline** under the player. Pink means tracked, amber means unsure (worth a look) and grey
   means not visible.
   - If the box is on the wrong person, click the right person and pick the name.
   - If the box is on someone, but your member isn't visible: click that box → "Not ‹name›".
   - If the member is hidden (for example behind a camera operator): click empty space →
     "‹name› isn't visible here".

   Each fix is a hard constraint. Usually one or two fixes are enough for a whole video.
4. **Choose the output.** Vertical 9:16 or horizontal 16:9. Framing: full body, knees up or waist up.
   Camera: lively, smooth or very calm. Optionally set a clip range with **I** / **O** (or the
   Start / End buttons).
5. The **Preview** panel plays the actual crop live. When it looks right, click **Render clip** and
   download the MP4.

Keys: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> one frame, <kbd>Shift</kbd>+arrows one second,
<kbd>I</kbd>/<kbd>O</kbd> clip in/out, <kbd>Esc</kbd> close menu.

Projects are kept in `workspace/`, so you can close the app and continue later. Delete a project from the
start page to free the space.

## Command line

```bash
# who is where at 0:30? (writes people_30.0s.jpg with numbered boxes, left to right)
python clip.py stage.mp4 --show 30

# follow person #4 at 0:30 and render a vertical full-body fancam
python clip.py stage.mp4 --at 30 --person 4 -o fancam.mp4

# horizontal, knees up, only 1:00–1:45, with an extra hint "at 1:12 she is person #2"
python clip.py stage.mp4 --at 30 --person 4 --format horizontal --framing knees \
       --start 60 --end 105 --fix 72:2 -o clip.mp4
```

The analysis is cached next to the video (`stage.mp4.clipper.npz`), so later runs skip it.

## How it works

| Step | File | What happens |
|---|---|---|
| Analyse | `clipper/analyze.py` | YOLO26 person detection on every frame, ByteTrack short-term tracking, camera-cut detection, and two appearance descriptors per person (deep features from a small CNN on head/torso/legs stripes, plus HSV outfit colour histograms). |
| Identify | `clipper/identity.py` | Small per-video classifiers learn "this member versus everyone else on stage" from your click (plus the same person followed a few seconds before and after). A Viterbi search then chooses at most one box per frame, balancing appearance, plausible motion and the possibility that the member is absent. It self-trains outwards in time from your clicks, and your fixes are hard constraints. |
| Camera | `clipper/camera.py` | Per shot, an L1-optimal camera path (as in YouTube's stabiliser, Grundmann et al. 2011). It minimises total movement, acceleration and jerk, keeps the member inside a dead zone, and uses a soft constraint so the whole body stays in frame. Zoom comes from a robust long-window body height. |
| Render | `clipper/render.py` | Sub-pixel crop and resize for each frame (no 1-pixel jitter on slow pans), encoded to H.264 with the original audio. |

## Tips and limits

- **Higher-resolution sources give better clips.** The crop is upscaled to 1080p. A 4K or 1080p stage video
  gives a sharp fancam, while a 360p source looks soft. Zoom is capped at 4× to limit blur.
- **Click the member in the main performance**, not during an unusual intro pose (sitting or lying down).
  If you start in the intro, add one more click later.
- Members in **identical outfits** with similar hair are the hardest case. Name all of them (one click
  each), which turns every member into a "not this one" example for the others.
- The tracker follows people, not faces, so it cannot tell who the member is by name. You name them.
