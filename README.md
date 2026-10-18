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
Analysis then runs many times faster. On a 4-core CPU without a GPU, a 4-minute video takes about 11
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
4. **Choose the output.**
   - **Format:** vertical 9:16 or horizontal 16:9.
   - **Framing:** full body, knees up, waist up, face (face and above), middle (chest to thighs) or legs
     (waist down). Body parts come from the pose model's keypoints, so the face stays framed even when
     the arms are raised.
   - **Box size:** 50–200%. Bigger shows more around the chosen part, smaller zooms in closer. The
     output keeps its 9:16 or 16:9 shape either way.
   - **Camera:** lively, smooth or very calm.
   - Optionally set a clip range with **I** / **O** (or the Start / End buttons).

   If a close-up has to enlarge a low-resolution video a lot, the preview warns you that the clip will
   look soft.
5. The **Preview** panel plays the actual crop live. When it looks right, click **Render clip** and
   download the MP4.

Keys: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> one frame, <kbd>Shift</kbd>+arrows one second,
<kbd>I</kbd>/<kbd>O</kbd> clip in/out, <kbd>[</kbd>/<kbd>]</kbd> box size, <kbd>Esc</kbd> close menu.

Projects are kept in `workspace/`, so you can close the app and continue later. Delete a project from the
start page to free the space.

## More tools

The tabs at the top of the app hold three more tools.

### Split Screen

Puts two pictures or videos side by side in one video.

- **Output shape:** 16:9 (1920×1080), 9:16, 1:1, 4:5, 21:9 or a custom size (320–3840 px).
- **Framing for each side:** zoom (1–4×) and horizontal/vertical position. Drag in the small framing
  view or directly on the preview, and scroll to zoom. Media always fills its side by cropping, never
  with black bars.
- **Middle edge:** a soft seam (0 for a hard edge) that fades transparency with a smooth curve.
  Direction can be both sides cross-fading, the left fading over the right, or the reverse.
- **Trimming:** a scrubber with **Start here / End here**, or type exact times.
- **When a clip ends:** **Loop** (repeats exactly the trimmed section, sound included), **Freeze** on
  the last frame, or go **Black**.
- **Length:** by default the length of the shortest clip that doesn't loop, or set it manually
  (1–120 s). Notes under each clip tell you if it will loop, freeze or go black, and for how long.
- **Sound:** each video has its own volume (0–150%), so you can use one, the other, or mix both. Short
  sound is padded with silence, so it never shortens the video. Videos with no sound track are
  detected and marked.
- **Preview:** one shared play/scrub bar keeps both sides in sync. Turn on **Sound** to hear the mix.
- **Render:** 60 fps H.264 MP4 with AAC sound. It renders on your PC with ffmpeg, which is much faster
  than in-browser rendering and isn't limited by browser memory.

### Overlay

Puts one picture or video on top of another. The controls are the same as Split Screen, plus:

- **Top layer opacity** (0–100%).
- **Blend mode:** normal, screen, lighten, add, overlay, multiply, darken or difference.
- **Swap layers.**
- Pictures with transparency (PNG) stay transparent.

### Caption Maker

Captions and blur regions on a picture, exported as PNG. It runs entirely in your browser.

- **Text boxes:**
  - drag to move and drag the handle to set the width; text wraps and the box grows downwards
  - 15 fonts, bundled with the app so they work offline
  - size, alignment, bold, italic, uppercase, text and outline colours, outline thickness, line height,
    rotation, and a dark or light background
- **Sync styles** makes style changes apply to every text box. **Apply this style to all** copies the
  style once.
- **Blur regions:** circle or rectangle, with position, size, strength and a soft edge fade. Sizes are
  in % of the picture's shorter side. A blur region also blurs any captions underneath it.
- **Editing:** undo/redo (Ctrl+Z / Ctrl+Shift+Z), duplicate (Ctrl+D), delete (Del), arrow-key nudging,
  snapping to the centre and to other items with guide lines, and layer ordering.
- **Export sizes:** original, 2× for sharper text (up to 3840 px), or a fixed long side of
  1080/2160/3840 px. The exact pixel size is shown before you download.
- **The preview matches the export.** Both are drawn by the same code and text is wrapped once at the
  picture's own resolution, so the PNG matches the screen line for line.

## Command line

```bash
# who is where at 0:30? (writes people_30.0s.jpg with numbered boxes, left to right)
python clip.py stage.mp4 --show 30

# follow person #4 at 0:30 and render a vertical full-body fancam
python clip.py stage.mp4 --at 30 --person 4 -o fancam.mp4

# horizontal, knees up, only 1:00–1:45, with an extra hint "at 1:12 she is person #2"
python clip.py stage.mp4 --at 30 --person 4 --format horizontal --framing knees \
       --start 60 --end 105 --fix 72:2 -o clip.mp4

# face close-up with a 30% bigger box
python clip.py stage.mp4 --at 30 --person 4 --framing face --size 1.3 -o face.mp4
```

The analysis is cached next to the video (`stage.mp4.clipper.npz`), so later runs skip it.

## How it works

| Step | File | What happens |
|---|---|---|
| Analyse | `clipper/analyze.py` | YOLO26 pose model on every 2nd frame (person boxes plus 17 body keypoints), ByteTrack short-term tracking, camera-cut detection on every frame, and two appearance descriptors per person (deep features from a small CNN on head/torso/legs stripes, plus HSV outfit colour histograms). |
| Identify | `clipper/identity.py` | Small per-video classifiers learn "this member versus everyone else on stage" from your click (plus the same person followed a few seconds before and after). A Viterbi search then chooses at most one box per frame, balancing appearance, plausible motion and the possibility that the member is absent. It self-trains outwards in time from your clicks, and your fixes are hard constraints. |
| Camera | `clipper/camera.py` | Body landmarks for the chosen framing (keypoints, with standard body proportions filling any the model couldn't see), then per shot an L1-optimal camera path (as in YouTube's stabiliser, Grundmann et al. 2011). It minimises total movement, acceleration and jerk, keeps the member inside a dead zone, and uses a soft constraint so the whole body stays in frame. Zoom comes from a robust long-window body height. |
| Render | `clipper/render.py` | Sub-pixel crop and resize for each frame (no 1-pixel jitter on slow pans), encoded to H.264 with the original audio. |

## Tips and limits

- **Higher-resolution sources give better clips.** The crop is upscaled to 1080p. A 4K or 1080p stage video
  gives a sharp fancam, while a 360p source looks soft. Zoom is capped at 4× to limit blur.
- **Click the member in the main performance**, not during an unusual intro pose (sitting or lying down).
  If you start in the intro, add one more click later.
- Members in **identical outfits** with similar hair are the hardest case. Name all of them (one click
  each), which turns every member into a "not this one" example for the others.
- The tracker follows people, not faces, so it cannot tell who the member is by name. You name them.
