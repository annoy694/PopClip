"""Pass 3: plan a virtual camera that follows the member like a human camera operator.

Per camera shot we solve an L1-optimal path (Grundmann et al., "Auto-Directed Video Stabilization
with Robust L1 Optimal Camera Paths"): penalising |velocity|, |acceleration| and |jerk| in L1 yields
paths made of holds, steady pans and smooth eases, i.e. minimal, deliberate camera moves. The member
only has to stay inside a dead zone around the centre, so small dance moves never shake the frame,
and a strongly-weighted soft constraint keeps her whole body inside the crop.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import linprog

from .analyze import Analysis
from .identity import MAX_SKIP, IdentityResult

# Output formats offered in the UI.
FORMATS = {
    "vertical": (1080, 1920),
    "horizontal": (1920, 1080),
}

# Framing presets. Each frames a body region from `top` to `bottom` (landmarks from body_landmarks),
# which takes `fill` of the crop height. The camera holds the `anchor` edge steady with `margin` of the
# crop height beyond it, centres on `x`, and keeps `half` (a half-width) inside the crop. `fit_width`
# zooms out when a wide pose would not fit.
FRAMINGS = {
    "full": dict(top="box_top", bottom="feet", fill=0.80, anchor="top", margin=0.09, x="box_x", half="box_hw", fit_width=True),
    "knees": dict(top="box_top", bottom="knee", fill=0.86, anchor="top", margin=0.07, x="box_x", half="box_hw", fit_width=True),
    "waist": dict(top="box_top", bottom="hip", fill=0.86, anchor="top", margin=0.08, x="torso_x", half="torso_hw", fit_width=True),
    "face": dict(top="head_top", bottom="chest", fill=0.85, anchor="top", margin=0.10, x="head_x", half="head_hw", fit_width=False),
    "middle": dict(top="neck", bottom="thigh", fill=0.92, anchor="center", margin=0.0, x="torso_x", half="torso_hw", fit_width=False),
    "legs": dict(top="waist", bottom="feet", fill=0.90, anchor="bottom", margin=0.05, x="legs_x", half="legs_hw", fit_width=False),
}

SMOOTHNESS = {"low": 0.4, "medium": 1.0, "high": 2.5}

SIZE_RANGE = (0.5, 2.0)  # user multiplier on the crop size; >1 shows more around the region
MAX_ZOOM = 8.0  # never crop smaller than 1/8 of the frame height
KP_CONF = 0.35  # keypoints below this confidence are ignored

# Where landmarks sit on a standing body, as a fraction of its height from the top of the head. Used to
# fill in landmarks the pose model could not see (and for analyses made without a pose model).
_CANON = {"face": 0.06, "shoulder": 0.18, "hip": 0.50, "knee": 0.73, "ankle": 0.96, "feet": 1.0}
_KP = {"face": [0, 1, 2, 3, 4], "shoulder": [5, 6], "hip": [11, 12], "knee": [13, 14], "ankle": [15, 16]}


@dataclass
class CameraPath:
    cx: np.ndarray  # (N,) crop centre x in source pixels
    cy: np.ndarray  # (N,) crop centre y
    h: np.ndarray  # (N,) crop height in source pixels (width = h * aspect)
    aspect: float

    @property
    def w(self) -> np.ndarray:
        return self.h * self.aspect

    def rects(self) -> np.ndarray:
        """(N, 4) x, y, w, h crop rectangles."""
        w = self.w
        return np.stack([self.cx - w / 2, self.cy - self.h / 2, w, self.h], axis=1)


def _fill_gaps(a: Analysis, values: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Linearly interpolate rows of `values` across short gaps (<= MAX_SKIP frames) inside each shot."""
    out = values.copy()
    for s, e in a.shots():
        idx = np.flatnonzero(have[s:e]) + s
        for p, q in zip(idx[:-1], idx[1:]):
            if 1 < q - p <= MAX_SKIP + 1:
                t = ((np.arange(p + 1, q) - p) / (q - p)).reshape(-1, *([1] * (values.ndim - 1)))
                out[p + 1 : q] = values[p] * (1 - t) + values[q] * t
    return out


def target_boxes(a: Analysis, res: IdentityResult) -> np.ndarray:
    """(N, 4) box of the member per frame; short gaps inside a shot are interpolated, absence is NaN."""
    boxes = np.full((a.frame_count, 4), np.nan, np.float64)
    have = res.det >= 0
    boxes[have] = a.det_box[res.det[have]]
    return _fill_gaps(a, boxes, have)


def body_landmarks(a: Analysis, res: IdentityResult) -> dict[str, np.ndarray]:
    """Per-frame body landmarks of the member (NaN when absent).

    y levels: box_top (includes raised hands), head_top, neck, chest, waist, hip, thigh, knee, feet.
    x centres: box_x, head_x, torso_x, legs_x; half-widths: box_hw, head_hw, torso_hw, legs_hw.
    Keypoints come from the pose model; anything it could not see is placed using standard body
    proportions fitted to the landmarks it did see (or to the box when it saw none).
    """
    N = a.frame_count
    have = res.det >= 0
    box = np.full((N, 4), np.nan)
    box[have] = a.det_box[res.det[have]]
    kp = np.zeros((N, 17, 3))
    if a.det_kpts is not None and len(a.det_kpts):
        kp[have] = a.det_kpts[res.det[have]].astype(np.float64)
    vis = kp[:, :, 2] > KP_CONF

    def mean(idx: list[int], coord: int) -> np.ndarray:
        m = vis[:, idx]
        n = m.sum(axis=1)
        return np.where(n > 0, (kp[:, idx, coord] * m).sum(axis=1) / np.maximum(n, 1), np.nan)

    seen = {k: mean(idx, 1) for k, idx in _KP.items()}
    seen["feet"] = box[:, 3]
    # Least-squares fit y = top + scale * canonical_position over the landmarks that were seen.
    c = np.stack([np.where(np.isfinite(seen[k]), _CANON[k], np.nan) for k in _CANON], axis=1)
    y = np.stack([seen[k] for k in _CANON], axis=1)
    n = np.isfinite(y).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames where the member is absent are all-NaN
        cm, ym = np.nanmean(c, axis=1), np.nanmean(y, axis=1)
    var = np.nansum((c - cm[:, None]) ** 2, axis=1)
    cov = np.nansum((c - cm[:, None]) * (y - ym[:, None]), axis=1)
    scale = np.where(var > 1e-4, cov / np.maximum(var, 1e-4), np.nan)
    ok = (n >= 2) & (scale > 0.25 * (box[:, 3] - box[:, 1]))
    scale = np.where(ok, scale, box[:, 3] - box[:, 1])
    top = np.where(ok, ym - scale * cm, box[:, 1])

    def level(k: str) -> np.ndarray:
        return np.where(np.isfinite(seen[k]), seen[k], top + scale * _CANON[k])

    face, shoulder, hip, knee = level("face"), level("shoulder"), level("hip"), level("knee")
    head_top = np.maximum(face - 0.55 * (shoulder - face), box[:, 1])  # eyes sit mid-head
    cx = (box[:, 0] + box[:, 2]) / 2

    def x_of(idx: list[int]) -> np.ndarray:
        x = mean(idx, 0)
        return np.where(np.isfinite(x), x, cx)

    def half_of(idx: list[int], fallback: float) -> np.ndarray:
        xs = np.where(vis[:, idx], kp[:, idx, 0], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = (np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)) / 2
        return np.where(np.isfinite(spread), spread + 0.04 * scale, fallback * scale)

    lm = {
        "box_top": box[:, 1],
        "head_top": head_top,
        "neck": shoulder - 0.05 * scale,
        "chest": shoulder + 0.35 * (hip - shoulder),
        "waist": hip - 0.20 * (hip - shoulder),
        "hip": hip,
        "thigh": hip + 0.45 * (knee - hip),
        "knee": knee + 0.04 * scale,
        "feet": box[:, 3],
        "box_x": cx,
        "head_x": x_of(_KP["face"]),
        "torso_x": x_of(_KP["shoulder"] + _KP["hip"]),
        "legs_x": x_of(_KP["hip"] + _KP["knee"] + _KP["ankle"]),
        "box_hw": (box[:, 2] - box[:, 0]) / 2,
        "head_hw": 0.08 * scale,
        "torso_hw": half_of(_KP["shoulder"] + _KP["hip"], 0.12),
        "legs_hw": half_of(_KP["hip"] + _KP["knee"] + _KP["ankle"], 0.12),
    }
    keys = list(lm)
    filled = _fill_gaps(a, np.stack([lm[k] for k in keys], axis=1), have)
    return {k: filled[:, i] for i, k in enumerate(keys)}


def _l1_path(
    target: np.ndarray,
    weight: np.ndarray,
    dead: np.ndarray,
    lo_req: np.ndarray,
    hi_req: np.ndarray,
    lo_bound: np.ndarray,
    hi_bound: np.ndarray,
    lam: tuple[float, float, float],
    contain_w: float = 50.0,
) -> np.ndarray:
    """Solve for a 1-D camera coordinate p_t.

    minimise  sum weight*max(0,|p-target|-dead) + contain_w*(violations) + lam1|D1 p| + lam2|D2 p| + lam3|D3 p|
    with soft containment  p <= lo_req + v1  and  p >= hi_req - v2  (NaN = no requirement)
    and hard bounds lo_bound <= p <= hi_bound.
    """
    n = len(target)
    if n == 1:
        t = target[0] if np.isfinite(target[0]) else (lo_bound[0] + hi_bound[0]) / 2
        return np.clip([t], lo_bound, hi_bound)
    tgt = np.where(np.isfinite(target), target, 0.0)
    wgt = np.where(np.isfinite(target), weight, 0.0)
    has_lo = np.isfinite(lo_req)
    has_hi = np.isfinite(hi_req)

    # Variable layout: [p (n) | e (n) | v1 (n) | v2 (n) | d1 (n-1) | d2 (n-2) | d3 (n-3)]
    orders = [k for k in (1, 2, 3) if n > k]
    sizes = [n, n, n, n] + [n - k for k in orders]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    nv = int(offs[-1])
    I = sparse.identity(n, format="csr")
    Z = lambda r, c: sparse.csr_matrix((r, c))  # noqa: E731

    def block(row_parts: dict[int, sparse.spmatrix], rows: int) -> sparse.csr_matrix:
        cols = [row_parts.get(k, Z(rows, sizes[k])) for k in range(len(sizes))]
        return sparse.hstack(cols, format="csr")

    A, b = [], []
    # dead-zone deviation:  p - e <= tgt + dead ;  -p - e <= -tgt + dead
    A += [block({0: I, 1: -I}, n), block({0: -I, 1: -I}, n)]
    b += [tgt + dead, -tgt + dead]
    # containment: p - v1 <= lo_req ;  -p - v2 <= -hi_req
    A += [block({0: I, 2: -I}, n), block({0: -I, 3: -I}, n)]
    b += [np.where(has_lo, lo_req, 1e6), np.where(has_hi, -hi_req, 1e6)]
    # derivatives: |D_k p| <= d_k
    for j, k in enumerate(orders):
        D = sparse.diags(
            [(-1) ** (k - i) * float(math.comb(k, i)) for i in range(k + 1)],
            list(range(k + 1)),
            shape=(n - k, n),
            format="csr",
        )
        Ik = sparse.identity(n - k, format="csr")
        A += [block({0: D, 4 + j: -Ik}, n - k), block({0: -D, 4 + j: -Ik}, n - k)]
        b += [np.zeros(n - k), np.zeros(n - k)]

    c = np.zeros(nv)
    c[offs[1] : offs[2]] = wgt
    c[offs[2] : offs[3]] = np.where(has_lo, contain_w, 0.0)
    c[offs[3] : offs[4]] = np.where(has_hi, contain_w, 0.0)
    for j, k in enumerate(orders):
        c[offs[4 + j] : offs[5 + j]] = lam[k - 1]

    bounds = np.zeros((nv, 2))
    bounds[:, 1] = np.inf
    bounds[:n, 0], bounds[:n, 1] = lo_bound, hi_bound
    res = linprog(c, A_ub=sparse.vstack(A, format="csr"), b_ub=np.concatenate(b), bounds=bounds, method="highs")
    if res.status != 0:
        # Fall back to a heavily smoothed target if the LP solver gives up.
        ok = np.isfinite(target)
        fill = np.interp(np.arange(n), np.flatnonzero(ok), target[ok]) if ok.any() else (lo_bound + hi_bound) / 2
        return np.clip(gaussian_filter1d(fill, 10), lo_bound, hi_bound)
    return res.x[:n]


def plan_camera(
    a: Analysis,
    res: IdentityResult,
    fmt: str = "vertical",
    framing: str = "full",
    smoothness: str = "medium",
    size: float = 1.0,
) -> CameraPath:
    out_w, out_h = FORMATS[fmt]
    aspect = out_w / out_h
    fr = FRAMINGS[framing]
    sm = SMOOTHNESS[smoothness]
    size = float(np.clip(size, *SIZE_RANGE))
    W, H = a.info.width, a.info.height
    fps = a.info.fps
    N = a.frame_count
    lm = body_landmarks(a, res)
    top, bottom, xc, half_w = lm[fr["top"]], lm[fr["bottom"]], lm[fr["x"]], lm[fr["half"]]
    present = np.isfinite(top) & np.isfinite(bottom)

    h_max = min(H, W / aspect)
    h_min = max(h_max / MAX_ZOOM, 40.0)
    cx = np.full(N, W / 2)
    cy = np.full(N, H / 2)
    hh = np.full(N, h_max)

    region_h = np.maximum(bottom - top, 1.0)
    # Zoom wanted per frame: the region fills `fill` of the crop (and wide poses still fit, if asked),
    # then the user's size multiplier. Width always follows height, so the output ratio never changes.
    need_h = region_h / fr["fill"]
    if fr["fit_width"]:
        need_h = np.maximum(need_h, 2 * half_w / (aspect * 0.80))
    need_h = need_h * size
    unit = H  # work in units of frame height so weights do not depend on resolution

    for s, e in a.shots():
        n = e - s
        pres = present[s:e]
        if not pres.any():
            continue  # member never visible in this shot: show the wide centre crop
        # --- zoom: robust long-window estimate (crouches and jumps must not pump the zoom), then L1-smoothed.
        nh = need_h[s:e].copy()
        idx = np.flatnonzero(pres)
        nh = np.interp(np.arange(n), idx, nh[idx])
        win = max(3, int(fps * 2.5)) | 1
        nh = percentile_filter(nh, 70, size=win, mode="nearest")
        nh = np.clip(nh, h_min, h_max)
        h_path = _l1_path(
            nh / unit,
            np.full(n, 1.0),
            nh / unit * 0.08,
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.full(n, h_min / unit),
            np.full(n, h_max / unit),
            lam=(40.0 * sm, 60.0 * sm, 200.0 * sm),
        ) * unit
        h_path = np.clip(gaussian_filter1d(h_path, 3), h_min, h_max)
        w_path = h_path * aspect

        # --- horizontal
        bx = xc[s:e]
        half = np.minimum(half_w[s:e], w_path / 2 * 0.92)  # arms may be clipped when the crop is narrow
        margin = w_path * 0.03
        x = _l1_path(
            bx / unit,
            np.full(n, 1.0),
            w_path / unit * (0.06 if aspect < 1 else 0.05),
            (bx - half - margin + w_path / 2) / unit,  # crop left <= body left - margin
            (bx + half + margin - w_path / 2) / unit,  # crop right >= body right + margin
            w_path / 2 / unit,
            (W - w_path / 2) / unit,
            lam=(8.0 * sm, 25.0 * sm, 120.0 * sm),
        ) * unit
        # --- vertical: hold the anchored edge of the region steady, keep the whole region in when it fits
        rt, rb = top[s:e], bottom[s:e]
        if fr["anchor"] == "top":
            ty = rt - fr["margin"] * h_path + h_path / 2
        elif fr["anchor"] == "bottom":
            ty = rb + fr["margin"] * h_path - h_path / 2
        else:
            ty = (rt + rb) / 2
        fits = (rb - rt) <= 0.96 * h_path
        keep_top = fits | (fr["anchor"] == "top")
        keep_bottom = fits | (fr["anchor"] == "bottom")
        top_req = np.where(keep_top, rt - 0.02 * h_path + h_path / 2, np.nan) / unit  # crop top above region top
        bottom_req = np.where(keep_bottom, rb + 0.02 * h_path - h_path / 2, np.nan) / unit  # crop bottom below it
        y = _l1_path(
            ty / unit,
            np.full(n, 1.0),
            h_path / unit * 0.05,
            top_req,
            bottom_req,
            h_path / 2 / unit,
            (H - h_path / 2) / unit,
            lam=(10.0 * sm, 30.0 * sm, 150.0 * sm),
        ) * unit

        # Round off the L1 corners a little so pans ease in and out.
        x = gaussian_filter1d(x, 2.0)
        y = gaussian_filter1d(y, 2.0)
        cx[s:e] = np.clip(x, w_path / 2, W - w_path / 2)
        cy[s:e] = np.clip(y, h_path / 2, H - h_path / 2)
        hh[s:e] = h_path
    return CameraPath(cx=cx, cy=cy, h=hh, aspect=aspect)
