"""Pass 3: plan a virtual camera that follows the member like a human camera operator.

Per camera shot we solve an L1-optimal path (Grundmann et al., "Auto-Directed Video Stabilization
with Robust L1 Optimal Camera Paths"): penalising |velocity|, |acceleration| and |jerk| in L1 yields
paths made of holds, steady pans and smooth eases, i.e. minimal, deliberate camera moves. The member
only has to stay inside a dead zone around the centre, so small dance moves never shake the frame,
and a strongly-weighted soft constraint keeps her whole body inside the crop.
"""

from __future__ import annotations

import math
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

# Framing presets: person height as a fraction of the crop height, and space kept above the head.
FRAMINGS = {
    "full": dict(fill=0.80, headroom=0.09),  # whole body, the classic fancam look
    "knees": dict(fill=1.15, headroom=0.07),  # head to about the knees
    "waist": dict(fill=1.75, headroom=0.08),  # upper body
}

SMOOTHNESS = {"low": 0.4, "medium": 1.0, "high": 2.5}

MAX_UPSCALE = 4.0  # never zoom in further than this (source pixels get too blurry)


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


def target_boxes(a: Analysis, res: IdentityResult) -> np.ndarray:
    """(N, 4) box of the member per frame; short gaps inside a shot are interpolated, absence is NaN."""
    N = a.frame_count
    boxes = np.full((N, 4), np.nan, np.float64)
    have = res.det >= 0
    boxes[have] = a.det_box[res.det[have]]
    for s, e in a.shots():
        idx = np.flatnonzero(have[s:e]) + s
        for p, q in zip(idx[:-1], idx[1:]):
            if 1 < q - p <= MAX_SKIP + 1:
                t = (np.arange(p + 1, q) - p) / (q - p)
                boxes[p + 1 : q] = boxes[p] * (1 - t[:, None]) + boxes[q] * t[:, None]
    return boxes


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
) -> CameraPath:
    out_w, out_h = FORMATS[fmt]
    aspect = out_w / out_h
    fr = FRAMINGS[framing]
    sm = SMOOTHNESS[smoothness]
    W, H = a.info.width, a.info.height
    fps = a.info.fps
    N = a.frame_count
    boxes = target_boxes(a, res)
    present = np.isfinite(boxes[:, 0])

    h_max = min(H, W / aspect)
    h_min = max(h_max / MAX_UPSCALE, out_h / MAX_UPSCALE / 2, 16.0)
    cx = np.full(N, W / 2)
    cy = np.full(N, H / 2)
    hh = np.full(N, h_max)

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    # Zoom wanted per frame: from body height (and body width so wide poses still fit).
    need_h = np.maximum(bh / fr["fill"], bw / (aspect * 0.80))
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

        b = boxes[s:e]
        # --- horizontal
        bcx = (b[:, 0] + b[:, 2]) / 2
        half = np.minimum(bw[s:e] / 2, w_path / 2 * 0.92)  # arms may be clipped when the crop is narrow
        margin = w_path * 0.03
        x = _l1_path(
            bcx / unit,
            np.full(n, 1.0),
            w_path / unit * (0.06 if aspect < 1 else 0.05),
            (bcx - half - margin + w_path / 2) / unit,  # crop left <= body left - margin
            (bcx + half + margin - w_path / 2) / unit,  # crop right >= body right + margin
            w_path / 2 / unit,
            (W - w_path / 2) / unit,
            lam=(8.0 * sm, 25.0 * sm, 120.0 * sm),
        ) * unit
        # --- vertical: head placed `headroom` below the crop top
        ty = b[:, 1] - fr["headroom"] * h_path + h_path / 2
        top_req = (b[:, 1] - 0.02 * h_path + h_path / 2) / unit  # crop top above the head
        bottom_req = (b[:, 3] - h_path / 2) / unit if fr["fill"] <= 1.0 else np.full(n, np.nan)  # feet inside
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
