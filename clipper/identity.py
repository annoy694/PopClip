"""Pass 2: decide which box is the chosen member in every frame.

Short-term tracker ids break whenever members cross, overlap, leave the frame or the camera cuts.
Instead of trusting them, we solve one global problem: pick at most one box per frame so that
  * appearance matches the member (deep features + outfit colours, learned from the user's clicks),
  * consecutive picks move plausibly (same tracker id is cheap, a jump to another person is expensive),
  * the member may be absent for a while (off stage, hidden, close-up of someone else),
  * every user correction is respected exactly.
This is a Viterbi/shortest-path search over all detections. The member's appearance model is then
re-learned from the confident part of the answer and the search runs again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyze import Analysis

NEG_INF = -1e9

# Scoring knobs, in log-likelihood units.
MAX_SKIP = 15  # frames a box may be missing between two picks while keeping motion continuity
SKIP_COST = 0.15  # per skipped frame
LEAVE_COST = 4.0  # going from a box to "absent"
ENTER_COST = 4.0  # coming back from "absent"
SAME_TRACK_BONUS = 1.5
SWITCH_TRACK_COST = 1.0
MOVE_COST = 14.0  # per (centre shift / box height) beyond what the frame gap allows
SIZE_COST = 6.0  # per |log(height ratio)|
CUT_SWITCH_COST = 0.5  # any box -> any box across a camera cut
EMISSION_CLIP = 6.0


@dataclass
class Member:
    name: str
    seeds: list[int] = field(default_factory=list)  # detection indices the user said ARE this member
    not_dets: list[int] = field(default_factory=list)  # detection indices the user said are NOT this member
    absent_frames: list[int] = field(default_factory=list)  # frames where the user said the member is not visible


@dataclass
class IdentityResult:
    det: np.ndarray  # (N,) int64 detection index picked per frame, -1 = absent
    score: np.ndarray  # (N,) float32 appearance evidence of the pick (calibrated log-odds), 0 when absent


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class _Features:
    """Appearance cues per detection: deep features, outfit colour per body stripe, whole-body colour."""

    def __init__(self, a: Analysis):
        deep = _l2n(a.det_deep.astype(np.float32))
        color = _l2n(a.det_color.astype(np.float32))
        # Whole-body colour: pose independent (sitting vs standing moves the outfit between stripes).
        parts = np.split(color**2, 3, axis=1)
        whole = _l2n(np.sqrt(sum(parts) / 3))
        self.cues = [deep, color, whole]
        self.x = np.concatenate([deep * np.sqrt(0.5), color * np.sqrt(0.25), whole * np.sqrt(0.25)], axis=1)

    def sims(self, gallery: np.ndarray, top_k: int = 5) -> np.ndarray:
        """Mean of the top-k cosine similarities of every detection to the gallery."""
        g = self.x[gallery]
        s = self.x @ g.T
        k = min(top_k, len(gallery))
        return np.sort(s, axis=1)[:, -k:].mean(axis=1)


def box_iou(b: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of box b with each row of others (x1 y1 x2 y2)."""
    x1 = np.maximum(b[0], others[:, 0])
    y1 = np.maximum(b[1], others[:, 1])
    x2 = np.minimum(b[2], others[:, 2])
    y2 = np.minimum(b[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
    return inter / (area(b) + area(others) - inter + 1e-6)


def _duplicates(b1: np.ndarray, b2: np.ndarray) -> bool:
    """One box (almost) inside the other: the detector fired twice on one person."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    smaller = min((b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter > 0.85 * smaller


def _chain(a: Analysis, d: int, direction: int, max_frames: int, max_skip: int = 3) -> list[int]:
    """Follow one person from detection d by box overlap, stopping as soon as it becomes ambiguous."""
    out = []
    cuts = set(a.cuts.tolist())
    box = a.det_box[d]
    f = int(a.det_frame[d])
    skipped = 0
    for _ in range(max_frames):
        nf = f + direction
        if nf < 0 or nf >= a.frame_count or (nf if direction > 0 else f) in cuts:
            break
        f = nf
        sl = a.frame_slice(f)
        if sl.stop == sl.start:
            skipped += 1
            if skipped > max_skip:
                break
            continue
        ious = box_iou(box, a.det_box[sl])
        order = np.argsort(ious)[::-1]
        best = ious[order[0]]
        second = ious[order[1]] if len(order) > 1 else 0.0
        if second >= 0.35 and _duplicates(a.det_box[sl.start + int(order[0])], a.det_box[sl.start + int(order[1])]):
            second = 0.0  # two boxes on the same person, not two people
        same_track = a.det_tid[d] >= 0 and a.det_tid[sl.start + int(order[0])] == a.det_tid[d]
        if best >= 0.5 and (second < 0.35 or same_track):
            d = sl.start + int(order[0])
            out.append(d)
            box = a.det_box[d]
            skipped = 0
        elif best < 0.3:
            skipped += 1  # briefly missed by the detector
            if skipped > max_skip:
                break
        else:
            break  # two people overlap: stop before we drift onto the wrong one
    return out


def features_for(a: Analysis) -> _Features:
    """Built once per loaded analysis; every re-track after a user click reuses it."""
    if getattr(a, "_features", None) is None:
        a._features = _Features(a)
    return a._features


def _seed_gallery(a: Analysis, seeds: list[int], window_s: float = 3.0) -> np.ndarray:
    """Detections that are almost surely the member: the clicked boxes plus the same person tracked a
    few seconds before and after by unambiguous box overlap."""
    win = int(a.info.fps * window_s)
    picked = set(seeds)
    for d in seeds:
        picked.update(_chain(a, d, +1, win))
        picked.update(_chain(a, d, -1, win))
    return np.array(sorted(picked), dtype=np.int64)


def _negatives(a: Analysis, positives: np.ndarray) -> np.ndarray:
    """Other people visible in the same frames as known positives (cannot be the member)."""
    frames = np.unique(a.det_frame[positives])
    mask = np.isin(a.det_frame, frames)
    mask[positives] = False
    return np.flatnonzero(mask)


def _train_classifier(x: np.ndarray, pos: np.ndarray, neg: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """Balanced L2-regularised logistic regression; returns weights with the bias as last element."""
    import torch

    X = torch.from_numpy(np.concatenate([x[pos], x[neg]])).float()
    X = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    sw = torch.cat([torch.full((len(pos),), 0.5 / len(pos)), torch.full((len(neg),), 0.5 / len(neg))])
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y, reduction="none") * sw).sum()
        loss = loss + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def _emissions(feats: _Features, gallery: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """Log-odds 'is the member' vs 'is someone else' for every detection.

    One small classifier per cue, each calibrated, then averaged: a cue that does not transfer to a new
    pose or camera angle (say, seated vs standing) cannot drown out the ones that do.
    """
    if len(negatives) < 3:  # nobody else to contrast with: fall back to plain similarity
        s = feats.sims(gallery)
        return np.clip((s - np.median(s)) * 20, -EMISSION_CLIP, EMISSION_CLIP).astype(np.float32)
    total = np.zeros(len(feats.x), np.float32)
    for x in feats.cues:
        w = _train_classifier(x, gallery, negatives)
        logit = x @ w[:-1] + w[-1]
        # Regularisation shrinks the raw logits: recentre and rescale so typical training positives and
        # negatives sit at +3 and -3, and "undecided" is 0.
        pos, neg = float(np.median(logit[gallery])), float(np.median(logit[negatives]))
        scaled = (logit - (pos + neg) / 2) * (6.0 / max(pos - neg, 1e-3))
        total += np.clip(scaled, -EMISSION_CLIP, EMISSION_CLIP)
    return (total / len(feats.cues)).astype(np.float32)


def _viterbi(
    a: Analysis,
    emission: np.ndarray,
    forced: dict[int, int],
) -> np.ndarray:
    """Best path through the detections. forced[frame] = detection index, or -1 to force 'absent'."""
    N = a.frame_count
    off = a.frame_offsets
    boxes = a.det_box
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    tid = a.det_tid
    is_cut = np.zeros(N + 1, bool)
    is_cut[a.cuts] = True
    shot_start = np.zeros(N, np.int64)
    for s, e in a.shots():
        shot_start[s:e] = s

    best = np.full(len(boxes), NEG_INF, np.float64)  # best path score ending at each detection
    back = np.full(len(boxes), -1, np.int64)  # predecessor detection, or -1 = came from absent / start
    absent = np.zeros(N, np.float64)  # best score ending with 'absent' at frame f
    absent_back = np.full(N, -2, np.int64)  # -2 = from absent at f-1, otherwise detection index

    for f in range(N):
        sl = slice(int(off[f]), int(off[f + 1]))
        n = sl.stop - sl.start
        prev_absent = absent[f - 1] if f > 0 else 0.0

        # --- absent at f
        best_leave, best_leave_i = NEG_INF, -1
        if f > 0:
            psl = slice(int(off[f - 1]), int(off[f]))
            if psl.stop > psl.start:
                i = int(np.argmax(best[psl])) + psl.start
                best_leave, best_leave_i = best[i] - LEAVE_COST, i
        if prev_absent >= best_leave:
            absent[f], absent_back[f] = prev_absent, -2
        else:
            absent[f], absent_back[f] = best_leave, best_leave_i

        if n:
            j = np.arange(sl.start, sl.stop)
            cand = np.full(n, prev_absent - ENTER_COST if f > 0 else 0.0)
            cand_back = np.full(n, -1, np.int64)
            if f > 0 and is_cut[f]:
                # New shot: anyone in the previous frame may continue as anyone here.
                psl = slice(int(off[f - 1]), int(off[f]))
                if psl.stop > psl.start:
                    i = int(np.argmax(best[psl])) + psl.start
                    v = best[i] - CUT_SWITCH_COST
                    better = v > cand
                    cand[better], cand_back[better] = v, i
            else:
                lo = max(int(shot_start[f]), f - MAX_SKIP)
                # Detections are stored frame-sorted, so all candidates from frames lo..f-1 are one slice.
                i = np.arange(int(off[lo]), int(off[f]))
                if len(i):
                    g = (f - a.det_frame[i]).astype(np.float64)[:, None]
                    d = np.hypot(cx[j][None, :] - cx[i][:, None], cy[j][None, :] - cy[i][:, None])
                    hh = np.maximum(h[j][None, :], h[i][:, None])
                    move = np.maximum(d / hh - 0.04 * g, 0.0)
                    size = np.abs(np.log(h[j][None, :] / h[i][:, None]))
                    same = (tid[i][:, None] == tid[j][None, :]) & (tid[i][:, None] >= 0)
                    trans = -MOVE_COST * move - SIZE_COST * size - SKIP_COST * (g - 1)
                    trans += np.where(same, SAME_TRACK_BONUS, -SWITCH_TRACK_COST)
                    tot = best[i][:, None] + trans
                    k = np.argmax(tot, axis=0)
                    v = tot[k, np.arange(n)]
                    better = v > cand
                    cand[better], cand_back[better] = v[better], i[k[better]]
            best[sl] = cand + emission[sl]
            back[sl] = cand_back

        if f in forced:
            want = forced[f]
            if want >= 0:
                keep = best[want]
                best[sl] = NEG_INF
                best[want] = keep if keep > NEG_INF / 2 else 0.0
                absent[f] = NEG_INF
            else:
                best[sl] = NEG_INF

    # Backtrack from the best final state.
    path = np.full(N, -1, np.int64)
    last = slice(int(off[N - 1]), int(off[N]))
    state = -2
    if last.stop > last.start and best[last].max() > absent[N - 1]:
        state = int(np.argmax(best[last])) + last.start
    f = N - 1
    while f >= 0:
        if state >= 0:
            path[f] = state
            prev = back[state]
            pf = f - 1
            if prev >= 0:
                pf = int(a.det_frame[prev])
                path[pf + 1 : f] = -1  # skipped frames are filled by interpolation later
                state = prev
                f = pf
                continue
            # came from absent (or the very start)
            state = -2
            f -= 1
        else:
            prev = absent_back[f]
            state = int(prev) if prev >= 0 else -2
            f -= 1
    return path


def _margin(a: Analysis, emission: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Emission of each picked detection minus the best other detection in the same frame."""
    out = np.empty(len(picks), np.float32)
    for k, d in enumerate(picks):
        sl = a.frame_slice(int(a.det_frame[d]))
        e = emission[sl]
        others = np.delete(e, d - sl.start)
        out[k] = emission[d] - (others.max() if len(others) else -EMISSION_CLIP)
    return out


def identify(a: Analysis, member: Member, others: list[Member] | None = None) -> IdentityResult:
    if not member.seeds:
        raise ValueError("Pick the member in at least one frame first.")
    feats = features_for(a)
    other_seeds = [d for m in (others or []) for d in m.seeds]
    blocked = np.array(other_seeds + member.not_dets, np.int64)

    forced: dict[int, int] = {}
    for f in member.absent_frames:
        forced[int(f)] = -1
    for d in member.seeds:
        forced[int(a.det_frame[d])] = int(d)

    seed_gallery = np.setdiff1d(_seed_gallery(a, member.seeds), blocked)
    other_gallery = _seed_gallery(a, other_seeds) if other_seeds else np.zeros(0, np.int64)
    seed_frames = a.det_frame[np.array(member.seeds)]
    gallery = seed_gallery
    # Self-training rounds: confident picks are learned only within a window around the user's clicks
    # that widens each round, so an early mistake far away cannot take over.
    windows = [15.0, 60.0, np.inf, None]
    for window in windows:
        negatives = np.setdiff1d(np.union1d(_negatives(a, gallery), other_gallery), gallery).astype(np.int64)
        emission = _emissions(feats, gallery, negatives)
        emission[member.not_dets] = NEG_INF / 4
        emission[other_seeds] = NEG_INF / 4
        path = _viterbi(a, emission, forced)
        if window is None:
            break
        picks = path[path >= 0]
        near = np.abs(a.det_frame[picks][:, None] - seed_frames[None, :]).min(axis=1) <= window * a.info.fps
        picks = picks[near]
        # A pick only counts when it clearly beats everybody else in its frame.
        conf = picks[(emission[picks] > 2.0) & (_margin(a, emission, picks) > 2.0)]
        if len(conf) > 600:
            conf = conf[np.linspace(0, len(conf) - 1, 600).astype(int)]
        gallery = np.setdiff1d(np.union1d(seed_gallery, conf), blocked).astype(np.int64)

    score = np.where(path >= 0, emission[np.maximum(path, 0)], 0.0).astype(np.float32)
    return IdentityResult(det=path, score=score)
