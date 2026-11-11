"""Online tray tracker: follows meat blobs across frames and picks ONE frame per tray.

Meat is found by colour on half-resolution frames, every `step`-th frame. Blobs are matched to
tracks by distance to where each tray should be now (last position + its speed): the belt can move
a tray ~90 half-res px per frame and also stops and restarts, so a track accepts a blob within
`max_jump_px` along the belt (and `max_dy_px` across it) of that prediction, nearest first.

Which frame gets measured: by default (`pick_zone_px` unset), the frame where the tray is fully in
view (meat clear of both frame edges) and closest to the frame centre; trays that never come within
`max_centre_dist` of the centre are skipped as fragments.

Optional, OFF by default until checked on the live stream: the pick zone R1 (`pick_zone_px`) is a band across the belt, centred,
wider than a tray by more than one frame's movement plus room for the next-tray check on both
ends. A tray is measured from the frame where its meat lies completely inside R1 and is closest to
the frame centre. Half trays at R1's edges never qualify. A tray that came near the centre but was
never completely inside R1 (merged with a touching tray, or the belt jumped it past) is still
reported, as not measured (`Pick.in_zone = False`), so it is counted rather than silently lost.

Where along the belt (camera angle): the camera looks at the belt at an angle, so meat heaped above
the tray rim is shifted in the image away from the point straight under the camera (right of the
frame centre on the FPL camera). At the frame centre the tray's left end looks more covered than it is.
Two options, both off by default:
  `pick_x_px`      measure the one frame where the tray is nearest this x (full-res) instead of the
                   frame centre.
  `side_views_px`  [x_left_end, x_right_end]: keep two frames per tray, the one nearest each x. The
                   tray's left half is measured in the first (tray further right, its left end near
                   the point under the camera), its right half in the second (`Pick.views`).
  `dynamic_split_px` (e.g. 1000): take the rightmost frame with the tray in view (the first one, as the
                   belt runs right to left). If the tray is at x <= split there, that one frame is good
                   for both ends: one photo. If it is further right, its right end starts to look
                   over-covered, so the right half comes from the next frame: two photos. The next frame
                   is also the backup if the first one cuts the tray box. Overrides the other two.

Bare hands: skin passes the meat colour rule and a hand is the size of a tray. A track whose picked
blob is smooth and skin-coloured (`color.skin_like`) is dropped, not reported (`drop_bare_hands`).

Spare frames: the last few frames with the tray fully in view are kept (`Pick.spares`), so a tray
that is unreadable only because of its outline or the frame edge can be measured again on another
frame (`retry_frames`); one unreadable because a worker's hand is over it, on up to
`retry_frames_hand` frames (the hand often moves away within a frame or two).

The tray is emitted as soon as it has moved `emit_after_px` past its best position(s) (or when its
track ends), so results come out while the tray is still on the belt instead of after the video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from collections import deque

from .color import masks, meat_blobs, skin_like


@dataclass
class Pick:
    """One tray, at its most central frame."""
    tray_id: int
    frame_index: int
    timestamp: float | None
    c: tuple            # meat centroid in half-res px
    area: int
    frame: np.ndarray | None   # full-resolution BGR frame (raw, as read from the camera); None if not in zone
    half: np.ndarray | None    # same frame at half resolution
    in_zone: bool = True       # False: the tray was never completely inside the pick zone (not measured)
    views: list | None = None  # side_views_px only: [Pick for the left half, Pick for the right half] (None = not seen there)
    spares: list = field(default_factory=list)  # other frames with the tray in view, nearest first (retry_frames)
    backup: "Pick | None" = None  # pick_x_px only: the next frame (tray further left), measured instead if this one
                                  # cuts the tray box (edge_of_view: a tilted tray's corner past the frame edge)


@dataclass
class _Track:
    id: int
    last_f: int
    last_c: tuple
    best_d: float = float("inf")
    best: Pick | None = None
    emitted: bool = False
    first_c: tuple = field(default=(0.0, 0.0))
    vel: tuple = (0.0, 0.0)      # px per frame
    n_obs: int = 0
    any_d: float = float("inf")  # closest to centre while fully in view, inside R1 or not
    any_best: Pick | None = None
    vd: list = field(default_factory=list)      # per target x: distance of the best frame so far
    vbest: list = field(default_factory=list)   # per target x: that frame
    dyn_r: Pick | None = None                   # dynamic: rightmost (first) frame with the tray in view
    dyn_n: Pick | None = None                   # dynamic: the next one (tray further left)
    left_d: float = float("inf")                # pick_x_px: nearest frame to the target with the tray left of it
    left_best: Pick | None = None
    ring: deque = field(default_factory=lambda: deque(maxlen=8))   # recent frames with the tray in view (spares)


class TrayTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        z = cfg.pick_zone_px or (0, cfg.frame_w)
        self.zone = (z[0] / 2.0, z[1] / 2.0)       # half-res px
        centre = cfg.frame_w / 2.0
        if cfg.side_views_px:
            xs = [float(x) for x in cfg.side_views_px]
        else:
            xs = [float(cfg.pick_x_px) if cfg.pick_x_px is not None else centre]
        self.targets = [x / 2.0 for x in xs]       # half-res px
        self.primary = int(np.argmin([abs(x - centre) for x in xs]))   # the view nearest the frame centre
        self.active: list[_Track] = []
        self.n_tracks = 0
        self.n_emitted = 0
        self.n_bare_hands = 0                     # tracks dropped because the "meat" was a bare hand

    def _small(self, full):
        w = self.cfg.work_w
        h = int(full.shape[0] * w / full.shape[1])
        return cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)

    def _finish(self, tr: _Track):
        if tr.emitted:
            return None
        tr.emitted = True
        if self.cfg.dynamic_split_px is not None:
            if tr.dyn_r is None:
                p = tr.any_best if tr.any_best is not None and tr.any_d <= self.cfg.track_max_centre_dist else None
                if p is None:
                    return None
            else:
                p = tr.dyn_r
                p.backup = tr.dyn_n
                if tr.dyn_n is not None and p.c[0] * 2 > self.cfg.dynamic_split_px:
                    p.views = [tr.dyn_r, tr.dyn_n]          # left half from the first frame, right half from the next
        elif tr.best is not None and tr.best_d <= self.cfg.track_max_centre_dist:
            p = tr.best
            if (self.cfg.pick_x_px is not None and not self.cfg.side_views_px and tr.left_best is not None
                    and tr.left_best is not p and p.c[0] > self.targets[0]):
                p.backup = tr.left_best
            if len(self.targets) > 1:
                p.views = [b if b is not None and d <= self.cfg.track_max_centre_dist else None
                           for b, d in zip(tr.vbest, tr.vd)]
        elif tr.best is None and tr.any_best is not None and tr.any_d <= self.cfg.track_max_centre_dist:
            p = tr.any_best      # a tray, but never completely inside R1: report it, unmeasured
        else:
            return None          # never came near the centre: a fragment
        if self.cfg.drop_bare_hands and p.half is not None and self._bare_hand(p):
            self.n_bare_hands += 1
            return None          # a bare hand over the belt, not a tray
        n_spare = max(self.cfg.retry_frames, self.cfg.retry_frames_hand)
        if n_spare > 0:
            used = [p] + [v for v in (p.views or []) if v is not None] + ([p.backup] if p.backup is not None else [])
            ids = {id(x) for x in used}
            near = lambda q: min(abs(q.frame_index - u.frame_index) for u in used)
            p.spares = sorted((q for q in tr.ring if id(q) not in ids), key=near)[:n_spare]
        p.tray_id = self.n_emitted
        self.n_emitted += 1
        return p

    def _bare_hand(self, p):
        red, _ = masks(p.half)
        lab, blobs = meat_blobs(red, self.cfg.track_min_area)
        if not blobs:
            return False
        b = min(blobs, key=lambda b: np.hypot(b["c"][0] - p.c[0], b["c"][1] - p.c[1]))
        return skin_like(p.half, lab == b["id"])[0]

    def update(self, frame_index: int, full: np.ndarray, timestamp: float | None = None) -> list[Pick]:
        """Feed every frame in order. Returns the trays that are ready (usually none)."""
        cfg = self.cfg
        if frame_index % cfg.track_step:
            return []
        small = self._small(full)
        W = small.shape[1]
        red, _ = masks(small)
        _, blobs = meat_blobs(red, cfg.track_min_area)
        used, nxt, ready = set(), [], []
        # match tracks to blobs around each track's predicted position, closest pairs first
        cand = []
        for ti, tr in enumerate(self.active):
            dt = frame_index - tr.last_f
            px, py = tr.last_c[0] + tr.vel[0] * dt, tr.last_c[1] + tr.vel[1] * dt
            for j, b in enumerate(blobs):
                dx, dy = b["c"][0] - px, b["c"][1] - py
                if abs(dx) < cfg.track_max_jump_px and abs(dy) < cfg.track_max_dy_px:
                    cand.append((np.hypot(dx, dy), ti, j))
        match = {}
        for _, ti, j in sorted(cand):
            if ti not in match and j not in used:
                match[ti] = j
                used.add(j)
        for ti, tr in enumerate(self.active):
            best = match.get(ti)
            if best is not None:
                self._observe(tr, blobs[best], frame_index, full, small, timestamp, W)
                nxt.append(tr)
                if self._past_best(tr, blobs[best], W):
                    p = self._finish(tr)          # moved well past its most central frame
                    if p is not None:
                        ready.append(p)
            elif frame_index - tr.last_f <= 3 * cfg.track_step:
                nxt.append(tr)                    # tolerate a short drop-out
            else:
                p = self._finish(tr)
                if p is not None:
                    ready.append(p)
        for j, b in enumerate(blobs):
            if j not in used:
                tr = _Track(id=self.n_tracks, last_f=frame_index, last_c=tuple(b["c"]), first_c=tuple(b["c"]),
                            vd=[float("inf")] * len(self.targets), vbest=[None] * len(self.targets))
                self.n_tracks += 1
                self._observe(tr, b, frame_index, full, small, timestamp, W)
                nxt.append(tr)
        self.active = nxt
        return ready

    def _past_best(self, tr, b, W):
        """Early emit only when the best frame(s) are final: the tray has been near each target x and has
        now crossed to the other side of every target from where it came in, by emit_after_px,
        as a same-sized blob (touching trays merge into one bigger blob whose centroid jumps)."""
        cfg = self.cfg
        if cfg.dynamic_split_px is not None:
            return not tr.emitted and tr.dyn_n is not None and 0.75 < b["area"] / max(tr.dyn_r.area, 1) < 1.33
        if tr.emitted or tr.best is None or tr.best_d > cfg.track_max_centre_dist:
            return False
        if not 0.75 < b["area"] / max(tr.best.area, 1) < 1.33:
            return False
        for t, d in zip(self.targets, tr.vd):
            came_from = np.sign(tr.first_c[0] - t) or -1.0
            past = (b["c"][0] - t) * -came_from          # > 0 once on the far side of this target
            if not (past > cfg.track_emit_after_px and abs(b["c"][0] - t) > min(d, 1e9) + cfg.track_emit_after_px):
                return False
        return True

    def _observe(self, tr, b, f, full, small, ts, W):
        if tr.n_obs:
            dt = max(f - tr.last_f, 1)
            v = ((b["c"][0] - tr.last_c[0]) / dt, (b["c"][1] - tr.last_c[1]) / dt)
            tr.vel = v if tr.n_obs == 1 else (0.5 * (tr.vel[0] + v[0]), 0.5 * (tr.vel[1] + v[1]))
        tr.n_obs += 1
        tr.last_f, tr.last_c = f, tuple(b["c"])
        x, _, w, _ = b["bbox"]
        m = self.cfg.track_edge_margin
        if not (x > m and x + w < W - m):
            return                                    # not fully in view
        d = abs(b["c"][0] - self.targets[self.primary])
        if d < tr.any_d:
            tr.any_d = d
            tr.any_best = Pick(tray_id=-1, frame_index=f, timestamp=ts, c=tuple(b["c"]), area=b["area"],
                               frame=None, half=None, in_zone=False)
        if not (x >= self.zone[0] and x + w <= self.zone[1]):
            return                                    # not completely inside R1
        # the tray box (wider than its meat) must be in view too, with room for its corners when it is
        # tilted up to ~11 deg (trays run at -3 to -11 deg); off-centre targets would otherwise pick a cut-off tray
        # (dynamic and two-view modes: no tilt allowance; the fitted box decides later, and a first frame
        # that cuts the tray or fails its outline check falls back to the next one, so frames just right of
        # the spot, x 1000-1070, stay usable for the tray's left half)
        tilt = 0.0 if (self.cfg.dynamic_split_px is not None or self.cfg.side_views_px) else 0.2 * self.cfg.tray_h / 2
        hw = self.cfg.tray_w / 2 + tilt + 2 - self.cfg.edge_tolerance_px / 2
        if b["c"][0] - hw < 0 or b["c"][0] + hw > W:
            return
        pk = Pick(tray_id=-1, frame_index=f, timestamp=ts, c=tuple(b["c"]), area=b["area"], frame=full.copy(), half=small)
        if max(self.cfg.retry_frames, self.cfg.retry_frames_hand) > 0:
            tr.ring.append(pk)
        if self.cfg.dynamic_split_px is not None:
            if tr.dyn_r is None:
                tr.dyn_r = pk
            elif tr.dyn_n is None and b["c"][0] < tr.dyn_r.c[0] - 10:      # the belt moved on
                tr.dyn_n = pk
            return
        for i, t in enumerate(self.targets):
            di = abs(b["c"][0] - t)
            if di < tr.vd[i]:
                tr.vd[i], tr.vbest[i] = di, pk
        t0 = self.targets[0]
        if self.cfg.pick_x_px is not None and b["c"][0] <= t0 and t0 - b["c"][0] < tr.left_d:
            tr.left_d, tr.left_best = t0 - b["c"][0], pk
        tr.best_d, tr.best = tr.vd[self.primary], tr.vbest[self.primary]

    def flush(self) -> list[Pick]:
        """End of stream: emit trays still being tracked."""
        out = [p for p in (self._finish(tr) for tr in self.active) if p is not None]
        self.active = []
        return out
