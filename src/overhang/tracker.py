"""Online tray tracker: follows meat blobs across frames and picks ONE frame per tray.

Meat is found by colour on half-resolution frames, every `step`-th frame. Blobs are matched to
tracks by distance to where each tray should be now (last position + its speed): the belt can move
a tray ~90 half-res px per frame and also stops and restarts, so a track accepts a blob within
`max_jump_px` along the belt (and `max_dy_px` across it) of that prediction, nearest first. For each track the best frame is the one where
the tray is fully in view (meat clear of both frame edges) and closest to the frame centre.

The tray is emitted as soon as it has moved `emit_after_px` past that best position (or when its
track ends), so results come out while the tray is still on the belt instead of after the video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import masks, meat_blobs


@dataclass
class Pick:
    """One tray, at its most central frame."""
    tray_id: int
    frame_index: int
    timestamp: float | None
    c: tuple            # meat centroid in half-res px
    area: int
    frame: np.ndarray   # full-resolution BGR frame (raw, as read from the camera)
    half: np.ndarray    # same frame at half resolution


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


class TrayTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.active: list[_Track] = []
        self.n_tracks = 0
        self.n_emitted = 0

    def _small(self, full):
        w = self.cfg.work_w
        h = int(full.shape[0] * w / full.shape[1])
        return cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)

    def _finish(self, tr: _Track):
        if tr.emitted or tr.best is None:
            return None
        tr.emitted = True
        if tr.best_d > self.cfg.track_max_centre_dist:
            return None          # never came near the centre: probably a fragment
        p = tr.best
        p.tray_id = self.n_emitted
        self.n_emitted += 1
        return p

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
                tr = _Track(id=self.n_tracks, last_f=frame_index, last_c=tuple(b["c"]), first_c=tuple(b["c"]))
                self.n_tracks += 1
                self._observe(tr, b, frame_index, full, small, timestamp, W)
                nxt.append(tr)
        self.active = nxt
        return ready

    def _past_best(self, tr, b, W):
        """Early emit only when the best frame is final: the tray has been near the centre and has now
        crossed to the other side of the frame centre from where it came in, by emit_after_px,
        as a same-sized blob (touching trays merge into one bigger blob whose centroid jumps)."""
        cfg = self.cfg
        if tr.emitted or tr.best is None or tr.best_d > cfg.track_max_centre_dist:
            return False
        if not 0.75 < b["area"] / max(tr.best.area, 1) < 1.33:
            return False
        came_from = np.sign(tr.first_c[0] - W / 2) or -1.0
        past = (b["c"][0] - W / 2) * -came_from          # > 0 once on the far side of the centre
        return past > cfg.track_emit_after_px and abs(b["c"][0] - W / 2) > tr.best_d + cfg.track_emit_after_px

    def _observe(self, tr, b, f, full, small, ts, W):
        if tr.n_obs:
            dt = max(f - tr.last_f, 1)
            v = ((b["c"][0] - tr.last_c[0]) / dt, (b["c"][1] - tr.last_c[1]) / dt)
            tr.vel = v if tr.n_obs == 1 else (0.5 * (tr.vel[0] + v[0]), 0.5 * (tr.vel[1] + v[1]))
        tr.n_obs += 1
        tr.last_f, tr.last_c = f, tuple(b["c"])
        x, _, w, _ = b["bbox"]
        m = self.cfg.track_edge_margin
        if x > m and x + w < W - m:
            d = abs(b["c"][0] - W / 2)
            if d < tr.best_d:
                tr.best_d = d
                tr.best = Pick(tray_id=-1, frame_index=f, timestamp=ts, c=tuple(b["c"]), area=b["area"],
                               frame=full.copy(), half=small)

    def flush(self) -> list[Pick]:
        """End of stream: emit trays still being tracked."""
        out = [p for p in (self._finish(tr) for tr in self.active) if p is not None]
        self.active = []
        return out
