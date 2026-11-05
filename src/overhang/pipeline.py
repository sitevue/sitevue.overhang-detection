"""End-to-end, per-tray pipeline for a live camera or a video file.

    frames ─▶ TrayTracker ─▶ (one frame per tray) ─▶ SAM tray mask (GPU, main process)
                                                  ─▶ worker processes (CPU):
                                                       camera-angle correction
                                                       outer outline (green)
                                                       inner opening (magenta)
                                                       measure + classify
                                                  ─▶ TrayResult (dict), in tray order

`OverhangPipeline.run(source)` yields one result dict per tray; see docs/output.md for the fields.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .classify import light_for, overhang_class
from .color import masks, meat_blobs, smooth_mask
from .config import Config
from .inner import fit_opening
from .measure import measure_tray
from .outline import fit_outline
from .rectify import Rectifier
from .tracker import TrayTracker

_W: dict = {}            # per-worker state: config, rectifier, debug dir


def _init_worker(cfg: Config, debug_dir: str | None):
    _W["cfg"] = cfg
    _W["rect"] = Rectifier(cfg.scale_map, cfg.frame_w, cfg.frame_h)
    _W["debug_dir"] = debug_dir


def prepare(pick, sam, cfg: Config) -> dict:
    """Main process: SAM tray mask for one picked tray. Returns the payload for measure()."""
    t0 = time.perf_counter()
    half = pick.half
    red, _ = masks(half)
    sm = smooth_mask(half, red)
    lab, blobs = meat_blobs(red, 4000)
    payload = dict(tray_id=pick.tray_id, frame_index=pick.frame_index, timestamp=pick.timestamp,
                   full=pick.frame, half=half, red=red, tray=None, own=None, sam_score=0.0, timings_ms={})
    if blobs:
        j = int(np.argmin([np.hypot(b["c"][0] - pick.c[0], b["c"][1] - pick.c[1]) for b in blobs]))
        t1 = time.perf_counter()
        tray, score = sam.segment(half, blobs, lab, sm, j)
        payload["timings_ms"]["sam_model"] = (time.perf_counter() - t1) * 1000
        payload.update(tray=tray, own=lab == blobs[j]["id"], sam_score=score)
    payload["timings_ms"]["sam_total"] = (time.perf_counter() - t0) * 1000
    return payload


def measure(payload: dict, cfg: Config | None = None, rect: Rectifier | None = None, debug_dir: str | None = None) -> dict:
    """CPU stages for one tray (runs in a worker process). Returns the TrayResult dict."""
    cfg = cfg or _W["cfg"]
    rect = rect or _W["rect"]
    debug_dir = debug_dir if debug_dir is not None else _W.get("debug_dir")
    tm = dict(payload["timings_ms"])
    res = dict(tray_id=payload["tray_id"], frame_index=payload["frame_index"], timestamp=payload["timestamp"],
               overhang_class="unknown", light=None, reliable=False, unreliable_reasons=[],
               sam_score=round(float(payload["sam_score"]), 3))
    tray = payload["tray"]
    if tray is None or not tray.any():
        res["unreliable_reasons"] = ["no_tray_mask"]
        res["timings_ms"] = {k: round(v, 1) for k, v in tm.items()}
        return res
    t = time.perf_counter()
    full = rect.warp_full(payload["full"])
    half = rect.warp_half(payload["half"])
    tray_c, own_c, red_c = (rect.warp_mask_half(m) for m in (tray, payload["own"], payload["red"]))
    tm["correct"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ol = fit_outline(tray_c, own_c, red_c, half, cfg.tray_w, cfg.tray_h)
    tm["outline"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    outer_full = (ol["cx"] * 2, ol["cy"] * 2, ol["th"], cfg.tray_w * 2, cfg.tray_h * 2)
    corners, how, _ = fit_opening(full, red_c, outer_full, cfg)
    tm["inner"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    num, geom = measure_tray(full, own_c, ol, corners, how, cfg)
    tm["measure"] = (time.perf_counter() - t) * 1000

    cls = overhang_class(num["lip_cover_pct"], num["past_green_px"], cfg)
    reasons = [r for r, on in (("hand_over_tray", num["occluded"]), ("edge_of_view", num["edge_of_view"]),
                               ("outline_check", not ol["outline_ok"])) if on]
    res.update(overhang_class=cls, light=light_for(cls, cfg), reliable=not reasons, unreliable_reasons=reasons, **num)
    res.update(outline_ok=ol["outline_ok"], tray_spill_frac=round(ol["tray_spill_frac"], 3),
               belt_inside_frac=round(ol["belt_inside_frac"], 4),
               outer=dict(cx=round(outer_full[0], 1), cy=round(outer_full[1], 1), deg=round(float(np.rad2deg(ol["th"])), 2),
                          w=round(outer_full[3], 1), h=round(outer_full[4], 1)),
               inner_corners=np.round(corners.astype(np.float64), 1).tolist(), inner_walls=how)
    if debug_dir:
        from .render import draw
        t = time.perf_counter()
        cv2.imwrite(os.path.join(debug_dir, f"tray_{res['tray_id']:05d}.jpg"), draw(full, geom, res), [cv2.IMWRITE_JPEG_QUALITY, 88])
        tm["debug_image"] = (time.perf_counter() - t) * 1000
    res["timings_ms"] = {k: round(v, 1) for k, v in tm.items()}
    return res


def open_source(source):
    """Video file path, RTSP/HTTP URL, or camera index ("0")."""
    src = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"could not open video source {source!r}")
    return cap, isinstance(src, str) and os.path.exists(src)


class OverhangPipeline:
    def __init__(self, cfg: Config, workers: int | None = None, debug_dir: str | None = None, sam=None):
        self.cfg = cfg
        self.workers = cfg.workers if workers is None else workers
        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        if sam is None:
            from .sam import SamTray
            sam = SamTray(cfg.sam_model, cfg.sam_device, cfg.sam_dtype)
        self.sam = sam
        self.stats = dict(frames=0, track_ms=0.0, trays=0)

    def run(self, source, realtime: bool = False, max_trays: int | None = None, max_frames: int | None = None):
        """Yield one result dict per tray, in tray order. realtime=True paces a video file at camera fps."""
        cfg = self.cfg
        cap, is_file = open_source(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
        tracker = TrayTracker(cfg)
        pool = None
        if self.workers > 0:
            pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                                       initializer=_init_worker, initargs=(cfg, self.debug_dir))
        else:
            _init_worker(cfg, self.debug_dir)
        futures, done, next_out = {}, {}, 0
        submitted = 0

        def submit(pick):
            nonlocal submitted
            if max_trays is not None and submitted >= max_trays:
                return
            payload = prepare(pick, self.sam, cfg)
            submitted += 1
            if pool is None:
                done[pick.tray_id] = measure(payload)
            else:
                futures[pick.tray_id] = pool.submit(measure, payload)

        def collect(block):
            nonlocal next_out
            for k in list(futures):
                f = futures[k]
                if block or f.done():
                    try:
                        done[k] = f.result()
                    except Exception as e:     # one bad tray must not stop the line
                        done[k] = dict(tray_id=k, overhang_class="unknown", light=None, reliable=False,
                                       unreliable_reasons=[f"error: {type(e).__name__}: {e}"])
                    del futures[k]
            while next_out in done:
                yield done.pop(next_out)
                next_out += 1

        t_start = time.time()
        i = 0
        try:
            while max_frames is None or i < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if realtime and is_file:
                    time.sleep(max(0.0, t_start + i / fps - time.time()))
                ts = i / fps if is_file else time.time()
                t = time.perf_counter()
                picks = tracker.update(i, frame, ts)
                self.stats["track_ms"] += (time.perf_counter() - t) * 1000
                self.stats["frames"] += 1
                for p in picks:
                    submit(p)
                yield from collect(block=False)
                i += 1
                if max_trays is not None and submitted >= max_trays and not futures:
                    break
            for p in tracker.flush():
                submit(p)
            yield from collect(block=True)
        finally:
            cap.release()
            if pool is not None:
                pool.shutdown(cancel_futures=True)
            self.stats["trays"] = submitted
