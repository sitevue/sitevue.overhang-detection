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
        ranked = sam.segment_all(half, blobs, lab, sm, j) if hasattr(sam, "segment_all") else [sam.segment(half, blobs, lab, sm, j)]
        payload["timings_ms"]["sam_model"] = (time.perf_counter() - t1) * 1000
        if ranked and ranked[0][0] is not None:
            payload.update(tray=ranked[0][0], own=lab == blobs[j]["id"], sam_score=ranked[0][1],
                           alt_trays=[(m, s) for m, s in ranked[1:]])
    payload["timings_ms"]["sam_total"] = (time.perf_counter() - t0) * 1000
    return payload


GEOMETRY = ("outline_check", "edge_of_view")    # problems another SAM mask or another frame can fix
HAND = "hand_over_tray"                          # a worker's hand: another frame can fix it (not another mask)
# A retried box (another SAM mask, another frame) is a second guess: it is used only if it sits on the meat.
# On 10-29 the correct rescues had 0-6 % of the tray's meat outside the box (all readable trays: median 1.2 %,
# 99th percentile 8.7 %); the three wrong ones 22-53 % (a box that passes the outline check but is not on the
# tray, so "past green" became 58,000-337,000 px). Above this the tray stays unreadable.
RETRY_MAX_OUTSIDE_PCT = 15.0


def retry_ok(r: dict) -> bool:
    return bool(r.get("reliable")) and r.get("outside_pct_of_meat", 0.0) <= RETRY_MAX_OUTSIDE_PCT


def measure(payload: dict, cfg: Config | None = None, rect: Rectifier | None = None, debug_dir: str | None = None,
            _parts: bool = False) -> dict:
    """CPU stages for one tray (runs in a worker process). Returns the TrayResult dict.
    _parts=True (two-view mode): returns (result with the per-half numbers in "_halves", corrected frame, geometry).

    SAM returns three masks. If the box from the first fails only the outline / edge-of-view checks (often
    SAM's first mask covers some belt or the next tray, so the box lands in the wrong place), the other
    two are tried and the first that passes is used (`sam_mask` = its rank). Same frame, no extra GPU work."""
    alts = payload.get("alt_trays") or []
    first = out = _measure_one(payload, cfg, rect, debug_dir, _parts and True)
    res = out[0] if _parts else out
    if not alts or not res.get("unreliable_reasons") or not set(res["unreliable_reasons"]) <= set(GEOMETRY):
        return first
    for k, (m, sc) in enumerate(alts, 1):
        alt = dict(payload, tray=m, sam_score=sc)
        out = _measure_one(alt, cfg, rect, "", _parts)          # no review image for a try that may fail
        r = out[0] if _parts else out
        if retry_ok(r):
            if not _parts and (debug_dir or (debug_dir is None and _W.get("debug_dir"))):
                out = r = _measure_one(alt, cfg, rect, debug_dir, _parts)   # again, to write its review image
            r["sam_mask"] = k
            return out
    return first


def _measure_one(payload: dict, cfg: Config | None = None, rect: Rectifier | None = None, debug_dir: str | None = None,
                 _parts: bool = False) -> dict:
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
        return (res, None, None) if _parts else res
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

    halves = num.pop("halves")
    cls = overhang_class(num["lip_cover_pct"], num["past_green_px"], cfg)
    reasons = [r for r, on in (("hand_over_tray", num["occluded"]), ("edge_of_view", num["edge_of_view"]),
                               ("outline_check", not ol["outline_ok"])) if on]
    res.update(overhang_class=cls, light=light_for(cls, cfg), reliable=not reasons, unreliable_reasons=reasons, **num)
    res.update(outline_ok=ol["outline_ok"], tray_spill_frac=round(ol["tray_spill_frac"], 3),
               belt_inside_frac=round(ol["belt_inside_frac"], 4), other_meat_frac=round(ol["other_meat_frac"], 4),
               outer=dict(cx=round(outer_full[0], 1), cy=round(outer_full[1], 1), deg=round(float(np.rad2deg(ol["th"])), 2),
                          w=round(outer_full[3], 1), h=round(outer_full[4], 1)),
               inner_corners=np.round(corners.astype(np.float64), 1).tolist(), inner_walls=how)
    if _parts:
        res["_halves"] = halves
        res["timings_ms"] = {k: round(v, 1) for k, v in tm.items()}
        return res, full, geom
    if debug_dir:
        from .render import draw
        t = time.perf_counter()
        cv2.imwrite(os.path.join(debug_dir, f"tray_{res['tray_id']:05d}.jpg"), draw(full, geom, res), [cv2.IMWRITE_JPEG_QUALITY, 88])
        tm["debug_image"] = (time.perf_counter() - t) * 1000
    res["timings_ms"] = {k: round(v, 1) for k, v in tm.items()}
    return res


REASONS = ("hand_over_tray", "edge_of_view", "outline_check")


def measure_pair(pay_left: dict, pay_right: dict, cfg: Config | None = None, rect: Rectifier | None = None,
                 debug_dir: str | None = None) -> dict:
    """Two-view mode (`side_views_px`): the tray's left half from one frame, its right half from another.
    Each half is measured where the camera angle shifts it least. The result is the right-half view's
    result with the overhang numbers replaced by the combined ones; a problem in either view (hand, edge
    of view, outline check) makes the tray unreliable. If one view has no tray mask the other is used alone."""
    cfg = cfg or _W["cfg"]
    debug_dir = debug_dir if debug_dir is not None else _W.get("debug_dir")
    (a, fa, ga), (b, fb, gb) = (measure(p, cfg, rect, None, _parts=True) for p in (pay_left, pay_right))
    views = dict(left_half=dict(frame_index=a["frame_index"], cx=a.get("outer", {}).get("cx")),
                 right_half=dict(frame_index=b["frame_index"], cx=b.get("outer", {}).get("cx")))
    if "_halves" not in a or "_halves" not in b:
        one = b if "_halves" in b else a
        one.pop("_halves", None)
        one["views"] = dict(views, combined=False)
        return one
    # only one photo has a problem (a worker's hand over the tray in that photo, the tray cut by the frame
    # edge, or a failed outline check, e.g. SAM's mask spilling at x ~1050): read the whole tray from the
    # other, clean photo instead of giving up. On 10-29 the hand alone made 18 trays unreadable this way.
    bad_a, bad_b = list(a["unreliable_reasons"]), list(b["unreliable_reasons"])
    if bool(bad_a) != bool(bad_b):
        one, other_bad, key = (b, bad_a, "left_view_failed") if bad_a else (a, bad_b, "right_view_failed")
        one.pop("_halves", None)
        one["views"] = dict(views, combined=False, **{key: other_bad})
        return one
    L, R = a.pop("_halves")["left"], b.pop("_halves")["right"]
    band, band_meat = L["band_px"] + R["band_px"], L["band_meat_px"] + R["band_meat_px"]
    lip, past, meat = L["lip_px"] + R["lip_px"], L["past_green_px"] + R["past_green_px"], L["meat_px"] + R["meat_px"]
    res = dict(b)
    res.update(lip_cover_pct=round(100.0 * band_meat / max(band, 1), 2), outside_px=lip + past, past_green_px=past,
               lip_px=lip, outside_pct_of_meat=round(100.0 * (lip + past) / max(meat, 1), 2), meat_px=meat,
               lip_band_px=band, lip_band_meat_px=band_meat,
               outside_by_side_px={k: L["side"][k] + R["side"][k] for k in L["side"]},
               occluded=bool(a["occluded"] or b["occluded"]), edge_of_view=bool(a["edge_of_view"] or b["edge_of_view"]),
               outline_ok=bool(a["outline_ok"] and b["outline_ok"]),
               views=dict(views, combined=True))
    reasons = [r for r in REASONS if r in a["unreliable_reasons"] or r in b["unreliable_reasons"]]
    cls = overhang_class(res["lip_cover_pct"], res["past_green_px"], cfg)
    res.update(overhang_class=cls, light=light_for(cls, cfg), reliable=not reasons, unreliable_reasons=reasons)
    tm = {k: round(a["timings_ms"].get(k, 0) + b["timings_ms"].get(k, 0), 1) for k in set(a["timings_ms"]) | set(b["timings_ms"])}
    if debug_dir:
        from .render import draw
        t = time.perf_counter()
        ia, ib = draw(fa, ga, a), draw(fb, gb, res)
        h = max(ia.shape[0], ib.shape[0])
        img = np.hstack([cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in (ia, ib)])
        cv2.imwrite(os.path.join(debug_dir, f"tray_{res['tray_id']:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tm["debug_image"] = round((time.perf_counter() - t) * 1000, 1)
    res["timings_ms"] = tm
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
        backups = {}             # tray_id -> [other frames], measured instead if the tray is unreadable only for its geometry
        firsts = {}              # tray_id -> the first result, kept if no other frame reads the tray
        tries = {}               # tray_id -> other frames measured so far
        submitted = 0

        def submit(pick):
            nonlocal submitted
            if max_trays is not None and submitted >= max_trays:
                return
            submitted += 1
            if not pick.in_zone:      # counted, not measured: never completely inside the pick zone
                done[pick.tray_id] = dict(tray_id=pick.tray_id, frame_index=pick.frame_index, timestamp=pick.timestamp,
                                          overhang_class="unknown", light=None, reliable=False,
                                          unreliable_reasons=["not_in_pick_zone"])
                return
            views = [v for v in (pick.views or []) if v is not None]
            if pick.views is not None and len(views) == 2:      # two-view mode: each half from its own frame
                for v in views:
                    v.tray_id = pick.tray_id
                fn, args = measure_pair, (prepare(views[0], self.sam, cfg), prepare(views[1], self.sam, cfg))
            else:                                               # one frame (or only one view was seen)
                fn, args = measure, (prepare(pick, self.sam, cfg),)
            others = ([pick.backup] if pick.backup is not None else []) + list(pick.spares or [])
            if others:
                for o in others:
                    o.tray_id = pick.tray_id
                backups[pick.tray_id] = others
            if pool is None:
                done[pick.tray_id] = retry(pick.tray_id, fn(*args))
            else:
                futures[pick.tray_id] = pool.submit(fn, *args)

        def retry(k, res):
            """The tray is unreadable for its outline, the frame edge or a worker's hand: measure it on the next
            other frame (pick_x_px / dynamic: the backup frame first, then the spares, nearest first), until one
            reads it with its box on the meat. Up to retry_frames other frames, or retry_frames_hand when the
            first result had a hand. If none reads it, the first result stands."""
            if k in firsts and res.get("reliable") and not retry_ok(res):
                res = dict(res, reliable=False, unreliable_reasons=["outline_check"])   # retried box not on the meat
            fixable = set(GEOMETRY) | {HAND}
            if res.get("reliable") or not res.get("unreliable_reasons") or not set(res["unreliable_reasons"]) <= fixable:
                if k in firsts and not res.get("reliable"):
                    pass                                   # a later frame had another problem: keep trying below
                else:
                    backups.pop(k, None)
                    tries.pop(k, None)
                    if k in firsts and res.get("reliable"):
                        firsts.pop(k)
                        res["frame_retry"] = True      # read on another frame
                        return res
                    return firsts.pop(k) if k in firsts else res
            first = firsts.setdefault(k, res)
            limit = cfg.retry_frames_hand if HAND in first.get("unreliable_reasons", []) else cfg.retry_frames
            rest = backups.get(k) or []
            if not rest or tries.get(k, 0) >= limit:
                backups.pop(k, None)
                tries.pop(k, None)
                return firsts.pop(k)
            tries[k] = tries.get(k, 0) + 1
            b = rest.pop(0)
            payload = prepare(b, self.sam, cfg)
            if pool is None:
                return retry(k, measure(payload))
            futures[k] = pool.submit(measure, payload)
            return None

        def collect(block):
            nonlocal next_out
            while True:
                for k in list(futures):
                    f = futures[k]
                    if block or f.done():
                        del futures[k]
                        try:
                            r = retry(k, f.result())
                        except Exception as e:     # one bad tray must not stop the line
                            r = dict(tray_id=k, overhang_class="unknown", light=None, reliable=False,
                                     unreliable_reasons=[f"error: {type(e).__name__}: {e}"])
                        if r is not None:
                            done[k] = r
                if not (block and futures):
                    break
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
