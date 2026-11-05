"""Rebuild a merged run's corrected frame/mask cache from the session videos and the kept SAM masks,
without running SAM again.

usage: python overhang/rebuild_frames.py SCALE_MAP.json SESSIONS_DIR MASKS_DIR RUN
  MASKS_DIR/<session8>/{picks.json, NNN.npz}   raw (uncorrected) SAM masks per session tray
  RUN/picks.json                               merged picks (global idx order, with session + session_frame)
Writes RUN/cache/<g>.npz (corrected masks) and RUN/cache_full/<g>.png (corrected full-res frame).
"""
import collections, glob, json, os, sys
from multiprocessing import Pool
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rectify import Rectifier

MAP, SDIR, MDIR, RUN = sys.argv[1:5]


def one_session(item):
    s8, trays = item                     # trays: [(global idx, session frame)]
    R = Rectifier(MAP)
    sp = json.load(open(os.path.join(MDIR, s8, "picks.json")))
    k_of = {p["frame"]: k for k, p in enumerate(sp)}
    video = glob.glob(os.path.join(SDIR, s8 + "*.mp4"))[0]
    cap = cv2.VideoCapture(video); pos, full, done = 0, None, 0
    for g, f in sorted(trays, key=lambda t: t[1]):
        if full is None or f != pos - 1:
            while pos < f:
                cap.grab(); pos += 1
            ok, full = cap.read(); pos += 1
            if not ok:
                raise IOError(f"{s8}: could not read frame {f}")
        cv2.imwrite(os.path.join(RUN, "cache_full", f"{g:03d}.png"), R.warp_full(full), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        z = dict(np.load(os.path.join(MDIR, s8, f"{k_of[f]:03d}.npz")))
        for key in ("tray", "own", "red"):
            z[key] = R.warp_half(z[key].astype(np.uint8), nearest=True).astype(bool)
        np.savez_compressed(os.path.join(RUN, "cache", f"{g:03d}.npz"), **z)
        done += 1
    return s8, done


if __name__ == "__main__":
    for sub in ("cache", "cache_full"):
        os.makedirs(os.path.join(RUN, sub), exist_ok=True)
    by = collections.defaultdict(list)
    for g, p in enumerate(json.load(open(os.path.join(RUN, "picks.json")))):
        by[p["session"][:8]].append((g, p["session_frame"]))
    with Pool(min(len(by), 14)) as pool:
        for s8, n in pool.imap_unordered(one_session, sorted(by.items(), key=lambda kv: -len(kv[1]))):
            print("rebuilt", s8, n, flush=True)
