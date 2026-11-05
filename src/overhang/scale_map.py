"""Build the camera-angle correction (scale map) from frames of the empty belt.

Roller centres are found as yellow blobs. Neighbours in the same row give the horizontal
spacing, neighbours in the same column the vertical spacing, each at the midpoint between the
two rollers. A smooth surface (quadratic in x, linear in y) is fitted to each; scale = local
spacing / spacing at the frame centre.  Needed once per camera position (re-run if it moves).
"""
import json

import cv2
import numpy as np

from .rectify import design


def roller_pairs(path):
    f = cv2.imread(path)
    h, s, v = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
    yel = ((h > 18) & (h < 40) & (s > 100) & (v > 120)).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, st, cen = cv2.connectedComponentsWithStats(yel, 8)
    ok = (st[:, 4] > 150) & (st[:, 4] < 3000) & (st[:, 2] < 90) & (st[:, 3] < 70)
    c = cen[ok]
    hx, vy = [], []
    for x, y in c:
        d = c - (x, y)
        m = (d[:, 0] > 40) & (d[:, 0] < 95) & (np.abs(d[:, 1]) < 15)
        if m.any():
            j = np.argmin(np.where(m, d[:, 0], 1e9))
            hx.append((x + d[j, 0] / 2, y + d[j, 1] / 2, np.hypot(*d[j])))
        m = (d[:, 1] > 35) & (d[:, 1] < 85) & (np.abs(d[:, 0]) < 15)
        if m.any():
            j = np.argmin(np.where(m, d[:, 1], 1e9))
            vy.append((x + d[j, 0] / 2, y + d[j, 1] / 2, np.hypot(*d[j])))
    return np.array(hx), np.array(vy), f.shape[1], f.shape[0]


def _fit(pairs, ref):
    A = design(pairs[:, 0], pairs[:, 1], ref)
    b = pairs[:, 2]
    keep = np.ones(len(b), bool)
    for _ in range(4):
        c = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        r = b - A @ c
        keep = np.abs(r) < max(1.0, 3 * np.median(np.abs(r[keep])))
    return c, float(np.sqrt(np.mean(r[keep] ** 2))), int(keep.sum())


def build(out, frames):
    H, V = [], []
    for p in frames:
        a, b, fw, fh = roller_pairs(p)
        H.append(a)
        V.append(b)
    ref = (fw / 2.0, fh / 2.0)
    H, V = np.vstack(H), np.vstack(V)
    ch, rh, nh = _fit(H, ref)
    cv_, rv, nv = _fit(V, ref)
    r0 = design(np.array([ref[0]]), np.array([ref[1]]), ref)
    sh0, sv0 = float((r0 @ ch)[0]), float((r0 @ cv_)[0])
    with open(out, "w") as fh_:
        json.dump(dict(coef_h=ch.tolist(), coef_v=cv_.tolist(), ref=ref, pitch_h_ref=sh0, pitch_v_ref=sv0), fh_, indent=1)
    print(f"horizontal roller spacing: {nh} pairs, fit rms {rh:.2f} px, {sh0:.2f} px at the frame centre")
    print(f"vertical   roller spacing: {nv} pairs, fit rms {rv:.2f} px, {sv0:.2f} px at the frame centre")
    print(f"wrote {out}; point camera.scale_map in the config at it")
