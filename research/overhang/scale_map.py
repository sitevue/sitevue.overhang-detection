"""Apparent scale across the frame, from the belt rollers (a regular grid on the belt).

Roller centres are found as yellow blobs. Neighbours in the same row give the horizontal
spacing, neighbours in the same column (adjacent rows) the vertical spacing, each at the
midpoint between the two rollers. A smooth surface (quadratic in x, linear in y) is fitted to
each, and scale = local spacing / spacing at a reference point, so 1.00 = as big as at the
reference and 1.03 = things look 3% bigger there.
usage: python overhang/scale_map.py OUT_JSON FRAME.png [FRAME.png ...]   (empty-belt frames)
"""
import json, sys
import cv2
import numpy as np


def roller_pairs(path):
    f = cv2.imread(path); h, s, v = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
    yel = ((h > 18) & (h < 40) & (s > 100) & (v > 120)).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(yel, 8)
    ok = (st[:, 4] > 150) & (st[:, 4] < 3000) & (st[:, 2] < 90) & (st[:, 3] < 70)
    c = cen[ok]
    hx, vy = [], []
    for i, (x, y) in enumerate(c):
        d = c - (x, y)
        # right-hand neighbour in the same row
        m = (d[:, 0] > 40) & (d[:, 0] < 95) & (np.abs(d[:, 1]) < 15)
        if m.any():
            j = np.argmin(np.where(m, d[:, 0], 1e9)); hx.append((x + d[j, 0] / 2, y + d[j, 1] / 2, np.hypot(*d[j])))
        # neighbour below in the same column (next roller row in the same band)
        m = (d[:, 1] > 35) & (d[:, 1] < 85) & (np.abs(d[:, 0]) < 15)
        if m.any():
            j = np.argmin(np.where(m, d[:, 1], 1e9)); vy.append((x + d[j, 0] / 2, y + d[j, 1] / 2, np.hypot(*d[j])))
    return np.array(hx), np.array(vy)


def design(x, y):
    x = (x - 728) / 728.0; y = (y - 544) / 544.0
    return np.stack([np.ones_like(x), x, x * x, y, x * y], 1)


def fit(pairs):
    A = design(pairs[:, 0], pairs[:, 1]); b = pairs[:, 2]
    keep = np.ones(len(b), bool)
    for _ in range(4):
        c = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        r = b - A @ c; keep = np.abs(r) < max(1.0, 3 * np.median(np.abs(r[keep])))
    return c, float(np.sqrt(np.mean(r[keep] ** 2))), int(keep.sum())


def main():
    out = sys.argv[1]; H, V = [], []
    for p in sys.argv[2:]:
        a, b = roller_pairs(p); H.append(a); V.append(b)
    H, V = np.vstack(H), np.vstack(V)
    ch, rh, nh = fit(H); cv_, rv, nv = fit(V)
    ref = (728.0, 544.0)
    sh0 = float((design(np.array([ref[0]]), np.array([ref[1]])) @ ch)[0]); sv0 = float((design(np.array([ref[0]]), np.array([ref[1]])) @ cv_)[0])
    json.dump(dict(coef_h=ch.tolist(), coef_v=cv_.tolist(), ref=ref, pitch_h_ref=sh0, pitch_v_ref=sv0), open(out, "w"), indent=1)
    print(f"horizontal roller spacing: {nh} pairs, fit rms {rh:.2f} px, at frame centre {sh0:.2f} px")
    print(f"vertical   roller spacing: {nv} pairs, fit rms {rv:.2f} px, at frame centre {sv0:.2f} px")
    xs = np.array([150, 450, 728, 1000, 1300]); ys = np.array([250, 544, 850])
    for name, c, p0 in (("horizontal", ch, sh0), ("vertical", cv_, sv0)):
        print(f"  {name} scale (rows y=250/544/850, columns x=150..1300):")
        for y in ys:
            print("   y=%4d  " % y + "  ".join(f"{float((design(np.array([x]), np.array([y])) @ c)[0]) / p0:5.3f}" for x in xs))


if __name__ == "__main__":
    main()
