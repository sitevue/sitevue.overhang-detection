"""Inner opening as four independent wall lines (a quadrilateral), not a rigid rectangle.

- A wall that is well seen (enough trusted edge points spread along it) gets its own
  robust straight line, at its own angle.
- A wall hidden by meat is drawn PARALLEL to its opposite wall, offset by the known
  inner width/height (measured on clean trays).
- If both opposite walls are hidden, both come from the rigid fit (inner_fit.json).
Trusted points = edge points at least NEAR px from meat, or within AGREE px of the
rigid fit (same guard as guard_inner.py). Coordinates: full resolution.
Writes out/run1/inner_quad.json: per tray the 4 corners (TL, TR, BR, BL) and how each wall was set.
"""
import os as _os
RUN = _os.environ.get("RUN", "out/run1")
import json, sys
import cv2
import numpy as np

sys.path.insert(0, "overhang")
from measure_full import tray_uv

NEAR, AGREE = 25.0, 3.0
MIN_PTS, MIN_SPREAD = 40, 250.0
SIDES = ("left", "right", "top", "bottom")


def robust_line(p, vertical):
    """Fit x = a + b*y (vertical walls) or y = a + b*x (horizontal walls); iterative trimming."""
    s, t = (p[:, 1], p[:, 0]) if vertical else (p[:, 0], p[:, 1])
    A = np.stack([np.ones_like(s), s], 1)
    c = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(4):
        r = t - A @ c
        keep = np.abs(r) < max(1.5, 2.5 * np.median(np.abs(r)))
        c = np.linalg.lstsq(A[keep], t[keep], rcond=None)[0]
    rms = float(np.sqrt(np.mean((t - A @ c)[keep] ** 2)))
    return float(c[0]), float(c[1]), rms, int(keep.sum())


def as_normal_form(a, b, vertical):
    """Line as n.x = d with unit normal n."""
    if vertical:          # x - b*y = a
        n = np.array([1.0, -b])
    else:                 # -b*x + y = a
        n = np.array([-b, 1.0])
    k = np.linalg.norm(n)
    return n / k, a / k


def rect_lines(q):
    """Normal-form lines of the rigid inner rectangle, per side."""
    c, s = np.cos(q["th"]), np.sin(q["th"])
    ex, ey = np.array([c, s]), np.array([-s, c])          # tray u and v axes in the image
    C = np.array([q["cx"], q["cy"]])
    L = {}
    for side, (axis, sign) in dict(left=(ex, -1), right=(ex, 1), top=(ey, -1), bottom=(ey, 1)).items():
        half = q["W"] / 2 if side in ("left", "right") else q["H"] / 2
        n = axis
        L[side] = (n, float(n @ C + sign * half))
    return L


def shift_parallel(line, dist, toward):
    """Copy of a line moved by `dist` px along its normal, toward the point `toward` side sign."""
    n, d = line
    return n, d + toward * dist


def intersect(l1, l2):
    A = np.array([l1[0], l2[0]]); b = np.array([l1[1], l2[1]])
    return np.linalg.solve(A, b)


def main():
    size = json.load(open(f"{RUN}/inner_size.json")); W_in, H_in = size["W_in"], size["H_in"]
    P = json.load(open(f"{RUN}/inner_points.json")); F = json.load(open(f"{RUN}/inner_fit.json"))
    out, how_count = {}, {"measured": 0, "from_opposite": 0, "rigid": 0}
    for k, q in F.items():
        z = np.load(f"{RUN}/cache/{int(k):03d}.npz")
        red = cv2.resize(z["red"].astype(np.uint8), (1456, 1088), interpolation=cv2.INTER_NEAREST)
        dist = cv2.distanceTransform((1 - red).astype(np.uint8), cv2.DIST_L2, 3)
        R = rect_lines(q)
        C = np.array([q["cx"], q["cy"]])
        L, how, info = {}, {}, {}
        for sd in SIDES:
            p = np.array(P[k][sd]).reshape(-1, 4)
            if len(p):
                near = dist[np.clip(p[:, 1].astype(int), 0, 1087), np.clip(p[:, 0].astype(int), 0, 1455)] < NEAR
                u, v = tray_uv(p[:, 0], p[:, 1], q["cx"], q["cy"], q["th"])
                r = (q["W"] / 2 - np.abs(u)) if sd in ("left", "right") else (q["H"] / 2 - np.abs(v))
                p = p[~near | (np.abs(r) <= AGREE)]
            vertical = sd in ("left", "right")
            spread = np.ptp(p[:, 1] if vertical else p[:, 0]) if len(p) else 0.0
            if len(p) >= MIN_PTS and spread >= MIN_SPREAD:
                a, b, rms, n = robust_line(p[:, :2], vertical)
                ln = as_normal_form(a, b, vertical)
                # keep the normal pointing the same way as the rigid one (outward sign convention)
                if ln[0] @ R[sd][0] < 0:
                    ln = (-ln[0], -ln[1])
                L[sd], how[sd] = ln, "measured"
                info[sd] = dict(n=n, rms=round(rms, 2))
        for a_, b_, size_ in (("left", "right", W_in), ("top", "bottom", H_in)):
            for this, opp in ((a_, b_), (b_, a_)):
                if this in L or opp not in L:
                    continue
                n, d = L[opp]
                # move from the opposite wall across the opening by the known size, toward the centre side
                sgn = -1.0 if (n @ C - d) < 0 else 1.0       # which way the centre lies from that wall
                L[this], how[this] = (n, d + sgn * size_), "from_opposite"
            for this in (a_, b_):
                if this not in L:
                    L[this], how[this] = R[this], "rigid"
        corners = [intersect(L["top"], L["left"]), intersect(L["top"], L["right"]),
                   intersect(L["bottom"], L["right"]), intersect(L["bottom"], L["left"])]
        for h in how.values():
            how_count[h] += 1
        out[k] = dict(corners=[c.round(2).tolist() for c in corners], how=how, info=info)
    json.dump(out, open(f"{RUN}/inner_quad.json", "w"))
    print("walls set by:", how_count)


if __name__ == "__main__":
    main()
