"""Inner opening of the tray (where the flat lip drops into the tray) as ONE rigid
rectangle of known size, fitted at full camera resolution.

1. Points: on each side, short lines run inward from the fitted OUTER edge. On each
   line we look for the lip -> wall brightness drop, place it to a fraction of a
   pixel, and keep it only if
     - the line reaches at least a few px of wall before any meat,
     - the lip just before it is clearly brighter than the wall just after it,
     - the image gradient there points straight outward (perpendicular to that side),
       which rejects glare, shadows and meat texture.
2. Fit: a rectangle of the calibrated inner size (W_in x H_in), fitted to the points
   from ALL visible sides at once (trimmed least squares, as for the outer edge).
   A side hidden under meat is then set by the opposite side + the known size,
   not by the outer edge.

All coordinates here are FULL resolution (1456 x 1088) unless noted.
"""
import cv2
import numpy as np

from fit_edges import refine

SIDES = ("left", "right", "top", "bottom")
SIDE_GEOM = dict(left=(0, -1), right=(0, 1), top=(1, -1), bottom=(1, 1))


def prep(full_bgr):
    g = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return g, gx, gy


def edge_points(g, gx, gy, meat_full, cx, cy, th, W, H, side,
                n_lines=120, span=0.36, d_lo=10, d_hi=72, min_drop=0.8, min_contrast=3.0,
                max_angle_deg=25, wall_after=8, min_ratio=0.30, window=None):
    """Returns array (N, 4): x, y, t, d  (image coords, position along side, depth from outer edge)."""
    axis, sign = SIDE_GEOM[side]
    half, other = (W / 2, H) if axis == 0 else (H / 2, W)
    c, s = np.cos(th), np.sin(th)
    depth = d_hi + wall_after + 4
    t = np.linspace(-span, span, n_lines) * other
    d = np.arange(depth, dtype=np.float32)
    dd = (half - d)[:, None] * sign
    uu = np.broadcast_to(dd if axis == 0 else t[None, :], (depth, n_lines))
    vv = np.broadcast_to(dd if axis == 1 else t[None, :], (depth, n_lines))
    x = (cx + uu * c - vv * s).astype(np.float32)
    y = (cy + uu * s + vv * c).astype(np.float32)
    Hh, Ww = g.shape
    inb = (x >= 1) & (x < Ww - 2) & (y >= 1) & (y < Hh - 2)
    prof = cv2.remap(g, x, y, cv2.INTER_LINEAR)
    xi = np.clip(np.round(x).astype(int), 0, Ww - 1)
    yi = np.clip(np.round(y).astype(int), 0, Hh - 1)
    blocked = np.maximum.accumulate(meat_full[yi, xi] | ~inb, axis=0)
    clear_to = np.where(blocked.any(0), np.argmax(blocked, 0), depth)
    der = np.gradient(prof, axis=0)
    # outward normal of this side in image coords
    nu, nv = (sign, 0.0) if axis == 0 else (0.0, sign)
    nx, ny = nu * c - nv * s, nu * s + nv * c
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    out = []
    for j in range(n_lines):
        lo_j, hi_j = d_lo, d_hi
        if window is not None:                      # second pass: only near the predicted edge
            lo_j, hi_j = max(d_lo, int(window[j] - 8)), min(d_hi, int(window[j] + 9))
        top = min(hi_j, clear_to[j] - wall_after)
        if top <= lo_j + 2:
            continue
        seg = der[lo_j:top, j]
        i = int(np.argmin(seg))
        if i == 0 or i == len(seg) - 1:
            continue
        drop = -float(seg[i])
        if drop < min_drop:
            continue
        k = lo_j + i
        lip = prof[max(2, k - 10):k - 3, j].mean()
        wall = prof[k + 3:k + wall_after, j].mean()
        if lip - wall < min_contrast or (lip - wall) / max(lip, 1.0) < min_ratio:
            continue                                 # a real wall is much darker than the lip, not just shading
        # sub-pixel: parabola through the derivative minimum
        y0, y1, y2 = der[k - 1, j], der[k, j], der[k + 1, j]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
        off = float(np.clip(off, -0.5, 0.5))
        dk = k + off
        px = cx + ((half - dk) * sign if axis == 0 else t[j]) * c - ((half - dk) * sign if axis == 1 else t[j]) * s
        py = cy + ((half - dk) * sign if axis == 0 else t[j]) * s + ((half - dk) * sign if axis == 1 else t[j]) * c
        # direction check: brighter side (lip) must be outward
        ix, iy = int(round(px)), int(round(py))
        gvx, gvy = float(gx[iy, ix]), float(gy[iy, ix])
        mag = np.hypot(gvx, gvy)
        if mag < 1e-3 or (gvx * nx + gvy * ny) / mag < cos_max:
            continue
        out.append((px, py, float(t[j]), dk))
    return np.array(out, np.float64).reshape(-1, 4)


def all_points(g, gx, gy, meat_full, cx, cy, th, W, H, inner=None, n_lines=120, span=0.36):
    """inner: a fitted inner rect (cx, cy, th, W, H) to search near (second pass)."""
    pts = {}
    for sd in SIDES:
        win = None
        if inner is not None:
            axis, sign = SIDE_GEOM[sd]
            other = H if axis == 0 else W
            t = np.linspace(-span, span, n_lines) * other
            icx, icy, ith, iW, iH = inner
            c, s = np.cos(th), np.sin(th)
            # predicted depth of the inner edge from the outer edge, along each line
            ox, oy = icx - cx, icy - cy
            du, dv = ox * c + oy * s, -ox * s + oy * c          # inner centre in outer coords
            dth = ith - th
            if axis == 0:
                win = W / 2 - (sign * du + iW / 2) + sign * np.tan(dth) * (t - dv)
            else:
                win = H / 2 - (sign * dv + iH / 2) - sign * np.tan(dth) * (t - du)
        pts[sd] = edge_points(g, gx, gy, meat_full, cx, cy, th, W, H, sd, n_lines=n_lines, span=span, window=win)
    return pts


def initial_inner(cx, cy, th, W, H, lip):
    """Inner rect guess from the outer rect and per-side lip widths (full-res px)."""
    l, r, t_, b = (lip[k] for k in SIDES)
    du, dv = (l - r) / 2, (t_ - b) / 2
    c, s = np.cos(th), np.sin(th)
    return cx + du * c - dv * s, cy + du * s + dv * c, W - l - r, H - t_ - b


def fit_inner(pts, init, W_in=None, H_in=None, min_side=10):
    """pts: dict side -> (N,4). init: (cx, cy, th, W0, H0).
    With W_in/H_in given the size is fixed; otherwise it is fitted too."""
    P = np.vstack([p[:, :2] for p in pts.values() if len(p)]) if any(len(p) for p in pts.values()) else np.zeros((0, 2))
    cx, cy, th, W0, H0 = init
    if W_in is not None:
        W0, H0 = W_in, H_in
    if len(P) < 20:
        return None
    cx, cy, th, W, H, info = refine(P.astype(np.float32), cx, cy, th, W0, H0,
                                    fit_size=W_in is None, trim=0.15, min_side=min_side)
    return dict(cx=cx, cy=cy, th=th, W=W, H=H, rms=info["rms"], side_counts=info["side_counts"], n=info["n_pts"])
