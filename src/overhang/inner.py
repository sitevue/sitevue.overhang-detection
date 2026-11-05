"""Inner opening of the tray (magenta line): where the flat lip drops into the tray.

Per tray, at FULL resolution on the camera-corrected frame:
1. Edge points: short lines run inward from the outer outline on every side. On each line we look
   for the lip -> wall brightness drop, place it to sub-pixel, and keep it only if the line reaches
   wall before any meat, the wall is clearly darker than the lip (ratio >= 0.30), and the image
   gradient there points straight outward (rejects glare, shadows and meat texture).
2. Rigid fit: a rectangle of the calibrated inner size fitted to the points of all visible sides.
   Second pass: search again only +-8 px around that fit, and refit.
3. Guard: edge points within 25 px of meat (meat shadows look like the lip edge) must agree with
   the fit from clean points within 3 px, or they are dropped.
4. Four walls: each well-seen wall gets its own straight line; a wall hidden by meat is placed
   parallel to the opposite wall at the known inner width/height.
5. Shape: the quadrilateral is shifted along the tray's length and its corners rounded
   (both calibrated on hand-marked empty trays).
"""
import cv2
import numpy as np

from .color import tray_uv
from .outline import refine

SIDES = ("left", "right", "top", "bottom")
SIDE_GEOM = dict(left=(0, -1), right=(0, 1), top=(1, -1), bottom=(1, 1))
NEAR, AGREE = 25.0, 3.0              # guard: px from meat that makes a point suspect / allowed disagreement
MIN_PTS, MIN_SPREAD = 40, 250.0      # a wall is "measured" with this many trusted points spread this far


# ---------------------------------------------------------------- 1. edge points
def prep(full_bgr):
    g = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    return g, cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def edge_points(g, gx, gy, meat_full, cx, cy, th, W, H, side, n_lines=120, span=0.36, d_lo=10, d_hi=72,
                min_drop=0.8, min_contrast=3.0, max_angle_deg=25, wall_after=8, min_ratio=0.30, window=None):
    """Returns array (N, 4): x, y, t, d (image coords, position along side, depth from outer edge)."""
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
    nu, nv = (sign, 0.0) if axis == 0 else (0.0, sign)
    nx, ny = nu * c - nv * s, nu * s + nv * c
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    out = []
    for j in range(n_lines):
        lo_j, hi_j = d_lo, d_hi
        if window is not None:
            lo_j, hi_j = max(d_lo, int(window[j] - 8)), min(d_hi, int(window[j] + 9))
        top = min(hi_j, clear_to[j] - wall_after)
        if top <= lo_j + 2:
            continue
        seg = der[lo_j:top, j]
        i = int(np.argmin(seg))
        if i == 0 or i == len(seg) - 1:
            continue
        if -float(seg[i]) < min_drop:
            continue
        k = lo_j + i
        lip = prof[max(2, k - 10):k - 3, j].mean()
        wall = prof[k + 3:k + wall_after, j].mean()
        if lip - wall < min_contrast or (lip - wall) / max(lip, 1.0) < min_ratio:
            continue
        y0, y1, y2 = der[k - 1, j], der[k, j], der[k + 1, j]
        den = y0 - 2 * y1 + y2
        off = float(np.clip(0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0, -0.5, 0.5))
        dk = k + off
        pu = (half - dk) * sign if axis == 0 else t[j]
        pv = (half - dk) * sign if axis == 1 else t[j]
        px, py = cx + pu * c - pv * s, cy + pu * s + pv * c
        ix, iy = int(round(px)), int(round(py))
        gvx, gvy = float(gx[iy, ix]), float(gy[iy, ix])
        mag = np.hypot(gvx, gvy)
        if mag < 1e-3 or (gvx * nx + gvy * ny) / mag < cos_max:
            continue
        out.append((px, py, float(t[j]), dk))
    return np.array(out, np.float64).reshape(-1, 4)


def all_points(g, gx, gy, meat_full, cx, cy, th, W, H, inner=None, n_lines=120, span=0.36):
    """Edge points on all four sides; with `inner` (cx, cy, th, W, H) search only near it."""
    pts = {}
    for sd in SIDES:
        win = None
        if inner is not None:
            axis, sign = SIDE_GEOM[sd]
            other = H if axis == 0 else W
            t = np.linspace(-span, span, n_lines) * other
            icx, icy, ith, iW, iH = inner
            c, s = np.cos(th), np.sin(th)
            ox, oy = icx - cx, icy - cy
            du, dv = ox * c + oy * s, -ox * s + oy * c
            dth = ith - th
            if axis == 0:
                win = W / 2 - (sign * du + iW / 2) + sign * np.tan(dth) * (t - dv)
            else:
                win = H / 2 - (sign * dv + iH / 2) - sign * np.tan(dth) * (t - du)
        pts[sd] = edge_points(g, gx, gy, meat_full, cx, cy, th, W, H, sd, n_lines=n_lines, span=span, window=win)
    return pts


# ---------------------------------------------------------------- 2. rigid fit
def initial_inner(cx, cy, th, W, H, lip):
    l, r, t_, b = (lip[k] for k in SIDES)
    du, dv = (l - r) / 2, (t_ - b) / 2
    c, s = np.cos(th), np.sin(th)
    return cx + du * c - dv * s, cy + du * s + dv * c, W - l - r, H - t_ - b


def fit_inner(pts, init, W_in=None, H_in=None, min_side=10):
    have = [p[:, :2] for p in pts.values() if len(p)]
    P = np.vstack(have) if have else np.zeros((0, 2))
    cx, cy, th, W0, H0 = init
    if W_in is not None:
        W0, H0 = W_in, H_in
    if len(P) < 20:
        return None
    cx, cy, th, W, H, info = refine(P.astype(np.float32), cx, cy, th, W0, H0, fit_size=W_in is None, trim=0.15, min_side=min_side)
    return dict(cx=cx, cy=cy, th=th, W=W, H=H, rms=info["rms"], side_counts=info["side_counts"], n=info["n_pts"])


def _rigid(pts, init, outer, ii, W_in, H_in):
    """Fixed-size fit; a centre axis with no wall seen on either side stays at the outline-based guess."""
    f = fit_inner(pts, init, W_in, H_in)
    if f is None:
        return dict(src="default", cx=ii[0], cy=ii[1], th=outer[2], W=W_in, H=H_in)
    sc = dict(zip(("right", "left", "bottom", "top"), f["side_counts"]))
    c, s = np.cos(outer[2]), np.sin(outer[2])
    du = (f["cx"] - ii[0]) * c + (f["cy"] - ii[1]) * s
    dv = -(f["cx"] - ii[0]) * s + (f["cy"] - ii[1]) * c
    if sc["left"] < 10 and sc["right"] < 10:
        du = 0.0
    if sc["top"] < 10 and sc["bottom"] < 10:
        dv = 0.0
    return dict(src="".join(sd[0].upper() if sc[sd] >= 10 else sd[0] for sd in SIDES),
                cx=ii[0] + du * c - dv * s, cy=ii[1] + du * s + dv * c, th=f["th"], W=W_in, H=H_in, rms=f["rms"])


# ---------------------------------------------------------------- 3. guard
def _near_meat(pts, dist):
    Hh, Ww = dist.shape
    return {sd: (dist[np.clip(p[:, 1].astype(int), 0, Hh - 1), np.clip(p[:, 0].astype(int), 0, Ww - 1)] < NEAR
                 if len(p) else np.zeros(0, bool)) for sd, p in pts.items()}


def guard(pts, f0, dist, W_in, H_in):
    near = _near_meat(pts, dist)
    clean = {sd: pts[sd][~near[sd]] for sd in SIDES}
    use = dict(clean)
    for a, b in (("left", "right"), ("top", "bottom")):
        if len(clean[a]) < 10 and len(clean[b]) < 10:        # no clean support on this axis: keep all
            use[a], use[b] = pts[a], pts[b]
    fa = fit_inner(use, (f0["cx"], f0["cy"], f0["th"], W_in, H_in), W_in, H_in, min_side=8)
    if fa is None:
        return dict(f0, guard="none")
    final = {}
    for sd in SIDES:
        p = pts[sd]
        if not len(p):
            final[sd] = p
            continue
        u, v = tray_uv(p[:, 0], p[:, 1], fa["cx"], fa["cy"], fa["th"])
        r = (W_in / 2 - np.abs(u)) if sd in ("left", "right") else (H_in / 2 - np.abs(v))
        ok = ~near[sd] | (np.abs(r) <= AGREE)
        if use[sd] is pts[sd]:
            ok[:] = True
        final[sd] = p[ok]
    f = fit_inner(final, (fa["cx"], fa["cy"], fa["th"], W_in, H_in), W_in, H_in, min_side=8) or fa
    sc = dict(zip(("right", "left", "bottom", "top"), f["side_counts"]))
    return dict(src="".join(sd[0].upper() if sc[sd] >= 8 else sd[0] for sd in SIDES),
                cx=f["cx"], cy=f["cy"], th=f["th"], W=W_in, H=H_in, rms=f["rms"], guard="ok")


# ---------------------------------------------------------------- 4. four walls
def _robust_line(p, vertical):
    s, t = (p[:, 1], p[:, 0]) if vertical else (p[:, 0], p[:, 1])
    A = np.stack([np.ones_like(s), s], 1)
    c = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(4):
        r = t - A @ c
        keep = np.abs(r) < max(1.5, 2.5 * np.median(np.abs(r)))
        c = np.linalg.lstsq(A[keep], t[keep], rcond=None)[0]
    return float(c[0]), float(c[1])


def _normal_form(a, b, vertical):
    n = np.array([1.0, -b]) if vertical else np.array([-b, 1.0])
    k = np.linalg.norm(n)
    return n / k, a / k


def _rect_lines(q):
    c, s = np.cos(q["th"]), np.sin(q["th"])
    ex, ey = np.array([c, s]), np.array([-s, c])
    C = np.array([q["cx"], q["cy"]])
    L = {}
    for side, (axis, sign) in dict(left=(ex, -1), right=(ex, 1), top=(ey, -1), bottom=(ey, 1)).items():
        half = q["W"] / 2 if side in ("left", "right") else q["H"] / 2
        L[side] = (axis, float(axis @ C + sign * half))
    return L


def four_walls(pts, q, dist, W_in, H_in):
    """Corners (TL, TR, BR, BL) of the inner quadrilateral and how each wall was set."""
    R = _rect_lines(q)
    C = np.array([q["cx"], q["cy"]])
    near = _near_meat(pts, dist)
    L, how = {}, {}
    for sd in SIDES:
        p = pts[sd]
        if len(p):
            u, v = tray_uv(p[:, 0], p[:, 1], q["cx"], q["cy"], q["th"])
            r = (q["W"] / 2 - np.abs(u)) if sd in ("left", "right") else (q["H"] / 2 - np.abs(v))
            p = p[~near[sd] | (np.abs(r) <= AGREE)]
        vertical = sd in ("left", "right")
        spread = np.ptp(p[:, 1] if vertical else p[:, 0]) if len(p) else 0.0
        if len(p) >= MIN_PTS and spread >= MIN_SPREAD:
            ln = _normal_form(*_robust_line(p[:, :2], vertical), vertical)
            if ln[0] @ R[sd][0] < 0:
                ln = (-ln[0], -ln[1])
            L[sd], how[sd] = ln, "measured"
    for a_, b_, size_ in (("left", "right", W_in), ("top", "bottom", H_in)):
        for this, opp in ((a_, b_), (b_, a_)):
            if this in L or opp not in L:
                continue
            n, d = L[opp]
            sgn = -1.0 if (n @ C - d) < 0 else 1.0
            L[this], how[this] = (n, d + sgn * size_), "from_opposite"
        for this in (a_, b_):
            if this not in L:
                L[this], how[this] = R[this], "rigid"

    def cross(l1, l2):
        return np.linalg.solve(np.array([l1[0], l2[0]]), np.array([l1[1], l2[1]]))

    corners = np.array([cross(L["top"], L["left"]), cross(L["top"], L["right"]),
                        cross(L["bottom"], L["right"]), cross(L["bottom"], L["left"])], np.float32)
    return corners, how


# ---------------------------------------------------------------- all steps
def fit_opening(full_corr, red_half, outer_full, cfg):
    """Inner opening of one tray. outer_full = (cx, cy, th, W, H) at full res.
    Returns (corners (4, 2) full-res, how per wall, rigid fit dict)."""
    Hh, Ww = full_corr.shape[:2]
    red_full = cv2.resize(red_half.astype(np.uint8), (Ww, Hh), interpolation=cv2.INTER_NEAREST)
    meat = cv2.dilate(red_full, np.ones((7, 7), np.uint8)).astype(bool)
    dist = cv2.distanceTransform((1 - red_full).astype(np.uint8), cv2.DIST_L2, 3)
    g, gx, gy = prep(full_corr)
    W_in, H_in = cfg.inner_w, cfg.inner_h
    ix, iy, iW, iH = initial_inner(*outer_full, {k: v * 2 for k, v in cfg.lip_default.items()})
    ii = (ix, iy, outer_full[2], iW, iH)
    p1 = all_points(g, gx, gy, meat, *outer_full)
    f1 = _rigid(p1, ii, outer_full, ii, W_in, H_in)
    p2 = all_points(g, gx, gy, meat, *outer_full, inner=(f1["cx"], f1["cy"], f1["th"], f1["W"], f1["H"]))
    f2 = _rigid(p2, (f1["cx"], f1["cy"], f1["th"], f1["W"], f1["H"]), outer_full, ii, W_in, H_in)
    q = guard(p2, f2, dist, W_in, H_in)
    corners, how = four_walls(p2, q, dist, W_in, H_in)
    return corners, how, q


def opening_mask(corners, th, shape, shift_u, radius):
    """Inner opening as a filled mask: the quadrilateral shifted `shift_u` px along the tray's length,
    corners rounded to `radius` px. Returns (mask uint8, contour)."""
    poly = corners + shift_u * np.array([np.cos(th), np.sin(th)], np.float32)
    x0, y0 = np.floor(poly.min(0)).astype(int) - 8
    x1, y1 = np.ceil(poly.max(0)).astype(int) + 8
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, shape[1]), min(y1, shape[0])
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(sub, [np.round((poly - (x0, y0)) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    r = int(round(radius))
    if r > 0:   # opening with a disk rounds every convex corner to radius r, straight edges untouched
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.zeros(shape, np.uint8)
    mask[y0:y1, x0:x1] = sub
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = (max(cnts, key=cv2.contourArea) + (x0, y0)).astype(np.int32) if cnts else np.round(poly).astype(np.int32)
    return mask, contour
