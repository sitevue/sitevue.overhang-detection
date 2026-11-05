"""Find the tray's own zones per side from image brightness:

    outer edge -> flat LIP (bright) -> inner WALL (dark) -> FLOOR

For each side we sample brightness profiles going inward from the fitted outer
edge, only along lines the meat does not cover, take the median profile and find
  lip_w   = where brightness drops (lip -> wall)
  floor_d = where it rises again (wall -> floor)
Sides fully covered by meat return None (caller falls back to a default).
"""
import cv2
import numpy as np

D = np.arange(0, 64)          # inward distance samples, px
SIDES = ("left", "right", "top", "bottom")


def side_profiles(gray, own, cx, cy, th, W, H, n_lines=40, span=0.3, min_lines=8):
    c, s = np.cos(th), np.sin(th)
    meatd = cv2.dilate(own.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    Hh, Ww = gray.shape
    out = {}
    for side, (axis, sign, half, other) in dict(left=(0, -1, W / 2, H), right=(0, 1, W / 2, H),
                                                top=(1, -1, H / 2, W), bottom=(1, 1, H / 2, W)).items():
        t = np.linspace(-span, span, n_lines) * other
        dd = (half - D)[:, None] * sign
        uu = np.broadcast_to(dd if axis == 0 else t[None, :], (len(D), n_lines))
        vv = np.broadcast_to(dd if axis == 1 else t[None, :], (len(D), n_lines))
        x = np.round(cx + uu * c - vv * s).astype(int)
        y = np.round(cy + uu * s + vv * c).astype(int)
        inb = (x >= 0) & (x < Ww) & (y >= 0) & (y < Hh)
        xc, yc = np.clip(x, 0, Ww - 1), np.clip(y, 0, Hh - 1)
        # each line is usable from the outer edge until it first meets meat
        blocked = np.maximum.accumulate(meatd[yc, xc] | ~inb, axis=0)
        vals = np.where(blocked, np.nan, gray[yc, xc])
        cnt = (~blocked).sum(1)
        if cnt[24] < min_lines:            # need clear lines at least past the lip
            out[side] = None
            continue
        prof = np.nanmedian(np.where(cnt[:, None] >= min_lines, vals, np.nan), 1)
        out[side] = (prof, int(cnt[24]), int(np.argmax(cnt < min_lines)) if (cnt < min_lines).any() else len(D))
    return out


def find_zones(profile, valid_to):
    p = profile[:valid_to].astype(np.float32)
    p = np.convolve(np.pad(p, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    g = np.gradient(p)
    lo, hi = 6, min(32, len(g) - 1)
    lip = lo + int(np.argmin(g[lo:hi]))                 # strongest drop: lip -> wall
    drop = float(-g[lip])
    f_lo = lip + 6
    floor = None
    if f_lo < len(g) - 3:
        floor = f_lo + int(np.argmax(g[f_lo:len(g) - 1]))  # strongest rise: wall -> floor
        if g[floor] < 0.8:
            floor = None
    rise = float(g[floor]) if floor is not None else 0.0
    return dict(lip_w=lip, lip_drop=round(drop, 2), floor_d=floor, floor_rise=round(rise, 2))


def tray_zones(gray, own, cx, cy, th, W, H, min_drop=1.0):
    res = {}
    for side, pr in side_profiles(gray, own, cx, cy, th, W, H).items():
        if pr is None:
            res[side] = None
            continue
        z = find_zones(pr[0], pr[2])
        z["n_lines"] = pr[1]
        z["valid_to"] = pr[2]
        res[side] = z if z["lip_drop"] >= min_drop else None
    return res


# ---------------------------------------------------------------------------
# Per-side inner-edge LINE from every visible stretch of the lip
# ---------------------------------------------------------------------------
SIDE_GEOM = dict(left=(0, -1), right=(0, 1), top=(1, -1), bottom=(1, 1))


def inner_edge_points(gray, own, cx, cy, th, W, H, side, n_lines=90, span=0.42,
                      depth=48, lo=6, hi=34, min_drop=1.0, avg=3):
    """Along one side, look inward from the outer edge on many short lines.
    Each line that reaches the wall before meeting meat gives one point:
    (t = position along the side, d = lip width at that position)."""
    axis, sign = SIDE_GEOM[side]
    half, other = (W / 2, H) if axis == 0 else (H / 2, W)
    c, s = np.cos(th), np.sin(th)
    meatd = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Hh, Ww = gray.shape
    t = np.linspace(-span, span, n_lines) * other
    d = np.arange(depth, dtype=np.float32)
    dd = (half - d)[:, None] * sign
    uu = np.broadcast_to(dd if axis == 0 else t[None, :], (depth, n_lines))
    vv = np.broadcast_to(dd if axis == 1 else t[None, :], (depth, n_lines))
    x = cx + uu * c - vv * s
    y = cy + uu * s + vv * c
    inb = (x >= 0) & (x < Ww - 1) & (y >= 0) & (y < Hh - 1)
    xi = np.clip(np.round(x).astype(int), 0, Ww - 1)
    yi = np.clip(np.round(y).astype(int), 0, Hh - 1)
    g = cv2.remap(gray, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
    blocked = np.maximum.accumulate(meatd[yi, xi] | ~inb, axis=0)
    clear_to = np.where(blocked.any(0), np.argmax(blocked, 0), depth)
    # average neighbouring lines to beat noise (only lines that are clear far enough)
    k = np.ones(avg) / avg
    gs = np.apply_along_axis(lambda col: np.convolve(np.pad(col, 2, mode="edge"), np.ones(5) / 5, mode="valid"), 0, g)
    gs = np.apply_along_axis(lambda row: np.convolve(np.pad(row, avg // 2, mode="edge"), k, mode="valid"), 1, gs)
    grad = np.gradient(gs, axis=0)
    pts = []
    for j in range(n_lines):
        top = min(hi, clear_to[j] - 5)          # need a few px of wall visible after the drop
        if top <= lo + 2:
            continue
        seg = grad[lo:top, j]
        i = int(np.argmin(seg)); drop = -float(seg[i])
        dl = lo + i
        if drop < min_drop or i == 0 or i == len(seg) - 1:
            continue
        lip_mean = gs[max(2, dl - 6):dl - 2, j].mean() if dl - 2 > 2 else gs[2:4, j].mean()
        wall_mean = gs[dl + 2:min(dl + 7, clear_to[j]), j].mean()
        if lip_mean - wall_mean < 3:             # lip must be clearly brighter than the wall
            continue
        pts.append((float(t[j]), float(dl), drop))
    return np.array(pts).reshape(-1, 3)


def fit_inner_line(pts, other_len, min_pts=8, min_spread=0.25, max_slope=0.08, iters=200, tol=2.5, seed=0):
    """RANSAC line d = a + b*t over (t, d) points. Returns (a, b, n_inliers) or None."""
    if len(pts) < min_pts:
        return None
    t, d = pts[:, 0], pts[:, 1]
    if (t.max() - t.min()) < min_spread * other_len:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if abs(t[i] - t[j]) < 5:
            continue
        b = (d[j] - d[i]) / (t[j] - t[i])
        if abs(b) > max_slope:
            continue
        a = d[i] - b * t[i]
        inl = np.abs(d - (a + b * t)) < tol
        if best is None or inl.sum() > best[2].sum():
            best = (a, b, inl)
    if best is None or best[2].sum() < min_pts:
        return None
    inl = best[2]
    A = np.stack([np.ones(inl.sum()), t[inl]], 1)
    a, b = np.linalg.lstsq(A, d[inl], rcond=None)[0]
    b = float(np.clip(b, -max_slope, max_slope))
    return float(a), b, int(inl.sum())


def inner_edges(gray, own, cx, cy, th, W, H):
    """Per side: dict(a, b, n, src='fit') or None when the side can't be measured."""
    out = {}
    for side in SIDES:
        pts = inner_edge_points(gray, own, cx, cy, th, W, H, side)
        other_len = H if side in ("left", "right") else W
        f = fit_inner_line(pts, other_len)
        out[side] = None if f is None else dict(a=round(f[0], 2), b=round(f[1], 4), n=f[2], n_pts=len(pts))
    return out
