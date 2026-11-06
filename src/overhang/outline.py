"""Outer tray outline (green box): a rectangle of the known tray size fitted to the SAM tray mask.

1. Coarse-to-fine search for the rectangle best explained by (tray | meat) that contains the tray.
2. If SAM's mask is longer than a tray (it spilled into a neighbour), re-anchor on the other end.
3. Refine against the trusted part of SAM's outer edge (away from meat, corners and frame border).
All coordinates are HALF resolution, camera-corrected.
"""
import cv2
import numpy as np

from .color import masks, meat_blobs, tray_uv

TOL = 3                    # px tolerance on the tray outline
MAX_TILT_DEG = 30.0        # trays travel lengthwise; a box tilted more than this is wrong
BELT_INSIDE_MAX = 0.01     # more than 1% belt-roller yellow inside the box -> box is misplaced


def fit_full(tray, own, W, H):
    fill = tray | own
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx0, cy0), (w0, h0), a0 = cv2.minAreaRect(np.vstack(cnts))
    if w0 < h0:
        a0 += 90
    a0 = (a0 + 90) % 180 - 90
    # trays travel lengthwise along the belt: turn a sideways start back to lengthwise
    if abs(a0) > 45:
        a0 -= 90 * np.sign(a0)
    ty, tx = np.nonzero(tray)
    sel = np.random.default_rng(0).choice(len(tx), min(2500, len(tx)), replace=False)
    tx, ty = tx[sel].astype(np.float32), ty[sel].astype(np.float32)
    gu, gv = np.meshgrid(np.linspace(-0.49, 0.49, 36) * W, np.linspace(-0.49, 0.49, 28) * H)
    gu, gv = gu.ravel(), gv.ravel()
    Hh, Ww = fill.shape

    def score(cx, cy, th):
        c, s = np.cos(th), np.sin(th)
        x = cx + gu * c - gv * s
        y = cy + gu * s + gv * c
        inb = (x >= 0) & (x < Ww) & (y >= 0) & (y < Hh)
        f = fill[np.clip(y.astype(int), 0, Hh - 1), np.clip(x.astype(int), 0, Ww - 1)] & inb
        empty = (~f & inb).mean()
        u, v = tray_uv(tx, ty, cx, cy, th)
        out = ((np.abs(u) > W / 2 + TOL) | (np.abs(v) > H / 2 + TOL)).mean()
        return -empty - 2.0 * out

    best = (score(cx0, cy0, np.deg2rad(a0)), cx0, cy0, np.deg2rad(a0))
    for span, step, aspan, astep in ((70, 7, 6, 1.5), (8, 2, 1.5, 0.5), (2, 1, 0.5, 0.25)):
        _, bx, by, bt = best
        for dx in np.arange(-span, span + 1e-6, step):
            for dy in np.arange(-span, span + 1e-6, step):
                for da in np.arange(-aspan, aspan + 1e-6, astep):
                    sc = score(bx + dx, by + dy, bt + np.deg2rad(da))
                    if sc > best[0]:
                        best = (sc, bx + dx, by + dy, bt + np.deg2rad(da))
    return best


def trusted_edge(tray, own, other, cx, cy, th, W, H, corner=28, spill=12, near_meat=6, near_other=45):
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cnts]).astype(np.float32)
    Hh, Ww = tray.shape
    k = np.ones((2 * near_meat + 1,) * 2, np.uint8)
    k2 = np.ones((2 * near_other + 1,) * 2, np.uint8)
    meat_near = (cv2.dilate(own.astype(np.uint8), k) | cv2.dilate(other.astype(np.uint8), k2)).astype(bool)
    x, y = pts[:, 0], pts[:, 1]
    ok = ~meat_near[y.astype(int), x.astype(int)]
    ok &= (x > 3) & (x < Ww - 4) & (y > 3) & (y < Hh - 4)
    u, v = tray_uv(x, y, cx, cy, th)
    ok &= (np.abs(u) < W / 2 + spill) & (np.abs(v) < H / 2 + spill)
    ok &= ~((np.abs(u) > W / 2 - corner) & (np.abs(v) > H / 2 - corner))
    return pts[ok]


def refine(pts, cx, cy, th, W, H, fit_size=False, iters=15, trim=0.2, min_side=15):
    """Trimmed Gauss-Newton fit of a rectangle to edge points. Returns (cx, cy, th, W, H, info)."""
    p = np.array([cx, cy, th, W, H], np.float64)
    info = {}
    for _ in range(iters):
        cx, cy, th, W, H = p
        c, s = np.cos(th), np.sin(th)
        x, y = pts[:, 0], pts[:, 1]
        u = (x - cx) * c + (y - cy) * s
        v = -(x - cx) * s + (y - cy) * c
        d = np.stack([np.abs(u - W / 2), np.abs(u + W / 2), np.abs(v - H / 2), np.abs(v + H / 2)], 1)
        side = np.argmin(d, 1)
        counts = np.bincount(side, minlength=4)
        use = counts[side] >= min_side
        sg = np.where(side % 2 == 0, 1.0, -1.0)
        isu = side < 2
        r = np.where(isu, sg * u - W / 2, sg * v - H / 2)
        du = np.stack([-c * np.ones_like(u), -s * np.ones_like(u), v], 1)
        dv = np.stack([s * np.ones_like(v), -c * np.ones_like(v), -u], 1)
        J = np.where(isu[:, None], sg[:, None] * du, sg[:, None] * dv)
        if fit_size:
            J = np.hstack([J, np.where(isu, -0.5, 0.0)[:, None], np.where(~isu, -0.5, 0.0)[:, None]])
        keep = use.copy()
        if keep.sum() < 20:
            break
        thr = np.quantile(np.abs(r[keep]), 1 - trim)
        keep &= np.abs(r) <= max(thr, 1.5)
        A, b = J[keep], -r[keep]
        delta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]) * len(b), A.T @ b)
        p[: len(delta)] += delta
        if np.abs(delta).max() < 1e-3:
            break
    cx, cy, th, W, H = p
    info.update(n_pts=int(len(pts)), side_counts=counts.tolist(),
                rms=float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else None)
    return cx, cy, th, W, H, info


def anchor_spill(tray, own, other, cx, cy, th, W, H, excess=15):
    """If SAM's mask is longer than a tray, the end facing the nearest neighbouring meat is spill:
    re-anchor on the opposite end. Returns (cx, cy, bad_sides)."""
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cnts]).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    u, _ = tray_uv(pts[:, 0], pts[:, 1], cx, cy, th)
    oy, ox = np.nonzero(other)
    ou, _ = tray_uv(ox, oy, cx, cy, th)
    bad, du = [], 0.0
    lo, hi = np.percentile(u, [0.5, 99.5])
    if hi - lo > W + excess:
        d_hi = np.min(ou[ou > hi - 40] - hi, initial=1e9) if len(ou) else 1e9
        d_lo = np.min(lo - ou[ou < lo + 40], initial=1e9) if len(ou) else 1e9
        if d_hi < d_lo:
            du = lo + W / 2
            bad.append("right")
        else:
            du = hi - W / 2
            bad.append("left")
    return cx + du * c, cy + du * s, bad


def drop_sides(pts, cx, cy, th, W, H, bad, corner=28):
    if not bad or len(pts) == 0:
        return pts
    u, v = tray_uv(pts[:, 0], pts[:, 1], cx, cy, th)
    keep = np.ones(len(pts), bool)
    if "right" in bad:
        keep &= ~(u > W / 2 - corner)
    if "left" in bad:
        keep &= ~(u < -W / 2 + corner)
    if "bottom" in bad:
        keep &= ~(v > H / 2 - corner)
    if "top" in bad:
        keep &= ~(v < -H / 2 + corner)
    return pts[keep]


def fit_outline(tray, own, red, half_bgr, W, H):
    """Fit the outer outline. tray/own/red: half-res corrected masks; half_bgr: corrected half frame.
    Returns a dict with cx, cy, th (half-res px / rad) and quality checks."""
    sc, cx, cy, th = fit_full(tray, own, W, H)
    # neighbouring meat = other large red blobs only (not the red belt markers)
    _, oblobs = meat_blobs(red, 4000)
    lab_all = cv2.connectedComponents(red.astype(np.uint8), connectivity=8)[1]
    other = np.isin(lab_all, [b["id"] for b in oblobs]) & ~own
    cx, cy, bad = anchor_spill(tray, own, other, cx, cy, th, W, H)
    pts = drop_sides(trusted_edge(tray, own, other, cx, cy, th, W, H), cx, cy, th, W, H, bad)
    out = dict(spill_sides="/".join(bad))
    if len(pts) >= 40:
        cx, cy, th, _, _, info = refine(pts, cx, cy, th, W, H)
        out.update(edge_pts=info["n_pts"], edge_rms=info["rms"])
    out.update(fit_score=float(sc), cx=float(cx), cy=float(cy), th=float(th))
    # visible tray area far outside the outline => SAM mask spilled (e.g. into a neighbour tray)
    ty, tx = np.nonzero(tray)
    tu, tv = tray_uv(tx, ty, cx, cy, th)
    out["tray_spill_frac"] = float(((np.abs(tu) > W / 2 + 10) | (np.abs(tv) > H / 2 + 10)).mean())
    # belt rollers (yellow) visible inside the box mean the box is misplaced or spans two trays
    _, yel = masks(half_bgr)
    bu, bv = np.meshgrid(np.linspace(-0.46, 0.46, 46) * W, np.linspace(-0.44, 0.44, 34) * H)
    bx = cx + bu * np.cos(th) - bv * np.sin(th)
    by = cy + bu * np.sin(th) + bv * np.cos(th)
    inb = (bx >= 0) & (bx < yel.shape[1]) & (by >= 0) & (by < yel.shape[0])
    out["belt_inside_frac"] = float(yel[by[inb].astype(int), bx[inb].astype(int)].mean()) if inb.any() else 0.0
    out["tilt_ok"] = abs(np.rad2deg(th)) <= MAX_TILT_DEG
    out["outline_ok"] = outline_ok(out["belt_inside_frac"], th)
    return out


def outline_ok(belt_inside_frac, th):
    """Is the green box on a tray? Only two signs reliably mean it is not: belt rollers visible
    inside the box (it straddles two trays or sits on empty belt) and a box tilted past 30 deg.
    SAM spill (tray_spill_frac) and a low fit_score are NOT used: on 48 by-eye-checked flagged trays
    from grind-01 on 2026-11-06, the box was right on all 31 hand-free trays flagged only for those;
    the box fit already ignores the spilled part of SAM's mask. Both stay in the result as diagnostics."""
    return bool(belt_inside_frac <= BELT_INSIDE_MAX and abs(np.rad2deg(th)) <= MAX_TILT_DEG)
