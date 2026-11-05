"""Refine a tray rectangle against the SAM mask's *trusted* outer edge.

Trusted edge = points on the tray mask's outer contour that are
  - not next to this tray's meat (there the mask edge is the meat edge, not the tray's),
  - not next to other meat or the image border,
  - not far outside the starting rectangle (SAM spill into a neighbour),
  - not on the rounded corners.
Each point is assigned to its nearest side; we solve a trimmed Gauss-Newton
least squares for (cx, cy, theta) and optionally (W, H).
"""
import cv2
import numpy as np


def trusted_edge(tray, own, other, cx, cy, th, W, H, corner=28, spill=12, near_meat=6, near_other=45):
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cnts]).astype(np.float32)
    Hh, Ww = tray.shape
    k = np.ones((2 * near_meat + 1,) * 2, np.uint8)
    # a neighbouring tray's meat nearby means SAM may have merged the two trays' rims
    # on that side, so distrust edges much further away from other meat
    k2 = np.ones((2 * near_other + 1,) * 2, np.uint8)
    meat_near = (cv2.dilate(own.astype(np.uint8), k) | cv2.dilate(other.astype(np.uint8), k2)).astype(bool)
    x, y = pts[:, 0], pts[:, 1]
    ok = ~meat_near[y.astype(int), x.astype(int)]
    ok &= (x > 3) & (x < Ww - 4) & (y > 3) & (y < Hh - 4)
    c, s = np.cos(th), np.sin(th)
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    ok &= (np.abs(u) < W / 2 + spill) & (np.abs(v) < H / 2 + spill)
    corner_zone = (np.abs(u) > W / 2 - corner) & (np.abs(v) > H / 2 - corner)
    ok &= ~corner_zone
    return pts[ok]


def refine(pts, cx, cy, th, W, H, fit_size=False, iters=15, trim=0.2, min_side=15):
    """Returns (cx, cy, th, W, H, info)."""
    p = np.array([cx, cy, th, W, H], np.float64)
    info = {}
    for _ in range(iters):
        cx, cy, th, W, H = p
        c, s = np.cos(th), np.sin(th)
        x, y = pts[:, 0], pts[:, 1]
        u = (x - cx) * c + (y - cy) * s
        v = -(x - cx) * s + (y - cy) * c
        # nearest side: 0 right(+u) 1 left(-u) 2 bottom(+v) 3 top(-v)
        d = np.stack([np.abs(u - W / 2), np.abs(u + W / 2), np.abs(v - H / 2), np.abs(v + H / 2)], 1)
        side = np.argmin(d, 1)
        counts = np.bincount(side, minlength=4)
        use = counts[side] >= min_side
        sg = np.where(side % 2 == 0, 1.0, -1.0)           # +1 for right/bottom
        isu = side < 2
        # residual: signed distance outward from the side
        r = np.where(isu, sg * u - W / 2, sg * v - H / 2)
        # Jacobians of u, v wrt (cx, cy, th)
        du = np.stack([-c * np.ones_like(u), -s * np.ones_like(u), v], 1)
        dv = np.stack([s * np.ones_like(v), -c * np.ones_like(v), -u], 1)
        J = np.where(isu[:, None], sg[:, None] * du, sg[:, None] * dv)
        if fit_size:
            dW = np.where(isu, -0.5, 0.0)[:, None]
            dH = np.where(~isu, -0.5, 0.0)[:, None]
            J = np.hstack([J, dW, dH])
        # trimmed least squares
        keep = use.copy()
        if keep.sum() < 20:
            break
        thr = np.quantile(np.abs(r[keep]), 1 - trim)
        keep &= np.abs(r) <= max(thr, 1.5)
        A, b = J[keep], -r[keep]
        # need constraints along both axes; ridge keeps unconstrained params put
        lam = 1e-3 * np.eye(A.shape[1])
        delta = np.linalg.solve(A.T @ A + lam * len(b), A.T @ b)
        p[: len(delta)] += delta
        if np.abs(delta).max() < 1e-3:
            break
    cx, cy, th, W, H = p
    info.update(n_pts=int(len(pts)), side_counts=counts.tolist(),
                rms=float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else None)
    return cx, cy, th, W, H, info


def anchor_spill(tray, own, other, cx, cy, th, W, H, excess=15):
    """If SAM's mask is longer than a tray along an axis, assume the end facing
    the nearest neighbouring meat is spill: re-anchor the rectangle on the
    opposite end and report that side as untrusted.
    Returns (cx, cy, bad_sides) with sides named right/left/bottom/top."""
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in cnts]).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    u = (pts[:, 0] - cx) * c + (pts[:, 1] - cy) * s
    v = -(pts[:, 0] - cx) * s + (pts[:, 1] - cy) * c
    oy, ox = np.nonzero(other)
    ou = (ox - cx) * c + (oy - cy) * s
    ov = -(ox - cx) * s + (oy - cy) * c
    bad = []
    du = dv = 0.0
    # trays only butt end-to-end along the belt, so only the long axis can spill
    for axis, vals, ovals, size, names in ((0, u, ou, W, ("right", "left")),):
        lo, hi = np.percentile(vals, [0.5, 99.5])
        if hi - lo <= size + excess:
            continue
        # distance from each mask end to the nearest other meat along this axis
        d_hi = np.min(ovals[ovals > hi - 40] - hi, initial=1e9) if len(ovals) else 1e9
        d_lo = np.min(lo - ovals[ovals < lo + 40], initial=1e9) if len(ovals) else 1e9
        if d_hi < d_lo:           # spill toward +axis: keep the low end
            shift = (lo + size / 2)
            bad.append(names[0])
        else:
            shift = (hi - size / 2)
            bad.append(names[1])
        if axis == 0:
            du = shift
        else:
            dv = shift
    cx2 = cx + du * c - dv * s
    cy2 = cy + du * s + dv * c
    return cx2, cy2, bad


def drop_sides(pts, cx, cy, th, W, H, bad, corner=28):
    if not bad or len(pts) == 0:
        return pts
    c, s = np.cos(th), np.sin(th)
    u = (pts[:, 0] - cx) * c + (pts[:, 1] - cy) * s
    v = -(pts[:, 0] - cx) * s + (pts[:, 1] - cy) * c
    keep = np.ones(len(pts), bool)
    if "right" in bad: keep &= ~(u > W / 2 - corner)
    if "left" in bad: keep &= ~(u < -W / 2 + corner)
    if "bottom" in bad: keep &= ~(v > H / 2 - corner)
    if "top" in bad: keep &= ~(v < -H / 2 + corner)
    return pts[keep]
