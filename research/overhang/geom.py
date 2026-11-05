"""Tray/meat geometry for overhang detection (classical CV, no labels).

Works on frames resized to WORK_W wide (default 728 = half of 1456).
"""
import cv2
import numpy as np

WORK_W = 728


def masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = ((h < 12) | (h > 160)) & (s > 70) & (v > 60)
    yel = (h > 18) & (h < 40) & (s > 100) & (v > 120)
    red = red.astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return red.astype(bool), yel


def meat_blobs(red, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out.append(dict(id=i, area=int(st[i, 4]), bbox=st[i, :4].tolist(),
                            c=cen[i].tolist()))
    return lab, out


def rect_corners(cx, cy, w, h, th):
    c, s = np.cos(th), np.sin(th)
    ux, uy = np.array([c, s]), np.array([-s, c])
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([[cx + a * ux[0] + b * uy[0], cy + a * ux[1] + b * uy[1]] for a, b in pts],
                    np.float32)


def smooth_mask(bgr, red):
    """Tray rims are matte and edge-free; the belt is dense with edges."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    ed = cv2.Canny(g, 20, 50).astype(np.float32) / 255
    d = cv2.blur(ed, (13, 13))
    sm = ((d < 0.04) & ~red).astype(np.uint8)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return sm.astype(bool)


class TrayFitter:
    """Fixed-size rotated rectangle that is full of (meat | edge-free rim),
    has little rim-like area just outside it, covers no yellow rollers and no
    other tray's meat."""

    def __init__(self, tray_w, tray_h, ring=14):
        self.w, self.h = tray_w, tray_h
        gx, gy = np.meshgrid(np.linspace(-0.49, 0.49, 40) * tray_w,
                             np.linspace(-0.49, 0.49, 28) * tray_h)
        self.g = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        # band just inside the border (where the rim should be)
        gi = self.g[(np.abs(self.g[:, 0]) > tray_w / 2 - 30) | (np.abs(self.g[:, 1]) > tray_h / 2 - 30)]
        self.gi = gi
        ring_pts = []
        for d in (4, ring):
            for t in np.linspace(-0.5, 0.5, 60):
                ring_pts += [(t * tray_w, -tray_h / 2 - d), (t * tray_w, tray_h / 2 + d)]
            for t in np.linspace(-0.5, 0.5, 40):
                ring_pts += [(-tray_w / 2 - d, t * tray_h), (tray_w / 2 + d, t * tray_h)]
        self.r = np.array(ring_pts, np.float32)

    @staticmethod
    def _xform(pts, cx, cy, th):
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        x = cx[:, None] + pts[None, :, 0] * c - pts[None, :, 1] * s
        y = cy[:, None] + pts[None, :, 0] * s + pts[None, :, 1] * c
        return x, y

    def _sample(self, img, pts, cx, cy, th):
        H, W = img.shape
        x, y = self._xform(pts, cx, cy, th)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xi = np.clip(np.round(x).astype(int), 0, W - 1)
        yi = np.clip(np.round(y).astype(int), 0, H - 1)
        return img[yi, xi], inb

    def score(self, cx, cy, th, fill, yel, other):
        f_in, inb = self._sample(fill, self.g, cx, cy, th)
        fill_in = (f_in & inb).sum(1) / np.maximum(inb.sum(1), 1)
        f_rim, inb2 = self._sample(fill, self.gi, cx, cy, th)
        rim_in = (f_rim & inb2).sum(1) / np.maximum(inb2.sum(1), 1)
        y_in, _ = self._sample(yel, self.g, cx, cy, th)
        o_in, _ = self._sample(other, self.g, cx, cy, th)
        f_out, inb3 = self._sample(fill & ~other, self.r, cx, cy, th)
        out = (f_out & inb3).sum(1) / np.maximum(inb3.sum(1), 1)
        return fill_in + rim_in - 1.0 * out - 25 * y_in.mean(1) - 4 * o_in.mean(1)

    def fit(self, fill, yel, other_meat, c0, lane_cy=None, lane_span=40, max_deg=6):
        fill = fill.astype(bool); yel = yel.astype(bool); other_meat = other_meat.astype(bool)
        spans = [(120, 6, np.deg2rad(max_deg), np.deg2rad(1)),
                 (10, 2, np.deg2rad(2), np.deg2rad(0.5)),
                 (2, 1, np.deg2rad(0.5), np.deg2rad(0.25))]
        cx0, cy0, th0 = c0[0], (c0[1] if lane_cy is None else lane_cy), 0.0
        first = True
        for span, step, aspan, astep in spans:
            xs = np.arange(cx0 - span, cx0 + span + 1e-6, step)
            ys = np.arange(cy0 - (lane_span if first and lane_cy is not None else span),
                           cy0 + (lane_span if first and lane_cy is not None else span) + 1e-6, step)
            first = False
            ths = np.arange(th0 - aspan, th0 + aspan + 1e-9, astep)
            X, Y, T = (a.ravel() for a in np.meshgrid(xs, ys, ths, indexing="ij"))
            sc = np.concatenate([self.score(X[i:i + 3000], Y[i:i + 3000], T[i:i + 3000],
                                            fill, yel, other_meat)
                                 for i in range(0, len(X), 3000)])
            k = int(np.argmax(sc))
            cx0, cy0, th0 = X[k], Y[k], T[k]
        return float(cx0), float(cy0), float(th0), float(sc[k])


def measure(meat_mask, other_trays_mask, cx, cy, th, w, h, band_px):
    """Meat outside the fitted tray rectangle, within band_px of it."""
    H, W = meat_mask.shape
    yy, xx = np.nonzero(meat_mask & ~other_trays_mask)
    c, s = np.cos(th), np.sin(th)
    u = (xx - cx) * c + (yy - cy) * s        # along tray long axis
    v = -(xx - cx) * s + (yy - cy) * c       # along short axis
    du = np.abs(u) - w / 2
    dv = np.abs(v) - h / 2
    outside = (du > 0) | (dv > 0)
    near = (du < band_px) & (dv < band_px)
    sel = outside & near
    res = dict(out_px=int(sel.sum()))
    inside = ~outside
    res["in_px"] = int(inside.sum())
    sides = {"end_neg": sel & (u < 0) & (du > 0), "end_pos": sel & (u > 0) & (du > 0),
             "side_neg": sel & (v < 0) & (dv > 0), "side_pos": sel & (v > 0) & (dv > 0)}
    for k, m in sides.items():
        d = np.where(k.startswith("end"), du, dv)[m]
        res[k + "_px"] = int(m.sum())
        res[k + "_maxd"] = float(np.percentile(d, 98)) if m.sum() > 20 else 0.0
    # user's idea: centroid offset between tray and the meat assigned to it
    own = (inside | sel)
    if own.sum():
        mu, mv = u[own].mean(), v[own].mean()
        res["centroid_off_u"] = float(mu)
        res["centroid_off_v"] = float(mv)
    res["out_mask_xy"] = (xx[sel], yy[sel])
    return res
