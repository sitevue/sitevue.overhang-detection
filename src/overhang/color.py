"""Colour masks and small geometry helpers (classical CV, no learned model)."""
import cv2
import numpy as np


def masks(bgr):
    """Meat (red) and belt-roller (yellow) masks of a BGR frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = ((h < 12) | (h > 160)) & (s > 70) & (v > 60)
    yel = (h > 18) & (h < 40) & (s > 100) & (v > 120)
    red = red.astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return red.astype(bool), yel


def meat_blobs(red, min_area):
    """Connected meat regions of at least min_area px: (label image, [dict(id, area, bbox, c)])."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out.append(dict(id=i, area=int(st[i, 4]), bbox=st[i, :4].tolist(), c=cen[i].tolist()))
    return lab, out


# A bare hand (skin) passes the meat colour rule, and a hand is the size of a tray, so the tracker would
# count it as one. Ground meat is full of fine edges, skin is smooth and more orange. On 10-29: 600 random
# real trays had a median edge strength (|Laplacian|, half-res) of 16-22 and hue -2..+2; six bare-hand
# blobs had 2-4 and +7..+10. Both must hold, so a blurred real tray (smooth but red) is kept.
SKIN_MAX_TEXTURE = 10.0
SKIN_MIN_HUE = 4.0


def skin_like(bgr, blob):
    """True if the meat blob (bool mask on bgr) looks like bare skin, not ground meat. Returns (skin, texture, hue)."""
    m = cv2.erode(blob.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if m.sum() < 500:
        return False, None, None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))[m]))
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0][m].astype(np.float32)
    hue = float(np.median(np.where(h > 90, h - 180, h)))      # red hue around 0
    return bool(tex < SKIN_MAX_TEXTURE and hue >= SKIN_MIN_HUE), tex, hue


def smooth_mask(bgr, red):
    """Edge-free, non-meat pixels: tray rims are matte, the belt is dense with edges."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    ed = cv2.Canny(g, 20, 50).astype(np.float32) / 255
    d = cv2.blur(ed, (13, 13))
    sm = ((d < 0.04) & ~red).astype(np.uint8)
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return sm.astype(bool)


def rect_corners(cx, cy, w, h, th):
    """Corners (TL, TR, BR, BL) of a w x h rectangle centred at (cx, cy), rotated by th (rad)."""
    c, s = np.cos(th), np.sin(th)
    ux, uy = np.array([c, s]), np.array([-s, c])
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([[cx + a * ux[0] + b * uy[0], cy + a * ux[1] + b * uy[1]] for a, b in pts], np.float32)


def tray_uv(xs, ys, cx, cy, th):
    """Image points -> tray coordinates (u along the tray's length, v across it)."""
    c, s = np.cos(th), np.sin(th)
    return (xs - cx) * c + (ys - cy) * s, -(xs - cx) * s + (ys - cy) * c
