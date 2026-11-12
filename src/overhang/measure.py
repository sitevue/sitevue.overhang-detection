"""Overhang numbers for one tray, on the full-resolution camera-corrected frame.

  lip_cover_pct   % of the lip band (between the magenta inner opening and the green outer edge)
                  that meat covers
  outside_px      this tray's meat pixels outside the magenta line (on the lip + past the edge)
  past_green_px   meat pixels past the green outer edge

Before counting, two kinds of pixels are handled:
  next tray   touching trays' meat merges into one blob. Just past each end of the tray we look for
              belt rollers; no rollers means another tray is there, and meat more than 10 px past our
              outer edge on that end is that tray's meat (dropped).
  hand        a worker's blue glove or white sleeve over the tray makes the numbers unreliable (flag).
              Only a glove over the tray's central area counts: a hand holding the tray's edge or
              corner leaves the meat and lip visible and the numbers fine.
"""
import cv2
import numpy as np

from .color import masks, rect_corners, tray_uv
from .inner import opening_mask

S = 2.0                      # full-res px per half-res px
NB_NEAR, NB_FAR = 20, 140    # px past this tray's end where the next tray would be
NB_YELLOW = 0.03             # below this share of belt yellow among non-meat pixels -> a tray is there
NB_CUT = 10                  # meat more than this past our outer edge, on a side with a next tray, is that tray's
OUTER_TOL = 6                # px tolerance on the green edge
# Glove check, tuned on 770 grind-01 trays from 2026-11-06 (checked by eye): live gloves are often
# in shadow (brightness ~60-80), so the colour rule accepts darker blue than the 10-29 one did
# (saturation >= 150, brightness >= 110). The belt's own blue is mostly low-saturation (median 75),
# hence saturation >= 120. Only the central 84% of the tray box is scored, so a hand at the edge or
# corner (meat fully visible) does not count; 3% of that area is the line between "touching the
# edge" (0-2.5%) and "over the tray" (4% and up).
GLOVE_HUE = (100, 130)
GLOVE_MIN_SAT, GLOVE_MIN_VAL = 120, 35
GLOVE_AREA = 0.42            # half-size of the scored area, as a fraction of the tray (0.42 -> central 84%)
GLOVE_MAX = 0.03             # glove share of that area above which the tray is "hand over tray"
SLEEVE_MAX = 0.05            # white sleeve share of the whole tray box


def measure_tray(full, own_half, outline, corners, how, cfg):
    """full: corrected full-res BGR; own_half: this tray's meat (half-res corrected mask);
    outline: dict cx, cy, th (half-res); corners: inner quadrilateral (full-res).
    Returns (numbers dict, geometry dict for drawing)."""
    Hf, Wf = full.shape[:2]
    red_full, yel_full = masks(full)
    own_up = cv2.resize(own_half.astype(np.uint8), (Wf, Hf), interpolation=cv2.INTER_NEAREST)
    own = red_full & cv2.dilate(own_up, np.ones((9, 9), np.uint8)).astype(bool)
    cx, cy, th = outline["cx"] * S, outline["cy"] * S, outline["th"]
    W, H = cfg.tray_w * S, cfg.tray_h * S
    ys, xs = np.nonzero(own)
    c_, s_ = np.cos(th), np.sin(th)

    # --- next tray along the belt
    def window_says_tray(lo, hi, sign):
        """None = can't tell (off frame or all meat); True = no belt rollers (a tray); False = belt."""
        u_, v_ = np.meshgrid(np.linspace(lo, hi, 24), np.linspace(-0.42, 0.42, 30) * H)
        uu = sign * (W / 2 + u_)
        px_, py_ = cx + uu * c_ - v_ * s_, cy + uu * s_ + v_ * c_
        inb = (px_ >= 0) & (px_ < Wf) & (py_ >= 0) & (py_ < Hf)
        if inb.mean() < 0.5:
            return None
        xi, yi = px_[inb].astype(int), py_[inb].astype(int)
        nonred = ~red_full[yi, xi]
        if nonred.mean() < 0.15:
            return None
        return bool(yel_full[yi, xi][nonred].mean() < NB_YELLOW)

    nb = {}
    for name, sign in (("left", -1), ("right", 1)):
        near = window_says_tray(NB_NEAR, NB_FAR, sign)
        if near is None:
            # all meat right past our end: our meat sliding off, or the next tray's? A next tray is
            # still there 300-500 px out; slid-off meat gives way to belt. Unknown -> keep the meat.
            near = bool(window_says_tray(300, 500, sign))
        nb[name] = near
    uo0, _ = tray_uv(xs, ys, cx, cy, th)
    keep = np.ones(len(xs), bool)
    for name, sign in (("left", -1), ("right", 1)):
        if nb[name]:
            keep &= ~((sign * uo0 - W / 2) > NB_CUT)
    n_dropped = int((~keep).sum())
    xs, ys = xs[keep], ys[keep]

    # --- hand over the tray (blue glove / white sleeve)
    def sample_hsv(half_extent):
        ru, rv = np.meshgrid(np.linspace(-half_extent, half_extent, 40) * W, np.linspace(-half_extent, half_extent, 30) * H)
        rx = np.clip((cx + ru * c_ - rv * s_).astype(int), 0, Wf - 1)
        ry = np.clip((cy + ru * s_ + rv * c_).astype(int), 0, Hf - 1)
        return cv2.split(cv2.cvtColor(np.ascontiguousarray(full[ry, rx]), cv2.COLOR_BGR2HSV))
    hh, ss, vv = sample_hsv(GLOVE_AREA)
    glove = float(((hh >= GLOVE_HUE[0]) & (hh <= GLOVE_HUE[1]) & (ss >= GLOVE_MIN_SAT) & (vv >= GLOVE_MIN_VAL)).mean())
    _, ss, vv = sample_hsv(0.5)
    sleeve = float(((ss < 35) & (vv >= 200)).mean())
    occluded = glove > GLOVE_MAX or sleeve > SLEEVE_MAX

    # --- inside / lip / past the edge
    inner_raw, inner_contour = opening_mask(corners, th, (Hf, Wf), cfg.inner_shift_u, cfg.inner_corner_r)
    inside = cv2.dilate(inner_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # 2 px tolerance
    past_inner = ~inside[ys, xs].astype(bool)
    uo, vo = tray_uv(xs, ys, cx, cy, th)
    past_outer = (np.abs(uo) > W / 2 + OUTER_TOL) | (np.abs(vo) > H / 2 + OUTER_TOL)
    lip = past_inner & ~past_outer
    beyond = past_outer
    outer_m = np.zeros((Hf, Wf), np.uint8)
    cv2.fillPoly(outer_m, [np.round(rect_corners(cx, cy, W, H, th)).astype(np.int32)], 1)
    band = outer_m.astype(bool) & ~inner_raw.astype(bool)
    band_px = int(band.sum())
    band_meat = int(band[ys, xs].sum())
    n_lip, n_bey = int(lip.sum()), int(beyond.sum())

    # where the meat outside the magenta line is (for the operator: which way it is sliding)
    side = np.where(np.abs(uo) / W > np.abs(vo) / H, np.where(uo < 0, "left", "right"), np.where(vo < 0, "top", "bottom"))
    out_side = {sd: int((past_inner & (side == sd)).sum()) for sd in ("left", "right", "top", "bottom")}

    # the same numbers per half of the tray along the belt (u < 0 = left half), so that two views of
    # the tray can be combined: each half from the frame where it is least shifted by the camera angle
    by, bx = np.nonzero(band)
    ub, _ = tray_uv(bx, by, cx, cy, th)
    on_band = band[ys, xs]
    halves = {}
    sides = ("left", "right", "top", "bottom")
    for name, m, mb in (("left", uo < 0, ub < 0), ("right", uo >= 0, ub >= 0)):
        per = lambda mask: {sd: int((mask & m & (side == sd)).sum()) for sd in sides}
        h = dict(band_px=int(mb.sum()), meat_px=int(m.sum()), side=per(past_inner),
                 band_meat_side=per(on_band), lip_side=per(lip), past_side=per(beyond))
        h.update(band_meat_px=sum(h["band_meat_side"].values()), lip_px=sum(h["lip_side"].values()),
                 past_green_px=sum(h["past_side"].values()))
        halves[name] = h

    walls = "".join(sd[0].upper() if how[sd] == "measured" else sd[0] for sd in ("left", "right", "top", "bottom"))
    # only the tray itself (its outer box) has to be in view. The belt past each end is not needed:
    # when less than half of the next-tray strip is in view, that end is treated as having no
    # neighbour (its meat is kept). On ~29,000 trays (10-29, 11-05, 11-06) the 17 that this makes
    # readable with one end unseen all read correctly by eye, except one box on two touching
    # trays, which the outline check now catches (other_meat_frac).
    # edge_tolerance_px (default 0): a corner this far out is less than the rim's width, nothing measured is lost
    tol = cfg.edge_tolerance_px
    edge_of_view = not all(-tol <= cx + u * c_ - v * s_ < Wf + tol and -tol <= cy + u * s_ + v * c_ < Hf + tol
                           for u in (-W / 2, W / 2) for v in (-H / 2, H / 2))
    num = dict(
        lip_cover_pct=round(100.0 * band_meat / max(band_px, 1), 2),
        outside_px=n_lip + n_bey,
        past_green_px=n_bey,
        lip_px=n_lip,
        outside_pct_of_meat=round(100.0 * (n_lip + n_bey) / max(len(xs), 1), 2),
        meat_px=int(len(xs)),
        lip_band_px=band_px,
        lip_band_meat_px=band_meat,
        outside_by_side_px=out_side,
        meat_offset_px=dict(along=round(float(uo.mean()), 1) if len(uo) else 0.0,
                            across=round(float(vo.mean()), 1) if len(vo) else 0.0),
        walls_measured=walls,
        next_tray_left=nb["left"], next_tray_right=nb["right"], dropped_next_tray_px=n_dropped,
        glove_frac=round(glove, 4), sleeve_frac=round(sleeve, 4),
        occluded=occluded, edge_of_view=edge_of_view,
        halves=halves,
    )
    geom = dict(xs=xs, ys=ys, lip=lip, beyond=beyond, outer=(cx, cy, W, H, th), inner_contour=inner_contour)
    return num, geom

