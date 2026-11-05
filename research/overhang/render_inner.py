"""Full-res overlay for every tray: magenta = fitted inner opening, meat outside it
blended at alpha 0.3 (orange = on the lip, inside the outer edge; yellow = past the
outer edge). Thin green = outer outline. Writes one crop per tray + a summary CSV.

Per tray it reports
  lip_cover_pct  % of the lip band (between magenta and green) covered by meat
  outside_px     meat pixels outside the magenta line (lip + past green)
  large          1 when meat past the green box >= LARGE_PX camera pixels (large overhang)"""
import os as _os
RUN = _os.environ.get("RUN", "out/run1")
import csv, sys, json, os, cv2, numpy as np
sys.path.insert(0, "overhang")
from geom import masks, rect_corners
from measure_full import TRAY_W, TRAY_H, tray_uv

S = 2.0; ALPHA = 0.3   # overlay opacity: lower = more see-through
NB_NEAR, NB_FAR = 20, 140   # camera px past this tray's end where the next tray would be
NB_YELLOW = 0.03            # below this share of belt-yellow among non-meat pixels -> a tray is there
NB_CUT = 10                 # meat more than this past our outer edge, on a side with a next tray, is that tray's
ORANGE = np.array([0, 140, 255], np.float32); YELLOW = np.array([0, 255, 255], np.float32)
out_dir = f"{RUN}/inner_overlays"; os.makedirs(out_dir, exist_ok=True)
F = json.load(open(f"{RUN}/inner_fit.json"))
Q = json.load(open(f"{RUN}/inner_quad.json")) if os.path.exists(f"{RUN}/inner_quad.json") else None
INNER_SHIFT_U = float(os.environ.get("INNER_SHIFT_U", 4.0))   # px along the tray's length (+ = right); from hand-marked empty trays
INNER_CORNER_R = float(os.environ.get("INNER_CORNER_R", 45.0))  # inner corner radius, px (circle fit on the empty tray)
LARGE_PX = int(os.environ.get("LARGE_PX", 10000))   # meat px past the green box at or above this = large overhang (~top 1% on 10-29)
RED = (60, 60, 255)
NO_IMAGE = os.environ.get("NO_IMAGE") == "1"   # measure only: no drawing, no image written


def inner_opening_mask(corners, th, shape):
    """Inner opening as a rounded quadrilateral: the four wall lines, shifted by INNER_SHIFT_U along
    the tray's length, with corners rounded to INNER_CORNER_R. Returns (mask, contour)."""
    poly = corners + INNER_SHIFT_U * np.array([np.cos(th), np.sin(th)], np.float32)
    x0, y0 = np.floor(poly.min(0)).astype(int) - 8; x1, y1 = np.ceil(poly.max(0)).astype(int) + 8
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, shape[1]), min(y1, shape[0])
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(sub, [np.round((poly - (x0, y0)) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    r = int(round(INNER_CORNER_R))
    if r > 0:   # opening with a disk = round every convex corner to radius r, straight edges untouched
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.zeros(shape, np.uint8); mask[y0:y1, x0:x1] = sub
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = (max(cnts, key=cv2.contourArea) + (x0, y0)).astype(np.int32) if cnts else np.round(poly).astype(np.int32)
    return mask, contour


def render_one(r):
    k = int(r["idx"]); full = cv2.imread(f"{RUN}/cache_full/{k:03d}.png"); z = np.load(f"{RUN}/cache/{k:03d}.npz")
    # this tray's meat at full res: full-res colour mask, limited to (dilated) this tray's blob
    red_full, yel_full = masks(full)
    own_up = cv2.resize(z["own"].astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    own = red_full & cv2.dilate(own_up, np.ones((9, 9), np.uint8)).astype(bool)
    cx, cy, th = float(r["tray_cx"]) * S, float(r["tray_cy"]) * S, np.deg2rad(float(r["tray_deg"]))
    W, H = TRAY_W * S, TRAY_H * S
    q = F[str(k)]
    ys, xs = np.nonzero(own)
    # --- the next tray along the belt -------------------------------------------------------
    # Touching trays' meat merges into one blob. Look just past each end of this tray: if the
    # belt's yellow rollers show there it is open belt; if not (dark tray rim or meat), another
    # tray sits there, and meat past this tray's outer edge on that end belongs to that tray.
    c_, s_ = np.cos(th), np.sin(th)
    gu, gv = np.meshgrid(np.linspace(NB_NEAR, NB_FAR, 24), np.linspace(-0.42, 0.42, 30) * H)
    def window_says_tray(lo, hi, sign):
        """None = can't tell (off frame or all meat); True = no belt rollers (a tray); False = belt."""
        u_, v_ = np.meshgrid(np.linspace(lo, hi, 24), np.linspace(-0.42, 0.42, 30) * H)
        uu = sign * (W / 2 + u_); px_ = cx + uu * c_ - v_ * s_; py_ = cy + uu * s_ + v_ * c_
        inb = (px_ >= 0) & (px_ < full.shape[1]) & (py_ >= 0) & (py_ < full.shape[0])
        if inb.mean() < 0.5:
            return None
        xi = px_[inb].astype(int); yi = py_[inb].astype(int)
        nonred = ~red_full[yi, xi]
        if nonred.mean() < 0.15:
            return None
        return bool(yel_full[yi, xi][nonred].mean() < NB_YELLOW)

    nb = {}
    for name, sign in (("left", -1), ("right", 1)):
        near = window_says_tray(NB_NEAR, NB_FAR, sign)
        if near is None:
            # all meat right past our end: our meat sliding off, or the next tray's meat? A next
            # tray (~780 px long) is still there 300-500 px out; slid-off meat gives way to belt.
            far = window_says_tray(300, 500, sign)
            near = bool(far)          # unknown -> keep the meat (never hide a real overhang)
        nb[name] = near
    uo0, _ = tray_uv(xs, ys, cx, cy, th)
    keep = np.ones(len(xs), bool)
    for name, sign in (("left", -1), ("right", 1)):
        if nb[name]:
            keep &= ~((sign * uo0 - W / 2) > NB_CUT)
    n_dropped = int((~keep).sum()); xs, ys = xs[keep], ys[keep]
    # --- a worker's hand over the tray (blue glove / white sleeve) -------------------------------
    ru, rv = np.meshgrid(np.linspace(-0.5, 0.5, 40) * W, np.linspace(-0.5, 0.5, 30) * H)
    rx = np.clip((cx + ru * c_ - rv * s_).astype(int), 0, full.shape[1] - 1); ry = np.clip((cy + ru * s_ + rv * c_).astype(int), 0, full.shape[0] - 1)
    hh, ss, vv_ = cv2.split(cv2.cvtColor(np.ascontiguousarray(full[ry, rx]), cv2.COLOR_BGR2HSV))
    glove = float(((hh >= 100) & (hh <= 130) & (ss >= 150) & (vv_ >= 110)).mean()); sleeve = float(((ss < 35) & (vv_ >= 200)).mean())
    occluded = int(glove > 0.01 or sleeve > 0.05)
    if Q is not None:
        # four-line opening: each well-seen wall at its own angle, hidden walls parallel to the opposite one
        inner_raw, inner_contour = inner_opening_mask(np.array(Q[str(k)]["corners"], np.float32), th, full.shape[:2])
        inside = cv2.dilate(inner_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # 2 camera px tolerance
        past_inner = ~inside[ys, xs].astype(bool)
        how = Q[str(k)]["how"]; walls = "".join(sd[0].upper() if how[sd] == "measured" else sd[0] for sd in ("left", "right", "top", "bottom"))
    else:
        ui, vi = tray_uv(xs, ys, q["cx"], q["cy"], q["th"])
        past_inner = (np.abs(ui) > q["W"] / 2 + 2) | (np.abs(vi) > q["H"] / 2 + 2)     # 2 camera px tolerance
        walls = q["src"]
        inner_raw = np.zeros(full.shape[:2], np.uint8)
        cv2.fillPoly(inner_raw, [np.round(rect_corners(q["cx"], q["cy"], q["W"], q["H"], q["th"])).astype(np.int32)], 1)
    uo, vo = tray_uv(xs, ys, cx, cy, th)
    past_outer = (np.abs(uo) > W / 2 + 6) | (np.abs(vo) > H / 2 + 6)
    lip = past_inner & ~past_outer; beyond = past_outer
    # lip band = inside the green box but outside the (rounded) magenta opening; how much of it meat covers
    outer_m = np.zeros(full.shape[:2], np.uint8)
    cv2.fillPoly(outer_m, [np.round(rect_corners(cx, cy, W, H, th)).astype(np.int32)], 1)
    band = outer_m.astype(bool) & ~inner_raw.astype(bool)
    band_px = int(band.sum()); band_meat = int(band[ys, xs].sum())
    cover = 100.0 * band_meat / max(band_px, 1)
    n_lip, n_bey = int(lip.sum()), int(beyond.sum())
    pct = 100.0 * (n_lip + n_bey) / max(len(xs), 1)
    large = int(n_bey >= LARGE_PX)
    if not NO_IMAGE:   # overlay image for review; production (customer view) only needs the numbers
        vis = full.astype(np.float32)
        for m, col in ((lip, ORANGE), (beyond, YELLOW)):
            yy, xx = ys[m], xs[m]
            vis[yy, xx] = vis[yy, xx] * (1 - ALPHA) + col * ALPHA
        vis = vis.astype(np.uint8)
        cv2.polylines(vis, [rect_corners(cx, cy, W, H, th).astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
        if Q is not None:
            cv2.polylines(vis, [inner_contour], True, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis, [rect_corners(q["cx"], q["cy"], q["W"], q["H"], q["th"]).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        c = rect_corners(cx, cy, W, H, th)
        x0, y0 = np.maximum(c.min(0) - 60, 0).astype(int); x1, y1 = np.minimum(c.max(0) + 60, [full.shape[1], full.shape[0]]).astype(int)
        crop = vis[y0:y1, x0:x1].copy()
        ts = float(r['t_sec']); tlab = f"{int(ts//3600):02d}:{int(ts%3600//60):02d}:{ts%60:04.1f}" if ts >= 3600 else f"{ts}s"
        flags = ("  NEXT TRAY " + "".join(n[0].upper() for n in ("left", "right") if nb[n]) if (nb["left"] or nb["right"]) else "") + ("  HAND" if occluded else "")
        l1 = f"#{k}  lip covered {cover:.1f}%   outside magenta {n_lip + n_bey:,} px   past green {n_bey:,} px"
        l2 = f"t={tlab}  ({pct:.1f}% of meat outside magenta)  walls {walls}{flags}"
        bar = np.zeros((60, crop.shape[1], 3), np.uint8)
        cv2.putText(bar, l1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if large:
            (tw, _), _ = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(bar, "LARGE OVERHANG", (8 + tw + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2, cv2.LINE_AA)
            cv2.rectangle(crop, (1, 1), (crop.shape[1] - 2, crop.shape[0] - 2), RED, 4)
        cv2.putText(bar, l2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/tray_{k:03d}.jpg", np.vstack([bar, crop]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return dict(idx=k, session=r.get("session", ""), t_sec=float(r["t_sec"]),
                lip_cover_pct=round(cover, 2), outside_px=n_lip + n_bey, past_green_px=n_bey, large=large,
                lip_band_px=band_px, lip_band_meat_px=band_meat, lip_px=n_lip, beyond_px=n_bey,
                        outside_pct=round(pct, 2), meat_px=int(len(xs)), walls=walls,
                        next_tray_left=int(nb["left"]), next_tray_right=int(nb["right"]), dropped_next_tray_px=n_dropped,
                        glove_frac=round(glove, 4), white_frac=round(sleeve, 4), occluded=occluded)


if __name__ == "__main__":
    from multiprocessing import Pool
    rows = [r for r in csv.DictReader(open(f"{RUN}/results_full.csv")) if r["status"] != "no_tray"]
    with Pool(int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 1)))) as pool:
        summary = sorted(pool.imap_unordered(render_one, rows, chunksize=8), key=lambda d: d["idx"])
    with open(f"{RUN}/inner_overlay_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print("rendered", len(summary))
    o = np.array([s["outside_px"] for s in summary]); p = np.array([s["outside_pct"] for s in summary])
    c = np.array([s["lip_cover_pct"] for s in summary]); print("lip covered % p50/p90/p99:", np.percentile(c, [50, 90, 99]).round(1), " large:", sum(s["large"] for s in summary))
    print("outside magenta (camera px) p50/p90/max:", np.percentile(o, [50, 90]).round(0), o.max(), " % of meat p50/p90:", np.percentile(p, [50, 90]).round(2))
