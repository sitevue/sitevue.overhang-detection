"""Stage 2 with a known-size tray rectangle.

For each picked tray (picks.json from run_video.py):
  - SAM 2 tray mask (cached in OUTDIR/cache/)
  - fit a rectangle of the known tray size that is explained by (tray | meat)
    and contains the visible tray
  - classify this tray's meat:  inside well | on rim (inside tray outline,
    within RIM px of the edge) | beyond the tray outline
usage: python overhang/measure_full.py VIDEO OUTDIR [TRAY_W TRAY_H]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "overhang")
from geom import WORK_W, masks, meat_blobs, rect_corners, smooth_mask  # noqa: E402
from fit_edges import anchor_spill, drop_sides, refine, trusted_edge  # noqa: E402
from tray_zones import inner_edges, tray_zones  # noqa: E402

TRAY_W, TRAY_H = 389.6, 290.8     # median edge-fitted size of clean trays (calib_size.py), half-res px
if os.environ.get("TRAY_SIZE_JSON") and os.path.exists(os.environ["TRAY_SIZE_JSON"]):   # per-camera calibration
    _ts = json.load(open(os.environ["TRAY_SIZE_JSON"])); TRAY_W, TRAY_H = _ts["W"], _ts["H"]
RIM = 16                          # (legacy single value)
# apparent lip width per side, median over the video (tray_zones.py); left/bottom read
# smaller and right/top larger, i.e. the fitted outline sits ~3-4 px up-left of the true edge
LIP_DEFAULT = dict(left=10.4, right=18.9, top=15.3, bottom=12.9)   # median fitted inner-edge width per side (inner_edges)
LIP_MARGIN = 2                    # px given to fit error
MAX_TILT_DEG = 30.0              # trays travel lengthwise; a box tilted more than this is wrong
BELT_INSIDE_MAX = 0.01           # more than 1% belt-roller yellow inside the box -> box is misplaced
INNER_TOL = 1.0                   # half-res px (= 2 camera px) tolerance on the rigid inner opening
TOL = 3                           # tolerance on the tray outline


def get_masks(video, outdir, picks):
    cache = os.path.join(outdir, "cache")
    os.makedirs(cache, exist_ok=True)
    need = [k for k in range(len(picks)) if not os.path.exists(os.path.join(cache, f"{k:03d}.npz"))]
    if need:
        from sam_tray import load, segment
        proc, model = load()
        cap = cv2.VideoCapture(video)
        full_dir = os.path.join(outdir, "cache_full")
        os.makedirs(full_dir, exist_ok=True)
        # walk the video once, in frame order: grab() skips frames cheaply, no back-and-forth seeking
        pos, full = 0, None
        for k in sorted(need, key=lambda k_: picks[k_]["frame"]):
            p = picks[k]
            if full is None or p["frame"] != pos - 1:
                while pos < p["frame"]:
                    cap.grab(); pos += 1
                ok, full = cap.read(); pos += 1
            if not cv2.imwrite(os.path.join(full_dir, f"{k:03d}.png"), full, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise IOError("could not write full-res frame")
            f = cv2.resize(full, (WORK_W, int(full.shape[0] * WORK_W / full.shape[1])), interpolation=cv2.INTER_AREA)
            red, _ = masks(f)
            sm = smooth_mask(f, red)
            lab, blobs = meat_blobs(red, 4000)
            j = int(np.argmin([np.hypot(b["c"][0] - p["c"][0], b["c"][1] - p["c"][1]) for b in blobs]))
            seg = segment(proc, model, f, blobs, lab, sm, only=j)[0]
            own = lab == blobs[j]["id"]
            tray = seg["mask"] if seg else np.zeros_like(own)
            np.savez_compressed(os.path.join(cache, f"{k:03d}.npz"), tray=tray, own=own,
                                red=red, score=seg["score"] if seg else 0.0)
            cv2.imwrite(os.path.join(cache, f"{k:03d}.jpg"), f)
            print("sam", k, flush=True)


def tray_uv(xs, ys, cx, cy, th):
    c, s = np.cos(th), np.sin(th)
    return (xs - cx) * c + (ys - cy) * s, -(xs - cx) * s + (ys - cy) * c


def fit_full(tray, own, W=TRAY_W, H=TRAY_H):
    fill = tray | own
    cnts, _ = cv2.findContours(tray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx0, cy0), (w0, h0), a0 = cv2.minAreaRect(np.vstack(cnts))
    if w0 < h0:
        a0 += 90
    a0 = (a0 + 90) % 180 - 90
    # trays always travel lengthwise along the belt (long axis ~horizontal in this camera).
    # A tray cut off by the frame edge or partly covered can look taller than wide; turn such
    # a start back to lengthwise instead of fitting a sideways box.
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


def lip_lines(lip):
    """Normalise per-side lip spec to (a, b): width = a + b*t along the side.
    Accepts a number (constant width) or an inner_edges() entry."""
    out = {}
    for k in ("left", "right", "top", "bottom"):
        v = (lip or {}).get(k)
        if v is None:
            out[k] = (LIP_DEFAULT[k], 0.0)
        elif isinstance(v, dict):
            out[k] = (v["a"], v["b"])
        else:
            out[k] = (float(v), 0.0)
    return out


def classify(own, cx, cy, th, W=TRAY_W, H=TRAY_H, lip=None, inner=None):
    """inner: rigid inner opening (cx, cy, th, W, H) in the same (half-res) coords; when given,
    meat outside it (but inside the outer outline) counts as on the lip."""
    ln = lip_lines(lip)
    ys, xs = np.nonzero(own)
    u, v = tray_uv(xs, ys, cx, cy, th)
    du, dv = np.abs(u) - W / 2, np.abs(v) - H / 2       # >0 means beyond that edge
    beyond = (du > TOL) | (dv > TOL)
    # lip width at each pixel's position along its side (t = v for left/right, u for top/bottom)
    Lw = {k: np.maximum(a + b * (v if k in ("left", "right") else u) - LIP_MARGIN, 4) for k, (a, b) in ln.items()}
    rim = ~beyond & (((u < 0) & (du > -Lw["left"])) | ((u >= 0) & (du > -Lw["right"])) |
                     ((v < 0) & (dv > -Lw["top"])) | ((v >= 0) & (dv > -Lw["bottom"])))
    if inner is not None:
        icx, icy, ith, iW, iH = inner
        ui, vi = tray_uv(xs, ys, icx, icy, ith)
        rim = ~beyond & ((np.abs(ui) > iW / 2 + INNER_TOL) | (np.abs(vi) > iH / 2 + INNER_TOL))
    side = np.where(du / W > dv / H, np.where(u < 0, "left", "right"), np.where(v < 0, "top", "bottom"))
    res = dict(meat_px=len(xs), beyond_px=int(beyond.sum()), rim_px=int(rim.sum()),
               beyond_frac=round(beyond.mean(), 4), rim_frac=round(rim.mean(), 4))
    d = np.maximum(du, dv)
    res["beyond_maxd"] = round(float(np.percentile(d[beyond], 99)), 1) if beyond.sum() > 30 else 0.0
    for s in ("left", "right", "top", "bottom"):
        res[f"beyond_{s}_px"] = int((beyond & (side == s)).sum())
        res[f"rim_{s}_px"] = int((rim & (side == s)).sum())
    # the user's centroid idea, in tray coordinates (px, + = right/bottom)
    res["centroid_off_u"] = round(float(u.mean()), 1)
    res["centroid_off_v"] = round(float(v.mean()), 1)
    masks_ = dict(beyond=(xs[beyond], ys[beyond]), rim=(xs[rim], ys[rim]))
    return res, masks_


INNER_FIT = None


def main():
    global INNER_FIT
    video, outdir = sys.argv[1], sys.argv[2]
    pth = os.path.join(outdir, "inner_fit.json")
    INNER_FIT = json.load(open(pth)) if os.path.exists(pth) else None
    picks = json.load(open(os.path.join(outdir, "picks.json")))
    get_masks(video, outdir, picks)
    vdir = os.path.join(outdir, "full")
    os.makedirs(vdir, exist_ok=True)
    rows = []
    for k, p in enumerate(picks):
        z = np.load(os.path.join(outdir, "cache", f"{k:03d}.npz"))
        tray, own = z["tray"], z["own"]
        f = cv2.imread(os.path.join(outdir, "cache", f"{k:03d}.jpg"))
        row = dict(idx=k, frame=p["frame"], t_sec=round(p["frame"] / 10, 1), sam_score=round(float(z["score"]), 3))
        if not tray.any():
            row["status"] = "no_tray"
            rows.append(row)
            continue
        sc, cx, cy, th = fit_full(tray, own)
        # refine against the trusted part of SAM's outer tray edge
        # neighbouring meat = other large red blobs only (not the red belt markers)
        _, oblobs = meat_blobs(z["red"], 4000)
        lab_all = cv2.connectedComponents(z["red"].astype(np.uint8), connectivity=8)[1]
        other = np.isin(lab_all, [b["id"] for b in oblobs]) & ~own
        cx, cy, bad = anchor_spill(tray, own, other, cx, cy, th, TRAY_W, TRAY_H)
        pts = trusted_edge(tray, own, other, cx, cy, th, TRAY_W, TRAY_H)
        pts = drop_sides(pts, cx, cy, th, TRAY_W, TRAY_H, bad)
        row["spill_sides"] = "/".join(bad)
        if len(pts) >= 40:
            cx, cy, th, _, _, info = refine(pts, cx, cy, th, TRAY_W, TRAY_H)
            row.update(edge_pts=info["n_pts"], edge_rms=round(info["rms"], 2),
                       edge_sides="/".join(map(str, info["side_counts"])))
        row.update(fit_score=round(sc, 3), tray_cx=round(cx, 1), tray_cy=round(cy, 1), tray_deg=round(np.rad2deg(th), 1))
        # visible tray area far outside the fitted outline => SAM mask spilled (e.g. neighbour tray)
        ty, tx = np.nonzero(tray)
        tu, tv = tray_uv(tx, ty, cx, cy, th)
        row["tray_spill_frac"] = round(float(((np.abs(tu) > TRAY_W / 2 + 10) | (np.abs(tv) > TRAY_H / 2 + 10)).mean()), 3)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = inner_edges(g, own, cx, cy, th, TRAY_W, TRAY_H)   # fitted inner-edge line per side, or None
        lip = {sd: edges[sd] for sd in ("left", "right", "top", "bottom") if edges[sd]}
        ln = lip_lines(lip)
        row["lip_px"] = "/".join(f"{ln[k][0]:.1f}" for k in ("left", "right", "top", "bottom"))
        row["lip_src"] = "".join(sd[0].upper() if edges[sd] else sd[0] for sd in ("left", "right", "top", "bottom"))
        inner = None
        if INNER_FIT is not None and str(k) in INNER_FIT:
            q = INNER_FIT[str(k)]
            inner = (q["cx"] / 2, q["cy"] / 2, q["th"], q["W"] / 2, q["H"] / 2)
            row["inner_src"] = q["src"]
        res, mm = classify(own, cx, cy, th, lip=lip, inner=inner)
        row.update(res)
        # belt inside the box: if the outline is right, nothing inside it is belt. Belt rollers
        # (yellow) showing inside it mean the box is misplaced, rotated or spans two trays.
        _, yel_f = masks(f)
        bu, bv = np.meshgrid(np.linspace(-0.46, 0.46, 46) * TRAY_W, np.linspace(-0.44, 0.44, 34) * TRAY_H)
        bx = cx + bu * np.cos(th) - bv * np.sin(th); by = cy + bu * np.sin(th) + bv * np.cos(th)
        binb = (bx >= 0) & (bx < yel_f.shape[1]) & (by >= 0) & (by < yel_f.shape[0])
        row["belt_inside_frac"] = round(float(yel_f[by[binb].astype(int), bx[binb].astype(int)].mean()) if binb.any() else 0.0, 4)
        row["tray_deg_ok"] = int(abs(np.rad2deg(th)) <= MAX_TILT_DEG)
        row["status"] = "check" if (row["tray_spill_frac"] > 0.05 or sc < -0.25 or row["belt_inside_frac"] > BELT_INSIDE_MAX
                                    or not row["tray_deg_ok"]) else "ok"
        vis = f.copy()
        vis[tray] = (vis[tray] * 0.6 + np.array([255, 140, 0]) * 0.4).astype(np.uint8)
        vis[mm["rim"][1], mm["rim"][0]] = (0, 140, 255)
        vis[mm["beyond"][1], mm["beyond"][0]] = (0, 255, 255)
        cv2.polylines(vis, [rect_corners(cx, cy, TRAY_W, TRAY_H, th).astype(int)], True, (0, 255, 0), 2)
        # inner lip line, per side
        cth, sth = np.cos(th), np.sin(th)
        P = lambda a_, b_: (int(round(cx + a_ * cth - b_ * sth)), int(round(cy + a_ * sth + b_ * cth)))
        if inner is not None:
            cv2.polylines(vis, [rect_corners(*inner[:2], inner[3], inner[4], inner[2]).astype(int)], True, (255, 0, 255), 1)
        for sd, (a0, b0) in ([] if inner is not None else ln.items()):
            col = (255, 0, 255) if edges[sd] else (0, 120, 0)      # magenta = fitted, dark green = default
            if sd in ("left", "right"):
                sg = -1 if sd == "left" else 1
                ends = [P(sg * (TRAY_W / 2 - (a0 + b0 * t_)), t_) for t_ in (-TRAY_H / 2, TRAY_H / 2)]
            else:
                sg = -1 if sd == "top" else 1
                ends = [P(t_, sg * (TRAY_H / 2 - (a0 + b0 * t_))) for t_ in (-TRAY_W / 2, TRAY_W / 2)]
            cv2.line(vis, ends[0], ends[1], col, 1)
        txt = (f"#{k} t={row['t_sec']}s beyond={res['beyond_px']}px maxd={res['beyond_maxd']} "
               f"rim={100*res['rim_frac']:.0f}% {row['status']}")
        cv2.rectangle(vis, (0, 0), (WORK_W, 26), (0, 0, 0), -1)
        cv2.putText(vis, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(vdir, f"tray_{k:03d}.jpg"), vis)
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(outdir, "results_full.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))


if __name__ == "__main__":
    main()
