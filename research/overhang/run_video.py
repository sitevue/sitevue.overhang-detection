"""Overhang prototype over a video.

Pass 1 (cheap): colour-segment meat every frame, track meat blobs, pick one frame
per tray where it is fully in view and closest to the frame centre.
Pass 2: SAM 2 tray mask for that tray, tray footprint = convex hull of the tray
mask, overhang = this tray's meat outside the footprint.

usage: python overhang/run_video.py VIDEO OUTDIR
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "overhang")
from geom import WORK_W, masks, meat_blobs, smooth_mask  # noqa: E402
from sam_tray import load, segment  # noqa: E402

STEP = 2              # analyse every 2nd frame in pass 1 (5 fps)
MIN_AREA = 12000      # meat blob area (half-res px) to count as a tray
MATCH_PX = 90         # max centroid jump between analysed frames


def read_small(cap):
    ok, f = cap.read()
    if not ok:
        return None
    h = int(f.shape[0] * WORK_W / f.shape[1])
    return cv2.resize(f, (WORK_W, h), interpolation=cv2.INTER_AREA)


def pass1(path):
    cap = cv2.VideoCapture(path)
    tracks, active, i = [], [], 0
    while True:
        f = read_small(cap)
        if f is None:
            break
        if i % STEP == 0:
            H, W = f.shape[:2]
            red, _ = masks(f)
            _, blobs = meat_blobs(red, MIN_AREA)
            used, nxt = set(), []
            for tr in active:
                last = tr["obs"][-1]
                best, bd = None, MATCH_PX
                for j, b in enumerate(blobs):
                    d = np.hypot(b["c"][0] - last["c"][0], b["c"][1] - last["c"][1])
                    if j not in used and d < bd:
                        best, bd = j, d
                if best is not None:
                    used.add(best)
                    b = blobs[best]
                    tr["obs"].append(dict(f=i, c=b["c"], bbox=b["bbox"], area=b["area"]))
                    nxt.append(tr)
                elif i - last["f"] <= 3 * STEP:
                    nxt.append(tr)  # tolerate a short drop-out
            for j, b in enumerate(blobs):
                if j not in used:
                    tr = dict(id=len(tracks), obs=[dict(f=i, c=b["c"], bbox=b["bbox"], area=b["area"])])
                    tracks.append(tr)
                    nxt.append(tr)
            active = nxt
        i += 1
    W = WORK_W
    picks = []
    for tr in tracks:
        full = [o for o in tr["obs"] if o["bbox"][0] > 12 and o["bbox"][0] + o["bbox"][2] < W - 12]
        if not full:
            continue
        o = min(full, key=lambda o: abs(o["c"][0] - W / 2))
        if abs(o["c"][0] - W / 2) > 140:
            continue  # never came near the centre: probably a fragment
        picks.append(dict(track=tr["id"], frame=o["f"], c=o["c"], area=o["area"],
                          n_obs=len(tr["obs"]), dx=tr["obs"][-1]["c"][0] - tr["obs"][0]["c"][0]))
    return picks, i


def measure(tray_mask, own):
    cnts, _ = cv2.findContours(tray_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts))
    foot = np.zeros(tray_mask.shape, np.uint8)
    cv2.fillPoly(foot, [hull], 1)
    foot = cv2.dilate(foot, np.ones((5, 5), np.uint8)).astype(bool)   # 2 px tolerance
    out = own & ~foot
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    dist = cv2.distanceTransform((~foot).astype(np.uint8), cv2.DIST_L2, 5)
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(hull)
    if rw < rh:                      # make u the long axis
        rw, rh, ang = rh, rw, ang + 90
    th = np.deg2rad(ang)
    ys, xs = np.nonzero(out)
    u = (xs - rcx) * np.cos(th) + (ys - rcy) * np.sin(th)
    v = -(xs - rcx) * np.sin(th) + (ys - rcy) * np.cos(th)
    side = np.where(np.abs(u) / rw > np.abs(v) / rh, np.where(u < 0, "end_left", "end_right"),
                    np.where(v < 0, "side_top", "side_bottom"))
    res = dict(tray_w=round(rw, 1), tray_h=round(rh, 1), tray_angle=round(ang, 1),
               meat_px=int(own.sum()), out_px=int(out.sum()),
               out_frac=round(out.sum() / max(own.sum(), 1), 4),
               out_maxd=round(float(dist[out].max()) if out.any() else 0.0, 1))
    for s in ("end_left", "end_right", "side_top", "side_bottom"):
        m = side == s
        res[s + "_px"] = int(m.sum())
        res[s + "_maxd"] = round(float(dist[ys[m], xs[m]].max()) if m.any() else 0.0, 1)
    my, mx = np.nonzero(own & foot)
    fy, fx = np.nonzero(foot)
    res["centroid_off_x"] = round(float(mx.mean() - fx.mean()), 1) if len(mx) else None
    res["centroid_off_y"] = round(float(my.mean() - fy.mean()), 1) if len(my) else None
    return res, hull, out


def pass2(path, picks, outdir, proc, model):
    cap = cv2.VideoCapture(path)
    rows = []
    for k, p in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
        f = read_small(cap)
        red, _ = masks(f)
        sm = smooth_mask(f, red)
        lab, blobs = meat_blobs(red, 4000)
        j = int(np.argmin([np.hypot(b["c"][0] - p["c"][0], b["c"][1] - p["c"][1]) for b in blobs]))
        # SAM on this blob, the others act as negatives
        seg = segment(proc, model, f, blobs, lab, sm, only=j)[0]
        own = lab == blobs[j]["id"]
        row = dict(idx=k, track=p["track"], frame=p["frame"], t_sec=round(p["frame"] / 10, 1),
                   cx=round(p["c"][0]), dx_track=round(p["dx"]))
        if seg is None:
            row["status"] = "no_rim_points"
            rows.append(row)
            continue
        row["sam_score"] = round(seg["score"], 3)
        m, hull, out = measure(seg["mask"], own)
        row.update(m)
        row["status"] = "ok"
        vis = f.copy()
        tm = seg["mask"]
        vis[tm] = (vis[tm] * 0.55 + np.array([255, 140, 0]) * 0.45).astype(np.uint8)
        vis[out] = (0, 255, 255)
        cv2.polylines(vis, [hull], True, (0, 255, 0), 2)
        txt = f"#{k} t={row['t_sec']}s out={m['out_px']}px ({100*m['out_frac']:.1f}%) maxd={m['out_maxd']}"
        cv2.rectangle(vis, (0, 0), (WORK_W, 26), (0, 0, 0), -1)
        cv2.putText(vis, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, f"tray_{k:03d}.jpg"), vis)
        rows.append(row)
        print(txt, flush=True)
    return rows


if __name__ == "__main__":
    video, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    picks, n = pass1(video)
    print(f"pass1: {n} frames, {len(picks)} trays picked", flush=True)
    json.dump(picks, open(os.path.join(outdir, "picks.json"), "w"), indent=1)
    if os.environ.get("PICKS_ONLY"):
        sys.exit(0)            # measure_full.py does SAM + measurement
    proc, model = load()
    rows = pass2(video, picks, outdir, proc, model)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("done", len(rows))
