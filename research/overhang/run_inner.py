"""Two-pass rigid inner-opening fit on all trays: detect -> calibrate size -> fit -> re-detect near fit -> refit.

Images are loaded per tray inside worker processes (never all at once), so memory stays
flat however many trays there are. Same algorithm as before.
"""
import os as _os
RUN = _os.environ.get("RUN", "out/run1")
import csv, sys, json, time, cv2, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "overhang")
from inner_rigid import prep, all_points, initial_inner, fit_inner, SIDES
from measure_full import TRAY_W, TRAY_H, LIP_DEFAULT
S = 2.0
WORKERS = int(_os.environ.get("WORKERS", max(1, (_os.cpu_count() or 2) - 1)))


def outer_of(r):
    return (float(r["tray_cx"]) * S, float(r["tray_cy"]) * S, np.deg2rad(float(r["tray_deg"])), TRAY_W * S, TRAY_H * S)


def init_of(o):
    ix, iy, iW, iH = initial_inner(*o, {k_: v * S for k_, v in LIP_DEFAULT.items()})
    return (ix, iy, o[2], iW, iH)


def points_for(job):
    """Worker: load one tray's full-res frame + meat mask, return its inner-edge points."""
    k, o, prev = job
    full = cv2.imread(f"{RUN}/cache_full/{k:03d}.png")
    z = np.load(f"{RUN}/cache/{k:03d}.npz")
    meat = cv2.dilate(cv2.resize(z["red"].astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST),
                      np.ones((7, 7), np.uint8)).astype(bool)
    g, gx, gy = prep(full)
    inner = None if prev is None else (prev["cx"], prev["cy"], prev["th"], prev["W"], prev["H"])
    P = all_points(g, gx, gy, meat, *o, inner=inner)
    return k, P


def run_pass(rows, inner_prev, W_in, H_in, calibrate, pool):
    jobs = [(int(r["idx"]), outer_of(r), None if inner_prev is None else inner_prev.get(int(r["idx"]))) for r in rows]
    clean = {int(r["idx"]): int(r["beyond_px"]) == 0 for r in rows}
    outers = {j[0]: j[1] for j in jobs}
    pts, Ws, Hs = {}, [], []
    for k, P in pool.imap_unordered(points_for, jobs, chunksize=8):
        pts[k] = P
    inits = {}
    for k, o in outers.items():
        prev = None if inner_prev is None else inner_prev.get(k)
        inits[k] = init_of(o) if prev is None else (prev["cx"], prev["cy"], prev["th"], prev["W"], prev["H"])
        n = {sd: len(pts[k][sd]) for sd in SIDES}
        if calibrate and clean[k] and min(n.values()) >= 20:
            f = fit_inner(pts[k], inits[k])
            if f and min(f["side_counts"]) >= 15:
                Ws.append(f["W"]); Hs.append(f["H"])
    if calibrate:
        W_in, H_in = float(np.median(Ws)), float(np.median(Hs))
        print(f"  calibrated on {len(Ws)} trays: W_in {W_in:.1f} (p10-90 {np.percentile(Ws,10):.1f}-{np.percentile(Ws,90):.1f})  "
              f"H_in {H_in:.1f} (p10-90 {np.percentile(Hs,10):.1f}-{np.percentile(Hs,90):.1f})", flush=True)
    out, rms = {}, []
    for k, o in outers.items():
        ii = init_of(o)
        f = fit_inner(pts[k], inits[k], W_in, H_in)
        if f is None:
            out[k] = dict(src="default", cx=ii[0], cy=ii[1], th=o[2], W=W_in, H=H_in); continue
        sc = dict(zip(("right", "left", "bottom", "top"), f["side_counts"]))
        c, s = np.cos(o[2]), np.sin(o[2])
        du = (f["cx"] - ii[0]) * c + (f["cy"] - ii[1]) * s; dv = -(f["cx"] - ii[0]) * s + (f["cy"] - ii[1]) * c
        if sc["left"] < 10 and sc["right"] < 10: du = 0.0
        if sc["top"] < 10 and sc["bottom"] < 10: dv = 0.0
        out[k] = dict(src="".join(sd[0].upper() if sc[sd] >= 10 else sd[0] for sd in SIDES),
                      cx=ii[0] + du * c - dv * s, cy=ii[1] + du * s + dv * c, th=f["th"], W=W_in, H=H_in, rms=f["rms"])
        if f["rms"] is not None: rms.append(f["rms"])
    return pts, out, W_in, H_in, rms


if __name__ == "__main__":
    t0 = time.time()
    rows = [r for r in csv.DictReader(open(f"{RUN}/results_full.csv")) if r["status"] != "no_tray"]
    print(f"{len(rows)} trays, {WORKERS} workers", flush=True)
    with Pool(WORKERS) as pool:
        print("pass 1", flush=True); pts1, fit1, W_in, H_in, rms1 = run_pass(rows, None, None, None, True, pool)
        print(f"  rms p50/p90 {np.percentile(rms1,[50,90]).round(2)}  ({time.time()-t0:.0f}s)", flush=True)
        print("pass 2 (search +-8 px around pass-1 fit)", flush=True); pts2, fit2, W_in, H_in, rms2 = run_pass(rows, fit1, W_in, H_in, True, pool)
    print(f"  rms p50/p90 {np.percentile(rms2,[50,90]).round(2)}")
    cnt = {sd: sum(len(pts2[k][sd]) >= 10 for k in pts2) for sd in SIDES}; print("  sides with >=10 points:", cnt)
    srcs = [v["src"] for v in fit2.values()]; print("  sides used per tray:", {n: sum(sum(ch.isupper() for ch in x) == n for x in srcs) for n in range(5)})
    json.dump({str(k): v for k, v in fit2.items()}, open(f"{RUN}/inner_fit.json", "w"))
    json.dump({str(k): {sd: v[sd].tolist() for sd in SIDES} for k, v in pts2.items()}, open(f"{RUN}/inner_points.json", "w"))
    json.dump(dict(W_in=W_in, H_in=H_in), open(f"{RUN}/inner_size.json", "w"))
    print(f"{(time.time()-t0):.0f}s total")
