"""Detect inner-edge points on every tray (full res) and calibrate the inner opening size
from trays where all four inner walls are visible."""
import csv, sys, json, time, cv2, numpy as np
sys.path.insert(0, "overhang")
from inner_rigid import prep, all_points, initial_inner, fit_inner, SIDES
from measure_full import TRAY_W, TRAY_H, LIP_DEFAULT

rows = [r for r in csv.DictReader(open("out/run1/results_full.csv")) if r["status"] != "no_tray"]
S = 2.0
store, Ws, Hs, rms = {}, [], [], []
cnt = {s: 0 for s in SIDES}; t0 = time.time()
for r in rows:
    k = int(r["idx"])
    full = cv2.imread(f"out/run1/cache_full/{k:03d}.png")
    z = np.load(f"out/run1/cache/{k:03d}.npz")
    meat = cv2.resize(z["red"].astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    meat = cv2.dilate(meat, np.ones((7, 7), np.uint8)).astype(bool)
    g, gx, gy = prep(full)
    cx, cy, th = float(r["tray_cx"]) * S, float(r["tray_cy"]) * S, np.deg2rad(float(r["tray_deg"]))
    W, H = TRAY_W * S, TRAY_H * S
    pts = all_points(g, gx, gy, meat, cx, cy, th, W, H)
    store[k] = {sd: pts[sd].tolist() for sd in SIDES}
    n = {sd: len(pts[sd]) for sd in SIDES}
    for sd in SIDES: cnt[sd] += n[sd] >= 10
    if min(n.values()) >= 20 and int(r["beyond_px"]) == 0:
        init = (*initial_inner(cx, cy, th, W, H, {k_: v * S for k_, v in LIP_DEFAULT.items()})[:2], th,
                *initial_inner(cx, cy, th, W, H, {k_: v * S for k_, v in LIP_DEFAULT.items()})[2:])
        f = fit_inner(pts, init)
        if f and min(f["side_counts"]) >= 15:
            Ws.append(f["W"]); Hs.append(f["H"]); rms.append(f["rms"])
print(f"{(time.time()-t0)/len(rows)*1000:.0f} ms/tray")
print("sides with >=10 points:", cnt, "of", len(rows))
Ws, Hs = np.array(Ws), np.array(Hs)
print("calibration trays:", len(Ws), " W_in p10/50/90", np.percentile(Ws, [10, 50, 90]).round(1), " H_in", np.percentile(Hs, [10, 50, 90]).round(1), " fit rms median", np.median(rms).round(2))
json.dump(dict(W_in=float(np.median(Ws)), H_in=float(np.median(Hs)), n=len(Ws)), open("out/run1/inner_size.json", "w"))
json.dump(store, open("out/run1/inner_points.json", "w"))
