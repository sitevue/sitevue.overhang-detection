"""Fit the calibrated, fixed-size inner opening on every tray (full res) and save it."""
import csv, sys, json, numpy as np
sys.path.insert(0, "overhang")
from inner_rigid import initial_inner, fit_inner, SIDES
from measure_full import TRAY_W, TRAY_H, LIP_DEFAULT
S = 2.0
size = json.load(open("out/run1/inner_size.json")); W_in, H_in = size["W_in"], size["H_in"]
P = json.load(open("out/run1/inner_points.json"))
rows = [r for r in csv.DictReader(open("out/run1/results_full.csv")) if r["status"] != "no_tray"]
out, rms, nsides = {}, [], []
for r in rows:
    k = int(r["idx"]); pts = {sd: np.array(P[str(k)][sd]).reshape(-1, 4) for sd in SIDES}
    cx, cy, th = float(r["tray_cx"]) * S, float(r["tray_cy"]) * S, np.deg2rad(float(r["tray_deg"]))
    W, H = TRAY_W * S, TRAY_H * S
    ix, iy, _, _ = initial_inner(cx, cy, th, W, H, {k_: v * S for k_, v in LIP_DEFAULT.items()})
    f = fit_inner(pts, (ix, iy, th, W_in, H_in), W_in, H_in)
    if f is None:
        out[k] = dict(src="default", cx=ix, cy=iy, th=th, W=W_in, H=H_in); continue
    # sides that actually constrained the fit (refine order: right, left, bottom, top)
    sc = dict(zip(("right", "left", "bottom", "top"), f["side_counts"]))
    used = "".join(sd[0].upper() if sc[sd] >= 10 else sd[0] for sd in SIDES)
    # if neither end (left/right) was seen, keep the outer-based position along the long axis; same for top/bottom
    c, s = np.cos(th), np.sin(th)
    du = (f["cx"] - ix) * c + (f["cy"] - iy) * s; dv = -(f["cx"] - ix) * s + (f["cy"] - iy) * c
    if sc["left"] < 10 and sc["right"] < 10: du = 0.0
    if sc["top"] < 10 and sc["bottom"] < 10: dv = 0.0
    f["cx"], f["cy"] = ix + du * c - dv * s, iy + du * s + dv * c
    out[k] = dict(src=used, cx=f["cx"], cy=f["cy"], th=f["th"], W=W_in, H=H_in, rms=f["rms"])
    rms.append(f["rms"]); nsides.append(sum(ch.isupper() for ch in used))
json.dump({str(k): v for k, v in out.items()}, open("out/run1/inner_fit.json", "w"))
print("fitted", len(rms), "of", len(rows), " rms p50/p90", np.percentile(rms, [50, 90]).round(2),
      " sides used:", {n: nsides.count(n) for n in sorted(set(nsides))})
# angle difference inner vs outer
d = [np.rad2deg(out[int(r["idx"])]["th"] - np.deg2rad(float(r["tray_deg"]))) for r in rows if out[int(r["idx"])]["src"] != "default"]
print("inner minus outer angle, deg p10/50/90:", np.percentile(d, [10, 50, 90]).round(2))
