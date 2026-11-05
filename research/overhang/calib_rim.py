"""Per-side gap between tray outer edge and meat on clean trays, plus where the
visible tray floor/wall boundary sits (dark floor vs lighter rim) as a cross-check."""
import csv, sys, json, numpy as np
sys.path.insert(0, "overhang"); from fit_edges import trusted_edge, refine
from measure_full import TRAY_W, TRAY_H, tray_uv
S = json.load(open("out/run1/tray_size.json")); W, H = S["W"], S["H"]
rows = list(csv.DictReader(open("out/run1/results_full.csv")))
gaps = {k: [] for k in ("left", "right", "top", "bottom")}
for r in rows:
    if r["status"] != "ok" or int(r["beyond_px"]) > 0: continue
    z = np.load(f"out/run1/cache/{int(r['idx']):03d}.npz"); tray, own, red = z["tray"], z["own"], z["red"]
    cx, cy, th = float(r["tray_cx"]), float(r["tray_cy"]), np.deg2rad(float(r["tray_deg"]))
    pts = trusted_edge(tray, own, red & ~own, cx, cy, th, TRAY_W, TRAY_H)
    cx, cy, th, _, _, info = refine(pts, cx, cy, th, W, H)
    ys, xs = np.nonzero(own); u, v = tray_uv(xs, ys, cx, cy, th)
    mid_v = np.abs(v) < 0.3 * H; mid_u = np.abs(u) < 0.3 * W
    if mid_v.sum() > 100:
        gaps["left"].append(np.percentile(u[mid_v] + W / 2, 2)); gaps["right"].append(np.percentile(W / 2 - u[mid_v], 2))
    if mid_u.sum() > 100:
        gaps["top"].append(np.percentile(v[mid_u] + H / 2, 2)); gaps["bottom"].append(np.percentile(H / 2 - v[mid_u], 2))
for k, g in gaps.items():
    print(k, "n", len(g), "gap pct 5/10/25/50/75:", np.percentile(g, [5, 10, 25, 50, 75]).round(1))
