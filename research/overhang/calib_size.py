import os as _os
RUN = _os.environ.get("RUN", "out/run1")
import csv, sys, json, numpy as np
sys.path.insert(0, "overhang"); from fit_edges import trusted_edge, refine
from measure_full import TRAY_W, TRAY_H
rows = list(csv.DictReader(open(f"{RUN}/results_full.csv")))
Ws, Hs, rms = [], [], []
for r in rows:
    if r["status"] != "ok" or int(r["beyond_px"]) > 0: continue
    z = np.load(f"{RUN}/cache/{int(r['idx']):03d}.npz"); tray, own, red = z["tray"], z["own"], z["red"]
    cx, cy, th = float(r["tray_cx"]), float(r["tray_cy"]), np.deg2rad(float(r["tray_deg"]))
    pts = trusted_edge(tray, own, red & ~own, cx, cy, th, TRAY_W, TRAY_H)
    cx, cy, th, W, H, info = refine(pts, cx, cy, th, TRAY_W, TRAY_H, fit_size=True)
    if min(info["side_counts"]) >= 40:
        Ws.append(W); Hs.append(H); rms.append(info["rms"])
Ws, Hs = np.array(Ws), np.array(Hs)
print("n", len(Ws), "W pct", np.percentile(Ws, [10, 25, 50, 75, 90]).round(1), "H pct", np.percentile(Hs, [10, 25, 50, 75, 90]).round(1), "rms med", np.median(rms).round(2))
json.dump(dict(W=float(np.median(Ws)), H=float(np.median(Hs))), open(f"{RUN}/tray_size.json", "w"))
