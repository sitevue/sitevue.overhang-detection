import csv, sys, json, cv2, numpy as np
sys.path.insert(0, "overhang"); from fit_edges import trusted_edge, refine
from measure_full import TRAY_W, TRAY_H, tray_uv
S = json.load(open("out/run1/tray_size.json")); W, H = S["W"], S["H"]
rows = list(csv.DictReader(open("out/run1/results_full.csv")))
D = np.arange(0, 56)
prof = {k: [] for k in ("left", "right", "top", "bottom")}
for r in rows:
    if r["status"] != "ok" or int(r["beyond_px"]) > 0: continue
    k = int(r["idx"]); z = np.load(f"out/run1/cache/{k:03d}.npz"); tray, own, red = z["tray"], z["own"], z["red"]
    g = cv2.cvtColor(cv2.imread(f"out/run1/cache/{k:03d}.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cx, cy, th = float(r["tray_cx"]), float(r["tray_cy"]), np.deg2rad(float(r["tray_deg"]))
    pts = trusted_edge(tray, own, red & ~own, cx, cy, th, TRAY_W, TRAY_H)
    cx, cy, th, _, _, _ = refine(pts, cx, cy, th, W, H)
    c, s = np.cos(th), np.sin(th)
    meatd = cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    for side, (axis, sign, half, other) in dict(left=(0, -1, W/2, H), right=(0, 1, W/2, H), top=(1, -1, H/2, W), bottom=(1, 1, H/2, W)).items():
        t = np.linspace(-0.3, 0.3, 40) * other
        dd = half - D[:, None]                       # inward distance -> coordinate
        uu = (sign * dd) if axis == 0 else np.broadcast_to(t, (len(D), len(t)))
        vv = (sign * dd) if axis == 1 else np.broadcast_to(t, (len(D), len(t)))
        uu = np.broadcast_to(uu, (len(D), len(t))); vv = np.broadcast_to(vv, (len(D), len(t)))
        x = (cx + uu * c - vv * s).round().astype(int); y = (cy + uu * s + vv * c).round().astype(int)
        if x.min() < 0 or y.min() < 0 or x.max() >= g.shape[1] or y.max() >= g.shape[0]: continue
        col_ok = ~meatd[y, x].any(0)
        if col_ok.sum() < 10: continue
        gg = g[y, x][:, col_ok]
        prof[side].append(gg.mean(1))
for side, P in prof.items():
    if not P: print(side, "no empty strips"); continue
    m = np.median(np.array(P), 0)
    print(f"{side:6s} n={len(P):3d} ", " ".join(f"{int(v):3d}" for v in m[::2]))
