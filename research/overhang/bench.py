import sys, time, json, cv2, numpy as np, torch
sys.path.insert(0, "overhang")
from geom import masks, meat_blobs, smooth_mask, WORK_W
from sam_tray import load, segment, dev
from measure_full import fit_full, classify, TRAY_W, TRAY_H
from fit_edges import anchor_spill, trusted_edge, drop_sides, refine
from tray_zones import tray_zones
V = "/Users/Pratham/Downloads/overhang-1000-1010.mp4"
cap = cv2.VideoCapture(V); t_read = t_col = 0; n = 300
for i in range(n):
    t = time.perf_counter(); ok, f = cap.read(); f = cv2.resize(f, (WORK_W, 544), interpolation=cv2.INTER_AREA); t_read += time.perf_counter() - t
    t = time.perf_counter(); red, _ = masks(f); meat_blobs(red, 12000); t_col += time.perf_counter() - t
print(f"per frame: decode+resize {1000*t_read/n:.1f} ms, colour+blobs {1000*t_col/n:.1f} ms")
proc, model = load(); picks = json.load(open("out/run1/picks.json"))[:25]
ts = {"sam": [], "fit": [], "refine": [], "zones+classify": []}
for k, p in enumerate(picks):
    cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"]); ok, f = cap.read(); f = cv2.resize(f, (WORK_W, 544), interpolation=cv2.INTER_AREA)
    red, _ = masks(f); sm = smooth_mask(f, red); lab, blobs = meat_blobs(red, 4000)
    j = int(np.argmin([np.hypot(b["c"][0]-p["c"][0], b["c"][1]-p["c"][1]) for b in blobs])); own = lab == blobs[j]["id"]
    if dev == "mps": torch.mps.synchronize()
    t = time.perf_counter(); seg = segment(proc, model, f, blobs, lab, sm, only=j)[0]
    if dev == "mps": torch.mps.synchronize()
    ts["sam"].append(time.perf_counter() - t)
    if seg is None: continue
    tray = seg["mask"]; other = red & ~own
    t = time.perf_counter(); sc, cx, cy, th = fit_full(tray, own); ts["fit"].append(time.perf_counter() - t)
    t = time.perf_counter(); cx, cy, bad = anchor_spill(tray, own, other, cx, cy, th, TRAY_W, TRAY_H)
    pts = drop_sides(trusted_edge(tray, own, other, cx, cy, th, TRAY_W, TRAY_H), cx, cy, th, TRAY_W, TRAY_H, bad)
    if len(pts) >= 40: cx, cy, th, *_ = refine(pts, cx, cy, th, TRAY_W, TRAY_H)
    ts["refine"].append(time.perf_counter() - t)
    t = time.perf_counter(); tray_zones(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), own, cx, cy, th, TRAY_W, TRAY_H); classify(own, cx, cy, th)
    ts["zones+classify"].append(time.perf_counter() - t)
for k, v in ts.items():
    v = np.array(v[2:]) * 1000   # drop warm-up
    print(f"per tray: {k:15s} median {np.median(v):7.1f} ms  p90 {np.percentile(v, 90):7.1f} ms")
