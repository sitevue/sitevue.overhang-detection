"""Guarded inner-opening fit: good edges place the opening, edges near meat must agree.

Edge points within NEAR px of meat are suspect (meat casts a shadow on the lip that
looks like the lip -> wall drop). We fit the fixed-size opening on the clean points
only; the opposite side of a covered wall is then placed by the known size. Suspect
points are kept only if they lie within AGREE px of that fit, and we refit.
If an axis has no clean points on either wall, that axis falls back to all points."""
import os as _os
RUN = _os.environ.get("RUN", "out/run1")
import json, sys, numpy as np, cv2
sys.path.insert(0, "overhang")
from inner_rigid import fit_inner, SIDES
from measure_full import tray_uv
NEAR, AGREE = 25.0, 3.0
size = json.load(open(f"{RUN}/inner_size.json")); W_in, H_in = size["W_in"], size["H_in"]
P = json.load(open(f"{RUN}/inner_points.json")); F0 = json.load(open(f"{RUN}/inner_fit_v1.json"))
out, stats = {}, dict(rejected=0, kept_suspect=0, clean=0, axis_fallback=0)
for k, f0 in F0.items():
    z = np.load(f"{RUN}/cache/{int(k):03d}.npz")
    red = cv2.resize(z["red"].astype(np.uint8), (1456, 1088), interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform((1 - red).astype(np.uint8), cv2.DIST_L2, 3)
    pts = {sd: np.array(P[k][sd]).reshape(-1, 4) for sd in SIDES}
    near = {sd: dist[np.clip(p[:, 1].astype(int), 0, 1087), np.clip(p[:, 0].astype(int), 0, 1455)] < NEAR if len(p) else np.zeros(0, bool) for sd, p in pts.items()}
    clean = {sd: pts[sd][~near[sd]] for sd in SIDES}
    stats["clean"] += sum(len(v) for v in clean.values())
    init = (f0["cx"], f0["cy"], f0["th"], W_in, H_in)
    # axes with no clean support keep all their points (fallback = previous behaviour)
    use = dict(clean)
    for a, b in (("left", "right"), ("top", "bottom")):
        if len(clean[a]) < 10 and len(clean[b]) < 10:
            use[a], use[b] = pts[a], pts[b]; stats["axis_fallback"] += 1
    fa = fit_inner(use, init, W_in, H_in, min_side=8)
    if fa is None:
        out[k] = dict(f0, guard="none"); continue
    # admit suspect points that agree with the clean fit
    final = {}
    for sd in SIDES:
        p = pts[sd]
        if not len(p): final[sd] = p; continue
        u, v = tray_uv(p[:, 0], p[:, 1], fa["cx"], fa["cy"], fa["th"])
        r = (W_in / 2 - np.abs(u)) if sd in ("left", "right") else (H_in / 2 - np.abs(v))
        ok = ~near[sd] | (np.abs(r) <= AGREE)
        if use[sd] is pts[sd]: ok[:] = True          # fallback axis: keep everything
        stats["rejected"] += int((~ok).sum()); stats["kept_suspect"] += int((near[sd] & ok).sum())
        final[sd] = p[ok]
    f = fit_inner(final, (fa["cx"], fa["cy"], fa["th"], W_in, H_in), W_in, H_in, min_side=8) or fa
    sc = dict(zip(("right", "left", "bottom", "top"), f["side_counts"]))
    src = "".join(sd[0].upper() if sc[sd] >= 8 else sd[0] for sd in SIDES)
    out[k] = dict(src=src, cx=f["cx"], cy=f["cy"], th=f["th"], W=W_in, H=H_in, rms=f["rms"], guard="ok")
json.dump(out, open(f"{RUN}/inner_fit.json", "w"))
print(stats)
# how far did each opening move vs before?
mv = []
for k in out:
    a, b = F0[k], out[k]
    mv.append(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
mv = np.array(mv); print("opening moved (camera px): p50", np.median(mv).round(2), " p90", np.percentile(mv, 90).round(2), " >4px:", (mv > 4).sum(), " max", mv.max().round(1))
# re-check per-side disagreement
res = {s: [] for s in SIDES}
for k, v in out.items():
    for s in SIDES:
        p = np.array(P[k][s]).reshape(-1, 4)
        if len(p) < 10: continue
        u, w = tray_uv(p[:, 0], p[:, 1], v["cx"], v["cy"], v["th"])
        r = (v["W"] / 2 - np.abs(u)) if s in ("left", "right") else (v["H"] / 2 - np.abs(w))
        res[s].append(np.median(r))
for s in SIDES:
    r = np.array(res[s]); print(f"{s:6s} sides whose raw points sit >4px off the opening: {(np.abs(r) > 4).sum()}")
