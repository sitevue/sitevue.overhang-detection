"""Compare `overhang run` output with the research batch pipeline, tray by tray (matched on frame).

usage: python tools/compare_with_batch.py RUN.jsonl BATCH_SUMMARY.csv BATCH_PICKS.json SESSION_PREFIX
  BATCH_SUMMARY.csv  tray_overhang_summary.csv from the batch run (idx, lip_cover_pct, outside_px, past_green_px, reliable)
  BATCH_PICKS.json   merged picks.json of that run (session, session_frame per idx)
"""
import csv
import json
import sys

import numpy as np

run, summ, picks, sess = sys.argv[1:5]
R = {r["frame_index"]: r for r in map(json.loads, open(run))}
P = json.load(open(picks))
B = {}
for r in csv.DictReader(open(summ)):
    p = P[int(r["idx"])]
    if p["session"].startswith(sess):
        B[p["session_frame"]] = r
common = sorted(set(R) & set(B))
print(f"trays: new {len(R)}  batch {len(B)}  same frame {len(common)}  "
      f"only new {len(set(R) - set(B))}  only batch {len(set(B) - set(R))}")
# trays picked one or two frames apart are the same tray
near = {f: min(B, key=lambda g: abs(g - f)) for f in set(R) - set(B)}
near = {f: g for f, g in near.items() if abs(g - f) <= 4 and g not in R}
print(f"  of the rest, picked within 4 frames of a batch tray: {len(near)}")


def cls(lip, past):
    return "large" if past >= 10000 else "small" if lip >= 18.5 else "mild" if lip >= 10.5 else "none"


pairs = [(R[f], B[f]) for f in common]
for key, bkey in (("lip_cover_pct", "lip_cover_pct"), ("outside_px", "outside_px"), ("past_green_px", "past_green_px")):
    a = np.array([float(r.get(key, 0)) for r, _ in pairs])
    b = np.array([float(x[bkey]) for _, x in pairs])
    d = np.abs(a - b)
    print(f"{key:15s} |new - batch|: median {np.median(d):8.2f}  p90 {np.percentile(d, 90):8.2f}  max {d.max():9.1f}   "
          f"identical {np.mean(d < 1e-6) * 100:5.1f}%")
same_cls = np.mean([r["overhang_class"] == cls(float(b["lip_cover_pct"]), int(b["past_green_px"])) for r, b in pairs])
same_rel = np.mean([r["reliable"] == (b["reliable"] == "1") for r, b in pairs])
print(f"same class {same_cls * 100:.1f}%   same reliable flag {same_rel * 100:.1f}%")
conf = {}
for r, b in pairs:
    k = (cls(float(b["lip_cover_pct"]), int(b["past_green_px"])), r["overhang_class"])
    conf[k] = conf.get(k, 0) + 1
print("class changes (batch -> new):", {f"{a}->{b}": n for (a, b), n in sorted(conf.items()) if a != b})
