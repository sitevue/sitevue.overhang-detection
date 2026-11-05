"""Merge per-session runs into one run so the inner-opening calibration is pooled
over the whole day and every tray gets one global index.

usage: python overhang/merge_runs.py SESSIONS_JSON OUT_RUN RUN_DIR [RUN_DIR ...]
Each RUN_DIR is named after its session id (prefix match against SESSIONS_JSON keys).
t_sec in the merged results is the clock time in seconds since midnight
(session start + frame / 10), and a `session` column is added.
"""
import csv, json, os, shutil, sys

sessions = json.load(open(sys.argv[1]))
out = sys.argv[2]; runs = sys.argv[3:]
for sub in ("cache", "cache_full"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
picks_all, rows_all, k_new = [], [], 0
for rd in sorted(runs):
    sid = next(s for s in sessions if s.startswith(os.path.basename(rd.rstrip("/"))[:8]))
    t0 = sessions[sid]
    picks = json.load(open(os.path.join(rd, "picks.json")))
    rows = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(rd, "results_full.csv")))}
    for k, p in enumerate(picks):
        if k not in rows:
            continue
        r = dict(rows[k]); r["session"] = sid; r["session_frame"] = r.get("frame", p["frame"])
        r["idx"] = k_new; r["t_sec"] = round(t0 + p["frame"] / 10.0, 1)
        for src, dst in ((f"cache/{k:03d}.npz", f"cache/{k_new:03d}.npz"), (f"cache/{k:03d}.jpg", f"cache/{k_new:03d}.jpg"),
                         (f"cache_full/{k:03d}.png", f"cache_full/{k_new:03d}.png")):
            s = os.path.join(rd, src); d = os.path.join(out, dst)
            if os.path.exists(s) and not os.path.exists(d):
                os.link(s, d) if os.stat(rd).st_dev == os.stat(out).st_dev else shutil.copy2(s, d)
        p2 = dict(p); p2["session"] = sid; p2["session_frame"] = p["frame"]
        picks_all.append(p2); rows_all.append(r); k_new += 1
json.dump(picks_all, open(os.path.join(out, "picks.json"), "w"))
keys = list(dict.fromkeys(k for r in rows_all for k in r))
with open(os.path.join(out, "results_full.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows_all)
print("merged", k_new, "trays from", len(runs), "sessions")
