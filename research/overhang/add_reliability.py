"""Add reliability flags to the per-tray summary and print the recount.
usage: python overhang/add_reliability.py RESULTS_DIR SESSIONS_JSON
  edge_of_view  tray, or the 140 px past either end that the next-tray check needs, leaves the frame
  fit_check     outline flagged by the outline fit (SAM spill, box tilted > 30 deg, or belt rollers inside the box)
  reliable      no hand, fully in view, outline OK"""
import csv, json, math, os, sys, statistics as st, collections
D, SJ = sys.argv[1], sys.argv[2]
ts = json.load(open(os.path.join(D, "tray_size.json"))); W, H = ts["W"] * 2, ts["H"] * 2
R = {r["idx"]: r for r in csv.DictReader(open(os.path.join(D, "results_full.csv")))}
S = list(csv.DictReader(open(os.path.join(D, "inner_overlay_summary.csv"))))
def view_ok(r, far=140, FW=1456, FH=1088):
    cx, cy = float(r["tray_cx"]) * 2, float(r["tray_cy"]) * 2; th = math.radians(float(r["tray_deg"])); c, s = math.cos(th), math.sin(th)
    return all(0 <= cx + u * c - v * s < FW and 0 <= cy + u * s + v * c < FH for u in (-(W / 2 + far), W / 2 + far) for v in (-0.42 * H, 0.42 * H))
for row in S:
    r = R[row["idx"]]
    row["tray_deg"] = r["tray_deg"]; row["belt_inside_frac"] = r.get("belt_inside_frac", "")
    row["edge_of_view"] = int(not view_ok(r)); row["fit_check"] = int(r["status"] == "check")
    row["reliable"] = int(row["occluded"] == "0" and row["edge_of_view"] == 0 and row["fit_check"] == 0)
with open(os.path.join(D, "tray_overhang_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(S[0])); w.writeheader(); w.writerows(S)
deg = [abs(float(r["tray_deg"])) for r in R.values()]; belt = [float(r.get("belt_inside_frac") or 0) for r in R.values()]
print(f"outlines >45 deg: {sum(d > 45 for d in deg)}   >30 deg: {sum(d > 30 for d in deg)}   belt inside >1%: {sum(b > 0.01 for b in belt)}")
print("trays", len(S), " hand", sum(r["occluded"] == "1" for r in S), " edge of view", sum(r["edge_of_view"] for r in S),
      " outline check", sum(r["fit_check"] for r in S), " reliable", sum(r["reliable"] for r in S))
ok = [r for r in S if r["reliable"]]
pct = sorted(float(r["outside_pct"]) for r in ok); q = lambda p: pct[int(p * (len(pct) - 1))]
print(f"reliable: p50 {q(.5):.2f}  p90 {q(.9):.2f}  p99 {q(.99):.2f}  max {pct[-1]:.1f}")
for t in (1, 2, 3, 5, 10, 20): print(f"  > {t:2d}%: {sum(p > t for p in pct)}")
if "lip_cover_pct" in ok[0]:
    cv_ = sorted(float(r["lip_cover_pct"]) for r in ok); qc = lambda p: cv_[int(p * (len(cv_) - 1))]
    op = sorted(int(r["outside_px"]) for r in ok); qo = lambda p: op[int(p * (len(op) - 1))]
    print(f"reliable lip covered %: p50 {qc(.5):.1f}  p90 {qc(.9):.1f}  p99 {qc(.99):.1f}   outside magenta px: p50 {qo(.5)}  p90 {qo(.9)}  p99 {qo(.99)}")
    print(f"large overhang: {sum(r['large'] == '1' for r in ok)} reliable / {sum(r['large'] == '1' for r in S)} all")
sess = json.load(open(SJ)); by = collections.defaultdict(list)
for r in ok: by[r["session"]].append(float(r["outside_pct"]))
print("session start trays median p90 >3% >5%")
for sid, t0 in sorted(sess.items(), key=lambda x: x[1]):
    v = sorted(by.get(sid, []))
    if v: print(f"  {sid[:8]} {int(t0//3600):02d}:{int(t0%3600//60):02d} {len(v):4d} {st.median(v):5.2f} {v[int(.9*(len(v)-1))]:5.2f} {sum(x>3 for x in v):3d} {sum(x>5 for x in v):3d}")
