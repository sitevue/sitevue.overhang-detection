"""Flag trays where a worker's hand (blue glove) or white sleeve covers the tray, using the
overlay crops; write tray_overhang_summary.csv with glove_frac / white_frac / occluded and print a recount.
usage: python overhang/flag_occluded.py RESULTS_DIR SESSIONS_JSON"""
import cv2, numpy as np, csv, json, os, sys, statistics as st, collections
from multiprocessing import Pool
D = sys.argv[1]

def fr(k):
    im = cv2.imread(os.path.join(D, f"inner_overlays/tray_{int(k):03d}.jpg"))[34:]
    h, s, v = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    return k, float(((h >= 100) & (h <= 130) & (s >= 150) & (v >= 110)).mean()), float(((s < 35) & (v >= 200)).mean())

if __name__ == "__main__":
    S = list(csv.DictReader(open(os.path.join(D, "inner_overlay_summary.csv"))))
    with Pool(8) as p: res = {k: (g, w) for k, g, w in p.map(fr, [s["idx"] for s in S], chunksize=32)}
    for s in S:
        g, w = res[s["idx"]]; s["glove_frac"] = round(g, 4); s["white_frac"] = round(w, 4); s["occluded"] = int(g > 0.01 or w > 0.05)
    with open(os.path.join(D, "tray_overhang_summary.csv"), "w", newline="") as fh:
        W = csv.DictWriter(fh, fieldnames=list(S[0])); W.writeheader(); W.writerows(S)
    occ = [s for s in S if s["occluded"]]; ok = [s for s in S if not s["occluded"]]
    print("occluded by a hand/sleeve:", len(occ), "  clean trays:", len(ok))
    print("occluded per session:", dict(collections.Counter(s["session"][:8] for s in occ)))
    pct = sorted(float(s["outside_pct"]) for s in ok); q = lambda p: pct[int(p * (len(pct) - 1))]
    print(f"clean trays  pct outside inner edge: p50 {q(.5):.2f}  p90 {q(.9):.2f}  p99 {q(.99):.2f}  max {pct[-1]:.1f}")
    for t in (1, 2, 3, 5, 10, 20): print(f"  > {t:2d}%: {sum(p > t for p in pct)}")
    sess = json.load(open(sys.argv[2])); by = collections.defaultdict(list)
    for s in ok: by[s["session"]].append(float(s["outside_pct"]))
    print("session start trays median p90 >3% >5%")
    for sid, t0 in sorted(sess.items(), key=lambda x: x[1]):
        v = sorted(by.get(sid, []))
        if v: print(f"  {sid[:8]} {int(t0//3600):02d}:{int(t0%3600//60):02d} {len(v):4d} {st.median(v):5.2f} {v[int(.9*(len(v)-1))]:5.2f} {sum(x>3 for x in v):3d} {sum(x>5 for x in v):3d}")
    top = sorted(ok, key=lambda s: -float(s["outside_pct"]))[:16]
    print("top clean:", [(s["idx"], s["outside_pct"]) for s in top])
