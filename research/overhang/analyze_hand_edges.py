"""Compare hand-clicked inner/outer edge points on the empty-tray frames with the fits.
usage: python overhang/analyze_hand_edges.py EDGES_DIR OUT_JPG"""
import json, glob, os, sys, cv2, numpy as np
D, OUT = sys.argv[1], sys.argv[2]
T = {t["id"]: t for t in json.load(open("labeller/trays.json"))["trays"]}
SIDES = ("left", "right", "top", "bottom")

def uv(R, x, y):
    cx, cy, deg = R[:3]; th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c

def side_of(R, x, y):
    u, v = uv(R, x, y); W, H = R[3], R[4]
    return np.where(np.abs(u) / W > np.abs(v) / H, np.where(u < 0, "left", "right"), np.where(v < 0, "top", "bottom"))

def wall_fit(p, vertical):
    s, t = (p[:, 1], p[:, 0]) if vertical else (p[:, 0], p[:, 1])
    lin = np.polyfit(s, t, 1); r1 = t - np.polyval(lin, s)
    quad = np.polyfit(s, t, 2) if len(s) >= 5 else None
    r2 = t - np.polyval(quad, s) if quad is not None else r1
    span = np.ptp(s)
    sag = abs(quad[0]) * (span / 2) ** 2 if quad is not None else 0.0     # bow at mid-span vs the chord
    return lin, quad, float(np.sqrt(np.mean(r1 ** 2))), float(np.sqrt(np.mean(r2 ** 2))), float(np.abs(r1).max()), sag, span

tiles, rows = [], []
for path in sorted(glob.glob(os.path.join(D, "*.json"))):
    d = json.load(open(path)); d = d.get("data", d); tid = d["tray"]; t = T[tid]
    img = cv2.imread(f"labeller/img/raw/{tid}.jpg")
    R, I = t["rect"], t["inner"]
    print(f"\n== {tid}  (frame at {t['t']} s)  inner points {len(d['inner'])}, outer points {len(d['outer'])}")
    lines = {}
    for kind, pts, ref in (("inner", d["inner"], I), ("outer", d["outer"], R)):
        P = np.array(pts, float)
        if not len(P): continue
        sd = side_of(R, P[:, 0], P[:, 1])
        for side in SIDES:
            p = P[sd == side]
            if len(p) < 2: continue
            vert = side in ("left", "right")
            lin, quad, rms1, rms2, mx, sag, span = wall_fit(p, vert)
            # signed distance of the user's line from the fitted edge at the wall's middle, + = user's edge outside the fit
            u, v = uv(ref, p[:, 0], p[:, 1])
            off = (np.abs(u) - ref[3] / 2) if vert else (np.abs(v) - ref[4] / 2)
            lines[(kind, side)] = (lin, vert, p)
            print(f"  {kind:5s} {side:6s} n={len(p):2d} span {span:4.0f}px | scatter about a straight line: rms {rms1:4.1f} max {mx:4.1f} px"
                  f" | bow of best curve {sag:4.1f} px (curve rms {rms2:4.1f}) | your edge vs fit: {np.median(off):+5.1f} px")
    # opening size from the user's four inner lines, measured through the middle of the tray
    def at(kind, side, s):
        lin, vert, _ = lines[(kind, side)]; return np.polyval(lin, s)
    size = {}
    for kind in ("inner", "outer"):
        if all((kind, s) in lines for s in SIDES):
            cx, cy = R[0], R[1]
            ys = np.array([cy - 0.25 * R[4], cy, cy + 0.25 * R[4]]); xs = np.array([cx - 0.25 * R[3], cx, cx + 0.25 * R[3]])
            Wv = at(kind, "right", ys) - at(kind, "left", ys); Hv = at(kind, "bottom", xs) - at(kind, "top", xs)
            size[kind] = (Wv, Hv)
            print(f"  your {kind} size: width {Wv.mean():.1f} (top/mid/bottom {Wv.round(1).tolist()})  height {Hv.mean():.1f} (left/mid/right {Hv.round(1).tolist()})")
    print(f"  fitted inner {I[3]:.1f} x {I[4]:.1f}   fitted outer {R[3]:.1f} x {R[4]:.1f}")
    rows.append((tid, t["t"], size, I, R))
    # picture
    vis = img.copy()
    def poly(Rr):
        cx, cy, deg, W, H = Rr; th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
        return np.array([[cx + a * c - b * s, cy + a * s + b * c] for a, b in ((-W/2,-H/2),(W/2,-H/2),(W/2,H/2),(-W/2,H/2))], np.int32)
    cv2.polylines(vis, [poly(R)], True, (0, 200, 0), 1, cv2.LINE_AA); cv2.polylines(vis, [poly(I)], True, (255, 0, 255), 1, cv2.LINE_AA)
    for (kind, side), (lin, vert, p) in lines.items():
        col = (255, 255, 0) if kind == "inner" else (0, 180, 255)
        s0, s1 = (p[:, 1].min() - 40, p[:, 1].max() + 40) if vert else (p[:, 0].min() - 40, p[:, 0].max() + 40)
        a = (np.polyval(lin, s0), s0) if vert else (s0, np.polyval(lin, s0)); b = (np.polyval(lin, s1), s1) if vert else (s1, np.polyval(lin, s1))
        cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), col, 1, cv2.LINE_AA)
        for x, y in p: cv2.circle(vis, (int(x), int(y)), 4, col, -1); cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 0), 1)
    cv2.putText(vis, f"{tid}  t={t['t']}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(cv2.resize(vis, (int(vis.shape[1] * 620 / vis.shape[0]), 620)))
W = sum(t.shape[1] for t in tiles); canvas = np.zeros((620, W, 3), np.uint8); x = 0
for t_ in tiles: canvas[:, x:x + t_.shape[1]] = t_; x += t_.shape[1]
cv2.imwrite(OUT, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
print("\nSUMMARY (camera px)   frame  tray centre x   your inner WxH          fitted inner WxH")
for tid, ts, size, I, R in rows:
    xc = None
    if "inner" in size:
        print(f"  {tid}  t={ts}   {R[0]:6.0f} (crop)   {size['inner'][0].mean():6.1f} x {size['inner'][1].mean():6.1f}     {I[3]:6.1f} x {I[4]:6.1f}")
