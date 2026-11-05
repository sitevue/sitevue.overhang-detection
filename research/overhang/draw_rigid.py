"""Full-res check image: green = outer outline, magenta = rigid inner opening,
dashed orange = previous per-side inner lines, cyan = detected inner-edge points."""
import csv, sys, json, cv2, numpy as np
sys.path.insert(0, "overhang")
from geom import rect_corners
from measure_full import TRAY_W, TRAY_H, lip_lines
from inner_rigid import SIDES
S = 2.0
F = json.load(open("out/run1/inner_fit.json")); P = json.load(open("out/run1/inner_points.json"))
v4 = {r["idx"]: r for r in csv.DictReader(open("out/run1/results_full_v4.csv"))}
v5 = {r["idx"]: r for r in csv.DictReader(open("out/run1/results_full.csv"))}

def draw(k):
    r = v5[str(k)]; full = cv2.imread(f"out/run1/cache_full/{k:03d}.png")
    cx, cy, th = float(r["tray_cx"]) * S, float(r["tray_cy"]) * S, np.deg2rad(float(r["tray_deg"]))
    W, H = TRAY_W * S, TRAY_H * S
    vis = full.copy()
    # previous per-side lines (from v4 lip widths, constant per side)
    old = [float(x) * S for x in v4[str(k)]["lip_px"].split("/")]
    oc = rect_corners(cx + ((old[0] - old[1]) / 2) * np.cos(th) - ((old[2] - old[3]) / 2) * np.sin(th),
                      cy + ((old[0] - old[1]) / 2) * np.sin(th) + ((old[2] - old[3]) / 2) * np.cos(th),
                      W - old[0] - old[1], H - old[2] - old[3], th).astype(int)
    for i in range(4):
        a, b = oc[i], oc[(i + 1) % 4]
        for t0 in np.arange(0, 1, 0.06):
            p0 = (a + (b - a) * t0).astype(int); p1 = (a + (b - a) * min(1, t0 + 0.03)).astype(int)
            cv2.line(vis, tuple(p0), tuple(p1), (0, 140, 255), 1)
    cv2.polylines(vis, [rect_corners(cx, cy, W, H, th).astype(int)], True, (0, 220, 0), 2)
    q = F[str(k)]
    cv2.polylines(vis, [rect_corners(q["cx"], q["cy"], q["W"], q["H"], q["th"]).astype(int)], True, (255, 0, 255), 2)
    for sd in SIDES:
        for x, y, *_ in P[str(k)][sd]:
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 255, 0), -1)
    c = rect_corners(cx, cy, W, H, th)
    x0, y0 = np.maximum(c.min(0) - 40, 0).astype(int); x1, y1 = np.minimum(c.max(0) + 40, [full.shape[1], full.shape[0]]).astype(int)
    crop = cv2.resize(vis[y0:y1, x0:x1], (700, 540), interpolation=cv2.INTER_AREA)
    bar = np.zeros((30, 700, 3), np.uint8)
    a4 = int(v4[str(k)]["rim_px"]) + int(v4[str(k)]["beyond_px"]); a5 = int(r["rim_px"]) + int(r["beyond_px"])
    cv2.putText(bar, f"#{k} past inner: per-side {a4} -> rigid {a5}  sides {q['src']}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, crop])

def zoom(k, side):
    """Close-up of one side at full res."""
    r = v5[str(k)]; full = cv2.imread(f"out/run1/cache_full/{k:03d}.png")
    img = draw.__globals__["draw"]  # noqa
    return None

if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[2].split(",")]
    T = [draw(k) for k in ids]
    while len(T) % 3: T.append(np.zeros_like(T[0]))
    cv2.imwrite(sys.argv[1], np.vstack([np.hstack(T[i:i + 3]) for i in range(0, len(T), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
