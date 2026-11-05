import csv, sys, cv2, numpy as np
sys.path.insert(0, "overhang")
from tray_zones import inner_edges, inner_edge_points, SIDES, SIDE_GEOM
from measure_full import TRAY_W, TRAY_H, LIP_DEFAULT

def draw(k, r, scale=2):
    z = np.load(f"out/run1/cache/{k:03d}.npz"); f = cv2.imread(f"out/run1/cache/{k:03d}.jpg")
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cx, cy, th = float(r["tray_cx"]), float(r["tray_cy"]), np.deg2rad(float(r["tray_deg"]))
    W, H = TRAY_W, TRAY_H; c, s = np.cos(th), np.sin(th)
    P = lambda u, v: (int(round((cx + u * c - v * s) * scale)), int(round((cy + u * s + v * c) * scale)))
    vis = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners = [P(-W/2, -H/2), P(W/2, -H/2), P(W/2, H/2), P(-W/2, H/2)]
    cv2.polylines(vis, [np.array(corners)], True, (0, 220, 0), 2)
    E = inner_edges(g, z["own"], cx, cy, th, W, H)
    for side in SIDES:
        axis, sign = SIDE_GEOM[side]; half, other = (W/2, H) if axis == 0 else (H/2, W)
        pts = inner_edge_points(g, z["own"], cx, cy, th, W, H, side)
        for tt, dd, _ in pts:
            u, v = ((half - dd) * sign, tt) if axis == 0 else (tt, (half - dd) * sign)
            cv2.circle(vis, P(u, v), 3, (255, 255, 0), -1)
        e = E[side]
        ts = np.linspace(-0.48, 0.48, 2) * other
        if e:
            ds = e["a"] + e["b"] * ts; col, thick = (255, 0, 255), 2
        else:
            ds = np.full(2, LIP_DEFAULT[side] - 2.0); col, thick = (0, 165, 255), 1
        pp = [P((half - d_) * sign, t_) if axis == 0 else P(t_, (half - d_) * sign) for t_, d_ in zip(ts, ds)]
        cv2.line(vis, pp[0], pp[1], col, thick)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    lab = " ".join(f"{sd[0].upper()}:{'fit' if E[sd] else 'def'}" for sd in SIDES)
    cv2.putText(vis, f"#{k}  {lab}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # crop around the tray
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    x0, x1 = max(0, min(xs) - 30), min(vis.shape[1], max(xs) + 30); y0, y1 = max(30, min(ys) - 30), min(vis.shape[0], max(ys) + 30)
    crop = vis[y0:y1, x0:x1]
    return np.vstack([vis[0:30, x0:x1], crop])

if __name__ == "__main__":
    rows = {int(r["idx"]): r for r in csv.DictReader(open("out/run1/results_full.csv"))}
    ids = [int(a) for a in sys.argv[2].split(",")]
    tiles = [cv2.resize(draw(k, rows[k]), (700, 560)) for k in ids]
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
