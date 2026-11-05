import sys, time, cv2, numpy as np
sys.path.insert(0, "overhang"); from geom import *
TW, TH = float(sys.argv[1]), float(sys.argv[2])
fit = TrayFitter(TW, TH); tiles=[]
for p in sys.argv[3:]:
    f = cv2.imread(p)
    red, yel = masks(f); sm = smooth_mask(f, red); lab, blobs = meat_blobs(red, 4000)
    vis = f.copy()
    for b in blobs:
        own = lab == b["id"]; other = red & ~own
        t = time.time(); cx, cy, th, sc = fit.fit(sm | own, yel, other, b["c"], lane_cy=215)
        cv2.polylines(vis, [rect_corners(cx, cy, TW, TH, th).astype(int)], True, (0, 255, 0), 2)
        print(p, b["area"], round(cx), round(cy), round(np.rad2deg(th), 1), round(sc, 2), f"{time.time()-t:.2f}s")
    tiles.append(vis)
cv2.imwrite("explore/fits.jpg", np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles)-1, 2)]))
