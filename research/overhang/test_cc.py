import sys, cv2, numpy as np
sys.path.insert(0, "overhang"); from geom import *
tiles=[]
for p in sys.argv[1:]:
    f = cv2.imread(p); red, yel = masks(f); sm = smooth_mask(f, red); lab, blobs = meat_blobs(red, 4000)
    vis = f.copy()
    for b in blobs:
        own = lab == b["id"]; other = cv2.dilate((red & ~own).astype(np.uint8), np.ones((15,15),np.uint8)).astype(bool)
        fill = ((sm & ~other) | own).astype(np.uint8)
        fill = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
        fill = cv2.erode(fill, np.ones((3,3),np.uint8))
        n, l = cv2.connectedComponents(fill)
        ids = np.unique(l[own & (l > 0)])
        comp = np.isin(l, ids)
        cnts,_ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(cnts))
        (cx,cy),(w,h),a = cv2.minAreaRect(hull)
        cv2.drawContours(vis,[cv2.boxPoints(((cx,cy),(w,h),a)).astype(int)],0,(0,255,0),2)
        vis[comp & ~own] = (vis[comp & ~own]*0.5 + np.array([255,0,0])*0.5).astype(np.uint8)
        print(p, round(cx),round(cy), round(w),round(h), round(a,1))
    tiles.append(vis)
cv2.imwrite("explore/cc.jpg", np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles)-1, 2)]))
