"""Measure an EMPTY tray's outer outline and inner opening at full resolution, with the
sizes free (not fixed to the calibrated values).
usage: python overhang/measure_empty.py OUT_JPG FRAME.png:cx,cy [FRAME.png:cx,cy ...]
cx,cy = rough tray centre in full-res pixels (for the SAM prompt).
"""
import sys
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "overhang")
from sam_tray import load, dev
from fit_edges import refine
from inner_rigid import prep, all_points, initial_inner, fit_inner, SIDES
from geom import rect_corners
from measure_full import LIP_DEFAULT, TRAY_W, TRAY_H

proc, model = load()


@torch.no_grad()
def sam_mask(bgr, cx, cy):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    H_, W_ = bgr.shape[:2]
    # positives: floor, top lip, and the lip at BOTH ends (the ends are what SAM tends to drop);
    # negatives: belt above/below and just past each end
    pts = [[cx, cy], [cx, cy - 250], [cx, cy + 255], [cx - 355, cy], [cx + 355, cy]]
    neg = [[cx, max(20, cy - 430)], [cx, min(H_ - 20, cy + 430)],
           [max(8, cx - 470), cy], [min(W_ - 8, cx + 470), cy]]
    lbl = [1] * len(pts) + [0] * len(neg)
    inp = proc(images=img, input_points=[[pts + neg]], input_labels=[[lbl]], return_tensors="pt").to(dev)
    o = model(**inp, multimask_output=True)
    m = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy().astype(bool)
    sc = o.iou_scores.cpu()[0, 0].numpy()
    areas = m.reshape(3, -1).sum(1)
    k = int(np.argmax(sc * (areas > 0.15 * H_ * W_ * 0)))            # best score
    return m[int(np.argmax(sc))], float(sc.max())


def outer_fit(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), a = cv2.minAreaRect(c)
    if w < h:
        w, h, a = h, w, a + 90
    a = (a + 90) % 180 - 90
    th = np.deg2rad(a)
    pts = c[:, 0, :].astype(np.float32)
    # drop rounded corners, then fit width/height/position/angle to the straight edges
    co, so = np.cos(th), np.sin(th)
    u = (pts[:, 0] - cx) * co + (pts[:, 1] - cy) * so; v = -(pts[:, 0] - cx) * so + (pts[:, 1] - cy) * co
    keep = ~((np.abs(u) > w / 2 - 50) & (np.abs(v) > h / 2 - 50))
    cx, cy, th, W, H, info = refine(pts[keep], cx, cy, th, w, h, fit_size=True, trim=0.1)
    return cx, cy, th, W, H, info


def main():
    out_jpg = sys.argv[1]
    tiles, rows = [], []
    for arg in sys.argv[2:]:
        path, c = arg.split(":"); cx0, cy0 = map(float, c.split(","))
        full = cv2.imread(path)
        m, sc = sam_mask(full, cx0, cy0)
        ocx, ocy, oth, oW, oH, oinfo = outer_fit(m)
        g, gx, gy = prep(full)
        meat = np.zeros(full.shape[:2], bool)
        P = all_points(g, gx, gy, meat, ocx, ocy, oth, oW, oH)
        ix, iy, iW0, iH0 = initial_inner(ocx, ocy, oth, oW, oH, {k: v * 2 for k, v in LIP_DEFAULT.items()})
        f = fit_inner(P, (ix, iy, oth, iW0, iH0))
        n = {sd: len(P[sd]) for sd in SIDES}
        # lip width per side = distance between outer and inner rectangle edges (in the outer frame)
        c_, s_ = np.cos(oth), np.sin(oth)
        du = (f["cx"] - ocx) * c_ + (f["cy"] - ocy) * s_; dv = -(f["cx"] - ocx) * s_ + (f["cy"] - ocy) * c_
        lips = dict(left=(oW / 2 + du) - f["W"] / 2, right=(oW / 2 - du) - f["W"] / 2, top=(oH / 2 + dv) - f["H"] / 2, bottom=(oH / 2 - dv) - f["H"] / 2)
        rows.append(dict(path=path, outer=[ocx, ocy, float(np.rad2deg(oth)), oW, oH],
                         inner=[f["cx"], f["cy"], float(np.rad2deg(f["th"])), f["W"], f["H"]],
                         frame=path.split("/")[-1], sam=sc, outer_W=oW, outer_H=oH, outer_deg=np.rad2deg(oth), outer_rms=oinfo["rms"],
                         inner_W=f["W"], inner_H=f["H"], inner_rms=f["rms"], inner_pts=n, lips=lips))
        vis = full.copy()
        vis[m] = (vis[m] * 0.75 + np.array([255, 140, 0]) * 0.25).astype(np.uint8)
        cv2.polylines(vis, [rect_corners(ocx, ocy, oW, oH, oth).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(vis, [rect_corners(f["cx"], f["cy"], f["W"], f["H"], f["th"]).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        for sd in SIDES:
            for x, y, *_ in P[sd]:
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 255, 0), -1)
        cv2.putText(vis, f"{rows[-1]['frame']}  outer {oW:.1f} x {oH:.1f}  inner {f['W']:.1f} x {f['H']:.1f}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (728, 544)))
    for r in rows:
        print(f"{r['frame']}: SAM {r['sam']:.3f} | outer {r['outer_W']:.1f} x {r['outer_H']:.1f} px (deg {r['outer_deg']:.2f}, edge rms {r['outer_rms']:.2f}) "
              f"| inner {r['inner_W']:.1f} x {r['inner_H']:.1f} px (rms {r['inner_rms']:.2f}, points {r['inner_pts']}) "
              f"| lip L {r['lips']['left']:.1f} R {r['lips']['right']:.1f} T {r['lips']['top']:.1f} B {r['lips']['bottom']:.1f}")
    print(f"calibrated from 10-29 footage: outer {TRAY_W*2:.1f} x {TRAY_H*2:.1f}, inner 725.1 x 529.2")
    import json
    json.dump([{k: (v if k != "lips" else {a: float(b) for a, b in v.items()}) for k, v in r.items() if k != "inner_pts"} for r in rows],
              open(out_jpg.rsplit(".", 1)[0] + ".json", "w"), default=float, indent=1)
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out_jpg, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
