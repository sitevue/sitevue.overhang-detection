"""Review overlay for one tray (not for the customer view).

green = outer edge, magenta = inner opening; meat outside the magenta line blended at 30%:
orange = on the lip, yellow = past the outer edge. Large overhang gets a red label and frame.
"""
import cv2
import numpy as np

from .color import rect_corners

ALPHA = 0.3
ORANGE = np.array([0, 140, 255], np.float32)
YELLOW = np.array([0, 255, 255], np.float32)
RED = (60, 60, 255)


def draw(full, geom, result):
    """full: corrected full-res frame; geom: from measure_tray; result: TrayResult dict."""
    xs, ys = geom["xs"], geom["ys"]
    vis = full.astype(np.float32)
    for m, col in ((geom["lip"], ORANGE), (geom["beyond"], YELLOW)):
        yy, xx = ys[m], xs[m]
        vis[yy, xx] = vis[yy, xx] * (1 - ALPHA) + col * ALPHA
    vis = vis.astype(np.uint8)
    cx, cy, W, H, th = geom["outer"]
    c = rect_corners(cx, cy, W, H, th)
    cv2.polylines(vis, [c.astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.polylines(vis, [geom["inner_contour"]], True, (255, 0, 255), 2, cv2.LINE_AA)
    x0, y0 = np.maximum(c.min(0) - 60, 0).astype(int)
    x1, y1 = np.minimum(c.max(0) + 60, [full.shape[1], full.shape[0]]).astype(int)
    crop = vis[y0:y1, x0:x1].copy()
    r = result
    flags = []
    if r["next_tray_left"] or r["next_tray_right"]:
        flags.append("NEXT TRAY " + ("L" if r["next_tray_left"] else "") + ("R" if r["next_tray_right"] else ""))
    flags += [n for n, on in (("HAND", r["occluded"]), ("EDGE OF VIEW", r["edge_of_view"]), ("CHECK OUTLINE", not r["outline_ok"])) if on]
    l1 = f"#{r['tray_id']}  {r['overhang_class'].upper()}  lip covered {r['lip_cover_pct']:.1f}%   " \
         f"outside magenta {r['outside_px']:,} px   past green {r['past_green_px']:,} px"
    l2 = f"frame {r['frame_index']}  walls {r['walls_measured']}  " + "  ".join(flags)
    bar = np.zeros((60, crop.shape[1], 3), np.uint8)
    cv2.putText(bar, l1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED if r["overhang_class"] == "large" else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, l2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if r["overhang_class"] == "large":
        cv2.rectangle(crop, (1, 1), (crop.shape[1] - 2, crop.shape[0] - 2), RED, 4)
    return np.vstack([bar, crop])
