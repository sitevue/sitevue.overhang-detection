"""Edge of view: only a tray cut off by the frame edge counts, not a missing belt margin past its ends."""
import os

import cv2
import numpy as np

from overhang import Config
from overhang.color import rect_corners
from overhang.measure import measure_tray

CFG = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "fpl-grind.yaml"))


def _measure(cx_full, deg):
    """Corrected full frame with one tray (meat inside) centred at cx_full, tilted deg."""
    f = np.full((1088, 1456, 3), 60, np.uint8)
    for yy in range(20, 1088, 110):
        for xx in range(20, 1456, 120):
            cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
    th = np.deg2rad(deg)
    W, H = CFG.tray_w * 2, CFG.tray_h * 2
    cv2.fillPoly(f, [rect_corners(cx_full, 544, W, H, th).astype(np.int32)], (45, 45, 45))
    cv2.fillPoly(f, [rect_corners(cx_full, 544, W - 220, H - 200, th).astype(np.int32)], (40, 40, 200))
    own = np.zeros((544, 728), bool)
    cv2.fillPoly(own.view(np.uint8), [(rect_corners(cx_full, 544, W - 220, H - 200, th) / 2).astype(np.int32)], 1)
    corners = rect_corners(cx_full, 544, CFG.inner_w, CFG.inner_h, th)
    how = dict(left="measured", right="measured", top="measured", bottom="measured")
    num, _ = measure_tray(f, own, dict(cx=cx_full / 2, cy=272.0, th=th), corners, how, CFG)
    return num


def test_off_centre_tray_fully_in_view_is_readable():
    # Aaron's 19 live trays: outer cx 900-940, tilt -3 to -11 deg; the tray is in view, the 140 px
    # of belt past its right end is not
    for cx, deg in ((900, -3), (940, -3), (920, -11)):
        assert not _measure(cx, deg)["edge_of_view"]


def test_tray_cut_off_by_frame_edge_is_flagged():
    assert _measure(1090, -3)["edge_of_view"]
    assert _measure(360, 0)["edge_of_view"]
