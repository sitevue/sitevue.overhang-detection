"""Hand check: a glove over the tray's centre is flagged (also in shadow); a hand at the edge is not."""
import os

import cv2
import numpy as np
import pytest

from overhang import Config
from overhang.color import rect_corners
from overhang.measure import measure_tray

CFG = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "fpl-grind.yaml"))
DARK_GLOVE = (95, 40, 12)      # BGR: blue glove in shadow (brightness ~95)
BRIGHT_GLOVE = (230, 120, 30)  # BGR: blue glove in light


def _scene(glove_rect=None, colour=DARK_GLOVE):
    """Corrected full frame: belt, a tray centred in view with meat inside, optional glove patch."""
    f = np.full((1088, 1456, 3), 60, np.uint8)
    for yy in range(20, 1088, 110):
        for xx in range(20, 1456, 120):
            cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
    W, H = CFG.tray_w * 2, CFG.tray_h * 2
    cv2.fillPoly(f, [rect_corners(728, 544, W, H, 0.0).astype(np.int32)], (45, 45, 45))
    cv2.rectangle(f, (450, 330), (1000, 760), (40, 40, 200), -1)                  # meat
    if glove_rect:
        x0, y0, x1, y1 = glove_rect
        cv2.rectangle(f, (x0, y0), (x1, y1), colour, -1)
    own = np.zeros((544, 728), bool)
    own[165:380, 225:500] = True
    corners = rect_corners(728, 544, CFG.inner_w, CFG.inner_h, 0.0)
    how = dict(left="measured", right="measured", top="measured", bottom="measured")
    num, _ = measure_tray(f, own, dict(cx=364.0, cy=272.0, th=0.0), corners, how, CFG)
    return num


def test_glove_colours_are_blue_enough():
    for c in (DARK_GLOVE, BRIGHT_GLOVE):
        h, s, v = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0]
        assert 100 <= h <= 130 and s >= 120 and v >= 35


def test_no_hand():
    assert not _scene()["occluded"]


@pytest.mark.parametrize("colour", [DARK_GLOVE, BRIGHT_GLOVE])
def test_glove_over_centre_is_flagged(colour):
    assert _scene((600, 400, 860, 640), colour)["occluded"]       # glove over the meat


def test_hand_at_edge_is_not_flagged():
    # tray spans x 338-1118; a hand holding the right end, inside the outer 8% of the tray
    assert not _scene((1080, 380, 1180, 700))["occluded"]
