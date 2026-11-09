"""Outline check: only belt inside the box or a tilt over 30 deg make the outline doubtful."""
import numpy as np
import pytest

from overhang.outline import BELT_INSIDE_MAX, outline_ok


@pytest.mark.parametrize("belt,deg,ok", [
    (0.0, 0.0, True),
    (0.0, -12.0, True),
    (BELT_INSIDE_MAX, 29.9, True),
    (0.02, 0.0, False),          # box straddles two trays / sits on belt
    (0.0, 35.0, False),          # box turned sideways
    (0.0, -41.0, False),
])
def test_outline_ok(belt, deg, ok):
    assert outline_ok(belt, np.deg2rad(deg)) is ok


# --- box on two touching trays: the other tray's meat inside the box (half-res scenes)
import cv2  # noqa: E402

from overhang.color import meat_blobs  # noqa: E402
from overhang.outline import OTHER_MEAT_MAX, other_meat_frac  # noqa: E402

W, H = 390, 288   # tray size, half-res px


def _other_frac(blocks, own_idx, box_cx):
    """blocks: meat rectangles (x0, y0, x1, y1); own_idx: which is this tray's; box centred at box_cx."""
    red = np.zeros((544, 728), np.uint8)
    for x0, y0, x1, y1 in blocks:
        cv2.rectangle(red, (x0, y0), (x1, y1), 1, -1)
    lab, blobs = meat_blobs(red, 4000)
    x0, y0, x1, y1 = blocks[own_idx]
    own = np.zeros(red.shape, bool)
    own[y0:y1, x0:x1] = True
    other = np.isin(lab, [b["id"] for b in blobs]) & ~own
    return other_meat_frac(lab, other, own, box_cx, 272.0, 0.0, W, H)


def test_box_on_two_touching_trays_is_flagged():
    # trays touch at x = 364 (meat 180 px wide in each, 30 px of tray wall either side of the seam)
    left, right = (154, 152, 334, 392), (394, 152, 574, 392)
    f = _other_frac([left, right], 0, 364.0)          # box centred on the seam, our meat on the left
    assert f > OTHER_MEAT_MAX and not outline_ok(0.0, 0.0, f)


def test_correct_box_with_neighbour_grazing_the_end_is_ok():
    own, nb = (194, 152, 534, 392), (548, 152, 728, 392)   # neighbour's meat 6 px inside our box end
    f = _other_frac([own, nb], 0, 364.0)
    assert f <= OTHER_MEAT_MAX and outline_ok(0.0, 0.0, f)


def test_own_meat_torn_in_two_is_not_another_tray():
    a, b = (194, 152, 330, 392), (350, 152, 534, 392)       # one tray, meat split by a gap
    assert _other_frac([a, b], 1, 364.0) == 0.0


def test_other_tray_cut_by_frame_edge_still_counts():
    # box near the left edge on two trays; the left tray's meat runs off the frame, so what is
    # visible of it is mostly inside our box
    left, right = (0, 152, 150, 392), (210, 152, 390, 392)
    f = _other_frac([left, right], 1, 180.0)
    assert f > OTHER_MEAT_MAX
