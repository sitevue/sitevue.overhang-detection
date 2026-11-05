"""Fast tests: no SAM, no GPU, no video files."""
import os

import cv2
import numpy as np
import pytest

from overhang import CLASSES, Config, light_for, overhang_class
from overhang.inner import opening_mask
from overhang.rectify import Rectifier
from overhang.tracker import TrayTracker

ROOT = os.path.dirname(os.path.dirname(__file__))
CFG_PATH = os.path.join(ROOT, "configs", "fpl-grind.yaml")


@pytest.fixture(scope="module")
def cfg():
    return Config.load(CFG_PATH)


def test_config_loads(cfg):
    assert (cfg.frame_w, cfg.frame_h) == (1456, 1088)
    assert cfg.scale_map is not None and len(cfg.scale_map["coef_h"]) == 5
    assert cfg.mild_lip_cover_pct < cfg.small_lip_cover_pct
    assert set(cfg.light) == set(CLASSES)


@pytest.mark.parametrize("lip,past,expected", [
    (2.0, 0, "none"), (10.5, 0, "mild"), (18.4, 9999, "mild"), (18.5, 0, "small"),
    (0.0, 10000, "large"), (30.0, 50000, "large"),
])
def test_classes(cfg, lip, past, expected):
    assert overhang_class(lip, past, cfg) == expected
    assert light_for(expected, cfg) in ("green", "yellow", "red")


def test_opening_mask_rounds_corners():
    corners = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.float32)
    sharp, _ = opening_mask(corners, 0.0, (600, 700), 0.0, 0)
    round_, _ = opening_mask(corners, 0.0, (600, 700), 0.0, 45)
    assert abs(int(sharp.sum()) - 400 * 300) < 1500
    # four corners lose (1 - pi/4) r^2 each
    lost = int(sharp.sum()) - int(round_.sum())
    assert abs(lost - 4 * (1 - np.pi / 4) * 45 ** 2) < 300
    shifted, _ = opening_mask(corners, 0.0, (600, 700), 10.0, 0)
    assert np.nonzero(shifted)[1].min() == np.nonzero(sharp)[1].min() + 10


def test_rectifier_identity_and_centre(cfg):
    img = np.random.default_rng(0).integers(0, 255, (cfg.frame_h, cfg.frame_w, 3), np.uint8)
    assert Rectifier(None).warp_full(img) is img
    R = Rectifier(cfg.scale_map, cfg.frame_w, cfg.frame_h)
    out = R.warp_full(img)
    assert out.shape == img.shape
    # the frame centre stays (within the 4 px grid the map is built on)
    x, y = cfg.frame_w // 2, cfg.frame_h // 2
    assert abs(float(R.inv_full[0][y, x]) - x) < 2 and abs(float(R.inv_full[1][y, x]) - y) < 2
    # the left of the frame is stretched (things look smaller there in the camera)
    assert float(R.inv_full[0][y, 100]) > 100


def _frame(x_centre, w=1456, h=1088):
    """Grey belt with yellow rollers and one red 'meat' block centred at x_centre."""
    f = np.full((h, w, 3), 60, np.uint8)
    for yy in range(40, h, 120):
        for xx in range(30, w, 130):
            cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
    x0 = int(x_centre - 300)
    cv2.rectangle(f, (x0, 300), (x0 + 600, 760), (40, 40, 200), -1)
    return f


def test_tracker_one_pick_per_tray_near_centre(cfg):
    tr = TrayTracker(cfg)
    picks, emitted_at = [], None
    xs = list(range(-200, 1700, 25))           # tray moves left to right across the frame
    for i, x in enumerate(xs):
        new = tr.update(i, _frame(x), timestamp=i / 10)
        if new and emitted_at is None:
            emitted_at = x
        picks += new
    picks += tr.flush()
    assert len(picks) == 1
    p = picks[0]
    assert p.tray_id == 0
    # its most central frame: the block moves 25 half-res px between analysed frames
    assert abs(p.c[0] - cfg.work_w / 2) <= 12.5 + 1
    assert p.frame.shape == (1088, 1456, 3)
    # emitted while the tray is still on screen, shortly after it passed the centre
    assert emitted_at is not None and emitted_at < 1456 - 300
