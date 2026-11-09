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


def test_tracker_fast_belt_every_tray_once_near_centre(cfg):
    """FPL belt: trays move right to left ~180 full-res px per frame, a new tray every ~7 frames.
    Each tray must be picked exactly once, close enough to the centre for the edge-of-view check."""
    speed, spacing, n_trays = 180, 1260, 5          # full-res px per frame, px between tray centres
    picks = []
    tr = TrayTracker(cfg)
    n_frames = int((1456 + 400 + spacing * n_trays) / speed) + 4
    for i in range(n_frames):
        f = np.full((1088, 1456, 3), 60, np.uint8)
        for yy in range(40, 1088, 120):
            for xx in range(30, 1456, 130):
                cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
        for k in range(n_trays):
            xc = 1456 + 400 + k * spacing - i * speed
            cv2.rectangle(f, (int(xc - 300), 300), (int(xc + 300), 760), (40, 40, 200), -1)
        picks += tr.update(i, f, timestamp=i / 10)
    picks += tr.flush()
    assert len(picks) == n_trays
    # tray centre within ~half a frame step of the frame centre (full-res), well inside the
    # +-200 px the edge-of-view check allows
    assert all(abs(p.c[0] * 2 - 728) <= speed / 2 + 2 for p in picks)
    assert all(p.in_zone and p.frame is not None for p in picks)


def test_tracker_merged_trays_reported_not_measured(cfg):
    """Two touching trays = one long meat blob that never fits inside the pick zone R1:
    reported once as not in the zone (counted, not measured), never measured off-centre."""
    tr = TrayTracker(cfg)
    picks = []
    for i in range(30):
        f = np.full((1088, 1456, 3), 60, np.uint8)
        xc = 1456 + 700 - i * 120
        cv2.rectangle(f, (int(xc - 600), 300), (int(xc + 600), 760), (40, 40, 200), -1)   # 1200 px of meat
        picks += tr.update(i, f, timestamp=i / 10)
    picks += tr.flush()
    assert len(picks) == 1 and not picks[0].in_zone and picks[0].frame is None


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
