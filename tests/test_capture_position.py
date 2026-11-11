"""Camera angle: measure at a chosen x (pick_x_px), or each half of the tray from its own frame (side_views_px)."""
import dataclasses
import os

import cv2
import numpy as np

from overhang import Config
from overhang.color import rect_corners
from overhang.measure import measure_tray
from overhang.tracker import TrayTracker

CFG = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "fpl-grind.yaml"))
CENTRE = dataclasses.replace(CFG, side_views_px=None, edge_tolerance_px=0.0)   # the old single frame at the centre


def _belt_run(cfg, speed=180, spacing=1260, n_trays=4):
    """FPL belt: trays move right to left; returns the picks."""
    tr, picks = TrayTracker(cfg), []
    for i in range(int((1456 + 400 + spacing * n_trays) / speed) + 4):
        f = np.full((1088, 1456, 3), 60, np.uint8)
        for yy in range(40, 1088, 120):
            for xx in range(30, 1456, 130):
                cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
        for k in range(n_trays):
            xc = 1456 + 400 + k * spacing - i * speed
            cv2.rectangle(f, (int(xc - 300), 300), (int(xc + 300), 760), (40, 40, 200), -1)
        picks += tr.update(i, f, timestamp=i / 10)
    return picks + tr.flush()


def test_shipped_config_uses_two_views():
    assert CFG.side_views_px == (1000.0, 730.0) and CFG.edge_tolerance_px == 10.0


def test_single_frame_mode_picks_the_frame_centre():
    picks = _belt_run(CENTRE)
    assert len(picks) == 4 and all(abs(p.c[0] * 2 - 728) <= 92 for p in picks) and all(p.views is None for p in picks)


def test_pick_x_moves_the_measured_frame():
    picks = _belt_run(dataclasses.replace(CENTRE, pick_x_px=900))
    assert len(picks) == 4 and all(abs(p.c[0] * 2 - 900) <= 92 for p in picks)


def test_side_views_keep_two_frames_per_tray():
    picks = _belt_run(dataclasses.replace(CFG, side_views_px=(1000, 730)))
    assert len(picks) == 4
    for p in picks:
        left, right = p.views
        assert abs(left.c[0] * 2 - 1000) <= 92 and abs(right.c[0] * 2 - 730) <= 92
        assert left.frame_index < right.frame_index          # the belt runs right to left
        assert p is right                                     # the reported frame is the one nearest the centre


def _tray(meat_shift_left=0, meat_shift_right=0):
    """Corrected frame with one tray at the centre; meat reaches past the inner opening on either end."""
    f = np.full((1088, 1456, 3), 60, np.uint8)
    for yy in range(20, 1088, 110):
        for xx in range(20, 1456, 120):
            cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
    W, H = CFG.tray_w * 2, CFG.tray_h * 2
    cv2.fillPoly(f, [rect_corners(728, 544, W, H, 0).astype(np.int32)], (45, 45, 45))
    x0, x1 = 728 - CFG.inner_w / 2 + 20 - meat_shift_left, 728 + CFG.inner_w / 2 - 20 + meat_shift_right
    cv2.rectangle(f, (int(x0), 544 - 220), (int(x1), 544 + 220), (40, 40, 200), -1)
    own = np.zeros((544, 728), bool)
    own[(544 - 220) // 2:(544 + 220) // 2, int(x0) // 2:int(x1) // 2] = True
    corners = rect_corners(728, 544, CFG.inner_w, CFG.inner_h, 0)
    how = dict(left="measured", right="measured", top="measured", bottom="measured")
    num, _ = measure_tray(f, own, dict(cx=364.0, cy=272.0, th=0.0), corners, how, CFG)
    return num


def test_halves_add_up_to_the_whole_tray():
    n = _tray(meat_shift_left=60, meat_shift_right=30)
    h = n["halves"]
    assert h["left"]["band_px"] + h["right"]["band_px"] == n["lip_band_px"]
    assert h["left"]["band_meat_px"] + h["right"]["band_meat_px"] == n["lip_band_meat_px"]
    assert h["left"]["lip_px"] > h["right"]["lip_px"] > 0          # more meat on the left lip
    assert h["left"]["side"]["right"] == 0 and h["right"]["side"]["left"] == 0


def test_two_views_take_each_half_from_its_own_frame():
    """Left view: the left end looks clean. Right view (the old pick): the left end looks covered
    (camera angle) and the right end is clean. Combined = clean on both ends."""
    from overhang.pipeline import measure_pair
    import overhang.pipeline as pl
    clean, left_covered = _tray(), _tray(meat_shift_left=80)
    calls = iter([clean, left_covered])

    def fake_measure(payload, cfg=None, rect=None, debug_dir=None, _parts=False):
        n = dict(next(calls))
        halves = n.pop("halves")
        res = dict(tray_id=0, frame_index=payload["frame_index"], timestamp=0.0, overhang_class="none", light="green",
                   reliable=True, unreliable_reasons=[], outline_ok=True, outer=dict(cx=payload["cx"]),
                   timings_ms={}, _halves=halves, **n)
        return res, None, None

    orig = pl.measure
    pl.measure = fake_measure
    try:
        r = measure_pair(dict(frame_index=10, cx=1000.0), dict(frame_index=11, cx=730.0), cfg=CFG, debug_dir="")
    finally:
        pl.measure = orig
    assert r["views"]["combined"] and r["views"]["left_half"]["frame_index"] == 10
    assert r["lip_cover_pct"] == clean["lip_cover_pct"] < left_covered["lip_cover_pct"]
    assert r["frame_index"] == 11 and r["overhang_class"] == "none"


def test_view_frames_keep_the_whole_tray_box_in_view():
    """Belt 250 px per frame: the frame nearest x 1000 may cut the tray box; the view must not."""
    picks = _belt_run(dataclasses.replace(CFG, side_views_px=(1000, 730)), speed=250)
    hw = CFG.tray_w + 4                                      # full-res half width of the box
    assert picks and all(p.views[0].c[0] * 2 + hw <= 1456 for p in picks)


def test_pick_x_keeps_the_next_frame_as_a_backup():
    """If the frame nearest x 900 is right of it, the next frame (tray further left) is kept, so a tray
    whose box corner is cut there can be measured one frame later."""
    picks = _belt_run(dataclasses.replace(CENTRE, pick_x_px=900), speed=250)
    for p in picks:
        if p.c[0] * 2 > 900:
            assert p.backup is not None and p.backup.c[0] < p.c[0] and p.backup.frame_index == p.frame_index + 1
        else:
            assert p.backup is None
    assert all(p.backup is None for p in _belt_run(CENTRE, speed=250))   # single frame at the centre: no backup


def test_dynamic_one_photo_near_the_spot_two_when_further_right():
    for speed in (180, 250):
        cfg = dataclasses.replace(CFG, dynamic_split_px=1000, edge_tolerance_px=10)
        picks = _belt_run(cfg, speed=speed, n_trays=6)
        assert len(picks) == 6
        hw = CFG.tray_w + 4 - 10
        for p in picks:
            assert p.c[0] * 2 + hw <= 1456 + 1                         # the box (with tolerance) is in view
            assert p.backup is not None and p.backup.frame_index == p.frame_index + 1
            if p.c[0] * 2 > 1000:
                assert p.views == [p, p.backup]                        # left half here, right half one frame later
            else:
                assert p.views is None                                 # one photo is good for both ends


def test_corner_tolerance_only_when_set():
    assert Config().edge_tolerance_px == 0      # off unless the config sets it (the shipped config sets 10)


def _pair_with(bad_left, bad_right):
    from overhang.pipeline import measure_pair
    import overhang.pipeline as pl
    good = _tray()
    calls = iter([tuple(bad_left), tuple(bad_right)])

    def fake_measure(payload, cfg=None, rect=None, debug_dir=None, _parts=False):
        n = dict(good)
        halves = n.pop("halves")
        bad = list(next(calls))
        res = dict(tray_id=0, frame_index=payload["frame_index"], timestamp=0.0, overhang_class="none", light="green",
                   reliable=not bad, unreliable_reasons=bad, outline_ok="outline_check" not in bad, outer=dict(cx=payload["cx"]),
                   timings_ms={}, _halves=halves, **n)
        return res, None, None

    orig = pl.measure
    pl.measure = fake_measure
    try:
        return measure_pair(dict(frame_index=10, cx=1000.0), dict(frame_index=11, cx=730.0), cfg=CFG, debug_dir="")
    finally:
        pl.measure = orig


def test_hand_in_one_photo_uses_the_other():
    r = _pair_with(["hand_over_tray"], [])
    assert r["reliable"] and r["frame_index"] == 11 and r["views"]["left_view_failed"] == ["hand_over_tray"]
    r = _pair_with([], ["hand_over_tray"])
    assert r["reliable"] and r["frame_index"] == 10 and r["views"]["right_view_failed"] == ["hand_over_tray"]


def test_problem_in_both_photos_stays_unreadable():
    r = _pair_with(["hand_over_tray"], ["outline_check"])
    assert not r["reliable"] and set(r["unreliable_reasons"]) == {"hand_over_tray", "outline_check"}


def test_failed_left_view_falls_back_to_the_other_frame():
    from overhang.pipeline import measure_pair
    import overhang.pipeline as pl
    good = _tray()
    calls = iter([("outline_check",), ()])

    def fake_measure(payload, cfg=None, rect=None, debug_dir=None, _parts=False):
        n = dict(good)
        halves = n.pop("halves")
        bad = list(next(calls))
        res = dict(tray_id=0, frame_index=payload["frame_index"], timestamp=0.0, overhang_class="none", light="green",
                   reliable=not bad, unreliable_reasons=bad, outline_ok=not bad, outer=dict(cx=payload["cx"]),
                   timings_ms={}, _halves=halves, **n)
        return res, None, None

    orig = pl.measure
    pl.measure = fake_measure
    try:
        r = measure_pair(dict(frame_index=10, cx=1050.0), dict(frame_index=11, cx=800.0), cfg=CFG, debug_dir="")
    finally:
        pl.measure = orig
    assert r["reliable"] and r["frame_index"] == 11 and r["views"]["combined"] is False
    assert r["views"]["left_view_failed"] == ["outline_check"] and "_halves" not in r
