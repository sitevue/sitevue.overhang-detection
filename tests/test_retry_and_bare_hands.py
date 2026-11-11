"""Bare hands are not trays; a box that fails its checks is retried with SAM's other masks."""
import dataclasses
import os

import cv2
import numpy as np

import overhang.pipeline as pl
from overhang import Config
from overhang.color import skin_like
from overhang.tracker import TrayTracker

CFG = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "fpl-grind.yaml"))
RNG = np.random.default_rng(0)


def _belt():
    f = np.full((1088, 1456, 3), 60, np.uint8)
    for yy in range(40, 1088, 120):
        for xx in range(30, 1456, 130):
            cv2.rectangle(f, (xx, yy), (xx + 60, yy + 40), (0, 220, 240), -1)
    return f


def _meat(f, x0, y0, x1, y1):
    """Ground-meat texture: red with fine bright/dark strands."""
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    patch[:] = (45, 45, 190)
    noise = RNG.integers(-60, 60, patch.shape[:2]).astype(np.int16)
    patch[..., 2] = np.clip(190 + noise, 120, 255)
    patch[..., 1] = np.clip(45 + noise // 3, 10, 90)
    f[y0:y1, x0:x1] = patch


def _skin(f, x0, y0, x1, y1):
    """Bare hand: smooth, orange-red."""
    cv2.rectangle(f, (x0, y0), (x1, y1), (90, 120, 200), -1)
    return cv2.GaussianBlur(f, (9, 9), 0)


def test_skin_like_separates_skin_from_meat():
    f = _belt()
    _meat(f, 300, 300, 900, 760)
    blob = np.zeros(f.shape[:2], bool)
    blob[300:760, 300:900] = True
    assert not skin_like(f, blob)[0]
    g = _skin(_belt(), 300, 300, 900, 760)
    assert skin_like(g, blob)[0]


def test_blurred_red_tray_is_not_skin():
    f = _belt()
    cv2.rectangle(f, (300, 300), (900, 760), (40, 40, 200), -1)          # smooth but pure red
    blob = np.zeros(f.shape[:2], bool)
    blob[300:760, 300:900] = True
    assert not skin_like(f, blob)[0]


def _run(cfg, painter):
    tr, picks = TrayTracker(cfg), []
    for i in range(int((1456 + 800) / 180) + 4):
        f = _belt()
        xc = 1456 + 400 - i * 180
        x0, x1 = int(max(xc - 300, 0)), int(min(xc + 300, 1456))
        if x1 > x0 + 10:
            g = painter(f, x0, 300, x1, 760)
            f = f if g is None else g
        picks += tr.update(i, f, timestamp=i / 10)
    return picks + tr.flush(), tr


def test_tracker_drops_a_bare_hand_but_keeps_meat():
    picks, tr = _run(CFG, _skin)
    assert picks == [] and tr.n_bare_hands == 1
    picks, tr = _run(CFG, _meat)
    assert len(picks) == 1 and tr.n_bare_hands == 0
    picks, tr = _run(dataclasses.replace(CFG, drop_bare_hands=False), _skin)
    assert len(picks) == 1


def test_tracker_keeps_spare_frames():
    picks, _ = _run(CFG, _meat)
    p = picks[0]
    assert 1 <= len(p.spares) <= max(CFG.retry_frames, CFG.retry_frames_hand)
    assert all(s.frame_index != p.frame_index for s in p.spares)
    gaps = [abs(s.frame_index - p.frame_index) for s in p.spares]
    assert gaps == sorted(gaps)                                          # nearest first
    picks, _ = _run(dataclasses.replace(CFG, retry_frames=0, retry_frames_hand=0), _meat)
    assert picks[0].spares == []


def _fake_one(results):
    calls = []

    def one(payload, cfg=None, rect=None, debug_dir=None, _parts=False):
        calls.append(payload["sam_score"])
        return dict(results[payload["sam_score"]])
    return one, calls


def test_failed_box_retries_other_sam_masks():
    orig = pl._measure_one
    try:
        pl._measure_one, calls = _fake_one({0.9: dict(reliable=False, unreliable_reasons=["outline_check"]),
                                            0.8: dict(reliable=False, unreliable_reasons=["outline_check"]),
                                            0.7: dict(reliable=True, unreliable_reasons=[], outside_pct_of_meat=2.0)})
        r = pl.measure(dict(sam_score=0.9, alt_trays=[(None, 0.8), (None, 0.7)]), cfg=CFG, debug_dir="")
        assert r["reliable"] and r["sam_mask"] == 2 and calls == [0.9, 0.8, 0.7]
        # a hand is not a box problem: no retry
        pl._measure_one, calls = _fake_one({0.9: dict(reliable=False, unreliable_reasons=["hand_over_tray"]),
                                            0.8: dict(reliable=True, unreliable_reasons=[])})
        r = pl.measure(dict(sam_score=0.9, alt_trays=[(None, 0.8)]), cfg=CFG, debug_dir="")
        assert not r["reliable"] and calls == [0.9]
        # nothing passes: the first result stands
        pl._measure_one, calls = _fake_one({0.9: dict(reliable=False, unreliable_reasons=["edge_of_view"], n=1),
                                            0.8: dict(reliable=False, unreliable_reasons=["outline_check"], n=2)})
        r = pl.measure(dict(sam_score=0.9, alt_trays=[(None, 0.8)]), cfg=CFG, debug_dir="")
        assert r["n"] == 1
    finally:
        pl._measure_one = orig


def test_retried_box_must_sit_on_the_meat():
    """A second-guess mask that passes the outline check but leaves most of the meat outside its box is rejected."""
    orig = pl._measure_one
    try:
        pl._measure_one, calls = _fake_one({0.9: dict(reliable=False, unreliable_reasons=["outline_check"], n=1),
                                            0.8: dict(reliable=True, unreliable_reasons=[], outside_pct_of_meat=53.2, n=2),
                                            0.7: dict(reliable=True, unreliable_reasons=[], outside_pct_of_meat=4.0, n=3)})
        r = pl.measure(dict(sam_score=0.9, alt_trays=[(None, 0.8), (None, 0.7)]), cfg=CFG, debug_dir="")
        assert r["n"] == 3 and r["sam_mask"] == 2
        pl._measure_one, calls = _fake_one({0.9: dict(reliable=False, unreliable_reasons=["outline_check"], n=1),
                                            0.8: dict(reliable=True, unreliable_reasons=[], outside_pct_of_meat=34.3, n=2)})
        r = pl.measure(dict(sam_score=0.9, alt_trays=[(None, 0.8)]), cfg=CFG, debug_dir="")
        assert r["n"] == 1 and not r["reliable"]
    finally:
        pl._measure_one = orig


def _video(tmp_path):
    """A short synthetic belt video with one meat tray passing right to left."""
    path = str(tmp_path / "belt.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1456, 1088))
    for i in range(int((1456 + 800) / 180) + 4):
        f = _belt()
        xc = 1456 + 400 - i * 180
        x0, x1 = int(max(xc - 300, 0)), int(min(xc + 300, 1456))
        if x1 > x0 + 10:
            _meat(f, x0, 300, x1, 760)
        w.write(f)
    w.release()
    return path


def _pipeline_with(tmp_path, monkeypatch, verdict):
    """Run the pipeline on the synthetic video with SAM and the measurement faked: verdict(frame_index, n) -> reasons."""
    calls = []
    monkeypatch.setattr(pl, "prepare", lambda pick, sam, cfg: dict(frame_index=pick.frame_index, tray_id=pick.tray_id))

    def fake_measure(payload, *a, **k):
        calls.append(payload["frame_index"])
        bad = verdict(payload["frame_index"], len(calls))
        return dict(tray_id=payload["tray_id"], frame_index=payload["frame_index"], reliable=not bad,
                    unreliable_reasons=bad, overhang_class="none", outside_pct_of_meat=1.0)
    monkeypatch.setattr(pl, "measure", fake_measure)
    one_photo = dataclasses.replace(CFG, side_views_px=None, edge_tolerance_px=0.0)   # the retry is the same with two views
    out = list(pl.OverhangPipeline(one_photo, workers=0, sam=object()).run(_video(tmp_path)))
    return out, calls


def test_hand_in_the_picked_frame_is_read_on_another_frame(tmp_path, monkeypatch):
    out, calls = _pipeline_with(tmp_path, monkeypatch, lambda fi, n: ["hand_over_tray"] if n == 1 else [])
    assert len(out) == 1 and out[0]["reliable"] and out[0]["frame_retry"] and len(calls) == 2
    assert out[0]["frame_index"] != calls[0]


def test_hand_in_every_frame_stays_unreadable_after_at_most_retry_frames_hand(tmp_path, monkeypatch):
    out, calls = _pipeline_with(tmp_path, monkeypatch, lambda fi, n: ["hand_over_tray"])
    assert len(out) == 1 and not out[0]["reliable"] and out[0]["unreliable_reasons"] == ["hand_over_tray"]
    assert out[0]["frame_index"] == calls[0] and 2 <= len(calls) <= 1 + CFG.retry_frames_hand


def test_outline_problem_gets_fewer_retries_than_a_hand(tmp_path, monkeypatch):
    out, calls = _pipeline_with(tmp_path, monkeypatch, lambda fi, n: ["outline_check"])
    assert not out[0]["reliable"] and len(calls) <= 1 + CFG.retry_frames


import pytest


@pytest.mark.parametrize("failed", ["left_view_failed", "right_view_failed"])
def test_one_photo_failed_reads_the_whole_tray_from_the_centre_frame(tmp_path, monkeypatch, failed):
    """Two views: the pair fell back to the right-half photo (left one failed); the centre frame is used instead."""
    monkeypatch.setattr(pl, "prepare", lambda pick, sam, cfg: dict(frame_index=pick.frame_index, tray_id=pick.tray_id))

    def fake_pair(pa, pb, *a, **k):
        return dict(tray_id=pb["tray_id"], frame_index=pb["frame_index"], reliable=True, unreliable_reasons=[],
                    overhang_class="mild", outside_pct_of_meat=2.0,
                    views=dict(left_half=dict(frame_index=pa["frame_index"]), right_half=dict(frame_index=pb["frame_index"]),
                               combined=False, **{failed: ["edge_of_view"]}))
    single = []

    def fake_measure(payload, *a, **k):
        single.append(payload["frame_index"])
        return dict(tray_id=payload["tray_id"], frame_index=payload["frame_index"], reliable=True, unreliable_reasons=[],
                    overhang_class="none", outside_pct_of_meat=1.0, outer=dict(cx=728.0))
    monkeypatch.setattr(pl, "measure_pair", fake_pair)
    monkeypatch.setattr(pl, "measure", fake_measure)
    out = list(pl.OverhangPipeline(CFG, workers=0, sam=object()).run(_video(tmp_path)))
    assert len(out) == 1 and len(single) == 1
    r = out[0]
    assert r["overhang_class"] == "none" and r["views"]["whole_tray_from"] == "centre"
    assert r["frame_index"] == single[0] == r["views"]["centre"]["frame_index"]
