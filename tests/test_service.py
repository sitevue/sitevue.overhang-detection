"""Service endpoints with a fake pipeline (no camera, no SAM)."""
import os
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from overhang import Config  # noqa: E402
from overhang import pipeline as pipeline_mod  # noqa: E402
from overhang.service import create_app  # noqa: E402

CFG = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "fpl-grind.yaml"))


def fake_result(i, cls):
    return dict(tray_id=i, frame_index=10 * i, timestamp=i * 1.0, overhang_class=cls, light=CFG.light[cls],
                reliable=True, unreliable_reasons=[], lip_cover_pct=5.0, outside_px=100, past_green_px=0,
                outside_by_side_px=dict(left=10, right=90, top=0, bottom=0), meat_offset_px=dict(along=3.0, across=-1.0))


class FakePipeline:
    def __init__(self, *a, **k):
        pass

    delay = 0.0

    def run(self, source, realtime=False):
        for i, c in enumerate(["none", "none", "mild", "large"]):
            time.sleep(self.delay)
            yield fake_result(i, c)


def test_endpoints(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "OverhangPipeline", FakePipeline)
    app = create_app(CFG, "fake.mp4", workers=0)
    with TestClient(app) as c:
        for _ in range(50):
            if c.get("/health").json()["status"] == "finished":
                break
            time.sleep(0.05)
        h = c.get("/health").json()
        assert h["status"] == "finished" and h["trays"] == 4
        assert c.get("/api/trays/latest").json()["overhang_class"] == "large"
        assert [r["tray_id"] for r in c.get("/api/trays?limit=2").json()] == [3, 2]
        s = c.get("/api/summary").json()
        assert s["counts"] == dict(none=2, mild=1, small=0, large=1)
        assert s["latest"]["light"] == CFG.light["large"]
        assert c.get("/api/config").json()["classes"]["large_past_green_px"] == CFG.large_past_green_px


def test_websocket_pushes_trays(monkeypatch):
    class Slow(FakePipeline):
        delay = 0.3
    monkeypatch.setattr(pipeline_mod, "OverhangPipeline", Slow)
    app = create_app(CFG, "fake.mp4", workers=0)
    with TestClient(app) as c:
        with c.websocket_connect("/ws/trays") as ws:
            got = [ws.receive_json()["overhang_class"] for _ in range(2)]
        assert all(g in ("none", "mild", "large") for g in got)
