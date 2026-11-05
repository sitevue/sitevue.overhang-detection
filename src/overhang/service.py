"""HTTP + WebSocket service for the customer UI.

Runs the pipeline on one video source in a background thread and serves its results:

  GET  /health                  service and pipeline state
  GET  /api/trays/latest        last tray result
  GET  /api/trays?limit=50      recent tray results, newest first
  GET  /api/summary?window=100  class counts / shares over the last N reliable trays + current light
  GET  /api/config              class cut-offs and light colours in use
  WS   /ws/trays                pushes every tray result as JSON as soon as it is ready

Start it with:  overhang serve SOURCE --config configs/fpl-grind.yaml --port 8000
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter, deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .classify import CLASSES
from .config import Config


def create_app(cfg: Config, source, workers: int | None = None, realtime: bool = False,
               keep: int = 2000, cors_origins: list[str] | None = None, debug_dir: str | None = None, loop_video=False):
    app = FastAPI(title="Overhang detection", version="0.1.0")
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET"], allow_headers=["*"])
    state = dict(results=deque(maxlen=keep), status="starting", error=None, started=time.time(), trays=0)
    subscribers: set[asyncio.Queue] = set()
    lock = threading.Lock()

    def publish(r):
        with lock:
            state["results"].append(r)
            state["trays"] += 1
        loop = app.state.loop
        for q in list(subscribers):
            loop.call_soon_threadsafe(_put_latest, q, r)

    def worker():
        from .pipeline import OverhangPipeline
        try:
            pipe = OverhangPipeline(cfg, workers=workers, debug_dir=debug_dir)
            state["status"] = "running"
            while True:
                for r in pipe.run(source, realtime=realtime):
                    publish(r)
                if not loop_video:
                    break
            state["status"] = "finished"
        except Exception as e:  # surface the failure on /health instead of dying silently
            state["status"], state["error"] = "error", f"{type(e).__name__}: {e}"

    @app.on_event("startup")
    async def _start():
        app.state.loop = asyncio.get_running_loop()
        threading.Thread(target=worker, daemon=True, name="overhang-pipeline").start()

    @app.get("/health")
    def health():
        return dict(status=state["status"], error=state["error"], trays=state["trays"],
                    uptime_s=round(time.time() - state["started"], 1), source=str(source))

    @app.get("/api/trays/latest")
    def latest():
        with lock:
            return state["results"][-1] if state["results"] else None

    @app.get("/api/trays")
    def recent(limit: int = 50):
        with lock:
            return list(state["results"])[-limit:][::-1]

    @app.get("/api/summary")
    def summary(window: int = 100):
        with lock:
            rel = [r for r in state["results"] if r.get("reliable")][-window:]
            last = state["results"][-1] if state["results"] else None
        n = len(rel)
        cnt = Counter(r["overhang_class"] for r in rel)
        side = Counter()
        for r in rel:
            for k, v in (r.get("outside_by_side_px") or {}).items():
                side[k] += v
        return dict(window=n,
                    counts={c: cnt.get(c, 0) for c in CLASSES},
                    share_pct={c: round(100.0 * cnt.get(c, 0) / n, 1) if n else 0.0 for c in CLASSES},
                    outside_by_side_px=dict(side),
                    mean_meat_offset_px=dict(
                        along=round(sum(r["meat_offset_px"]["along"] for r in rel) / n, 1) if n else 0.0,
                        across=round(sum(r["meat_offset_px"]["across"] for r in rel) / n, 1) if n else 0.0),
                    latest=dict(tray_id=last["tray_id"], overhang_class=last["overhang_class"], light=last["light"],
                                reliable=last["reliable"]) if last else None)

    @app.get("/api/config")
    def config():
        return dict(classes=dict(mild_lip_cover_pct=cfg.mild_lip_cover_pct, small_lip_cover_pct=cfg.small_lip_cover_pct,
                                 large_past_green_px=cfg.large_past_green_px), light=cfg.light)

    @app.websocket("/ws/trays")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        subscribers.add(q)
        try:
            while True:
                await sock.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            subscribers.discard(q)

    return app


def _put_latest(q: asyncio.Queue, item):
    """Never block the pipeline on a slow client: drop that client's oldest message instead."""
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(item)
