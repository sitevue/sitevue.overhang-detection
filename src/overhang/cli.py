"""Command line:

  overhang run SOURCE      results as JSON lines (stdout or --out), optional review images
  overhang serve SOURCE    HTTP + WebSocket service for the customer UI
  overhang bench SOURCE    per-stage timing on this machine (does it keep up with the belt?)
  overhang scale-map OUT FRAME [FRAME ...]   build the camera-angle correction from empty-belt frames

SOURCE is a video file, an RTSP/HTTP URL, or a camera index (0).
"""
import argparse
import json
import os
import statistics as st
import sys
import time

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "fpl-grind.yaml")


def _cfg(path):
    from .config import Config
    return Config.load(path)


def cmd_run(a):
    from .pipeline import OverhangPipeline
    cfg = _cfg(a.config)
    pipe = OverhangPipeline(cfg, workers=a.workers, debug_dir=a.debug_dir)
    out = open(a.out, "w") if a.out else sys.stdout
    for r in pipe.run(a.source, realtime=a.realtime, max_trays=a.max_trays):
        out.write(json.dumps(r) + "\n")
        out.flush()
        if a.out:
            print(f"#{r['tray_id']:5d}  {r['overhang_class']:7s}  light {str(r['light']):6s}  "
                  f"lip {r.get('lip_cover_pct', 0):5.1f}%  outside {r.get('outside_px', 0):7,d} px  "
                  f"past green {r.get('past_green_px', 0):7,d} px  {'' if r['reliable'] else 'unreliable: ' + ','.join(r['unreliable_reasons'])}",
                  file=sys.stderr)


def cmd_serve(a):
    import uvicorn
    from .service import create_app
    app = create_app(_cfg(a.config), a.source, workers=a.workers, realtime=a.realtime,
                     cors_origins=a.cors_origin or None, debug_dir=a.debug_dir, loop_video=a.loop)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def cmd_bench(a):
    """Runs the pipeline with ONE worker process so every stage is timed on one CPU core, then
    reports per-tray costs and whether that keeps up with the given belt rate."""
    from .pipeline import OverhangPipeline
    cfg = _cfg(a.config)
    pipe = OverhangPipeline(cfg, workers=0)
    t0 = time.time()
    rows = list(pipe.run(a.source, max_trays=a.max_trays))
    wall = time.time() - t0
    rows = [r for r in rows if "timings_ms" in r]
    keys = list(dict.fromkeys(k for r in rows for k in r["timings_ms"]))
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
    print(f"machine: {os.uname().nodename}  cpu cores: {os.cpu_count()}  gpu: {dev}  sam dtype: {cfg.sam_dtype}")
    print(f"{len(rows)} trays, {pipe.stats['frames']} frames, {wall:.1f} s wall time")
    per_frame = pipe.stats["track_ms"] / max(pipe.stats["frames"], 1)
    print(f"  {'tracking (per frame)':22s} {per_frame:7.1f} ms")
    total = 0.0
    for k in keys:
        v = [r["timings_ms"][k] for r in rows if k in r["timings_ms"]]
        m = st.median(v)
        if k != "sam_model":            # sam_model is part of sam_total
            total += m
        print(f"  {k:22s} {m:7.1f} ms median   {sorted(v)[int(0.9 * (len(v) - 1))]:7.1f} ms p90")
    per_tray_cpu = total - st.median([r["timings_ms"].get("sam_total", 0) for r in rows])
    need = a.trays_per_min / 60.0
    print(f"per tray: {total:.0f} ms in total ({per_tray_cpu:.0f} ms in the CPU workers)")
    print(f"belt at {a.trays_per_min:.0f} trays/min needs 1 tray every {1000 / need:.0f} ms")
    cores = max(1, int(per_tray_cpu * need / 1000) + 1)
    print(f"-> CPU workers needed: {cores} (config runtime.workers), tracking uses {per_frame * cfg.fps / 10:.1f}% of one core")


def cmd_scale_map(a):
    from .scale_map import build
    build(a.out, a.frames)


def main(argv=None):
    p = argparse.ArgumentParser(prog="overhang", description="Tray overhang detection")
    p.add_argument("--config", default=os.environ.get("OVERHANG_CONFIG", DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="results as JSON lines")
    r.add_argument("source")
    r.add_argument("--out", help="write JSON lines here (default: stdout)")
    r.add_argument("--debug-dir", help="also write a review image per tray here")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--realtime", action="store_true", help="pace a video file at camera speed")
    r.add_argument("--max-trays", type=int, default=None)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="HTTP + WebSocket service")
    s.add_argument("source")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--realtime", action="store_true", help="pace a video file at camera speed (demo)")
    s.add_argument("--loop", action="store_true", help="restart a video file when it ends (demo)")
    s.add_argument("--cors-origin", action="append", help="allowed browser origin for the UI (repeatable)")
    s.add_argument("--debug-dir")
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", help="per-stage timing on this machine")
    b.add_argument("source")
    b.add_argument("--max-trays", type=int, default=150)
    b.add_argument("--trays-per-min", type=float, default=60.0, help="belt rate to check against (busiest minute on 10-29: 59)")
    b.set_defaults(fn=cmd_bench)

    m = sub.add_parser("scale-map", help="camera-angle correction from empty-belt frames")
    m.add_argument("out")
    m.add_argument("frames", nargs="+")
    m.set_defaults(fn=cmd_scale_map)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
