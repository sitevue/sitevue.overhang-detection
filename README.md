# sitevue.overhang-detection

Tray overhang detection for the FPL grind line. A camera above the belt sees every tray; for
each tray this tells you how much meat hangs over the tray's lip, as a customer-facing class and
a floor-light colour:

| Class | Light | Meaning (default cut-offs, `configs/fpl-grind.yaml`) | Share on 10-29 |
|---|---|---|---|
| `none` | green | lip covered < 10.5% | ~77% |
| `mild` | yellow | lip covered 10.5–18.5% | ~18% |
| `small` | yellow | lip covered ≥ 18.5% | ~3–4% |
| `large` | red | ≥ 10,000 px of meat past the tray's outer edge | ~1% |

"Lip covered" = % of the band between the tray's inner opening and its outer edge that meat
covers. A clean tray reads ~5%.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
# 1. torch for your GPU (RTX 50-series / CUDA 13 shown; CPU-only also works, slowly)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
# 2. this package
pip install -e ".[sam,service,dev]"

# results as JSON lines, plus review images
overhang run path/to/video.mp4 --out results.jsonl --debug-dir review/

# service for the customer UI (HTTP + WebSocket on :8000)
overhang serve rtsp://CAMERA/stream --cors-origin http://localhost:3000

# does this machine keep up with the belt?
overhang bench path/to/video.mp4 --max-trays 150
```

The SAM 2 weights (`facebook/sam2.1-hiera-small`, ~180 MB) download from Hugging Face on first
use. All commands take `--config` (default `configs/fpl-grind.yaml`, or `$OVERHANG_CONFIG`).

## For the UI

Start with **[docs/integration.md](docs/integration.md)** (endpoints, WebSocket example) and
**[docs/output.md](docs/output.md)** (every field of a tray result). The UI needs only
`overhang_class`, `light` and `reliable` per tray; `/api/summary` gives the line view.

```python
from overhang import Config, OverhangPipeline

cfg = Config.load("configs/fpl-grind.yaml")
for tray in OverhangPipeline(cfg).run("rtsp://camera/stream"):
    print(tray["tray_id"], tray["overhang_class"], tray["light"], tray["reliable"])
```

## Layout

```
configs/fpl-grind.yaml     everything line-specific: tray sizes, camera correction, class cut-offs, lights
configs/scale_maps/        camera-angle correction maps (from the belt rollers)
src/overhang/
  pipeline.py              OverhangPipeline: frames -> one result per tray (GPU + CPU worker pool)
  tracker.py               one frame per tray, while the tray is still on the belt
  sam.py                   SAM 2 tray mask (the only learned model / GPU step)
  rectify.py               camera-angle correction
  outline.py               outer tray edge (green)
  inner.py                 inner opening (magenta)
  measure.py               lip covered %, px outside magenta, px past green, next-tray and hand checks
  classify.py              class + light from the numbers
  render.py                review overlay image (not for customers)
  service.py               FastAPI service: REST + WebSocket
  cli.py                   `overhang run | serve | bench | scale-map`
docs/                      pipeline.md (how it works, speed), output.md, integration.md, calibration.md
tests/                     fast tests (no GPU / video needed): `pytest`
tools/compare_with_batch.py  check results against the research batch run
Dockerfile                 GPU service image for the site machine
```

## Status

- Validated on the 2026-10-29 export (busiest session, 695 trays): same trays and 98.6% same
  class as the research batch pipeline; ~0.33 s per tray on a GB10, 88 s for 21 min of video.
- Not yet run on the site machine (RTX 5070). Expect the GPU to be fine (SAM uses 0.55 GB and
  <10% of the time); run `overhang bench` there to size `runtime.workers`.
- Class cut-offs are a first guess matched to the line's slide (77/18/3/1 %), not to hand labels.
- Measurements are in camera pixels, not mm.

See [docs/pipeline.md](docs/pipeline.md) for the method and [docs/calibration.md](docs/calibration.md)
to set up another line or camera.
