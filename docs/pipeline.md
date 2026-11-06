# How it works

One camera above the belt (1456 × 1088, 10 fps). For every tray the pipeline finds the tray's
outer edge (green) and inner opening (magenta, where the flat lip drops into the tray), then
measures how much of this tray's meat lies outside the inner opening.

```
camera frames
   │  every 2nd frame, half resolution
   ▼
1. tracker.py   meat blobs by colour, tracked across frames; one frame per tray, the one where
   │            the tray is fully in view and closest to the frame centre. Emitted once the tray
   │            has crossed the centre (≈ 0.5–1 s later), while it is still on the belt.
   ▼
2. sam.py       SAM 2.1 (hiera-small) tray mask. Prompts are automatic: points on the visible
   │            tray rim (positive) and on meat (negative). The only learned model; the only GPU step.
   ▼  ── handed to a CPU worker process ──
3. rectify.py   camera-angle correction (the camera looks at the belt at a slight angle: a tray is
   │            ~8% smaller on the left of the frame). Built from the belt rollers, a regular grid.
   ▼
4. outline.py   outer edge: rectangle of the known tray size fitted to the SAM mask; re-anchored
   │            if SAM spilled into a neighbour; refined on the trusted part of SAM's edge.
   ▼
5. inner.py     inner opening, full resolution: lip→wall brightness edges on each side, a rigid fit
   │            of the known inner size, a second pass near that fit, a guard against meat shadows,
   │            then four wall lines (a wall hidden by meat is placed parallel to the opposite wall
   │            at the known size), shifted 4 px and with 45 px rounded corners.
   ▼
6. measure.py   meat outside the opening: lip covered %, px outside magenta, px past green.
   │            Drops the next tray's meat when trays touch; flags a hand over the tray.
   ▼
7. classify.py  none / mild / small / large + light colour (cut-offs in the config)
```

## Speed

Measured on a GB10 (ARM CPU, NVIDIA GB10 GPU), busiest 10-29 session (21 min of video,
1,506 trays; the belt runs at ~90 trays/min when moving):

| Stage | Where | Median per tray |
|---|---|---|
| Tracking | CPU, main process | 1.8 ms per frame (≈2% of one core at 10 fps) |
| SAM 2 tray mask (fp32) | GPU | 84 ms (bf16: ~35 ms) |
| Camera-angle correction | CPU worker | 1.5 ms |
| Outer outline | CPU worker | 148 ms |
| Inner opening | CPU worker | 14 ms |
| Measure + classify | CPU worker | 87 ms |
| **Total** | | **~0.33 s per tray** |

At ~90 trays/min the belt needs 1.5 trays/s, i.e. ~0.4 s of CPU per second: 1 worker keeps up;
the default is 3 for headroom.

The site machine (RTX 5070, 12 GB) has not been benchmarked yet. SAM uses 0.55 GB of GPU
memory, and the GPU is <10% of the per-tray time, so the GPU is not the limit; the CPU is.
Run `overhang bench VIDEO` on the machine to check (prints the number of workers needed).

## Belt speed and tracking (fixed 2026-11-05)

The belt moves a tray ~180 full-res px per frame at 10 fps (a tray crosses the view in ~6 frames)
and stops and restarts. The first tracker looked at every 2nd frame and matched blobs within 90
half-res px of their last position, so tracks broke at every frame: it found only ~46% of the trays
(695 of 1,506 on the busiest 10-29 session) and picked them at random positions, so 26% failed the
edge-of-view check. The tracker now looks at every frame and matches each blob to where its tray
should be (last position + speed). Same session: 1,506 trays (each tray in a hand-checked 5 s
stretch picked exactly once), edge of view 1%, unreliable 5% (was 30%).

Before that fix, this repo reproduced the research batch run exactly (same trays, 98.6% same
class); the measurement code is unchanged, only which trays and frames get picked. The 10-29 batch
numbers (4,396 trays, the 77/18/3/1 class split) come from the old tracker and so cover under half
the trays, from biased positions.

## Known limits

- Pixels, not millimetres. Cut-offs are in camera pixels after correction; a different camera
  height or lens needs new calibration (see calibration.md).
- The outer edge is placed from the known tray size where meat covers the rim; "past green" on
  that end depends on that placement.
- A tray partly out of view, under a hand, or with a doubtful outline is still measured but
  marked `reliable: false`.
- The class cut-offs are a first guess that reproduces the line's 77/18/3/1 % split on 10-29; they
  are not yet matched to hand labels.
