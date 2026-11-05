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
695 trays, belt averaging 33 trays/min, busiest minute 59):

| Stage | Where | Median per tray |
|---|---|---|
| Tracking | CPU, main process | 1.8 ms per frame (≈2% of one core at 10 fps) |
| SAM 2 tray mask (fp32) | GPU | 84 ms (bf16: ~35 ms) |
| Camera-angle correction | CPU worker | 1.5 ms |
| Outer outline | CPU worker | 148 ms |
| Inner opening | CPU worker | 14 ms |
| Measure + classify | CPU worker | 87 ms |
| **Total** | | **~0.33 s per tray** |

With 6 workers the whole 21-minute session ran in 88 s (~14× real time). The belt needs at most
~1 tray/s, so one CPU worker would just keep up; the default is 3 for headroom.

The site machine (RTX 5070, 12 GB) has not been benchmarked yet. SAM uses 0.55 GB of GPU
memory, and the GPU is <10% of the per-tray time, so the GPU is not the limit; the CPU is.
Run `overhang bench VIDEO` on the machine to check (prints the number of workers needed).

## Agreement with the research batch run

The batch pipeline ([research/](../research/README.md)) calibrated the inner opening size over the whole
day and processed trays after the video ended. This repo fixes those calibrations in the config
and runs tray by tray. On the busiest 10-29 session: same 695 trays, 693 picked from the same
frame, 98.6% same class, same reliable flag on every tray; lip covered % differs by a median 0.06
points (8 trays by more than 1 point). Check with `tools/compare_with_batch.py`.

## Known limits

- Pixels, not millimetres. Cut-offs are in camera pixels after correction; a different camera
  height or lens needs new calibration (see calibration.md).
- The outer edge is placed from the known tray size where meat covers the rim; "past green" on
  that end depends on that placement.
- A tray partly out of view, under a hand, or with a doubtful outline is still measured but
  marked `reliable: false`.
- The class cut-offs are a first guess that reproduces the line's 77/18/3/1 % split on 10-29; they
  are not yet matched to hand labels.
