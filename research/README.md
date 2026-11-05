# Research code (batch pipeline, calibration, experiments)

How the method in `src/overhang/` was developed and calibrated on the FPL grind 2026-10-29 export.
Kept for reference and for **re-calibrating** a line; the product code is `src/overhang/`.
These scripts are not packaged, not tested, and expect the layout they were written in:
run them from `research/` (so the scripts sit in `overhang/`), with videos and outputs in
folders you pass as arguments or set via the `RUN` environment variable.

## Getting the data

`overhang/prepare_sessions.sh EXPORT/overhang OUT_DIR` turns an S3 export (6 s `.ts` segments per
session) into one `.mp4` per session plus `sessions.json` (session start, seconds since midnight).
The S3 sync command is at the top of the script. Segments are joined without re-encoding;
checked to reproduce the 10-29 session videos frame for frame.

## Batch pipeline (whole export, day-level calibration)

`overhang/gb10_job.sh SCALE_MAP.json TRAY_SIZE.json` ran the full export on a GB10:

| Step | Script | What it does |
|---|---|---|
| 1 | `run_video.py` (`PICKS_ONLY=1`) | Track trays, one frame per tray → `picks.json` |
| 1 | `measure_full.py` | SAM 2 tray mask per pick (cached), outer outline → `results_full.csv` |
| 2 | `rectify.py` | Camera-angle correction of cached frames and masks |
| 3 | `measure_full.py` | Outer outline again on corrected frames with the corrected tray size |
| 4 | `merge_runs.py` | Merge sessions; global tray index, clock time |
| 4 | `run_inner.py` | Inner opening: two passes, **calibrates the inner size over all clean trays** |
| 4 | `guard_inner.py` | Guard against meat-shadow edge points |
| 4 | `quad_inner.py` | Four-wall inner quadrilateral |
| 4 | `render_inner.py` | Numbers + review overlays (`NO_IMAGE=1` = numbers only) |
| 4 | `make_sheets.py`, `add_reliability.py` | Contact sheets; reliability flags + summary CSV |

`rebuild_frames.py` rebuilds the corrected frame cache from videos + kept SAM masks (no SAM rerun).
`aws_*.sh` + `infra/overhang-batch.yml` are the earlier AWS (CloudFormation + SSM) version of the same job.

## Calibration

| Script | Calibrates |
|---|---|
| `scale_map.py` | Camera-angle correction from roller spacing (now also `overhang scale-map`) |
| `calib_size.py` | Outer tray size (median edge-fitted size of clean trays) |
| `run_inner.py` (calibration pass) | Inner opening size (median rigid fit, size free, on clean trays) |
| `measure_empty.py`, `analyze_hand_edges.py`, `score_edges.py` | Inner edge on empty trays vs hand-marked edges → corner radius 45 px, shift +4 px |
| `calib_inner.py`, `calib_rim.py`, `calib_wall.py` | Earlier lip/wall calibration experiments |
| `labeller/index.html` + `trays.json` | The labelling / edge-marking page (images not included) |
| `labeller/hand_edges.json` | Hand-marked inner (and some outer) edge points on 3 empty trays (e041-e043, 2026-11-04 clip at 81.0-81.2 s), in labeller image px (same frame as `trays.json` `rect`/`inner`) |
| `labeller/tray_labels.json` | 218 trays labelled none/mild/small/large by eye (first test clip): mild 150, small 50, none 15, large 3. Use to tune the class cut-offs |

## Timing

`bench_stages.sh` (per stage, one core, batch scripts) and `bench_sam.py` (SAM fp32 vs bf16 on the
GPU). For the product pipeline use `overhang bench` instead.

## Older experiments

`apply_inner.py`, `draw_inner.py`, `draw_rigid.py`, `tray_zones.py`, `cache_full.py`,
`flag_occluded.py`, `bench.py`, `test_cc.py`, `test_fit.py`, `explore/*.py`: earlier versions and
one-off checks (colour thresholds, timeline of meat coverage, first tray fits).
