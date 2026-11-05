# Calibration (per camera / line)

Everything specific to a line lives in one YAML file (`configs/fpl-grind.yaml`). Re-check it
when the camera moves, the camera height or lens changes, or a different tray is used.

| Setting | What it is | How it was measured (FPL grind, 10-29) |
|---|---|---|
| `camera.scale_map` | Camera-angle correction | `overhang scale-map` on empty-belt frames (roller spacing 61→67 px left→right, fit rms ~0.4 px) |
| `tray.outer_size_half_px` | Outer tray size, half-res corrected px | Median size of edge-fitted clean trays: 389.97 × 288.14 |
| `tray.inner_size_px` | Inner opening, full-res corrected px | Median rigid inner fit on clean trays (no meat outside): 724.0 × 525.7 |
| `tray.inner_corner_radius_px` | Rounded inner corners | Circle fit on an empty tray: 45 px |
| `tray.inner_shift_px` | Inner opening offset along the tray | Hand-marked inner edges on 3 empty trays: +4 px |
| `classes.*` | Customer cut-offs | Chosen to split 10-29 reliable trays like the line's slide (77/18/3/1 %) |

## Camera-angle correction

1. Save 3–5 full-resolution frames of the **empty belt** (no trays), e.g. with ffmpeg:
   `ffmpeg -ss 00:01:20 -i video.mp4 -frames:v 1 empty1.png`
2. `overhang scale-map configs/scale_maps/LINE-DATE.json empty1.png empty2.png empty3.png`
3. Point `camera.scale_map` at the new file.

The camera must not move between building the map and running; if it does, rebuild the map.

## Tray sizes

Run the pipeline on a video with a good number of clean trays (`overhang run VIDEO --out r.jsonl
--debug-dir img`) and look at the review images: the green box should sit on the tray's outer
edge and the magenta line on the lip→wall drop on clean sides. The size calibration scripts used
for 10-29 are in [research/](../research/README.md): `calib_size.py` (outer size), the
`run_inner.py` calibration pass (inner size), `measure_empty.py` and the edge-marking page in
`research/labeller/` (corner radius and shift from hand-marked empty trays).

## Class cut-offs

The numbers behind each class (`lip_cover_pct`, `past_green_px`) are in every result. To tune:
run a day's video with `--out`, label a sample of trays (none/mild/small/large) by eye, and pick
the cut-offs that agree best. Changing the cut-offs needs only a config change and a restart.
