# How it works

One camera above the belt (1456 × 1088, 10 fps). For every tray the pipeline finds the tray's
outer edge (green) and inner opening (magenta, where the flat lip drops into the tray), then
measures how much of this tray's meat lies outside the inner opening.

```
camera frames
   │  every 2nd frame, half resolution
   ▼
1. tracker.py   meat blobs by colour, tracked across frames. Two frames per tray (`side_views_px`,
   │            camera angle, below): the tray's left half from the frame where it is near x 1000,
   │            its right half from the frame near x 550, with the whole tray in view. Emitted once
   │            the tray has passed both (≈ 0.5–1 s later), while it is still on the belt.
   │            (Without `side_views_px`: one frame, the tray closest to the frame centre.)
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

## Camera angle: where along the belt to measure

The camera correction (step 3) is built from the belt rollers, so it is exact for things lying on the
belt. Meat heaped above the tray rim is higher than the belt, and it shows shifted in the image away
from the point straight under the camera. On the FPL camera that point is right of the frame centre
(about x 850-900). At the frame centre, where the tray is picked by default, the tray's left end
looks more covered than it is: meat appears to slide over the left lip. One frame earlier (tray
further right) the same lip shows clean. The opposite happens at x ~1000 to the right end.

Measured on 218 trays of three 10-29 sessions that could be followed through every frame:

| Tray centre x | ~730 (default pick) | ~990 (one frame earlier) |
|---|---|---|
| Median lip cover | 9.7 % | 3.8 % |
| Extra meat seen past the left lip (median) | 3,500-5,200 px | 0 |
| Class none / mild / small / large | 57 / 36 / 5 / 2 % | 90 / 6 / 1 / 2 % |

The whole pipeline in each mode, on the first 338-400 trays of the same three sessions (1,138 trays):

| Mode | Reliable | none / mild / small / large (reliable trays) | Median lip cover |
|---|---|---|---|
| Frame centre (default) | 99.6 % | 69.8 / 23.9 / 4.0 / 2.4 % | 6.6 % |
| `pick_x_px: 900` | 99.1 % | 86.2 / 9.6 / 2.4 / 1.9 % | 4.5 % |
| `side_views_px: [1000, 730]` | 99.1 % | 85.9 / 10.4 / 2.1 / 1.6 % | 4.3 % |

Checked by eye, the trays that enter or leave `large`: with meat pushed to the left end, the default
reads `large` (the extra "past green" is the camera angle) and both options read `mild`. With meat pushed
to the right end, `pick_x_px` reads `large` (now the right end is exaggerated: 3 trays) and two views
keep the default's `small`/`mild`. Two views is the one that treats both ends alike.

Later run on the same 1,138 trays, with the fixes for trays at the frame edge (a box corner may be up to
10 px out, `edge_tolerance_px: 10`; a first frame that cuts the tray or fails its outline check falls
back to the next frame):

| Mode | Unreadable | none / mild / small / large | Median lip cover | SAM runs per tray |
|---|---|---|---|---|
| Frame centre (default) | 4 | 69.8 / 23.9 / 4.0 / 2.4 % | 6.6 % | 1 |
| `pick_x_px: 900` + next frame as backup | 4 | 85.8 / 9.8 / 2.4 / 2.0 % | 4.5 % | 1 (+1 for 0.5 %) |
| `dynamic_split_px: 1000` | 3 | 87.8 / 8.0 / 2.2 / 1.9 % | 4.0 % | 1 for 82 %, 2 for 13 %, fallback 5 % |
| `side_views_px: [1000, 730]` | 4 | 86.4 / 9.7 / 2.2 / 1.7 % | 4.2 % | 2 (98 %) |

One photo is never right for both ends: each end is exaggerated by its own distance from the point under
the camera, and the two ends are ~780 px apart. Checked by eye, the 9 trays where `dynamic` reads a class
higher than two views are one-photo trays at x 790-930 with the meat against the right end (2 of them
`large`, as with `pick_x_px`); two views measures that end at x ~730 and reads it lower. Two views is the
recommended mode.

**Right half at x 550, not 730.** Each end is exaggerated by its own distance from the point under the camera
(~x 850-900), so the right half is best measured with the tray well left of the centre. On a recorded live hour
(11-11 12:00 UTC, 2,724 trays) and the 10-29 day, `[1000, 730]` gave the same frame for both photos on 36-41 %
of trays (the two targets were closer than one frame step), so those trays were really one photo at x ~730, and
trays with meat pushed to the right end read "large" (17,000-72,000 px past green; by eye the meat stops at the
tray's end). At `[1000, 550]` almost every tray gets two different frames and large went from 2.8 % to 1.5 %
(live hour) and 0.9 % to 0.6 % (10-29). If the left-half photo fails (usually the tray cut at x ~1050), the
whole tray is read from the frame nearest the centre, not from the right-half photo, whose left end would look
over-covered; the same when the right-half photo fails (the left-half photo would over-cover the right end).

**The shipped config uses two views** (`side_views_px: [1000, 550]`, with `edge_tolerance_px: 10`). The
class cut-offs were tuned at the frame centre, so with two views fewer trays read mild or small (10-29:
mild 16.8 % -> 6.0 %, small 2.6 % -> 1.2 %). `side_views_px: null` and `edge_tolerance_px: 0` go back to the
single frame at the centre. The options:

- `pick_x_px` (e.g. 900): measure the frame where the tray is nearest this x. Same cost. The belt moves
  180-250 px per frame, so the frame lands anywhere within about ±100 px of it.
- `side_views_px` (e.g. [1000, 550]): two frames per tray. The tray's left half is measured in the
  frame nearest the first x (tray right of the camera point, its left end near it), the right half in
  the frame nearest the second, and the numbers are added up per half. Two SAM runs per tray (~+85 ms
  on the GPU). A problem in either frame (hand, edge of view, outline check) makes the tray unreliable.

## Bare hands and retries

- **Bare hands are not trays.** Skin passes the meat colour rule, and a hand is the size of a tray, so a
  worker's bare hand over the belt was tracked and reported as an unreadable tray. Ground meat is full of
  fine edges and skin is smooth and more orange: on 10-29, 600 random real trays had a median edge strength
  of 16-22 and hue -2..+2, six bare hands 2-4 and +7..+10. A track whose picked blob has texture < 10 **and**
  hue >= 4 is dropped (`drop_bare_hands`); a blurred real tray (smooth but red) is kept.
- **A wrong box is retried.** Most outline / edge-of-view failures are a wrong first box: SAM's first mask
  covers some belt or the next tray. SAM returns three masks; when the first fails only these checks, the
  other two are tried (no extra GPU work). If none passes, the tray is measured on up to `retry_frames`
  other frames with the tray in view (one SAM run each, only for these trays).
- **A hand over the tray is retried on other frames.** A worker's hand often moves away within a frame or
  two, so a tray unreadable because of a hand is measured on up to `retry_frames_hand` (4) other frames with
  the tray in view, nearest first; the first one without a hand, with a good box on the meat, is used. (Not
  with SAM's other masks: a different mask does not move the hand.) On the 10-29 day this took unreadable
  trays from 163 to 117 (today's mode) and 127 to 101 (two views), with no readable tray changed; 12 random
  trays read this way all had the tray clearly in view and the box on it (checked by eye).
  A retried box is a second guess, so it is used only if it sits on the meat: at most 15 % of the tray's meat
  outside it (`RETRY_MAX_OUTSIDE_PCT`). On the 10-29 day the correct rescues had 0-6 % outside (all readable
  trays: median 1.2 %, 99th percentile 8.7 %); three wrong ones had 22-53 % and read "large" with
  58,000-337,000 px past green. Without the guard, the retry took the day from 185 to 161 unreadable
  (today's mode) and 144 to 126 (two views); the guard puts those three back to unreadable.

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
