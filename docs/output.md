# Output: one result per tray

`overhang run` writes one JSON object per line; the service sends the same object on
`/ws/trays` and returns it from `/api/trays*`. Results come out in tray order, about 0.5–1 s
after the tray passes the centre of the camera view.

## What the customer UI needs

| Field | Type | Meaning |
|---|---|---|
| `tray_id` | int | Running tray number since the pipeline started (0, 1, 2, …) |
| `timestamp` | float | Seconds into the video for a file; Unix time for a live camera |
| `overhang_class` | `"none"` \| `"mild"` \| `"small"` \| `"large"` \| `"unknown"` | The customer-facing class |
| `light` | `"green"` \| `"yellow"` \| `"red"` \| `null` | Floor-light colour for that class (config `light:`) |
| `reliable` | bool | `false` when the numbers can't be trusted for this tray |
| `unreliable_reasons` | list of str | `hand_over_tray`, `edge_of_view`, `outline_check`, `no_tray_mask`, `not_in_pick_zone`, or `error: …` |

`unknown` (with `light: null`) means no tray outline could be found for that tray (e.g. the tray mask
failed), or (only when the optional pick zone `pick_zone_px` is set, off by default) that the tray was never completely inside the pick zone (`not_in_pick_zone`: e.g. two
touching trays merged into one blob, or the belt jumped it past). Such a tray is counted but not
measured, and its result has only `tray_id`, `frame_index`, `timestamp`, `overhang_class`, `light`,
`reliable` and `unreliable_reasons`. Treat it as "no reading", not as a good tray.

Suggested UI rule: show the class for every tray, but grey it out or mark it when
`reliable` is `false`; only reliable trays count in the summary (`/api/summary` already does this).

## How the class is decided

Three measured numbers, cut-offs in the config (`classes:`):

| Field | Meaning |
|---|---|
| `lip_cover_pct` | % of the tray lip (the band between the inner opening and the outer edge) covered by meat |
| `outside_px` | Meat pixels outside the inner opening (on the lip + past the outer edge) |
| `past_green_px` | Meat pixels past the tray's outer edge |

```
large  if past_green_px >= large_past_green_px         (10,000)
small  elif lip_cover_pct >= small_lip_cover_pct       (18.5)
mild   elif lip_cover_pct >= mild_lip_cover_pct        (10.5)
none   otherwise
```

A clean tray still reads ~5% lip covered (meat touching the inner edge, corners), so 5% is the
normal floor, not zero. Pixels are camera pixels after camera-angle correction (not mm yet).

## For the operator (which way is the meat going?)

| Field | Meaning |
|---|---|
| `outside_by_side_px` | `{left, right, top, bottom}`: meat outside the inner opening, per side of the tray (left/right = ends along the belt) |
| `meat_offset_px` | `{along, across}`: mean meat position relative to the tray centre (+ = right / down in the image) |

`/api/summary` adds these up over recent trays, so a UI can say e.g. "meat drifting to the right end".

## Diagnostics (not for the customer)

| Field | Meaning |
|---|---|
| `lip_px`, `outside_pct_of_meat`, `meat_px` | Meat on the lip only; outside px as % of the tray's meat; total meat px |
| `lip_band_px`, `lip_band_meat_px` | Lip band area and the meat on it (`lip_cover_pct` = ratio) |
| `walls_measured` | Inner walls seen directly: capital L R T B = measured, lower case = placed from the opposite wall |
| `inner_walls` | Same, per side: `measured` / `from_opposite` / `rigid` |
| `next_tray_left/right` | Another tray touches this one on that end; `dropped_next_tray_px` = its meat removed |
| `glove_frac`, `sleeve_frac`, `occluded` | Hand over the tray: `glove_frac` = blue-glove share of the tray's central 84% (a hand holding the edge doesn't count), `occluded` when it is over 3% or a white sleeve covers over 5% of the tray |
| `edge_of_view` | The tray (its outer box) is cut off by the frame edge. The belt past each end need not be in view: if it isn't, the next-tray check treats that end as having no neighbour |
| `outline_ok`, `tray_spill_frac`, `belt_inside_frac`, `other_meat_frac` | Outer outline checks. `outline_ok` is false only when belt rollers show inside the box (`belt_inside_frac` > 1%: box on two trays or on belt), another tray's meat covers over 5% of the box (`other_meat_frac`: box on two touching trays, where no belt shows between them), or the box is tilted past 30°. `tray_spill_frac` (SAM's mask leaking past the box) is a diagnostic only |
| `outer` | Outer box: centre, angle, size (corrected full-res px) |
| `inner_corners` | Inner opening corners TL, TR, BR, BL (before the 4 px shift and corner rounding) |
| `views` | Only with `side_views_px` (two frames per tray): `{left_half: {frame_index, cx}, right_half: {frame_index, cx}, combined}`. The tray's left half is measured in the first frame, its right half in the second; `frame_index`, `outer` and the other diagnostics are from the right-half frame. `combined: false` = only one photo was used for the whole tray: the other had no tray mask, or a problem the clean one did not (`left_view_failed` / `right_view_failed` lists it: a hand over the tray, edge of view, outline check). With `dynamic_split_px`, `views` is present only for trays that got two photos |
| `sam_mask` | Present when SAM's first mask gave a box that failed the outline / edge-of-view check and another of its three masks passed: 1 or 2 = the mask used |
| `frame_retry` | `true` when the tray was unreadable in its own frame(s) (outline, frame edge, or a hand over it) and was read on another frame with the tray in view (`retry_frames`, `retry_frames_hand`) |
| `sam_score`, `timings_ms` | SAM confidence; time per stage on this machine |

## Example

```json
{
  "tray_id": 4, "frame_index": 216, "timestamp": 21.6,
  "overhang_class": "mild", "light": "yellow", "reliable": true, "unreliable_reasons": [],
  "lip_cover_pct": 15.41, "outside_px": 11493, "past_green_px": 1251,
  "outside_by_side_px": {"left": 10665, "right": 0, "top": 716, "bottom": 112},
  "meat_offset_px": {"along": -51.7, "across": 1.7},
  "sam_score": 0.875, "lip_px": 10242, "outside_pct_of_meat": 3.58, "meat_px": 321249,
  "lip_band_px": 68443, "lip_band_meat_px": 10545, "walls_measured": "lRTB",
  "next_tray_left": false, "next_tray_right": false, "dropped_next_tray_px": 0,
  "glove_frac": 0.0, "sleeve_frac": 0.0, "occluded": false, "edge_of_view": false,
  "outline_ok": true, "tray_spill_frac": 0.0, "belt_inside_frac": 0.0,
  "outer": {"cx": 744.9, "cy": 460.8, "deg": -3.25, "w": 779.9, "h": 576.3},
  "inner_corners": [[368.3, 210.0], [1091.9, 181.9], [1123.3, 707.4], [400.1, 742.5]],
  "inner_walls": {"right": "measured", "top": "measured", "bottom": "measured", "left": "from_opposite"},
  "timings_ms": {"sam_model": 81.8, "sam_total": 84.0, "correct": 1.5, "outline": 147.9, "inner": 14.2, "measure": 87.2}
}
```
