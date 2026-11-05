Tray overlays - FPL grind overhang camera, 2026-10-29 export
One full-resolution crop per tray picked from the footage.

Lines
  green (thin)   tray outer edge (SAM 2 tray mask, fitted as a rectangle of known size)
  magenta        tray inner edge = where the flat lip drops into the tray.
                 Each wall seen clearly gets its own fitted line; a wall hidden by meat
                 is drawn parallel to the opposite wall at the known inner width/height.
Meat outside the magenta line, blended at 30% opacity
  orange         meat on the lip (between the inner and outer edge)
  yellow         meat past the tray's outer edge

Caption, line 1 (the three headline numbers)
  lip covered      % of the lip band (the area between the magenta and green lines) covered by meat
  outside magenta  meat pixels outside the magenta line (orange + yellow), camera pixels
  past green       meat pixels past the green outer edge (yellow), camera pixels
  LARGE OVERHANG   shown in red, with a red frame round the image, when past green >= 10,000 px
                   (about the top 1% of trays on 10-29; the cut-off is a setting, LARGE_PX)
Caption, line 2: time of day (from the video timestamps), meat outside magenta as % of the tray's
  meat, which walls were measured on this tray (capital L R T B = measured; lower case = placed
  parallel to the opposite wall at the known size), and flags.

summary.csv has the same numbers for every tray: lip_cover_pct, outside_px, past_green_px, large,
plus lip_band_px / lip_band_meat_px (the band's area and the meat on it) and the older columns.

Flags in the caption
  NEXT TRAY L/R  another tray touches this one on that end (no belt rollers visible just past it).
                 Meat more than 10 camera px past this tray's outer edge on that end is treated as
                 the next tray's meat and not counted.
  HAND           a worker's glove or sleeve covers part of the tray; its numbers are not reliable.
summary.csv adds next_tray_left/right, dropped_next_tray_px, glove_frac, white_frac and occluded.

Camera-angle correction (corrected run)
  The camera views the belt at a slight angle: things look ~8% smaller at the left of the frame
  than at the centre. Frames are corrected with a map built from the belt rollers (a regular
  grid), so overlays are in "frame-centre pixels": a tray is the same size anywhere in view.
  Overlay images are therefore slightly stretched on the left compared with the raw video.

Inner opening shape
  The magenta inner edge has rounded corners (radius 45 px, measured on an empty tray) and is
  shifted 4 px towards the right end of the tray (calibrated from hand-marked edges on empty trays).
