#!/bin/bash
# Full overhang pipeline on the GB10 (sitevue-gb10-04), for one export.
# usage: bash overhang/gb10_job.sh SCALE_MAP.json TRAY_SIZE.json
# Writes ~/overhang/DONE or ~/overhang/FAILED when it ends; log in ~/overhang/gb10_job.log
set -euo pipefail
cd ~/overhang
MAP="$1"; TS="$2"; V=sessions
LOG=gb10_job.log; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
rm -f DONE FAILED
trap 'echo "FAILED at line $LINENO"; touch FAILED' ERR
echo "=== $(date -u) start  map=$MAP  tray_size=$TS"

# 1) per session: pick one frame per tray, SAM tray mask, cache half- and full-res frames (4 sessions at a time)
mkdir -p runs
ls $V/*.mp4 | xargs -P 4 -I{} bash -c '
  set -e; f="{}"; s=$(basename "$f" .mp4); d=runs/${s:0:8}; mkdir -p "$d"
  [ -f "$d/picks.json" ] || RUN=$d PICKS_ONLY=1 .venv/bin/python overhang/run_video.py "$f" "$d" > "$d.log" 2>&1
  RUN=$d .venv/bin/python overhang/measure_full.py "$f" "$d" >> "$d.log" 2>&1
  echo "sam done $s $(.venv/bin/python -c "import json;print(len(json.load(open(\"$d/picks.json\"))))") trays"'
echo "=== $(date -u) SAM done"

# 2) camera-angle correction of cached frames and masks
rm -rf rruns; mkdir -p rruns
ls -d runs/*/ | xargs -P 8 -I{} bash -c 'set -e; d="{}"; s=$(basename "$d"); .venv/bin/python overhang/rectify.py '"$MAP"' "$d" "rruns/$s" > "rruns/$s.rect.log" 2>&1; echo "corrected $s"'
echo "=== $(date -u) corrected"

# 3) outer outline on corrected frames, with the corrected tray size
ls $V/*.mp4 | xargs -P 8 -I{} bash -c '
  set -e; f="{}"; s=$(basename "$f" .mp4); d=rruns/${s:0:8}
  RUN=$d TRAY_SIZE_JSON='"$TS"' .venv/bin/python overhang/measure_full.py "$f" "$d" > "$d.m.log" 2>&1; echo "outlined $s"'
echo "=== $(date -u) outlines done"

# 4) pooled: inner opening, guarded fit, four-wall lines, measurement + overlays, sheets
rm -rf allr
.venv/bin/python overhang/merge_runs.py $V/sessions.json allr rruns/*/
export RUN=allr TRAY_SIZE_JSON="$TS"
cp "$TS" allr/tray_size.json
.venv/bin/python overhang/run_inner.py
cp allr/inner_fit.json allr/inner_fit_v1.json
.venv/bin/python overhang/guard_inner.py
.venv/bin/python overhang/quad_inner.py
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py allr
cp overhang/README_overlays.txt allr/inner_overlays/README.txt
echo "=== $(date -u) finished"
touch DONE
