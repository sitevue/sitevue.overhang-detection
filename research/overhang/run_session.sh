#!/bin/bash
# usage: run_session.sh VIDEO RUNDIR [TRAY_SIZE_JSON]
# picks -> SAM + outer fit -> full-res cache. Inner calibration/fit/render are run separately.
set -e
cd "$(dirname "$0")/.."
VIDEO="$1"; export RUN="$2"; mkdir -p "$RUN"
[ -n "$3" ] && export TRAY_SIZE_JSON="$3"
[ -f "$RUN/picks.json" ] || PICKS_ONLY=1 .venv/bin/python overhang/run_video.py "$VIDEO" "$RUN" 2>&1 | grep -v -i "warn\|redirect"
.venv/bin/python overhang/measure_full.py "$VIDEO" "$RUN" 2>&1 | grep -v -i "warn\|redirect\|sam2_video\|Loading weights\|^sam \|nanmedian"
.venv/bin/python overhang/cache_full.py "$VIDEO"
