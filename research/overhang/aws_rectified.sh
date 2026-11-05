#!/bin/bash
# Re-run 10-29 with the camera-angle correction: correct each session's cached frames and SAM
# masks, recalibrate the tray size, then outline fit -> inner edge -> measurement -> overlays.
set -euo pipefail
BUCKET="$1"; MAP="$2"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/results/FAILED"' ERR
echo "=== $(date -u) corrected run start"
PY=/opt/pytorch/bin/python; V=sessions
rm -rf rruns allr; mkdir -p rruns
ls -d runs/*/ | xargs -P 6 -I{} bash -c 'd="{}"; s=$(basename "$d"); .venv/bin/python overhang/rectify.py '"$MAP"' "$d" "rruns/$s" > "rruns/$s.rect.log" 2>&1; echo "corrected $s"'
CAL=rruns/276975df
RUN=$CAL .venv/bin/python overhang/measure_full.py $V/276975df-3951-4877-945b-73d96508b2ce.mp4 $CAL > $CAL.m1.log 2>&1
RUN=$CAL .venv/bin/python overhang/calib_size.py
cp $CAL/tray_size.json tray_size_corrected.json; cat tray_size_corrected.json; echo
ls $V/*.mp4 | xargs -P 4 -I{} bash -c '
  f="{}"; s=$(basename "$f" .mp4); d=rruns/${s:0:8}
  RUN=$d TRAY_SIZE_JSON=tray_size_corrected.json .venv/bin/python overhang/measure_full.py "$f" "$d" > "$d.m2.log" 2>&1
  echo "measured $s"'
.venv/bin/python overhang/merge_runs.py $V/sessions.json allr rruns/*/
export RUN=allr TRAY_SIZE_JSON=tray_size_corrected.json
cp tray_size_corrected.json allr/tray_size.json
.venv/bin/python overhang/run_inner.py
cp allr/inner_fit.json allr/inner_fit_v1.json
.venv/bin/python overhang/guard_inner.py
.venv/bin/python overhang/quad_inner.py
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py allr
echo "=== $(date -u) rendered"
cp overhang/README_overlays.txt allr/inner_overlays/README.txt
cp allr/inner_overlay_summary.csv allr/inner_overlays/summary.csv
rm -f tray_overlays_corrected.zip
$PY -c "import shutil; shutil.make_archive('tray_overlays_corrected', 'zip', 'allr', 'inner_overlays')"
P=results_corrected
aws s3 cp tray_overlays_corrected.zip "s3://$BUCKET/$P/tray_overlays.zip" --only-show-errors
aws s3 sync allr/sheets "s3://$BUCKET/$P/sheets/" --delete --only-show-errors
for f in results_full.csv inner_overlay_summary.csv inner_fit.json inner_quad.json inner_size.json picks.json tray_size.json; do aws s3 cp "allr/$f" "s3://$BUCKET/$P/$f" --only-show-errors; done
aws s3 cp "$LOG" "s3://$BUCKET/$P/job.log"
echo done | aws s3 cp - "s3://$BUCKET/$P/DONE"
echo "=== $(date -u) finished"
