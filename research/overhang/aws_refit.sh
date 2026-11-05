#!/bin/bash
# Re-run from the outline fit onward (SAM masks are reused from runs/*/cache).
set -euo pipefail
BUCKET="$1"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/results/FAILED"' ERR
echo "=== $(date -u) refit start"
PY=/opt/pytorch/bin/python; V=sessions
ls $V/*.mp4 | xargs -P 4 -I{} bash -c '
  f="{}"; s=$(basename "$f" .mp4); d=runs/${s:0:8}
  RUN=$d TRAY_SIZE_JSON=tray_size.json .venv/bin/python overhang/measure_full.py "$f" "$d" >> "$d.refit.log" 2>&1
  echo "refit $s"'
echo "=== $(date -u) outlines refitted"
rm -rf all
.venv/bin/python overhang/merge_runs.py $V/sessions.json all runs/*/
export RUN=all TRAY_SIZE_JSON=tray_size.json
.venv/bin/python overhang/run_inner.py
cp all/inner_fit.json all/inner_fit_v1.json
.venv/bin/python overhang/guard_inner.py
.venv/bin/python overhang/quad_inner.py
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py all
echo "=== $(date -u) rendered"
cp overhang/README_overlays.txt all/inner_overlays/README.txt
cp all/inner_overlay_summary.csv all/inner_overlays/summary.csv
rm -f tray_overlays.zip
$PY -c "import shutil; shutil.make_archive('tray_overlays', 'zip', 'all', 'inner_overlays')"
aws s3 cp tray_overlays.zip "s3://$BUCKET/results/tray_overlays.zip" --only-show-errors
aws s3 sync all/sheets "s3://$BUCKET/results/sheets/" --delete --only-show-errors
for f in results_full.csv inner_overlay_summary.csv inner_fit.json inner_quad.json inner_size.json picks.json; do aws s3 cp "all/$f" "s3://$BUCKET/results/$f" --only-show-errors; done
aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"
echo done | aws s3 cp - "s3://$BUCKET/results/DONE"
echo "=== $(date -u) finished"
