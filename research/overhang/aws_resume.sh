#!/bin/bash
# Runs on the overhang-batch GPU instance. usage: aws_job.sh BUCKET
# Pulls code + session videos from s3://BUCKET/input/, runs the whole overhang pipeline,
# pushes results to s3://BUCKET/results/ and writes s3://BUCKET/results/DONE (or FAILED).
set -euo pipefail
BUCKET="$1"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/results/FAILED"' ERR
echo "=== $(date -u) resume start"

PY=/opt/pytorch/bin/python; [ -x "$PY" ] || PY=python3
V=sessions; R=runs
echo "=== resume from pooled inner-edge step"
# 3) one pooled run: inner opening calibration + guarded fit + four-wall lines + overlays
.venv/bin/python overhang/merge_runs.py $V/sessions.json all $R/*/
export RUN=all TRAY_SIZE_JSON=tray_size.json
.venv/bin/python overhang/run_inner.py
cp all/inner_fit.json all/inner_fit_v1.json
.venv/bin/python overhang/guard_inner.py
.venv/bin/python overhang/quad_inner.py
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py all
echo "=== $(date -u) rendered"

# 4) package + upload
cp overhang/README_overlays.txt all/inner_overlays/README.txt
cp all/inner_overlay_summary.csv all/inner_overlays/summary.csv
$PY -c "import shutil; shutil.make_archive('tray_overlays', 'zip', 'all', 'inner_overlays')"
aws s3 cp tray_overlays.zip "s3://$BUCKET/results/tray_overlays.zip"
aws s3 sync all/sheets "s3://$BUCKET/results/sheets/" --only-show-errors
for f in results_full.csv inner_overlay_summary.csv inner_fit.json inner_quad.json inner_size.json picks.json; do aws s3 cp "all/$f" "s3://$BUCKET/results/$f" --only-show-errors; done
aws s3 cp tray_size.json "s3://$BUCKET/results/tray_size.json"
aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"
echo done | aws s3 cp - "s3://$BUCKET/results/DONE"
echo "=== $(date -u) finished"
