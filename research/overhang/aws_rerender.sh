#!/bin/bash
# Re-run only the measurement/overlay step on the existing pooled run (SAM + edge fits untouched).
set -euo pipefail
BUCKET="$1"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/results/FAILED"' ERR
echo "=== $(date -u) rerender start"
PY=/opt/pytorch/bin/python
export RUN=all TRAY_SIZE_JSON=tray_size.json
rm -rf all/inner_overlays all/sheets
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py all
cp overhang/README_overlays.txt all/inner_overlays/README.txt
cp all/inner_overlay_summary.csv all/inner_overlays/summary.csv
rm -f tray_overlays.zip
$PY -c "import shutil; shutil.make_archive('tray_overlays', 'zip', 'all', 'inner_overlays')"
aws s3 cp tray_overlays.zip "s3://$BUCKET/results/tray_overlays.zip" --only-show-errors
aws s3 sync all/sheets "s3://$BUCKET/results/sheets/" --delete --only-show-errors
aws s3 cp all/inner_overlay_summary.csv "s3://$BUCKET/results/inner_overlay_summary.csv" --only-show-errors
aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"
echo done | aws s3 cp - "s3://$BUCKET/results/DONE"
echo "=== $(date -u) finished"
