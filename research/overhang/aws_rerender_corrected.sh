#!/bin/bash
# Re-run only measurement + overlays on the corrected pooled run (allr).
set -euo pipefail
BUCKET="$1"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
P=results_corrected
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/$P/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/$P/FAILED"' ERR
echo "=== $(date -u) rerender (rounded corners + shift) start"
PY=/opt/pytorch/bin/python
export RUN=allr TRAY_SIZE_JSON=tray_size_corrected.json
rm -rf allr/inner_overlays allr/sheets
.venv/bin/python overhang/render_inner.py
.venv/bin/python overhang/make_sheets.py allr
cp overhang/README_overlays.txt allr/inner_overlays/README.txt
cp allr/inner_overlay_summary.csv allr/inner_overlays/summary.csv
rm -f tray_overlays_corrected.zip
$PY -c "import shutil; shutil.make_archive('tray_overlays_corrected', 'zip', 'allr', 'inner_overlays')"
aws s3 cp tray_overlays_corrected.zip "s3://$BUCKET/$P/tray_overlays.zip" --only-show-errors
aws s3 sync allr/sheets "s3://$BUCKET/$P/sheets/" --delete --only-show-errors
aws s3 cp allr/inner_overlay_summary.csv "s3://$BUCKET/$P/inner_overlay_summary.csv" --only-show-errors
aws s3 cp "$LOG" "s3://$BUCKET/$P/job.log"
echo done | aws s3 cp - "s3://$BUCKET/$P/DONE"
echo "=== $(date -u) finished"
