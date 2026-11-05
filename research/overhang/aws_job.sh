#!/bin/bash
# Runs on the overhang-batch GPU instance. usage: aws_job.sh BUCKET
# Pulls code + session videos from s3://BUCKET/input/, runs the whole overhang pipeline,
# pushes results to s3://BUCKET/results/ and writes s3://BUCKET/results/DONE (or FAILED).
set -euo pipefail
BUCKET="$1"; W=/opt/overhang; cd "$W"
LOG="$W/job.log"; exec > >(tee -a "$LOG") 2>&1
trap 'echo "FAILED at line $LINENO"; aws s3 cp "$LOG" "s3://$BUCKET/results/job.log"; echo failed | aws s3 cp - "s3://$BUCKET/results/FAILED"' ERR
echo "=== $(date -u) start"

aws s3 cp "s3://$BUCKET/input/code.tgz" . && tar xzf code.tgz
aws s3 sync "s3://$BUCKET/input/sessions/" sessions/ --only-show-errors
echo "=== $(date -u) data in: $(ls sessions/*.mp4 | wc -l) videos"

# python: the DLAMI ships a PyTorch venv
PY=/opt/pytorch/bin/python; [ -x "$PY" ] || PY=python3
$PY -m pip install -q "transformers==5.22.1" opencv-python-headless pillow
# scripts call .venv/bin/python: link the whole venv dir so Python finds its pyvenv.cfg and site-packages
rm -rf .venv && ln -s "$(dirname "$(dirname "$PY")")" .venv
.venv/bin/python -c "import cv2, transformers; print('cv2', cv2.__version__, 'transformers', transformers.__version__)"
$PY -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

V=sessions; R=runs; mkdir -p "$R"
CAL=276975df-3951-4877-945b-73d96508b2ce

# 1) calibrate the outer tray size on the longest early session
RUN=$R/${CAL:0:8} PICKS_ONLY=1 .venv/bin/python overhang/run_video.py "$V/$CAL.mp4" "$R/${CAL:0:8}"
RUN=$R/${CAL:0:8} .venv/bin/python overhang/measure_full.py "$V/$CAL.mp4" "$R/${CAL:0:8}" | grep -v "^sam " | tail -2
RUN=$R/${CAL:0:8} .venv/bin/python overhang/calib_size.py
cp "$R/${CAL:0:8}/tray_size.json" tray_size.json; cat tray_size.json
echo "=== $(date -u) calibrated"

# 2) every session: pick trays, SAM + outer fit with the calibrated size (3 at a time, one GPU)
ls $V/*.mp4 | xargs -P 3 -I{} bash -c '
  f="{}"; s=$(basename "$f" .mp4); d=runs/${s:0:8}; mkdir -p "$d"
  [ -f "$d/picks.json" ] || RUN=$d PICKS_ONLY=1 .venv/bin/python overhang/run_video.py "$f" "$d" > "$d.log" 2>&1
  RUN=$d TRAY_SIZE_JSON=tray_size.json .venv/bin/python overhang/measure_full.py "$f" "$d" >> "$d.log" 2>&1
  echo "done $s $(.venv/bin/python -c "import json;print(len(json.load(open(\"$d/picks.json\"))))") trays"'
echo "=== $(date -u) all sessions measured"

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
