#!/bin/bash
# Time every stage of the overhang pipeline on one session, one worker per stage (no parallelism),
# so the numbers are per-tray cost on one CPU core + the GPU.
# usage: bash overhang/bench_stages.sh SESSION8 SCALE_MAP.json TRAY_SIZE.json
set -euo pipefail
cd ~/overhang
S=$1; MAP=$2; TS=$3; B=bench/$S; rm -rf bench; mkdir -p $B
V=$(ls sessions/$S*.mp4); export WORKERS=1
t() { local a=$(date +%s.%N); "$@" > bench/$(echo "$*" | md5sum | cut -c1-6).log 2>&1; echo "$(echo "$(date +%s.%N) - $a" | bc)"; }
N_FRAMES=$(.venv/bin/python -c "import cv2;c=cv2.VideoCapture('$V');print(int(c.get(cv2.CAP_PROP_FRAME_COUNT)))")
T_DEC=$(t .venv/bin/python -c "
import cv2;c=cv2.VideoCapture('$V')
while c.grab(): pass")
T_TRACK=$(RUN=$B PICKS_ONLY=1 t .venv/bin/python overhang/run_video.py $V $B)
N=$(.venv/bin/python -c "import json;print(len(json.load(open('$B/picks.json'))))")
T_SAM=$(RUN=$B t .venv/bin/python overhang/measure_full.py $V $B)      # frame grab + SAM + first outline (cache written)
R=bench/rect/$S; mkdir -p bench/rect
T_RECT=$(t .venv/bin/python overhang/rectify.py $MAP $B $R)
T_OUT=$(RUN=$R TRAY_SIZE_JSON=$TS t .venv/bin/python overhang/measure_full.py $V $R)
.venv/bin/python overhang/merge_runs.py sessions/sessions.json bench/all $R/ > /dev/null
cp $TS bench/all/tray_size.json
export RUN=bench/all TRAY_SIZE_JSON=$TS
T_INNER=$(t .venv/bin/python overhang/run_inner.py)
cp bench/all/inner_fit.json bench/all/inner_fit_v1.json
T_GQ=$( { a=$(date +%s.%N); .venv/bin/python overhang/guard_inner.py >/dev/null 2>&1; .venv/bin/python overhang/quad_inner.py >/dev/null 2>&1; echo "$(date +%s.%N) - $a" | bc; } )
T_REND=$(t .venv/bin/python overhang/render_inner.py)
.venv/bin/python - <<E
N=$N; F=$N_FRAMES; dur=F/10
rows=[("decode all frames (10 fps)",$T_DEC,F),("tracking + tray picks (5 fps)",$T_TRACK,F),
      ("grab frame + SAM tray mask + outline",$T_SAM,N),("camera-angle correction",$T_RECT,N),("outline on corrected frame",$T_OUT,N),
      ("inner edge points + fit",$T_INNER,N),("guard + four-wall lines",$T_GQ,N),("measure + overlay image",$T_REND,N)]
print(f"session $S: {N} trays, {F} frames = {dur/60:.1f} min of video ({N/(dur/60):.1f} trays/min)")
for name,sec,n in rows:
    per = sec/n*1000
    print(f"{name:40s} {sec:7.1f} s total   {per:7.1f} ms per {'frame' if n==F else 'tray'}   {sec/dur*100:5.1f}% of real time")
E
