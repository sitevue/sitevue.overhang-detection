#!/bin/bash
# Turn an overhang-camera export (6 s .ts segments per session) into one .mp4 per session plus
# sessions.json (session id -> start time, seconds since midnight, from the first segment).
#
# Get the export first (needs the ML engineer SSO profile):
#   aws sso login --profile <ml-engineer profile>
#   aws s3 sync s3://sitevue-ml-results-prod/exports/fpl-grind/2026-10-29_30/ ./fpl-grind-2026-10-29 --profile <ml-engineer profile>
# Layout after sync:  fpl-grind-2026-10-29/overhang/SHA256SUMS
#                     fpl-grind-2026-10-29/overhang/2026-10-29/<session-uuid>/seg_00015163.ts ...
#
# usage: bash overhang/prepare_sessions.sh EXPORT/overhang OUT_DIR
#   e.g. bash overhang/prepare_sessions.sh ~/data/fpl-grind-2026-10-29/overhang ~/data/fpl-grind-2026-10-29/overhang/sessions
# Joins segments without re-encoding (ffmpeg concat, -c copy): fast, frames are bit-identical.
set -euo pipefail
EXP=$(cd "$1" && pwd); OUT=$2; mkdir -p "$OUT"
if [ -f "$EXP/SHA256SUMS" ]; then
  echo "checking SHA256SUMS ..."
  (cd "$EXP" && { command -v sha256sum >/dev/null && sha256sum -c --quiet SHA256SUMS || shasum -a 256 -c --quiet SHA256SUMS; })
fi
echo "{" > "$OUT/sessions.json.tmp"; first=1
for d in "$EXP"/*/*/; do
  sid=$(basename "$d")
  ls "$d"seg_*.ts >/dev/null 2>&1 || continue
  list="$OUT/$sid.txt"
  ls "$d"seg_*.ts | sort | sed "s|.*|file '&'|" > "$list"
  t0=$(ffprobe -v error -show_entries format=start_time -of csv=p=0 "$(ls "$d"seg_*.ts | sort | head -1)")
  ffmpeg -v error -y -f concat -safe 0 -i "$list" -c copy "$OUT/$sid.mp4"
  [ $first -eq 1 ] || echo "," >> "$OUT/sessions.json.tmp"; first=0
  printf ' "%s": %s' "$sid" "$t0" >> "$OUT/sessions.json.tmp"
  echo "$sid  $(wc -l < "$list" | tr -d ' ') segments  start $t0 s"
done
printf '\n}\n' >> "$OUT/sessions.json.tmp"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump({k: round(float(v), 1) for k, v in d.items()}, open(sys.argv[2], 'w'), indent=1)" \
  "$OUT/sessions.json.tmp" "$OUT/sessions.json" && rm "$OUT/sessions.json.tmp"
echo "wrote $(ls "$OUT"/*.mp4 | wc -l | tr -d ' ') session videos + sessions.json in $OUT"
