"""Save the full-resolution frame of every picked tray: RUN/cache_full/NNN.png"""
import json, os, sys, cv2
RUN = os.environ.get("RUN", "out/run1"); video = sys.argv[1]
os.makedirs(f"{RUN}/cache_full", exist_ok=True)
picks = json.load(open(f"{RUN}/picks.json")); cap = cv2.VideoCapture(video)
for k, p in enumerate(picks):
    out = f"{RUN}/cache_full/{k:03d}.png"
    if os.path.exists(out): continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"]); ok, f = cap.read()
    if ok: cv2.imwrite(out, f, [cv2.IMWRITE_PNG_COMPRESSION, 1])
print("cached", len(picks))
