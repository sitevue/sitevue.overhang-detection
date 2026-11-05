"""Contact sheets (5 x 8 tiles) of RUN/inner_overlays in time order -> RUN/sheets/sheet_N.jpg"""
import csv, os, sys, cv2, numpy as np
RUN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN", "out/run1")
S = sorted(csv.DictReader(open(f"{RUN}/inner_overlay_summary.csv")), key=lambda s: float(s["t_sec"]))
TW, TH, COLS, ROWS = 400, 330, 5, 8
os.makedirs(f"{RUN}/sheets", exist_ok=True)
def clock(t):
    t = float(t); return f"{int(t//3600):02d}:{int(t%3600//60):02d}:{int(t%60):02d}" if t >= 3600 else f"{t}s"
def thumb(s):
    im = cv2.imread(f"{RUN}/inner_overlays/tray_{int(s['idx']):03d}.jpg")[34:]
    h, w = im.shape[:2]; f = min(TW / w, (TH - 26) / h)
    im = cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    tile = np.full((TH, TW, 3), 24, np.uint8)
    tile[26:26 + im.shape[0], (TW - im.shape[1]) // 2:(TW - im.shape[1]) // 2 + im.shape[1]] = im
    pct = float(s["outside_pct"]); col = (90, 200, 90) if pct < 1 else (0, 200, 255) if pct < 3 else (60, 60, 255)
    cv2.putText(tile, f"#{s['idx']}  {clock(s['t_sec'])}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(tile, f"{pct:.1f}% out", (TW - 105, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return tile
per = COLS * ROWS
for p in range(0, len(S), per):
    T = [thumb(s) for s in S[p:p + per]]
    while len(T) % COLS: T.append(np.full((TH, TW, 3), 24, np.uint8))
    cv2.imwrite(f"{RUN}/sheets/sheet_{p // per + 1:03d}.jpg", np.vstack([np.hstack(T[i:i + COLS]) for i in range(0, len(T), COLS)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print("sheets", (len(S) + per - 1) // per)
