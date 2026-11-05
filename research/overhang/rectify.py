"""Correct frames for the camera angle, using a roller-based scale map (scale_map.py).

In the corrected image every length is in "frame-centre pixels": a tray looks the same size
wherever it is, and its edges form a true rectangle. The corrected image keeps the original
size (1456 x 1088 full, 728 x 544 half) and is centred on the frame centre, so the rest of the
pipeline runs unchanged with the frame-centre tray sizes.

Forward map (image -> corrected):  X = 728 + integral_{728}^{x} dx' / sh(x', y)
                                    Y = 544 + integral_{544}^{y} dy' / sv(x, y')
The inverse needed by cv2.remap is found by fixed-point iteration.

usage (correct a whole run's cached frames and SAM masks):
  python overhang/rectify.py SCALE_MAP.json RUN_IN RUN_OUT
"""
import json, os, shutil, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_map import design

FW, FH = 1456, 1088


class Rectifier:
    def __init__(self, map_json, step=4):
        M = json.load(open(map_json))
        self.ch, self.cv = np.array(M["coef_h"]), np.array(M["coef_v"])
        self.p0h, self.p0v = M["pitch_h_ref"], M["pitch_v_ref"]
        self._build(step)

    def sh(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (design(x.ravel(), y.ravel()) @ self.ch).reshape(x.shape) / self.p0h

    def sv(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (design(x.ravel(), y.ravel()) @ self.cv).reshape(x.shape) / self.p0v

    def _build(self, step):
        # forward map on a coarse grid (cumulative integrals), then upsample
        xs = np.arange(0, FW + step, step, dtype=float); ys = np.arange(0, FH + step, step, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        ish = 1.0 / self.sh(gx, gy); isv = 1.0 / self.sv(gx, gy)
        X = np.cumsum(np.concatenate([np.zeros((len(ys), 1)), (ish[:, 1:] + ish[:, :-1]) / 2 * step], 1), 1)
        Y = np.cumsum(np.concatenate([np.zeros((1, len(xs))), (isv[1:] + isv[:-1]) / 2 * step], 0), 0)
        ix0 = int(round(FW / 2 / step)); iy0 = int(round(FH / 2 / step))
        X = X - X[:, ix0:ix0 + 1] + FW / 2; Y = Y - Y[iy0:iy0 + 1, :] + FH / 2
        self._grid = (xs, ys, X, Y)
        # forward map at full resolution (bilinear on the coarse grid)
        fx = cv2.resize(X.astype(np.float32), (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)[:FH, :FW]
        fy = cv2.resize(Y.astype(np.float32), (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)[:FH, :FW]
        self.fwd = (fx, fy)
        # inverse: for each corrected pixel (X, Y) find the source (x, y)
        TX, TY = np.meshgrid(np.arange(FW, dtype=np.float32), np.arange(FH, dtype=np.float32))
        sx, sy = TX.copy(), TY.copy()
        for _ in range(8):
            cx_ = np.clip(sx, 0, FW - 1); cy_ = np.clip(sy, 0, FH - 1)
            px = cv2.remap(fx, cx_, cy_, cv2.INTER_LINEAR); py = cv2.remap(fy, cx_, cy_, cv2.INTER_LINEAR)
            sx = sx + (TX - px) * self.sh(cx_, cy_).astype(np.float32)
            sy = sy + (TY - py) * self.sv(cx_, cy_).astype(np.float32)
        self.inv_full = (sx.astype(np.float32), sy.astype(np.float32))
        self.inv_half = (cv2.resize(sx, (FW // 2, FH // 2)) / 2, cv2.resize(sy, (FW // 2, FH // 2)) / 2)
        # residual check of the inversion
        cx_ = np.clip(sx, 0, FW - 1); cy_ = np.clip(sy, 0, FH - 1)
        self.inv_err = float(np.median(np.hypot(cv2.remap(fx, cx_, cy_, cv2.INTER_LINEAR) - TX, cv2.remap(fy, cx_, cy_, cv2.INTER_LINEAR) - TY)))

    def warp_full(self, img, nearest=False):
        return cv2.remap(img, *self.inv_full, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def warp_half(self, img, nearest=False):
        return cv2.remap(img, self.inv_half[0].astype(np.float32), self.inv_half[1].astype(np.float32),
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def forward(self, x, y):
        """image (full-res) points -> corrected points"""
        fx, fy = self.fwd
        x = np.clip(np.asarray(x, np.float32), 0, FW - 1); y = np.clip(np.asarray(y, np.float32), 0, FH - 1)
        return (cv2.remap(fx, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel(),
                cv2.remap(fy, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel())


def rectify_run(map_json, run_in, run_out):
    R = Rectifier(map_json)
    print(f"inverse map residual {R.inv_err:.3f} px")
    for sub in ("cache", "cache_full"):
        os.makedirs(os.path.join(run_out, sub), exist_ok=True)
    picks = json.load(open(os.path.join(run_in, "picks.json")))
    for k in range(len(picks)):
        src = os.path.join(run_in, "cache", f"{k:03d}.npz")
        if not os.path.exists(src):
            continue
        z = dict(np.load(src))
        for key in ("tray", "own", "red"):
            z[key] = R.warp_half(z[key].astype(np.uint8), nearest=True).astype(bool)
        np.savez_compressed(os.path.join(run_out, "cache", f"{k:03d}.npz"), **z)
        jpg = os.path.join(run_in, "cache", f"{k:03d}.jpg")
        if os.path.exists(jpg):
            cv2.imwrite(os.path.join(run_out, "cache", f"{k:03d}.jpg"), R.warp_half(cv2.imread(jpg)))
        png = os.path.join(run_in, "cache_full", f"{k:03d}.png")
        if os.path.exists(png):
            cv2.imwrite(os.path.join(run_out, "cache_full", f"{k:03d}.png"), R.warp_full(cv2.imread(png)), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    for f in ("picks.json",):
        shutil.copy(os.path.join(run_in, f), os.path.join(run_out, f))
    print("rectified", len(picks), "trays ->", run_out)


if __name__ == "__main__":
    rectify_run(*sys.argv[1:4])
