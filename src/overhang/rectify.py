"""Camera-angle correction from a roller-based scale map.

The camera looks at the belt at a slight angle, so a tray looks ~8% smaller at the left of the
frame than at the centre. The belt rollers form a regular grid; their spacing across the frame
gives a smooth scale map (quadratic in x, linear in y). Correcting with it puts every length in
"frame-centre pixels": a tray has the same size wherever it is, and its edges form a true
rectangle. The corrected image keeps the original frame size and is centred on the frame centre.

Forward map (image -> corrected):  X = cx + integral_{cx}^{x} dx' / sh(x', y)
                                    Y = cy + integral_{cy}^{y} dy' / sv(x, y')
The inverse needed by cv2.remap is found by fixed-point iteration.
"""
import cv2
import numpy as np


def design(x, y, ref=(728.0, 544.0)):
    """Basis of the scale surfaces: 1, x, x^2, y, x*y (normalised to the frame centre)."""
    x = (np.asarray(x, float) - ref[0]) / ref[0]
    y = (np.asarray(y, float) - ref[1]) / ref[1]
    return np.stack([np.ones_like(x), x, x * x, y, x * y], 1)


class Rectifier:
    def __init__(self, scale_map: dict | None, frame_w=1456, frame_h=1088, step=4):
        self.fw, self.fh = frame_w, frame_h
        self.identity = scale_map is None
        if self.identity:
            return
        M = scale_map
        self.ref = tuple(M.get("ref", (frame_w / 2, frame_h / 2)))
        self.ch, self.cv = np.array(M["coef_h"]), np.array(M["coef_v"])
        self.p0h, self.p0v = M["pitch_h_ref"], M["pitch_v_ref"]
        self._build(step)

    def sh(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (design(x.ravel(), y.ravel(), self.ref) @ self.ch).reshape(x.shape) / self.p0h

    def sv(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (design(x.ravel(), y.ravel(), self.ref) @ self.cv).reshape(x.shape) / self.p0v

    def _build(self, step):
        FW, FH = self.fw, self.fh
        xs = np.arange(0, FW + step, step, dtype=float)
        ys = np.arange(0, FH + step, step, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        ish = 1.0 / self.sh(gx, gy)
        isv = 1.0 / self.sv(gx, gy)
        X = np.cumsum(np.concatenate([np.zeros((len(ys), 1)), (ish[:, 1:] + ish[:, :-1]) / 2 * step], 1), 1)
        Y = np.cumsum(np.concatenate([np.zeros((1, len(xs))), (isv[1:] + isv[:-1]) / 2 * step], 0), 0)
        ix0 = int(round(FW / 2 / step))
        iy0 = int(round(FH / 2 / step))
        X = X - X[:, ix0:ix0 + 1] + FW / 2
        Y = Y - Y[iy0:iy0 + 1, :] + FH / 2
        fx = cv2.resize(X.astype(np.float32), (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)[:FH, :FW]
        fy = cv2.resize(Y.astype(np.float32), (len(xs) * step, len(ys) * step), interpolation=cv2.INTER_LINEAR)[:FH, :FW]
        self.fwd = (fx, fy)
        TX, TY = np.meshgrid(np.arange(FW, dtype=np.float32), np.arange(FH, dtype=np.float32))
        sx, sy = TX.copy(), TY.copy()
        for _ in range(8):
            cx_ = np.clip(sx, 0, FW - 1)
            cy_ = np.clip(sy, 0, FH - 1)
            px = cv2.remap(fx, cx_, cy_, cv2.INTER_LINEAR)
            py = cv2.remap(fy, cx_, cy_, cv2.INTER_LINEAR)
            sx = sx + (TX - px) * self.sh(cx_, cy_).astype(np.float32)
            sy = sy + (TY - py) * self.sv(cx_, cy_).astype(np.float32)
        self.inv_full = (sx.astype(np.float32), sy.astype(np.float32))
        self.inv_half = ((cv2.resize(sx, (FW // 2, FH // 2)) / 2).astype(np.float32),
                         (cv2.resize(sy, (FW // 2, FH // 2)) / 2).astype(np.float32))

    def warp_full(self, img, nearest=False):
        if self.identity:
            return img
        return cv2.remap(img, *self.inv_full, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def warp_half(self, img, nearest=False):
        if self.identity:
            return img
        return cv2.remap(img, *self.inv_half, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def warp_mask_half(self, mask):
        return self.warp_half(mask.astype(np.uint8), nearest=True).astype(bool)
