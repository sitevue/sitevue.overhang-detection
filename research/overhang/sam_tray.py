"""Segment trays with SAM 2 using automatic point prompts.

Meat: colour threshold (geom.masks).
Tray: SAM 2, positive points on the visible edge-free rim around each meat blob,
negative points on the meat itself and on neighbouring meat.
"""
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

sys.path.insert(0, "overhang")
from geom import masks, meat_blobs, smooth_mask  # noqa: E402

MODEL = "facebook/sam2.1-hiera-small"
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load():
    proc = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to(dev).eval()
    return proc, model


def rim_points(own, sm, k=4):
    """Pick k rim pixels just outside the meat blob, spread around it."""
    ring = cv2.dilate(own.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    near = cv2.dilate(own.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    cand = sm & ring & ~near
    ys, xs = np.nonzero(cand)
    if len(xs) == 0:
        return []
    cy, cx = np.nonzero(own)
    c = np.array([cx.mean(), cy.mean()])
    ang = np.arctan2(ys - c[1], xs - c[0])
    pts = []
    for a in np.linspace(-np.pi, np.pi, k, endpoint=False) + np.pi / k:
        sel = np.abs(np.angle(np.exp(1j * (ang - a)))) < np.pi / k
        if sel.sum() > 30:
            # most "interior" rim pixel of that sector (furthest from belt): take the median one
            i = np.flatnonzero(sel)[len(np.flatnonzero(sel)) // 2]
            pts.append([int(xs[i]), int(ys[i])])
    return pts


@torch.no_grad()
def segment(proc, model, bgr, blobs, lab, sm, only=None):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = []
    for b in (blobs if only is None else [blobs[only]]):
        own = lab == b["id"]
        pos = rim_points(own, sm)
        if not pos:
            out.append(None)
            continue
        neg = [[int(b["c"][0]), int(b["c"][1])]] + \
              [[int(o["c"][0]), int(o["c"][1])] for o in blobs if o is not b]
        pts = pos + neg
        lbl = [1] * len(pos) + [0] * len(neg)
        inp = proc(images=img, input_points=[[pts]], input_labels=[[lbl]],
                   return_tensors="pt").to(dev)
        o = model(**inp, multimask_output=True)
        m = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0]  # (3,H,W)
        sc = o.iou_scores.cpu()[0, 0].numpy()
        m = m.numpy().astype(bool)
        # prefer high IoU score, and masks that are not mostly meat
        meat_frac = np.array([(mi & own).sum() / max(mi.sum(), 1) for mi in m])
        k = int(np.argmax(sc - 0.5 * meat_frac))
        out.append(dict(mask=m[k], score=float(sc[k]), pos=pos, neg=neg))
    return out


if __name__ == "__main__":
    t = time.time()
    proc, model = load()
    print("loaded", f"{time.time()-t:.1f}s", dev)
    tiles = []
    for p in sys.argv[1:]:
        f = cv2.imread(p)
        red, yel = masks(f)
        sm = smooth_mask(f, red)
        lab, blobs = meat_blobs(red, 4000)
        t = time.time()
        res = segment(proc, model, f, blobs, lab, sm)
        print(p, len(blobs), "blobs", f"{time.time()-t:.1f}s")
        vis = f.copy()
        cols = [(255, 120, 0), (0, 200, 255), (255, 0, 255), (0, 255, 120)]
        for i, r in enumerate(res):
            if r is None:
                continue
            col = np.array(cols[i % 4])
            vis[r["mask"]] = (vis[r["mask"]] * 0.45 + col * 0.55).astype(np.uint8)
            for x, y in r["pos"]:
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            for x, y in r["neg"]:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        tiles.append(vis)
    cv2.imwrite("explore/sam.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]))
