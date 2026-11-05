"""Tray mask with SAM 2 (the only learned model in the pipeline, and the only GPU step).

Point prompts are made automatically: positive points on the visible, edge-free tray rim just
outside this tray's meat, negative points on this tray's meat and on neighbouring meat.
"""
from __future__ import annotations

import cv2
import numpy as np


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
            i = np.flatnonzero(sel)[len(np.flatnonzero(sel)) // 2]
            pts.append([int(xs[i]), int(ys[i])])
    return pts


class SamTray:
    def __init__(self, model_name="facebook/sam2.1-hiera-small", device="auto", dtype="float32"):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.proc = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name).to(device).to(self.dtype).eval()

    def segment(self, half_bgr, blobs, lab, sm, j):
        """Tray mask for blob j (half-res). Returns (mask, score), or (None, 0.0) without rim points."""
        from PIL import Image

        torch = self.torch
        b = blobs[j]
        own = lab == b["id"]
        pos = rim_points(own, sm)
        if not pos:
            return None, 0.0
        neg = [[int(b["c"][0]), int(b["c"][1])]] + [[int(o["c"][0]), int(o["c"][1])] for o in blobs if o is not b]
        pts = pos + neg
        lbl = [1] * len(pos) + [0] * len(neg)
        img = Image.fromarray(cv2.cvtColor(half_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inp = self.proc(images=img, input_points=[[pts]], input_labels=[[lbl]], return_tensors="pt").to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
            o = self.model(**inp, multimask_output=True)
            m = self.proc.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"])[0][0]   # (3, H, W)
            sc = o.iou_scores.float().cpu()[0, 0].numpy()
        m = m.numpy().astype(bool)
        # prefer a high IoU score, and masks that are not mostly meat
        meat_frac = np.array([(mi & own).sum() / max(mi.sum(), 1) for mi in m])
        k = int(np.argmax(sc - 0.5 * meat_frac))
        return m[k], float(sc[k])
