"""Score the model's tray outline and lip line against hand-clicked edge points.

Input: the `edges` collection of the labelling page, saved as JSON files
(ArtifactData list with out_dir=...), plus labeller/trays.json for the model's
geometry in crop pixels.

For every clicked point we take its distance to the model's line on the same
side. Reported per side and overall, in camera pixels (full-res 1456 px wide),
and in mm when --tray-mm W H (real outer tray size) is given.

usage: python overhang/score_edges.py EDGES_DIR [--tray-mm 230 175]
"""
import argparse
import glob
import json
import os

import numpy as np

SIDES = ("left", "right", "top", "bottom")
LIP_I = dict(left=0, right=1, top=2, bottom=3)


def uv(rect, x, y):
    cx, cy, deg = rect[:3]
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c


def side_of(rect, u, v):
    W, H = rect[3], rect[4]
    return np.where(np.abs(u) / W > np.abs(v) / H, np.where(u < 0, "left", "right"), np.where(v < 0, "top", "bottom"))


def offsets(rect, pts, insets):
    """+ = model line lies outside the clicked point (model too big there)."""
    if not len(pts):
        return np.array([]), np.array([])
    p = np.asarray(pts, float)
    u, v = uv(rect, p[:, 0], p[:, 1])
    sd = side_of(rect, u, v)
    W, H = rect[3], rect[4]
    ins = np.array([insets[LIP_I[s]] for s in sd])
    d = np.where((sd == "left") | (sd == "right"), np.abs(u) - (W / 2 - ins), np.abs(v) - (H / 2 - ins))
    return -d, sd


def summarize(name, vals, sides, scale, unit):
    if not len(vals):
        print(f"{name}: no points")
        return
    v = vals * scale
    print(f"{name}: {len(v)} points  bias {np.median(v):+.1f} {unit}  |err| median {np.median(np.abs(v)):.1f}  p90 {np.percentile(np.abs(v), 90):.1f}  max {np.abs(v).max():.1f}")
    for s in SIDES:
        m = sides == s
        if m.any():
            print(f"   {s:6s} n={m.sum():3d}  bias {np.median(v[m]):+.1f}  |err| median {np.median(np.abs(v[m])):.1f}  p90 {np.percentile(np.abs(v[m]), 90):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges_dir")
    ap.add_argument("--trays", default="labeller/trays.json")
    ap.add_argument("--tray-mm", nargs=2, type=float, help="real outer tray size W H in mm")
    a = ap.parse_args()
    trays = {t["id"]: t for t in json.load(open(a.trays))["trays"]}
    docs = [json.load(open(p)) for p in glob.glob(os.path.join(a.edges_dir, "**", "*.json"), recursive=True)]
    docs = [d.get("data", d) for d in docs]
    print(f"{len(docs)} marked trays from {len({d['user'] for d in docs})} people")
    out_v, out_s, in_v, in_s, per_tray = [], [], [], [], []
    mm_per_cam = None
    for d in docs:
        t = trays[d["tray"]]
        o, so = offsets(t["rect"], d["outer"], [0, 0, 0, 0])
        if "inner" in t and len(d.get("inner", [])):
            # lip edge vs the rigid inner opening; side taken from the outer outline
            q = np.asarray(d["inner"], float); u0, v0 = uv(t["rect"], q[:, 0], q[:, 1]); si = side_of(t["rect"], u0, v0)
            ui, vi = uv(t["inner"], q[:, 0], q[:, 1])
            i = -np.where((si == "left") | (si == "right"), np.abs(ui) - t["inner"][3] / 2, np.abs(vi) - t["inner"][4] / 2)
        else:
            i, si = offsets(t["rect"], d.get("inner", []), t["lip"])
        out_v.append(o / t["f"]); out_s.append(so); in_v.append(i / t["f"]); in_s.append(si)
        per_tray.append((d["tray"], float(np.median(np.abs(o / t["f"]))) if len(o) else None))
        if a.tray_mm:   # camera px -> mm from the model's tray size in camera px
            mm_per_cam = a.tray_mm[0] / (t["rect"][3] / t["f"])
    out_v, out_s = np.concatenate(out_v), np.concatenate(out_s)
    in_v = np.concatenate(in_v) if in_v else np.array([])
    in_s = np.concatenate(in_s) if in_s else np.array([])
    unit, scale = ("mm", mm_per_cam) if mm_per_cam else ("camera px", 1.0)
    summarize("Outer edge", out_v, out_s, scale, unit)
    summarize("Lip inner edge", in_v, in_s, scale, unit)
    worst = sorted([p for p in per_tray if p[1] is not None], key=lambda p: -p[1])[:8]
    print("worst trays (median |outer err|, camera px):", ", ".join(f"{k} {v:.1f}" for k, v in worst))


if __name__ == "__main__":
    main()
