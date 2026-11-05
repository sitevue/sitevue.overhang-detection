"""Line / camera configuration, loaded from a YAML file (see configs/fpl-grind.yaml)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class Config:
    frame_w: int = 1456
    frame_h: int = 1088
    fps: float = 10.0
    scale_map: dict | None = None

    tray_w: float = 389.97            # outer tray size, half-res corrected px
    tray_h: float = 288.14
    inner_w: float = 724.01           # inner opening, full-res corrected px
    inner_h: float = 525.71
    inner_corner_r: float = 45.0
    inner_shift_u: float = 4.0
    lip_default: dict = field(default_factory=lambda: dict(left=10.4, right=18.9, top=15.3, bottom=12.9))

    track_step: int = 2
    track_min_area: int = 12000
    track_match_px: float = 90.0
    track_edge_margin: int = 12
    track_max_centre_dist: float = 140.0
    track_emit_after_px: float = 60.0

    sam_model: str = "facebook/sam2.1-hiera-small"
    sam_dtype: str = "float32"
    sam_device: str = "auto"

    mild_lip_cover_pct: float = 10.5
    small_lip_cover_pct: float = 18.5
    large_past_green_px: int = 10000
    light: dict = field(default_factory=lambda: dict(none="green", mild="yellow", small="yellow", large="red"))

    workers: int = 3

    @property
    def work_w(self) -> int:
        """Width of the half-resolution working frames."""
        return self.frame_w // 2

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            y = yaml.safe_load(fh) or {}
        base = os.path.dirname(os.path.abspath(path))
        cam, tray, trk = y.get("camera", {}), y.get("tray", {}), y.get("tracking", {})
        sam, cls_, rt = y.get("sam", {}), y.get("classes", {}), y.get("runtime", {})
        kw = {}
        if "frame_size" in cam:
            kw["frame_w"], kw["frame_h"] = (int(v) for v in cam["frame_size"])
        if "fps" in cam:
            kw["fps"] = float(cam["fps"])
        if cam.get("scale_map"):
            with open(os.path.join(base, cam["scale_map"])) as fh:
                kw["scale_map"] = json.load(fh)
        if "outer_size_half_px" in tray:
            kw["tray_w"], kw["tray_h"] = (float(v) for v in tray["outer_size_half_px"])
        if "inner_size_px" in tray:
            kw["inner_w"], kw["inner_h"] = (float(v) for v in tray["inner_size_px"])
        for src, dst in (("inner_corner_radius_px", "inner_corner_r"), ("inner_shift_px", "inner_shift_u")):
            if src in tray:
                kw[dst] = float(tray[src])
        if "lip_default_half_px" in tray:
            kw["lip_default"] = {k: float(v) for k, v in tray["lip_default_half_px"].items()}
        for k in ("step", "min_area", "match_px", "edge_margin", "max_centre_dist", "emit_after_px"):
            if k in trk:
                kw["track_" + k] = trk[k]
        for k in ("model", "dtype", "device"):
            if k in sam:
                kw["sam_" + k] = sam[k]
        for k in ("mild_lip_cover_pct", "small_lip_cover_pct", "large_past_green_px"):
            if k in cls_:
                kw[k] = cls_[k]
        if "light" in y:
            kw["light"] = dict(y["light"])
        if "workers" in rt:
            kw["workers"] = int(rt["workers"])
        return cls(**kw)
