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

    track_step: int = 1
    track_min_area: int = 12000
    track_max_jump_px: float = 220.0
    track_max_dy_px: float = 40.0
    track_edge_margin: int = 12
    track_max_centre_dist: float = 140.0
    track_emit_after_px: float = 60.0
    pick_zone_px: tuple | None = None   # R1, full-res x range, e.g. (168, 1288); None = off (whole frame)
    pick_x_px: float | None = None      # measure the frame where the tray is nearest this x (full-res); None = frame centre
    side_views_px: tuple | None = None  # (x for the left half, x for the right half): two frames per tray; None = off
    dynamic_split_px: float | None = None  # rightmost in-view frame; one photo if the tray is at x <= this there, else two
    edge_tolerance_px: float = 0.0      # a tray box corner may be this far past the frame edge before edge_of_view
    drop_bare_hands: bool = True        # a "tray" whose meat is really a bare hand (smooth, skin-coloured) is not reported
    retry_frames: int = 2               # a tray unreadable only for outline / edge of view is re-measured on up to this many other frames

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
        for k in ("step", "min_area", "max_jump_px", "max_dy_px", "edge_margin", "max_centre_dist", "emit_after_px"):
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
        if "pick_zone_px" in trk:
            kw["pick_zone_px"] = tuple(trk["pick_zone_px"]) if trk["pick_zone_px"] else None
        if trk.get("pick_x_px") is not None:
            kw["pick_x_px"] = float(trk["pick_x_px"])
        if trk.get("side_views_px"):
            kw["side_views_px"] = tuple(float(v) for v in trk["side_views_px"])
        if trk.get("dynamic_split_px") is not None:
            kw["dynamic_split_px"] = float(trk["dynamic_split_px"])
        if trk.get("edge_tolerance_px") is not None:
            kw["edge_tolerance_px"] = float(trk["edge_tolerance_px"])
        if "drop_bare_hands" in trk:
            kw["drop_bare_hands"] = bool(trk["drop_bare_hands"])
        if "retry_frames" in trk:
            kw["retry_frames"] = int(trk["retry_frames"])
        if "workers" in rt:
            kw["workers"] = int(rt["workers"])
        return cls(**kw)
