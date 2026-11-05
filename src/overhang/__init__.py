"""Tray overhang detection for the FPL grind line.

    from overhang import Config, OverhangPipeline
    cfg = Config.load("configs/fpl-grind.yaml")
    for tray in OverhangPipeline(cfg).run("rtsp://camera/stream"):
        print(tray["overhang_class"], tray["light"], tray["lip_cover_pct"])
"""
from .classify import CLASSES, light_for, overhang_class
from .config import Config

__version__ = "0.1.0"
__all__ = ["CLASSES", "Config", "OverhangPipeline", "light_for", "overhang_class", "__version__"]


def __getattr__(name):            # keep `import overhang` light (no torch) until the pipeline is needed
    if name == "OverhangPipeline":
        from .pipeline import OverhangPipeline
        return OverhangPipeline
    raise AttributeError(name)
