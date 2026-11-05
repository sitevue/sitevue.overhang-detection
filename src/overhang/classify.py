"""Customer classes and floor light, from the measured numbers. Cut-offs live in the config."""

CLASSES = ("none", "mild", "small", "large")


def overhang_class(lip_cover_pct: float, past_green_px: int, cfg) -> str:
    if past_green_px >= cfg.large_past_green_px:
        return "large"
    if lip_cover_pct >= cfg.small_lip_cover_pct:
        return "small"
    if lip_cover_pct >= cfg.mild_lip_cover_pct:
        return "mild"
    return "none"


def light_for(cls: str, cfg) -> str:
    return cfg.light.get(cls, "green")
