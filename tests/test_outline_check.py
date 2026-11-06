"""Outline check: only belt inside the box or a tilt over 30 deg make the outline doubtful."""
import numpy as np
import pytest

from overhang.outline import BELT_INSIDE_MAX, outline_ok


@pytest.mark.parametrize("belt,deg,ok", [
    (0.0, 0.0, True),
    (0.0, -12.0, True),
    (BELT_INSIDE_MAX, 29.9, True),
    (0.02, 0.0, False),          # box straddles two trays / sits on belt
    (0.0, 35.0, False),          # box turned sideways
    (0.0, -41.0, False),
])
def test_outline_ok(belt, deg, ok):
    assert outline_ok(belt, np.deg2rad(deg)) is ok
