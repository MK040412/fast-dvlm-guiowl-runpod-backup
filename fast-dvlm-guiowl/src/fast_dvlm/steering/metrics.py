
from __future__ import annotations
import math


def coord_l2(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)
