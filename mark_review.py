from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import cv2
import math

@dataclass
class MarkReview:
    crossed_out: bool
    erasure_like: bool
    reason: str

def inspect_mark(gray, cx, cy, radius) -> MarkReview:
    h, w = gray.shape[:2]
    r = max(5, int(radius * 0.8))
    x0, x1 = max(0, cx-r), min(w, cx+r+1)
    y0, y1 = max(0, cy-r), min(h, cy+r+1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return MarkReview(False, False, "")

    # Erasures tend to be patchy gray rather than uniformly dark.
    mean = float(np.mean(roi))
    std = float(np.std(roi))
    erasure_like = (125 < mean < 220 and std > 35)

    edges = cv2.Canny(roi, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=7,
                            minLineLength=max(6, int(radius*0.8)),
                            maxLineGap=3)
    diag_count = 0
    if lines is not None:
        for ln in lines[:, 0, :]:
            x1l, y1l, x2l, y2l = ln
            angle = abs(math.degrees(math.atan2(y2l-y1l, x2l-x1l))) % 180
            if 20 <= angle <= 70 or 110 <= angle <= 160:
                diag_count += 1
    crossed = diag_count >= 2

    reason = ""
    if crossed:
        reason = "possible crossed-out mark"
    elif erasure_like:
        reason = "possible erased/corrected mark"
    return MarkReview(crossed, erasure_like, reason)
