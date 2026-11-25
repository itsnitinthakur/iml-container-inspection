"""Synthetic side-view images of an IML container on the belt, with optional defects."""
from __future__ import annotations

import cv2
import numpy as np

BELT = (96, 45, 160)      # HSV measured from the factory photos
LABEL = (95, 240, 200)
LID = (15, 90, 240)
PRINT = (10, 25, 225)
WALL = (20, 10, 200)


def _bgr(hsv):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def make(defect: str | None = None, seed: int = 0, w: int = 1200, h: int = 900) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dx, dy = rng.integers(-8, 9, 2)
    img = np.full((h, w, 3), _bgr(BELT), np.uint8)
    img[: h // 3] = _bgr(WALL)
    cx, top, bot = w // 2 + dx, 330 + dy, 690 + dy
    half_top, half_bot = 210, 170
    label = np.array([[cx - half_top, top], [cx + half_top, top], [cx + half_bot, bot], [cx - half_bot, bot]])
    if defect == "label_shift":                       # label skewed in the mould
        label = label + np.array([[0, 0], [0, 55], [0, 0], [0, 0]])
    cv2.fillPoly(img, [label], _bgr(LABEL))
    for i in range(3):                                # diagonal white print
        y = top + 80 + i * 100
        cv2.line(img, (cx - 150, y + 60), (cx + 120, y - 40), _bgr(PRINT), 22)
    if defect != "no_lid":
        cv2.rectangle(img, (cx - 235, top - 75), (cx + 235, top + 5), _bgr(LID), cv2.FILLED)
    if defect == "pink_streak":
        cv2.line(img, (cx + 40, top + 150), (cx + 120, top + 200), _bgr((170, 120, 220)), 9)
    if defect == "black_speck_lid":
        cv2.circle(img, (cx - 60, top - 35), 7, (20, 20, 20), cv2.FILLED)
    if defect == "black_speck_label":
        cv2.circle(img, (cx + 70, top + 260), 6, (15, 15, 15), cv2.FILLED)
    if defect == "torn_label":                        # piece of label missing at the side
        cv2.fillPoly(img, [np.array([[cx + half_top - 5, top + 40], [cx + 80, top + 150],
                                     [cx + half_top - 20, top + 260]])], _bgr(BELT))
    if defect == "no_container":
        img[:] = _bgr(BELT)
    noise = rng.normal(0, 3, img.shape)
    gain = rng.uniform(0.95, 1.05)
    return np.clip(img.astype(float) * gain + noise, 0, 255).astype(np.uint8)
