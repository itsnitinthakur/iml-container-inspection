"""OpenCV defect inspection for IML (in-mould label) containers.

Each corner camera sees one side of the container: the cream lid on top and the
cyan IML label below it. For every image we:

1. find the label body (saturated cyan) and the lid (cream) by colour,
2. run fixed rules  - no container, lid missing, spots / foreign colour on lid or label,
                      torn / missing label area,
3. run the learned "golden sample" model - every measurement is compared with the
   mean / std learned from good containers (teach mode). Anything more than
   ``z_max`` standard deviations away is a defect (shifted label, deformed
   container, wrong lid, wrong product ...).

Model = one JSON file per product (small / medium / large), with statistics per camera.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

DEFAULTS = {
    "roi": None,                    # [x, y, w, h] in pixels, None = whole image
    "work_width": 1000,             # images are resized to this width before inspection
    "label_hsv_low": [86, 150, 60],  # cyan IML label
    "label_hsv_high": [108, 255, 255],
    "lid_hsv_low": [5, 35, 120],    # cream / off-white lid
    "lid_hsv_high": [35, 150, 255],
    "white_s_max": 55,              # white print on label: low saturation ...
    "white_v_min": 90,              # ... and not dark
    "check_bare_area": False,       # needs white LED lighting; warm light makes print look cream
    "bare_s_min": 65,               # cream plastic showing through label is more saturated than print
    "dark_v_max": 60,               # black specks / burn marks
    "foreign_s_min": 60,            # coloured contamination (pink, red, brown ...)
    "min_container_area": 0.03,     # label area as fraction of ROI area
    "min_spot_area_px": 25,         # smallest spot counted (at work_width scale)
    "edge_margin_px": 6,            # ignore this many px at mask edges (reflections/shadows)
    "lid_band_ratio": 0.30,         # lid search band height as fraction of label height
    "min_lid_fill": 0.35,           # lid pixels / lid band (rule)
    "min_label_solidity": 0.88,     # area / convex hull (rule) - torn / folded label
    "z_max": 4.0,                   # model: allowed deviation from taught good samples
    "min_std_frac": 0.03,           # model: std floor as fraction of mean (avoids over-tight models)
}

MODEL_FEATURES = ["label_area", "label_aspect", "label_solidity", "label_fill",
                  "lid_fill", "lid_width_ratio", "label_cx", "label_cy", "top_edge_angle"]


@dataclass
class Defect:
    code: str
    detail: str
    bbox: Optional[tuple[int, int, int, int]] = None  # in original image pixels


@dataclass
class Result:
    camera: str
    ok: bool
    defects: list[Defect] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    ms: float = 0.0
    annotated: Optional[np.ndarray] = None

    @property
    def codes(self) -> list[str]:
        return [d.code for d in self.defects]


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _blobs(mask: np.ndarray, min_area: int):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [tuple(int(v) for v in stats[i, :4]) + (int(stats[i, 4]),)
            for i in range(1, n) if stats[i, 4] >= min_area]


class Inspector:
    def __init__(self, params: Optional[dict] = None, model: Optional[dict] = None,
                 camera_params: Optional[dict] = None):
        self.p = {**DEFAULTS, **(params or {})}
        self.camera_params = camera_params or {}   # {camera_name: {param: value}} e.g. per-camera ROI
        self.model = model or {}   # {camera_name: {feature: [mean, std]}}

    def params_for(self, camera: str) -> dict:
        return {**self.p, **self.camera_params.get(camera, {})}

    # ------------------------------------------------------------------ model I/O
    @staticmethod
    def load_model(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model, f, indent=2)

    def teach(self, camera: str, images: list[np.ndarray]) -> dict:
        """Learn mean/std of every feature from GOOD containers for one camera."""
        rows = []
        for img in images:
            feats, defects = self._measure(img, self.params_for(camera))[:2]
            if any(d.code == "no_container" for d in defects):
                continue
            rows.append([feats[k] for k in MODEL_FEATURES])
        if len(rows) < 3:
            raise ValueError(f"{camera}: need at least 3 good images with a container, got {len(rows)}")
        arr = np.array(rows, dtype=float)
        mean, std = arr.mean(0), arr.std(0)
        stats = {}
        for i, k in enumerate(MODEL_FEATURES):
            floor = self.params_for(camera)["min_std_frac"] * max(abs(mean[i]), 1e-3)
            if k == "top_edge_angle":
                floor = max(floor, 1.0)      # degrees
            stats[k] = [float(mean[i]), float(max(std[i], floor))]
        stats["_samples"] = len(rows)
        self.model[camera] = stats
        return stats

    # ------------------------------------------------------------------ inspection
    def inspect(self, camera: str, img: Optional[np.ndarray], annotate: bool = True) -> Result:
        t0 = cv2.getTickCount()
        if img is None:
            return Result(camera, False, [Defect("no_image", "camera returned no frame")])
        p = self.params_for(camera)
        feats, defects, dbg = self._measure(img, p)

        stats = self.model.get(camera)
        if stats and not any(d.code == "no_container" for d in defects):
            for k in MODEL_FEATURES:
                if k not in stats:
                    continue
                mean, std = stats[k]
                z = abs(feats[k] - mean) / std
                feats[f"z_{k}"] = round(z, 2)
                if z > p["z_max"]:
                    defects.append(Defect(f"abnormal_{k}",
                                          f"{k}={feats[k]:.3f} expected {mean:.3f}±{std:.3f} (z={z:.1f})"))

        res = Result(camera, not defects, defects, feats)
        res.ms = (cv2.getTickCount() - t0) * 1000 / cv2.getTickFrequency()
        if annotate:
            res.annotated = self._draw(img, res, dbg)
        return res

    # ------------------------------------------------------------------ measurements
    @staticmethod
    def _measure(img: np.ndarray, p: dict):
        ox, oy = 0, 0
        if p["roi"]:
            x, y, w, h = p["roi"]
            img = img[y:y + h, x:x + w]
            ox, oy = x, y
        scale = p["work_width"] / img.shape[1] if img.shape[1] > p["work_width"] else 1.0
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
        H, W = small.shape[:2]
        k = max(3, int(round(W / 200)) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        def to_orig(b):
            bx, by, bw, bh = b[:4]
            return (int(bx / scale) + ox, int(by / scale) + oy, int(bw / scale), int(bh / scale))

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)
        cyan = cv2.inRange(hsv, np.array(p["label_hsv_low"]), np.array(p["label_hsv_high"]))
        cream = cv2.inRange(hsv, np.array(p["lid_hsv_low"]), np.array(p["lid_hsv_high"]))
        white = ((sch <= p["white_s_max"]) & (vch >= p["white_v_min"])).astype(np.uint8) * 255
        dark = (vch <= p["dark_v_max"]).astype(np.uint8) * 255
        # coloured pixels that are neither label cyan nor lid cream -> pink / red / brown / green contamination
        foreign = ((sch >= p["foreign_s_min"]) & (cyan == 0) & (cream == 0)).astype(np.uint8) * 255

        feats = {k: 0.0 for k in MODEL_FEATURES}
        defects: list[Defect] = []
        dbg: dict = {"scale": scale, "off": (ox, oy)}

        # ---- label body: close the white print holes, keep largest blob
        body_mask = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, kern, iterations=4)
        body = _largest_contour(body_mask)
        if body is None or cv2.contourArea(body) < p["min_container_area"] * H * W:
            defects.append(Defect("no_container", "label / container not found in view"))
            return feats, defects, dbg
        body_filled = np.zeros((H, W), np.uint8)
        cv2.drawContours(body_filled, [body], -1, 255, cv2.FILLED)
        area = cv2.contourArea(body)
        hull_area = cv2.contourArea(cv2.convexHull(body)) or 1.0
        bx, by, bw, bh = cv2.boundingRect(body)
        m = cv2.moments(body)
        dbg["body"] = body

        feats["label_area"] = area / (H * W)
        feats["label_aspect"] = bw / max(bh, 1)
        feats["label_solidity"] = area / hull_area
        feats["label_cx"] = (m["m10"] / m["m00"]) / W
        feats["label_cy"] = (m["m01"] / m["m00"]) / H
        feats["label_fill"] = float(cv2.countNonZero(cv2.bitwise_and(cyan | white, body_filled))) / area

        # top edge angle of the label (label shifted / skewed in the mould)
        cols = np.arange(bx + bw // 10, bx + bw - bw // 10)
        tops = [(c, int(np.argmax(body_filled[:, c] > 0))) for c in cols if body_filled[:, c].any()]
        if len(tops) > 5:
            cs, rs = np.array(tops).T
            slope = np.polyfit(cs, rs, 1)[0]
            feats["top_edge_angle"] = math.degrees(math.atan(slope))

        if feats["label_solidity"] < p["min_label_solidity"]:
            defects.append(Defect("label_damaged", f"label solidity {feats['label_solidity']:.2f} "
                                  f"< {p['min_label_solidity']} (torn / folded / missing piece)",
                                  to_orig((bx, by, bw, bh))))

        # ---- lid: cream pixels in a band above the label
        band_h = int(bh * p["lid_band_ratio"])
        y0 = max(0, by - band_h)
        x0, x1 = max(0, bx - bw // 8), min(W, bx + bw + bw // 8)
        band = np.zeros((H, W), np.uint8)
        band[y0:by + max(2, bh // 20), x0:x1] = 255
        lid_raw = cv2.bitwise_and(cream, band)
        lid_raw = cv2.morphologyEx(lid_raw, cv2.MORPH_OPEN, kern, iterations=1)
        lid_raw = cv2.morphologyEx(lid_raw, cv2.MORPH_CLOSE, kern, iterations=2)
        lid = _largest_contour(lid_raw)
        band_area = max(1, (by - y0) * (x1 - x0))
        lid_filled = np.zeros((H, W), np.uint8)
        if lid is not None:
            cv2.drawContours(lid_filled, [lid], -1, 255, cv2.FILLED)
            lx, ly, lw, lh = cv2.boundingRect(lid)
            feats["lid_fill"] = cv2.contourArea(lid) / band_area
            feats["lid_width_ratio"] = lw / max(bw, 1)
            dbg["lid"] = lid
        if feats["lid_fill"] < p["min_lid_fill"]:
            defects.append(Defect("lid_missing", f"lid fill {feats['lid_fill']:.2f} < {p['min_lid_fill']}",
                                  to_orig((x0, y0, x1 - x0, max(1, by - y0)))))

        # ---- spots / contamination inside label and lid (eroded so edges/shadows don't count)
        er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p["edge_margin_px"] + 1,) * 2)
        inner_label = cv2.erode(body_filled, er)
        inner_lid = cv2.erode(lid_filled, er)
        min_a = p["min_spot_area_px"]
        checks = [
            ("label_spot_dark", dark, inner_label),
            ("label_foreign_colour", foreign, inner_label),
            # bare plastic (lid-coloured cream) showing through a torn / missing label piece
            ("label_bare_area", cv2.bitwise_and(cream, (sch >= p["bare_s_min"]).astype(np.uint8) * 255), inner_label),
            ("lid_spot_dark", dark, inner_lid),
            ("lid_foreign_colour", foreign, inner_lid),
        ]
        if not p["check_bare_area"]:
            checks = [c for c in checks if c[0] != "label_bare_area"]
        spots = []
        for code, mask, region in checks:
            mm = cv2.morphologyEx(cv2.bitwise_and(mask, region), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            for b in _blobs(mm, min_a):
                defects.append(Defect(code, f"{b[4]} px spot", to_orig(b)))
                spots.append(b)
        feats["spot_count"] = len(spots)
        dbg["spots"] = spots
        return feats, defects, dbg

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _draw(img: np.ndarray, res: Result, dbg: dict) -> np.ndarray:
        out = img.copy()
        s, (ox, oy) = dbg.get("scale", 1.0), dbg.get("off", (0, 0))
        lw = max(2, img.shape[1] // 400)

        def up(c):
            return (c.astype(np.float32) / s + np.array([ox, oy], np.float32)).astype(np.int32)

        if "body" in dbg:
            cv2.drawContours(out, [up(dbg["body"])], -1, (255, 200, 0), lw)
        if "lid" in dbg:
            cv2.drawContours(out, [up(dbg["lid"])], -1, (0, 220, 255), lw)
        for d in res.defects:
            if d.bbox:
                x, y, w, h = d.bbox
                cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), lw)
        colour = (0, 200, 0) if res.ok else (0, 0, 255)
        fs = img.shape[1] / 900
        cv2.putText(out, f"{res.camera}: {'OK' if res.ok else 'REJECT'}", (15, int(45 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, colour, max(2, lw))
        for i, code in enumerate(sorted(set(res.codes))[:6]):
            cv2.putText(out, code, (15, int((90 + 38 * i) * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * fs, (0, 0, 255), max(1, lw - 1))
        return out
