"""Production loop: trigger -> 4 images -> inspect in parallel -> reject -> count / log."""
from __future__ import annotations

import csv
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from .cameras import CameraGroup
from .detector import Inspector, Result
from .reject import RejectOutput

log = logging.getLogger(__name__)


class Counters:
    def __init__(self):
        self.total = self.good = self.rejected = 0
        self.by_defect: Counter = Counter()
        self.started = time.time()

    def add(self, ok: bool, codes: list[str]) -> None:
        self.total += 1
        if ok:
            self.good += 1
        else:
            self.rejected += 1
            self.by_defect.update(set(codes))

    def summary(self) -> str:
        rate = 100 * self.rejected / self.total if self.total else 0
        top = ", ".join(f"{k}:{v}" for k, v in self.by_defect.most_common(5))
        return f"total {self.total} | good {self.good} | reject {self.rejected} ({rate:.1f}%) | {top}"


def mosaic(images: list[np.ndarray], width: int = 1280) -> np.ndarray:
    """2x2 view of the 4 camera images."""
    cell_w = width // 2
    cells = []
    for img in images[:4]:
        h = int(img.shape[0] * cell_w / img.shape[1])
        cells.append(cv2.resize(img, (cell_w, h)))
    while len(cells) < 4:
        cells.append(np.zeros_like(cells[0]))
    ch = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


class Runner:
    def __init__(self, cfg: dict, cameras: CameraGroup, inspector: Inspector, reject: RejectOutput):
        self.cfg, self.cams, self.ins, self.reject = cfg, cameras, inspector, reject
        out = cfg.get("output", {})
        self.save_dir = out.get("save_dir", "results")
        self.save_good = out.get("save_good_images", False)
        self.show = out.get("show_window", True)
        self.counters = Counters()
        self.pool = ThreadPoolExecutor(max_workers=len(cameras.cameras))
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, f"log_{datetime.now():%Y%m%d}.csv")
        new = not os.path.exists(self.csv_path)
        self.csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.csv = csv.writer(self.csv_f)
        if new:
            self.csv.writerow(["time", "container_no", "result", "defects", "cameras_failed", "inspect_ms"])

    # ---------------------------------------------------------------- one container
    def inspect_cycle(self, frames: dict) -> tuple[bool, list[Result]]:
        t0 = time.perf_counter()
        futs = [self.pool.submit(self.ins.inspect, name, img) for name, img in frames.items()]
        results = [f.result() for f in futs]
        ok = all(r.ok for r in results)
        if not ok:
            self.reject.reject()
        ms = (time.perf_counter() - t0) * 1000
        codes = [c for r in results for c in r.codes]
        self.counters.add(ok, codes)
        n = self.counters.total
        self.csv.writerow([datetime.now().isoformat(timespec="milliseconds"), n, "OK" if ok else "REJECT",
                           ";".join(sorted(set(codes))), ";".join(r.camera for r in results if not r.ok),
                           f"{ms:.0f}"])
        self.csv_f.flush()
        if not ok or self.save_good:
            self._save(n, ok, results)
        log.info("#%d %s in %.0f ms %s | %s", n, "OK" if ok else "REJECT", ms,
                 sorted(set(codes)) or "", self.counters.summary())
        return ok, results

    def _save(self, n: int, ok: bool, results: list[Result]) -> None:
        day = os.path.join(self.save_dir, datetime.now().strftime("%Y%m%d"), "ok" if ok else "reject")
        os.makedirs(day, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        for r in results:
            if r.annotated is not None:
                cv2.imwrite(os.path.join(day, f"{stamp}_{n:06d}_{r.camera}.jpg"), r.annotated,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _display(self, results: list[Result], ok: bool) -> bool:
        imgs = [r.annotated if r.annotated is not None else np.zeros((480, 640, 3), np.uint8) for r in results]
        view = mosaic(imgs)
        bar = np.full((60, view.shape[1], 3), (0, 140, 0) if ok else (0, 0, 180), np.uint8)
        cv2.putText(bar, self.counters.summary(), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("IML inspection  (q = quit)", np.vstack([bar, view]))
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    # ---------------------------------------------------------------- main loop
    def run(self, max_cycles: int = 0) -> None:
        cam = self.cfg["cameras"]
        trigger = cam.get("trigger", "hardware")
        timeout = int(cam.get("grab_timeout_ms", 12000))
        interval = float(cam.get("software_trigger_interval_s", 5.0))
        log.info("running: trigger=%s, %d cameras, CSV %s", trigger, len(self.cams.cameras), self.csv_path)
        try:
            while True:
                t_cycle = time.time()
                if trigger == "software":
                    self.cams.clear()
                    self.cams.trigger_all()
                frames = self.cams.grab_all(timeout)
                if trigger == "hardware" and all(f is None for f in frames.values()):
                    continue  # no container arrived within the timeout: keep waiting
                ok, results = self.inspect_cycle(frames)
                if self.show and not self._display(results, ok):
                    break
                if max_cycles and self.counters.total >= max_cycles:
                    break
                if trigger in ("software", "continuous"):
                    time.sleep(max(0.0, interval - (time.time() - t_cycle)))
        except KeyboardInterrupt:
            pass
        finally:
            log.info("FINAL: %s", self.counters.summary())
            self.csv_f.close()
            self.pool.shutdown(wait=False)
            if self.show:
                cv2.destroyAllWindows()
