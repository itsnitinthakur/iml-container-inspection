"""Camera back-ends.

* MindVisionCamera - industrial MindVision (MV-SUA / MV-GE ...) cameras through the
  official ``mvsdk.py`` wrapper that ships with the MindVision SDK.
* OpenCVCamera     - any USB / RTSP camera OpenCV can open (handy for bench tests).
* FileCamera       - replays images from a folder (offline testing / teaching).

All cameras return BGR ``numpy`` images (H, W, 3).
"""
from __future__ import annotations

import glob
import logging
import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

TRIGGER_CONTINUOUS, TRIGGER_SOFTWARE, TRIGGER_HARDWARE = 0, 1, 2


class CameraError(RuntimeError):
    pass


class BaseCamera:
    name: str = "camera"

    def open(self) -> None: ...
    def close(self) -> None: ...

    def software_trigger(self) -> None:
        """Fire one exposure (only meaningful in software-trigger mode)."""

    def clear_buffer(self) -> None:
        """Drop stale frames so the next grab belongs to the next container."""

    def grab(self, timeout_ms: int) -> Optional[np.ndarray]:
        """Return the next frame or ``None`` on timeout."""
        raise NotImplementedError

    def set_output(self, index: int, state: bool) -> None:
        raise CameraError(f"{self.name}: digital output not supported")


# --------------------------------------------------------------------------- MindVision
def _import_mvsdk():
    try:
        import mvsdk  # noqa: F401  (copied from the MindVision SDK "Demo/python" folder)
        return mvsdk
    except ImportError as exc:  # pragma: no cover - depends on the factory PC
        raise CameraError(
            "mvsdk.py not found. Install the MindVision SDK and copy mvsdk.py from "
            "<SDK>/Demo/python next to main.py (or add it to PYTHONPATH)."
        ) from exc


class MindVisionCamera(BaseCamera):
    def __init__(self, name: str, serial: str = "", index: int = 0, trigger: str = "hardware",
                 exposure_us: Optional[float] = None, gain: Optional[int] = None):
        self.name, self.serial, self.index = name, str(serial or ""), index
        self.trigger = {"continuous": TRIGGER_CONTINUOUS, "software": TRIGGER_SOFTWARE,
                        "hardware": TRIGGER_HARDWARE}[trigger]
        self.exposure_us, self.gain = exposure_us, gain
        self.h = None
        self.buf = None
        self.mono = False
        self._lock = threading.Lock()

    def open(self) -> None:
        mv = _import_mvsdk()
        devices = mv.CameraEnumerateDevice()
        if not devices:
            raise CameraError("No MindVision camera found")
        dev = None
        if self.serial:
            dev = next((d for d in devices if d.GetSn() == self.serial), None)
            if dev is None:
                raise CameraError(f"{self.name}: camera with serial {self.serial} not found; "
                                  f"available: {[d.GetSn() for d in devices]}")
        else:
            if self.index >= len(devices):
                raise CameraError(f"{self.name}: index {self.index} but only {len(devices)} cameras")
            dev = devices[self.index]
        try:
            self.h = mv.CameraInit(dev, -1, -1)
        except mv.CameraException as e:
            raise CameraError(f"{self.name}: CameraInit failed ({e.error_code}): {e.message}") from e

        cap = mv.CameraGetCapability(self.h)
        self.mono = cap.sIspCapacity.bMonoSensor != 0
        mv.CameraSetIspOutFormat(self.h, mv.CAMERA_MEDIA_TYPE_MONO8 if self.mono else mv.CAMERA_MEDIA_TYPE_BGR8)
        mv.CameraSetTriggerMode(self.h, self.trigger)
        if self.exposure_us:
            mv.CameraSetAeState(self.h, 0)
            mv.CameraSetExposureTime(self.h, float(self.exposure_us))
        if self.gain is not None:
            mv.CameraSetAnalogGain(self.h, int(self.gain))
        size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if self.mono else 3)
        self.buf = mv.CameraAlignMalloc(size, 16)
        mv.CameraPlay(self.h)
        log.info("%s: opened %s (SN %s, %s)", self.name, dev.GetFriendlyName(), dev.GetSn(),
                 "mono" if self.mono else "color")
        if self.mono:
            log.warning("%s: mono sensor - colour checks (pink/foreign colour) will not work", self.name)

    def close(self) -> None:
        if self.h is None:
            return
        mv = _import_mvsdk()
        mv.CameraUnInit(self.h)
        mv.CameraAlignFree(self.buf)
        self.h = self.buf = None

    def software_trigger(self) -> None:
        _import_mvsdk().CameraSoftTrigger(self.h)

    def clear_buffer(self) -> None:
        _import_mvsdk().CameraClearBuffer(self.h)

    def grab(self, timeout_ms: int) -> Optional[np.ndarray]:
        mv = _import_mvsdk()
        with self._lock:
            try:
                raw, head = mv.CameraGetImageBuffer(self.h, int(timeout_ms))
            except mv.CameraException as e:
                if e.error_code == mv.CAMERA_STATUS_TIME_OUT:
                    return None
                raise CameraError(f"{self.name}: grab failed ({e.error_code}): {e.message}") from e
            try:
                mv.CameraImageProcess(self.h, raw, self.buf, head)
            finally:
                mv.CameraReleaseImageBuffer(self.h, raw)
            if platform.system() == "Windows":  # Windows SDK delivers bottom-up frames
                mv.CameraFlipFrameBuffer(self.buf, head, 1)
            data = (mv.c_ubyte * head.uBytes).from_address(self.buf)
            ch = 1 if head.uiMediaType == mv.CAMERA_MEDIA_TYPE_MONO8 else 3
            img = np.frombuffer(data, dtype=np.uint8).reshape(head.iHeight, head.iWidth, ch).copy()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if ch == 1 else img

    def set_output(self, index: int, state: bool) -> None:
        # MindVision cameras expose opto-isolated outputs; usable to drive a reject valve relay.
        _import_mvsdk().CameraSetIOState(self.h, int(index), 1 if state else 0)


# --------------------------------------------------------------------------- OpenCV / files
class OpenCVCamera(BaseCamera):
    def __init__(self, name: str, source=0):
        self.name, self.source, self.cap = name, source, None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise CameraError(f"{self.name}: cannot open {self.source!r}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def grab(self, timeout_ms: int) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None


class FileCamera(BaseCamera):
    """Replays ``<folder>/*.jpg|png|bmp`` in name order (loops forever)."""

    def __init__(self, name: str, folder: str, loop: bool = True):
        self.name, self.folder, self.loop = name, folder, loop
        self.files: list[str] = []
        self.pos = 0

    def open(self) -> None:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.files = sorted(f for e in exts for f in glob.glob(os.path.join(self.folder, e)))
        if not self.files:
            raise CameraError(f"{self.name}: no images in {self.folder}")

    def grab(self, timeout_ms: int) -> Optional[np.ndarray]:
        if self.pos >= len(self.files):
            if not self.loop:
                return None
            self.pos = 0
        img = cv2.imread(self.files[self.pos])
        self.pos += 1
        return img

    @property
    def current_file(self) -> str:
        return self.files[self.pos - 1] if self.pos else ""


# --------------------------------------------------------------------------- group of 4
class CameraGroup:
    """The 4 corner cameras that look at one container."""

    def __init__(self, cameras: list[BaseCamera]):
        self.cameras = cameras
        self.pool = ThreadPoolExecutor(max_workers=len(cameras))

    def open(self) -> None:
        for c in self.cameras:
            c.open()

    def close(self) -> None:
        for c in self.cameras:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                log.exception("closing %s", c.name)
        self.pool.shutdown(wait=False)

    def clear(self) -> None:
        for c in self.cameras:
            c.clear_buffer()

    def trigger_all(self) -> None:
        for c in self.cameras:
            c.software_trigger()

    def grab_all(self, timeout_ms: int) -> dict[str, Optional[np.ndarray]]:
        """Grab one frame from every camera in parallel."""
        futures = {c.name: self.pool.submit(c.grab, timeout_ms) for c in self.cameras}
        return {name: f.result() for name, f in futures.items()}


def build_cameras(cfg: dict) -> CameraGroup:
    cam_cfg = cfg["cameras"]
    backend = cam_cfg.get("backend", "mindvision")
    cams: list[BaseCamera] = []
    for i, d in enumerate(cam_cfg["devices"]):
        name = d.get("name", f"cam{i + 1}")
        if backend == "mindvision":
            cams.append(MindVisionCamera(name, d.get("serial", ""), d.get("index", i),
                                         cam_cfg.get("trigger", "hardware"),
                                         d.get("exposure_us", cam_cfg.get("exposure_us")),
                                         d.get("gain", cam_cfg.get("gain"))))
        elif backend == "opencv":
            cams.append(OpenCVCamera(name, d.get("source", i)))
        elif backend == "files":
            cams.append(FileCamera(name, os.path.join(cam_cfg.get("files_dir", "samples"), name)))
        else:
            raise ValueError(f"unknown camera backend {backend!r}")
    return CameraGroup(cams)
