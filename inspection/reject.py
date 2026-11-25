"""Reject outputs - tell the machine to push a bad container off the belt.

type: none        only log (dry run / commissioning)
type: camera_io   MindVision camera opto output -> relay -> reject valve
type: serial      send bytes to a USB relay board or PLC serial port
type: modbus_tcp  write a coil on a PLC (pip install pymodbus)

Every output pulses: ON, wait ``pulse_ms``, OFF. ``delay_ms`` waits first so the
container can travel from the cameras to the reject pusher.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class RejectOutput:
    def __init__(self, delay_ms: int = 0, pulse_ms: int = 200):
        self.delay_ms, self.pulse_ms = delay_ms, pulse_ms

    def _set(self, on: bool) -> None:
        raise NotImplementedError

    def _pulse(self) -> None:
        try:
            if self.delay_ms:
                time.sleep(self.delay_ms / 1000)
            self._set(True)
            time.sleep(self.pulse_ms / 1000)
            self._set(False)
        except Exception:  # noqa: BLE001 - never kill the inspection loop
            log.exception("reject output failed")

    def reject(self) -> None:
        """Non-blocking: the next container can be inspected while the pusher fires."""
        threading.Thread(target=self._pulse, daemon=True).start()

    def close(self) -> None: ...


class NoReject(RejectOutput):
    def _set(self, on: bool) -> None:
        if on:
            log.warning("REJECT (dry run - no output configured)")


class CameraIOReject(RejectOutput):
    def __init__(self, camera, output_index: int = 0, **kw):
        super().__init__(**kw)
        self.camera, self.index = camera, output_index

    def _set(self, on: bool) -> None:
        self.camera.set_output(self.index, on)


class SerialReject(RejectOutput):
    def __init__(self, port: str, baud: int = 9600, on_hex: str = "A00101A2",
                 off_hex: str = "A00100A1", **kw):
        super().__init__(**kw)
        import serial  # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.on, self.off = bytes.fromhex(on_hex), bytes.fromhex(off_hex)  # defaults: common LCUS USB relay

    def _set(self, on: bool) -> None:
        self.ser.write(self.on if on else self.off)

    def close(self) -> None:
        self.ser.close()


class ModbusReject(RejectOutput):
    def __init__(self, host: str, port: int = 502, coil: int = 0, unit: int = 1, **kw):
        super().__init__(**kw)
        from pymodbus.client import ModbusTcpClient  # pip install pymodbus
        self.client = ModbusTcpClient(host, port=port)
        self.client.connect()
        self.coil, self.unit = coil, unit

    def _set(self, on: bool) -> None:
        self.client.write_coil(self.coil, on, slave=self.unit)

    def close(self) -> None:
        self.client.close()


def build_reject(cfg: dict, cameras) -> RejectOutput:
    r = dict(cfg.get("reject") or {"type": "none"})
    kind = r.pop("type", "none")
    timing = {"delay_ms": r.pop("delay_ms", 0), "pulse_ms": r.pop("pulse_ms", 200)}
    if kind == "none":
        return NoReject(**timing)
    if kind == "camera_io":
        idx = r.pop("camera", 0)
        return CameraIOReject(cameras.cameras[idx], r.pop("output_index", 0), **timing)
    if kind == "serial":
        return SerialReject(**r, **timing)
    if kind == "modbus_tcp":
        return ModbusReject(**r, **timing)
    raise ValueError(f"unknown reject type {kind!r}")
