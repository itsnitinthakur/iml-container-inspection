"""IML container vision inspection - 4 MindVision cameras, OpenCV, real-time reject.

  python main.py list-cameras
  python main.py capture --count 30            # save good-container images for teaching
  python main.py teach                         # learn the good-container model
  python main.py check images/*.jpg --camera cam1_front_left
  python main.py run                           # production
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time

import cv2
import yaml

from inspection.cameras import build_cameras, _import_mvsdk
from inspection.detector import Inspector
from inspection.reject import build_reject
from inspection.runner import Runner

log = logging.getLogger("iml")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_path(cfg: dict) -> str:
    return os.path.join(cfg.get("models_dir", "models"), f"{cfg.get('product', 'default')}.json")


def build_inspector(cfg: dict) -> Inspector:
    product = cfg.get("product", "default")
    params = {**(cfg.get("inspection") or {}), **((cfg.get("products") or {}).get(product) or {})}
    cam_params = {d["name"]: d.get("inspection", {}) | ({"roi": d["roi"]} if d.get("roi") else {})
                  for d in cfg["cameras"]["devices"]}
    return Inspector(params, Inspector.load_model(model_path(cfg)), cam_params)


def camera_names(cfg: dict) -> list[str]:
    return [d["name"] for d in cfg["cameras"]["devices"]]


# ------------------------------------------------------------------------- commands
def cmd_list_cameras(cfg, args):
    mv = _import_mvsdk()
    for i, d in enumerate(mv.CameraEnumerateDevice()):
        print(f"index {i}: {d.GetFriendlyName()}  SN={d.GetSn()}  port={d.GetPortType()}")


def cmd_capture(cfg, args):
    """Grab N containers from all 4 cameras into teach_dir/<camera>/ (only put GOOD containers on the belt)."""
    cams = build_cameras(cfg)
    cams.open()
    out = args.out or cfg.get("teach_dir", "teach_images")
    trig = cfg["cameras"].get("trigger", "hardware")
    try:
        n = 0
        while n < args.count:
            if trig == "software":
                input(f"[{n + 1}/{args.count}] place a GOOD container and press Enter...")
                cams.clear()
                cams.trigger_all()
            frames = cams.grab_all(int(cfg["cameras"].get("grab_timeout_ms", 12000)))
            if any(f is None for f in frames.values()):
                log.warning("not all cameras returned a frame - skipped")
                continue
            n += 1
            for name, img in frames.items():
                os.makedirs(os.path.join(out, name), exist_ok=True)
                cv2.imwrite(os.path.join(out, name, f"{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}.png"), img)
            log.info("captured %d/%d", n, args.count)
    finally:
        cams.close()


def cmd_teach(cfg, args):
    ins = build_inspector(cfg)
    src = args.images or cfg.get("teach_dir", "teach_images")
    for name in camera_names(cfg):
        files = sorted(glob.glob(os.path.join(src, name, "*.*")))
        imgs = [im for im in (cv2.imread(f) for f in files) if im is not None]
        stats = ins.teach(name, imgs)
        log.info("%s: learned from %d images", name, stats["_samples"])
        # sanity: every good image must now pass
        bad = [f for f, im in zip(files, imgs) if not ins.inspect(name, im, annotate=False).ok]
        if bad:
            log.warning("%s: %d of the good teach images still fail the rules (check lighting / ROI / "
                        "thresholds or remove bad samples): %s", name, len(bad), bad[:5])
    path = model_path(cfg)
    ins.save_model(path)
    log.info("model saved to %s", path)


def cmd_check(cfg, args):
    ins = build_inspector(cfg)
    os.makedirs(args.out, exist_ok=True)
    files = [f for pat in args.images for f in glob.glob(pat)]
    fails = 0
    for f in files:
        r = ins.inspect(args.camera, cv2.imread(f))
        fails += not r.ok
        print(f"{'OK    ' if r.ok else 'REJECT'} {f}  {r.ms:.0f} ms  " +
              "; ".join(f"{d.code} ({d.detail})" for d in r.defects))
        if r.annotated is not None:
            cv2.imwrite(os.path.join(args.out, os.path.basename(f)), r.annotated)
    print(f"{len(files) - fails} OK, {fails} REJECT - annotated images in {args.out}/")


def cmd_run(cfg, args):
    if args.no_window:
        cfg.setdefault("output", {})["show_window"] = False
    cams = build_cameras(cfg)
    cams.open()
    reject = build_reject(cfg, cams)
    ins = build_inspector(cfg)
    if not ins.model:
        log.warning("no model at %s - only fixed rules are active. Run 'capture' + 'teach' first.",
                    model_path(cfg))
    try:
        Runner(cfg, cams, ins, reject).run(max_cycles=args.cycles)
    finally:
        reject.close()
        cams.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-cameras")
    p = sub.add_parser("capture"); p.add_argument("--count", type=int, default=30); p.add_argument("--out")
    p = sub.add_parser("teach"); p.add_argument("--images", help="folder with <camera_name>/ sub-folders")
    p = sub.add_parser("check"); p.add_argument("images", nargs="+"); p.add_argument("--camera", default="cam1")
    p.add_argument("--out", default="check_out")
    p = sub.add_parser("run"); p.add_argument("--cycles", type=int, default=0); p.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    {"list-cameras": cmd_list_cameras, "capture": cmd_capture, "teach": cmd_teach,
     "check": cmd_check, "run": cmd_run}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
