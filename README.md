# IML Container Vision Inspection

Real-time defect inspection for IML (in-mould label) containers on the take-out conveyor.
It uses **4 MindVision industrial cameras**, one at each corner of the container, plus **OpenCV**. Every container is
checked from all four sides in about 40 ms. A bad container triggers a reject output, and every result is counted and logged.

```
 machine cycle (3-10 s) ──► trigger ──► cam1 cam2 cam3 cam4 ──► OpenCV inspection (parallel)
                                                                  │
                     reject valve / PLC ◄── any camera NG ◄───────┘──► counters + CSV + NG images
```

## What it detects

| Defect code | How |
|---|---|
| `no_container`, `no_image` | Label not found in view, or a camera returned no frame (fail-safe reject) |
| `lid_missing` | Too little cream lid above the label |
| `lid_spot_dark`, `label_spot_dark` | Black specks or burn marks |
| `lid_foreign_colour`, `label_foreign_colour` | Pink, red or brown streaks and contamination (e.g. the pink streak at the gate) |
| `label_damaged` | Torn, folded or missing piece of label (shape solidity) |
| `abnormal_*` | Learned "golden sample" model. Label shifted or skewed, wrong size, deformed container, lid not seated, wrong product |
| `label_bare_area` | Optional. Cream plastic visible through the label (needs white LED lighting) |

No AI training is needed. The model is the statistics of 20–50 **good** containers per camera
(`teach`). Each product (small / medium / large from the drawing) has its own model file.

| Product | Top Ø | Bottom Ø | Height |
|---|---|---|---|
| small | 64 mm | ~50 mm | 67 mm |
| medium | 114 mm | 92 mm | 106–107 mm |
| large | 135 mm | 108 mm | 132–135 mm |

## Install (factory PC, Windows or Linux)

1. Install the **MindVision camera SDK** from the camera supplier / mindvision.com.cn.
2. Copy `mvsdk.py` from `<SDK folder>/Demo/python/` into this folder. It is not included here because it is MindVision's file.
3. `pip install -r requirements.txt`
4. Use **colour** cameras with a short exposure (≤2 ms) and steady white LED lighting. Colour checks don't work on mono cameras.

## Use

```bash
python main.py list-cameras                 # get serial numbers -> put them in config.yaml
python main.py capture --count 30           # run 30 GOOD containers, saves teach_images/<camera>/
python main.py teach                        # builds models/<product>.json
python main.py check teach_images/cam1_front_left/*.png --camera cam1_front_left   # offline test
python main.py run                          # production: live 2x2 view, press q to quit
```

Change product: set `product: small|medium|large` in `config.yaml`, then run `capture` and `teach` once for that product.

## Triggering (cycle 3–10 s)

* **hardware** (recommended): wire the machine/robot "part released" signal, or a photo-sensor on the belt,
  to the trigger input of all 4 cameras. All 4 expose at the same moment and the program waits for the frames.
* **software**: the program triggers all cameras every `software_trigger_interval_s`.

## Reject output (`config.yaml → reject`)

| type | wiring |
|---|---|
| `none` | dry run, logs only (use for commissioning) |
| `camera_io` | MindVision camera output → relay → reject solenoid |
| `serial` | USB relay board (LCUS default codes) or PLC serial |
| `modbus_tcp` | writes a coil on the PLC (`pip install pymodbus`) |

`delay_ms` = travel time from the cameras to the pusher. `pulse_ms` = how long the valve is held on.

## Output

* `results/log_YYYYMMDD.csv`: one row per container (OK/REJECT, defect codes, failed cameras, time)
* `results/YYYYMMDD/reject/`: annotated images of every rejected container, from all 4 cameras
* Live window: 2x2 camera view with total / good / reject counters

## Tuning

* `roi` per camera: crop to the container area to avoid background objects.
* `z_max`: tolerance of the learned model. Raise it if good parts are rejected, lower it to be stricter.
* `min_spot_area_px`: smallest spot that rejects.
* HSV ranges: match the label and lid colours. Use `check` to see annotated results.

## Tests

```bash
pip install pytest && python -m pytest -q
```

The tests use synthetic images (`tests/synthetic.py`) covering good parts, pink streak, black specks, missing lid,
torn label, skewed label and missing container.
