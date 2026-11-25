import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspection.detector import Inspector  # noqa: E402
from inspection.runner import Counters, mosaic  # noqa: E402
from synthetic import make  # noqa: E402


@pytest.fixture(scope="module")
def taught():
    ins = Inspector()
    ins.teach("cam1", [make(seed=s) for s in range(12)])
    return ins


def test_good_containers_pass(taught):
    for s in range(100, 120):
        r = taught.inspect("cam1", make(seed=s))
        assert r.ok, (s, [(d.code, d.detail) for d in r.defects])


def test_rules_work_without_model():
    r = Inspector().inspect("cam1", make(seed=1))
    assert r.ok, r.codes


@pytest.mark.parametrize("defect, expected", [
    ("pink_streak", "label_foreign_colour"),
    ("black_speck_lid", "lid_spot_dark"),
    ("black_speck_label", "label_spot_dark"),
    ("no_lid", "lid_missing"),
    ("no_container", "no_container"),
])
def test_rule_defects(taught, defect, expected):
    r = taught.inspect("cam1", make(defect, seed=7))
    assert not r.ok
    assert expected in r.codes, r.codes


@pytest.mark.parametrize("defect", ["torn_label", "label_shift"])
def test_model_defects(taught, defect):
    r = taught.inspect("cam1", make(defect, seed=7))
    assert not r.ok, r.features


def test_missing_frame_rejects(taught):
    r = taught.inspect("cam1", None)
    assert not r.ok and r.codes == ["no_image"]


def test_model_roundtrip(taught, tmp_path):
    p = tmp_path / "m.json"
    taught.save_model(str(p))
    again = Inspector(model=Inspector.load_model(str(p)))
    assert again.inspect("cam1", make("label_shift", seed=3)).ok is False


def test_roi_per_camera(taught):
    img = make("black_speck_lid", seed=2)
    ins = Inspector(camera_params={"cam1": {"roi": [0, 0, 300, 900]}})   # ROI misses the container
    assert "no_container" in ins.inspect("cam1", img).codes


def test_counters_and_mosaic():
    c = Counters()
    c.add(True, [])
    c.add(False, ["lid_missing", "lid_missing"])
    assert (c.total, c.good, c.rejected, c.by_defect["lid_missing"]) == (2, 1, 1, 1)
    m = mosaic([np.zeros((480, 640, 3), np.uint8)] * 4, width=640)
    assert m.shape == (480, 640, 3)
