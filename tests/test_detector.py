from pathlib import Path

import cv2
import numpy as np
import pytest

from gaze_check.models import DETECTOR_MODEL, to_rgb

PERSON = Path(__file__).parent / "data" / "person.jpg"
pytestmark = pytest.mark.skipif(not DETECTOR_MODEL.exists(), reason="нет модели детектора")


@pytest.fixture(scope="module")
def detector():
    from gaze_check.detector import Detector
    return Detector(DETECTOR_MODEL)


def test_finds_person_in_workflow_format(detector):
    rgb = to_rgb(cv2.imdecode(np.fromfile(str(PERSON), np.uint8), cv2.IMREAD_COLOR))
    dets = detector.detect(rgb)
    person = [d for d in dets if d["class"] == "person"]
    assert person and person[0]["confidence"] > 0.8 and person[0]["class_id"] == 1
    d = person[0]
    assert set(d) == {"x", "y", "width", "height", "confidence", "class_id", "class"}
    h, w = rgb.shape[:2]
    assert 0 < d["x"] < w and 0 < d["y"] < h and d["width"] > w / 4


def test_empty_frame_finds_nothing(detector):
    assert detector.detect(np.full((720, 1280, 3), 200, np.uint8)) == []
