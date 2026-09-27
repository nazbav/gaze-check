import csv
import json
import time

import numpy as np

from gaze_check.eyes import Calibration
from gaze_check.recorder import Recorder


def test_recording_writes_all_files(tmp_path):
    cal = Calibration((0, 0, 1920, 1080), {}, error_px=50)
    screen = np.full((720, 1280, 3), 200, np.uint8)
    probe = lambda point: ("Code.exe", "site.html - Visual Studio Code")  # noqa: E731
    rec = Recorder(cal, grab=lambda: (screen.copy(), 1280 / 1920), folder=tmp_path / "s", probe=probe)
    t0 = time.monotonic()
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(45):  # 1.5 с: сначала смотрит в угол, потом в центр
        ts = t0 + i / 30
        point = (200, 150) if i < 20 else (960, 540)
        rec.camera_frame(frame, ts, frame.copy())
        rec.gaze(ts, {"hx": 0.5}, point, np.array(point, float), "looking_at_screen")
        time.sleep(1 / 30)
    folder = rec.stop()
    names = {p.name for p in folder.iterdir()}
    assert {"camera.mp4", "camera_trace.mp4", "screen_gaze.mp4", "gaze.csv", "fixations.csv", "heatmap.png",
            "session.json", "attention.csv", "zones.png"} <= names
    rows = list(csv.DictReader(open(folder / "fixations.csv", encoding="utf-8")))
    assert [(r["order"], r["x"]) for r in rows] == [("1", "200"), ("2", "960")]
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert info["fixations"] == 2 and info["gaze_samples"] == 45
    assert (folder / "camera.mp4").stat().st_size > 1000

    # зоны: сначала левый верхний угол, потом центр; программа и окно — от probe
    gaze = list(csv.DictReader(open(folder / "gaze.csv", encoding="utf-8")))
    assert list(gaze[0])[-5:] == ["zone", "app", "window", "dist_cm", "method"]
    assert (gaze[0]["zone"], gaze[-1]["zone"]) == ("верх-лево", "центр")
    assert gaze[0]["app"] == "Code.exe"
    attention = list(csv.DictReader(open(folder / "attention.csv", encoding="utf-8")))
    assert [a["zone"] for a in attention] == ["верх-лево", "центр"]
    assert attention[0]["window"] == "site.html - Visual Studio Code" and len(attention[0]["time"]) == 8
    assert abs(float(attention[0]["end"]) - float(attention[1]["start"])) < 1e-6
    assert info["zones"]["верх-лево"]["share"] > 0.3 and info["zones"]["центр"]["share"] > 0.3
    assert info["apps"]["Code.exe"]["share"] > 0.99
    assert info["windows"][0]["window"] == "site.html - Visual Studio Code"
    assert {"attention.csv", "zones.png"} <= set(info["files"])


def test_recording_without_calibration_marks_no_data(tmp_path):
    rec = Recorder(None, folder=tmp_path / "s")
    t0 = time.monotonic()
    for i in range(10):
        rec.gaze(t0 + i / 30, None, None, None, "face_not_visible")
    folder = rec.stop()
    attention = list(csv.DictReader(open(folder / "attention.csv", encoding="utf-8")))
    assert [a["zone"] for a in attention] == ["нет данных"]
    assert (folder / "zones.png").stat().st_size > 1000
