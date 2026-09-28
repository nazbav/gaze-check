"""Отчёт по лёгкому журналу — те же итоги, что у записи сеанса, плюс моргания и «смотрит в экран»."""
import csv
import json

from gaze_check.gazelog import BLINK, DOUBLE, FACE, LOOKING, GazeLog
from gaze_check.gazereport import build, report_dir
from gaze_check.zones import ALL_ZONES, zone_of

SCREEN = (0, 0, 2560, 1600)


def write(path):
    """10 с: 6 с смотрит в левый верх (Code), 1 с нет точки, 3 с в центр (chrome); 4 моргания, одно двойное."""
    log = GazeLog(SCREEN, 60.0, path=path, clock=lambda: 0.0)
    for i in range(150):
        t = i / 15
        point = (300, 200) if t < 6 else None if t < 7 else (1280, 800)
        zone = ALL_ZONES.index(zone_of(point, 2560, 1600))
        flags = FACE | (LOOKING if point else 0)
        log.sample(t, point, 55, flags, zone, "Code.exe — app.py" if t < 6 else "chrome.exe — сайт" if point else "")
        if i in (15, 45, 75, 120):
            log.event(t, BLINK, 150)
    log.event(8.0, DOUBLE)
    return log.close()


def test_report_like_recording(tmp_path):
    path = write(tmp_path / "gaze_x.gazelog")
    folder = build(path)
    assert folder == report_dir(path) == tmp_path / "gaze_x"
    for name in ("session.json", "attention.csv", "fixations.csv", "heatmap.png", "zones.png", "gaze.csv"):
        assert (folder / name).stat().st_size > 0, name
    s = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert 9 <= s["duration_s"] <= 10.5 and s["source"] == "gaze_x.gazelog"
    assert s["fixations"] >= 2 and s["first_fixations"][0]["x"] == 300
    assert s["zones"]["верх-лево"]["share"] > 0.5 and s["zones"]["центр"]["share"] > 0.2
    assert set(s["apps"]) >= {"Code.exe", "chrome.exe"} and "app.py" in s["windows"][0]["window"]
    assert s["blinks"] == 4 and s["double_blinks"] == 1 and s["blink_ms_median"] == 150
    assert 0.8 <= s["looking_share"] <= 0.95 and s["face_share"] == 1.0 and s["distance_cm_median"] == 55
    with open(folder / "attention.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["zone"] == "верх-лево" and rows[0]["app"] == "Code.exe" and rows[0]["time"]
