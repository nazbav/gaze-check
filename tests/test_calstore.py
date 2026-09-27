import math

import synth

from gaze_check.calstore import CalStore, title
from gaze_check.eyes import Calibration, calibration_points

DARK = {"face": 40.0, "side": 0.0, "warm": 0.5, "back": 0.0, "fps": 10.0}
LAMP = {"face": 150.0, "side": 0.2, "warm": 0.8, "back": 0.0, "fps": 30.0}


def cal():
    w, h = synth.SCREEN[2], synth.SCREEN[3]
    return Calibration.fit(synth.SCREEN, [(p, synth.feat(p)) for p in calibration_points(w, h) for _ in range(3)],
                           synth.SIZE_MM, loo=False)


def test_put_and_reload(tmp_path):
    store = CalStore(tmp_path / "cals", synth.SCREEN)
    a = store.put(cal(), "Integrated Camera", DARK)
    assert (tmp_path / "cals" / ("%s.json" % a.meta["id"])).exists()
    again = CalStore(tmp_path / "cals", synth.SCREEN)
    [b] = again.for_camera("Integrated Camera")
    assert b.meta["id"] == a.meta["id"] and b.meta["light"] == DARK
    assert again.for_camera("Phone") == []
    assert "темно" in title(b)


def test_migrates_old_file_once(tmp_path):
    old = tmp_path / "calibration.json"
    cal().save(old)
    before = old.read_bytes()
    store = CalStore(tmp_path / "cals", synth.SCREEN, legacy=old, camera="Integrated Camera")
    [m] = store.for_camera("Integrated Camera")
    assert m.meta["light"] is None and old.read_bytes() == before  # прежний файл не тронут
    assert "свет неизвестен" in title(m)
    again = CalStore(tmp_path / "cals", synth.SCREEN, legacy=old, camera="Integrated Camera")
    assert len(again.items) == 1  # второй раз не переезжает


def test_nearest_prefers_known_light(tmp_path):
    store = CalStore(tmp_path / "cals", synth.SCREEN)
    unknown = store.put(cal(), "cam", None)
    assert store.nearest("cam", DARK) == (unknown, math.inf)  # только «свет неизвестен» — берём её
    dark = store.put(cal(), "cam", DARK)
    lamp = store.put(cal(), "cam", LAMP)
    assert store.nearest("cam", dict(DARK, face=45.0))[0] is dark
    assert store.nearest("cam", LAMP)[0] is lamp
    assert store.same_light("cam", dict(DARK, face=50.0)) is dark
    assert store.same_light("cam", dict(DARK, face=160.0)) is None
    assert store.nearest("other", DARK) == (None, math.inf)
    assert store.nearest("cam", None)[0] is lamp  # свет ещё не понят — последняя выбранная


def test_replace_keeps_id_and_update_after_click(tmp_path):
    store = CalStore(tmp_path / "cals", synth.SCREEN)
    a = store.put(cal(), "cam", DARK)
    a2 = store.put(cal(), "cam", DARK, replace=a)
    assert a2.meta["id"] == a.meta["id"] and store.for_camera("cam") == [a2]
    p = (synth.SCREEN[2] // 2, synth.SCREEN[3] // 2)
    clicked = a2.with_click(p, synth.feat(p))
    assert clicked is not None and clicked.meta["id"] == a2.meta["id"]
    store.update(clicked)
    assert store.for_camera("cam") == [clicked] and len(list((tmp_path / "cals").glob("*.json"))) == 1
    assert len(CalStore(tmp_path / "cals", synth.SCREEN).items[0].clicks) == len(clicked.clicks)
