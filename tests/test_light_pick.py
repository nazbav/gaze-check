"""Выбор калибровки по свету (Engine._pick_light) и куда идёт новая калибровка (set_calibration)."""
import types

import synth

from gaze_check.app import Engine, State
from gaze_check.calstore import CalStore
from gaze_check.eyes import Calibration, calibration_points
from gaze_check.light import LightWatch

DARK = {"face": 40.0, "side": 0.0, "warm": 0.5, "back": 0.0, "fps": 10.0}
LAMP = {"face": 150.0, "side": 0.2, "warm": 0.8, "back": 0.0, "fps": 30.0}
ODD = {"face": 250.0, "side": -0.4, "warm": 2.0, "back": 1.5, "fps": 30.0}  # ни на что не похож


def cal():
    w, h = synth.SCREEN[2], synth.SCREEN[3]
    return Calibration.fit(synth.SCREEN, [(p, synth.feat(p)) for p in calibration_points(w, h) for _ in range(3)],
                           synth.SIZE_MM, loo=False)


def engine(tmp_path, lights):
    """Движок без камеры и потоков: набор калибровок камеры "cam" с данными светами."""
    store = CalStore(tmp_path / "cals", synth.SCREEN)
    cals = [store.put(cal(), "cam", light) for light in lights]
    eng = Engine.__new__(Engine)
    told = []
    eng.store, eng.light, eng.camera, eng.calibration = store, LightWatch(), "cam", cals[0]
    eng.light_auto, eng.light_pick, eng.new_light_since, eng.new_light_told = True, None, None, False
    eng.state, eng.lead, eng.dirty, eng.saved_at, eng.light_logged = State(), "both", False, 0.0, None
    eng.journal = types.SimpleNamespace(notify=told.append)
    return eng, cals, told


def feed(eng, sig, t0, n=30):
    for i in range(n):
        eng.light.add(t0 + i, dict(sig))


def test_switches_to_nearer_light_after_two_checks(tmp_path):
    eng, (dark, lamp), _ = engine(tmp_path, [DARK, LAMP])
    feed(eng, LAMP, 0)
    eng._pick_light(30)
    assert eng.calibration is dark  # одна проверка — ещё нет
    eng._pick_light(40)
    assert eng.calibration is lamp
    assert "светло" in eng.state.snapshot()["accuracy"]
    eng._pick_light(50)
    assert eng.calibration is lamp  # и не скачет обратно


def test_close_lights_do_not_flip(tmp_path):
    eng, (dark, lamp), _ = engine(tmp_path, [DARK, dict(DARK, face=70.0)])
    feed(eng, dict(DARK, face=56.0), 0)  # чуть ближе ко второй, но меньше запаса
    for t in (30, 40, 50):
        eng._pick_light(t)
    assert eng.calibration is dark


def test_unknown_light_reminds_once(tmp_path):
    eng, _, told = engine(tmp_path, [DARK])
    feed(eng, ODD, 0)
    for t in (30, 60, 95, 120):
        eng._pick_light(t)
    assert len(told) == 1 and "Ctrl+Alt+C" in told[0]
    feed(eng, DARK, 200, 70)  # свет снова знакомый — можно напомнить снова, когда опять сменится
    eng._pick_light(270)
    assert not eng.new_light_told


def test_manual_choice_turns_auto_off(tmp_path):
    eng, (dark, lamp), _ = engine(tmp_path, [DARK, LAMP])
    assert eng.choose_calibration(lamp.meta["id"]) and eng.calibration is lamp and not eng.light_auto
    feed(eng, DARK, 0)
    eng._pick_light(30)
    eng._pick_light(40)
    assert eng.calibration is lamp


def test_quick_adjust_same_light_extends_new_light_adds(tmp_path):
    eng, (dark,), _ = engine(tmp_path, [DARK])
    feed(eng, DARK, 0)
    same = cal()
    eng.set_calibration(same, since=0, quick=True)
    assert same.meta["id"] == dark.meta["id"] and len(eng.store.items) == 1  # дополнила ту же
    feed(eng, LAMP, 100)
    new = cal()
    eng.set_calibration(new, since=100, quick=True)
    assert new.meta["id"] != dark.meta["id"] and new.meta["light"]["face"] == LAMP["face"]
    assert len(eng.store.items) == 2 and eng.calibration is new


def test_full_calibration_replaces_same_light(tmp_path):
    eng, (dark, lamp), _ = engine(tmp_path, [DARK, LAMP])
    feed(eng, dict(DARK, face=45.0), 0)
    again = cal()
    eng.set_calibration(again, since=0)
    assert again.meta["id"] == dark.meta["id"] and len(eng.store.items) == 2
