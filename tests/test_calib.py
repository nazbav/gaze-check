import threading
import time
import types

import cv2
import numpy as np
import synth

import gaze_check.calib as calib
from gaze_check.app import State


def test_pose_hint():
    frame = (1280, 720)
    assert not calib.pose_hint({})[0]
    ok, text = calib.pose_hint({"face": True, "valid": True, "marks": [(640, 100)], "frame": frame})
    assert not ok and "откиньте" in text
    ok, text = calib.pose_hint({"face": True, "valid": True, "marks": [(640, 600)], "frame": frame})
    assert not ok and "к себе" in text
    assert calib.pose_hint({"face": True, "valid": True, "marks": [(640, 330)], "frame": frame})[0]


def test_light_stripes_clear_of_targets():
    """Белые полосы подсветки — на каждом экране калибровки и не задевают ни одну метку (радиус до 28 px)."""
    from gaze_check.eyes import calibration_points, check_points, edge_points
    for w, h in ((2560, 1600), (1920, 1080), (1366, 768), (3840, 2160)):
        xs = [p[0] for p in calibration_points(w, h) + edge_points(w, h) + check_points(w, h)]
        xs.append(round(calib.HEAD_POINT[0] * w))
        stripes = calib.light_stripes(w)
        assert len(stripes) == 2
        for x0, x1 in stripes:
            assert x1 - x0 >= 0.07 * w
            for x in xs:
                assert x + 28 < x0 or x - 28 > x1, (w, x, x0, x1)
    scr = calib.Screen.__new__(calib.Screen)
    scr.w, scr.h = 2560, 1600
    img = scr.canvas()
    (a0, a1), (b0, b1) = calib.light_stripes(2560)
    assert (img[:, a0:a1] == 255).all() and (img[:, b0:b1] == 255).all()
    assert (img[:, 1280] == 24).all()


def test_calibration_flow_fits_checks_and_saves(monkeypatch, tmp_path):
    """Экран калибровки без окна: 9 точек → голова → ближе → дальше → проверка → Enter;
    глаза синтетические."""
    monkeypatch.setattr(calib, "SETTLE", 0.05)
    monkeypatch.setattr(calib, "COLLECT", 0.15)
    monkeypatch.setattr(calib, "HEAD_MOVE", 0.3)
    monkeypatch.setattr(calib, "primary_monitor", lambda: synth.SCREEN)
    monkeypatch.setattr(calib, "screen_mm", lambda monitor=None: synth.SIZE_MM)
    shown, current = {"n": 0, "last": None}, {}

    class FakeScreen(calib.Screen):
        def __init__(self, monitor):
            self.left, self.top, self.w, self.h = monitor
            self.font = 25

        def show(self, img, wait=15):
            shown["n"] += 1
            shown["last"] = img  # только последний: кадр 2560×1600 — 12 МБ
            time.sleep(0.005)
            return 13 if current.get("review") else 32  # пробел на подсказках, Enter на итоге

        def close(self):
            pass

    real_target, real_review = calib._target, calib._review

    def target(img, p, progress, color=(40, 40, 230)):
        current["p"] = p
        real_target(img, p, progress, color)

    def review(scr, cal, points, samples, checked):
        current["review"] = True
        current["checked"] = checked
        return real_review(scr, cal, points, samples, checked)

    monkeypatch.setattr(calib, "Screen", FakeScreen)
    monkeypatch.setattr(calib, "_target", target)
    monkeypatch.setattr(calib, "_review", review)

    state = State()
    state.eye = {"valid": True}
    saved = {}
    engine = types.SimpleNamespace(state=state, collect=None, calibration=None,
                                   set_calibration=lambda cal: saved.setdefault("cal", cal))
    stop = threading.Event()

    def eyes():  # «человек» честно смотрит на текущую точку
        rng = np.random.default_rng(0)
        while not stop.is_set():
            p = current.get("p")
            if engine.collect is not None and p:
                engine.collect.append((time.monotonic(), synth.feat(p, noise=0.005, rng=rng)))
            time.sleep(0.005)

    t = threading.Thread(target=eyes, daemon=True)
    t.start()
    try:
        cal = calib.run_calibration(engine)
    finally:
        stop.set()
    assert cal is saved["cal"] and cal.screen == synth.SCREEN
    assert len(current["checked"]) == 5 and cal.check_px is not None and cal.check_px < 60
    assert cal.jitter_px is not None and cal.jitter_px < 30  # дрожание на точках проверки посчитано
    assert cal.error_px < 60
    x, y = cal.predict(synth.feat((400, 700)))
    assert abs(x - 400) < 60 and abs(y - 700) < 60
    cv2.imwrite(str(tmp_path / "review.png"), shown["last"])
    assert engine.collect is None


def test_quick_adjust_adds_current_pose(monkeypatch):
    """Быстрая подстройка: 5 точек в текущей позе добавляются к калибровке."""
    from gaze_check.eyes import Calibration, calibration_points
    monkeypatch.setattr(calib, "SETTLE", 0.05)
    monkeypatch.setattr(calib, "COLLECT", 0.15)
    monkeypatch.setattr(calib, "primary_monitor", lambda: synth.SCREEN)
    current = {}

    class FakeScreen(calib.Screen):
        def __init__(self, monitor):
            self.left, self.top, self.w, self.h = monitor
            self.font = 25

        def show(self, img, wait=15):
            time.sleep(0.005)
            return 32

        def close(self):
            pass

    real_target = calib._target

    def target(img, p, progress, color=(40, 40, 230)):
        current["p"] = p
        real_target(img, p, progress, color)

    monkeypatch.setattr(calib, "Screen", FakeScreen)
    monkeypatch.setattr(calib, "_target", target)
    old = Calibration.fit(synth.SCREEN, [(p, synth.feat(p)) for p in calibration_points(2560, 1600) for _ in range(5)],
                          synth.SIZE_MM)
    state = State()
    state.eye = {"valid": True}
    saved = {}
    engine = types.SimpleNamespace(state=state, collect=None, calibration=old,
                                   set_calibration=lambda cal: saved.setdefault("cal", cal))
    stop = threading.Event()

    def eyes():  # в новой позе: голова ниже и дальше
        while not stop.is_set():
            p = current.get("p")
            if engine.collect is not None and p:
                engine.collect.append((time.monotonic(), synth.feat(p, eye=(0.0, -12.0, -55.0))))
            time.sleep(0.005)

    threading.Thread(target=eyes, daemon=True).start()
    try:
        cal = calib.run_quick(engine)
    finally:
        stop.set()
    assert cal is saved["cal"] and len(cal.points) > len(old.points)
    assert {p for p, _ in cal.points} == {p for p, _ in old.points}  # те же центр и углы — новые кадры к ним


def test_one_eye_calibration_is_added_not_replacing(monkeypatch):
    """Калибровка с закрытым вторым глазом (кадры без сети) добавляется к двухглазой."""
    import numpy as np
    from gaze_check.eyes import Calibration, calibration_points
    rng = np.random.default_rng(0)
    mix = rng.normal(0, 1, (40, 3))

    def net(p):
        return mix @ np.array([p[0] / 2560, p[1] / 1600, 1.0]) + rng.normal(0, 0.01, 40)

    two = [(p, {**synth.feat(p), "net": net(p)}) for p in calibration_points(2560, 1600) for _ in range(6)]
    old = Calibration.fit(synth.SCREEN, two, synth.SIZE_MM)
    assert old.method == "MGazeNet"
    one = [(p, synth.feat(p)) for p in calibration_points(2560, 1600) for _ in range(6)]
    assert calib.one_eye_session(one) and not calib.one_eye_session(two)
    monkeypatch.setattr(calib, "SETTLE", 0.01)
    monkeypatch.setattr(calib, "COLLECT", 0.05)
    monkeypatch.setattr(calib, "HEAD_MOVE", 0.05)
    monkeypatch.setattr(calib, "primary_monitor", lambda: synth.SCREEN)
    monkeypatch.setattr(calib, "screen_mm", lambda monitor=None: synth.SIZE_MM)
    current = {}

    class FakeScreen(calib.Screen):
        def __init__(self, monitor):
            self.left, self.top, self.w, self.h = monitor
            self.font = 25

        def show(self, img, wait=15):
            time.sleep(0.002)
            return 13 if current.get("review") else 32

        def close(self):
            pass

    real_target, real_review = calib._target, calib._review

    def target(img, p, progress, color=(40, 40, 230)):
        current["p"] = p
        real_target(img, p, progress, color)

    def review(*a):
        current["review"] = True
        return real_review(*a)

    monkeypatch.setattr(calib, "Screen", FakeScreen)
    monkeypatch.setattr(calib, "_target", target)
    monkeypatch.setattr(calib, "_review", review)
    state = State()
    state.eye = {"valid": True}
    saved = {}
    engine = types.SimpleNamespace(state=state, collect=None, calibration=old, lead="left",
                                   set_calibration=lambda cal: saved.setdefault("cal", cal))
    stop = threading.Event()

    def eyes():  # второй глаз закрыт: сеть не считается
        while not stop.is_set():
            if engine.collect is not None and current.get("p"):
                engine.collect.append((time.monotonic(), synth.feat(current["p"])))
            time.sleep(0.003)

    threading.Thread(target=eyes, daemon=True).start()
    try:
        cal = calib.run_calibration(engine)
    finally:
        stop.set()
    assert cal is saved["cal"] and cal.method == "MGazeNet"  # сеть осталась
    assert len(cal.points) > len(old.points)  # кадры по одному глазу добавились
