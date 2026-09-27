import numpy as np

from gaze_check.light import NEW, SAME, STEP, LightWatch, describe, distance, signature

PTS = np.array([[220, 120], [420, 120], [220, 380], [420, 380]], float)  # рамка «лица» в кадре


def frame(left=(120, 120, 120), right=(120, 120, 120), back=60):
    """Кадр BGR: фон и «лицо» из двух половин (левая и правая в кадре)."""
    f = np.full((480, 640, 3), back, np.uint8)
    f[120:381, 220:320] = left
    f[120:381, 320:421] = right
    return f


def test_signature_side_warm_face_back():
    even = signature(frame(), PTS, 30)
    assert abs(even["side"]) < 0.01 and even["fps"] == 30
    lit = signature(frame(left=(200, 200, 200), right=(80, 80, 80)), PTS, 30)
    assert lit["side"] > 0.2  # левая половина кадра — правая щека человека
    assert "свет справа" in describe(lit)
    warm = signature(frame(left=(60, 120, 200), right=(60, 120, 200)), PTS, 30)  # BGR: много красного — лампа
    cold = signature(frame(left=(200, 120, 90), right=(200, 120, 90)), PTS, 30)
    assert warm["warm"] > cold["warm"] + 0.5
    dark = signature(frame(left=(30, 30, 30), right=(30, 30, 30)), PTS, 10)
    assert dark["face"] < even["face"] - 60 and describe(dark).startswith("темно")
    assert signature(frame(back=230), PTS, 30)["back"] > even["back"] + 0.5  # светлый фон — окно за спиной


def test_no_face_no_signature():
    assert signature(frame(), None, 30) is None
    assert signature(frame(), np.array([[10, 10], [12, 12]], float), 30) is None  # лицо в пару пикселей


def test_distance_in_steps():
    a = {"face": 100.0, "side": 0.0, "warm": 0.4, "back": 0.0, "fps": 30.0}
    assert distance(a, a) == 0
    assert abs(distance(a, dict(a, face=100.0 + STEP["face"])) - 1) < 1e-9
    assert distance(a, None) == float("inf")
    assert SAME < NEW


def test_watch_median_needs_enough_samples():
    w = LightWatch()
    base = {"face": 100.0, "side": 0.0, "warm": 0.4, "back": 0.0, "fps": 30.0}
    for i in range(19):
        assert w.due(float(i))
        w.add(float(i), dict(base, face=100.0 + i))
    assert w.current() is None  # меньше 20 отпечатков — свет ещё не понят
    w.add(19.0, dict(base, face=500.0))  # выброс медиану не сдвигает
    cur = w.current()
    assert cur is not None and 105 <= cur["face"] <= 115
    assert not w.due(19.5)
    w.add(20.0, None)  # лица нет — не пишется
    assert len(w.samples) == 20
    assert w.between(0.0, 4.0, min_count=5)["face"] == 102.0
