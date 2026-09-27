import math

import numpy as np
import pytest
import synth

from gaze_check import eyes
from gaze_check.eyes import (EYES, Calibration, EyeModel, Fixations, OneEuro, ScreenGaze, calibration_points,
                             canonical_mesh, check_points, click_features)
from gaze_check.models import EYE_MODEL

S, MM = synth.SCREEN, synth.SIZE_MM


def _calibrate(rng, noise=0.01, head_move=True, extra=()):
    """Калибровка как на экране калибровки: 9 точек с неподвижной головой + центр с покачиванием."""
    samples = [(p, synth.feat(p, noise=noise, rng=rng)) for p in calibration_points(S[2], S[3]) for _ in range(20)]
    moving = []
    if head_move:
        c = (S[2] // 2, S[3] // 2)
        for i in range(40):
            head = synth.rot_y(15 * math.sin(i / 6)) @ synth.rot_x(10 * math.cos(i / 5))
            moving.append((c, synth.feat(c, head=head, noise=noise, rng=rng)))
    return Calibration.fit(S, samples + list(extra), MM, moving=moving)


def _miss(cal, point, **kw):
    x, y = cal.predict(synth.feat(point, **kw))
    return math.hypot(x - point[0], y - point[1])


def test_calibration_points_center_first_and_inside():
    pts = calibration_points(1920, 1080)
    assert pts[0] == (960, 540) and len(pts) == 9 and len(set(pts)) == 9
    assert all(0 < x < 1920 and 0 < y < 1080 for x, y in pts)
    xs, ys = sorted({x for x, _ in pts}), sorted({y for _, y in pts})
    assert xs[0] <= 40 and xs[-1] >= 1880 and ys[0] <= 40 and ys[-1] >= 1040  # у самой рамки, не 8% от края
    assert not set(check_points(1920, 1080)) & set(pts)


def test_calibration_recovers_person():
    cal = _calibrate(np.random.default_rng(0))
    assert _miss(cal, (700, 300)) < 25 and _miss(cal, (2300, 1400)) < 25
    assert cal.error_px is not None and cal.error_px < 60


def test_pose_change_after_calibration_keeps_point():
    """Главное: откалибровались в одной позе, потом наклонили крышку (камера смотрит на голову
    под другим углом — голова в пространстве камеры повернулась и сместилась) и откинулись."""
    cal = _calibrate(np.random.default_rng(1))
    tilt = synth.rot_x(-15)  # крышка откинута на 15°
    moved = dict(head=tilt @ synth.rot_y(10), eye=(6.0, -2.0, -52.0))
    for p in [(1280, 800), (300, 200), (2200, 1300), (400, 1450)]:
        assert _miss(cal, p, **moved) < 60, p  # ~0.8 см


def test_glances_away_do_not_spoil_calibration():
    """Во время калибровки человек иногда смотрит в телефон — эти кадры не должны её портить."""
    rng = np.random.default_rng(11)
    glances = [(p, synth.feat((p[0], p[1] + 1500), noise=0.01, rng=rng))  # смотрит сильно ниже экрана
               for p in calibration_points(S[2], S[3])[:4] for _ in range(6)]
    cal = _calibrate(rng, extra=glances)
    assert _miss(cal, (700, 300)) < 60 and _miss(cal, (2000, 1300)) < 60
    assert all(lo <= v <= hi for lo, v, hi in zip(eyes.LOW, cal.theta, eyes.HIGH))


def test_default_without_calibration_is_rough_but_on_screen():
    cal = Calibration.default(S, MM)
    assert not cal.calibrated
    x, y = cal.predict(synth.feat((1280, 800)))
    assert 0 < x < S[2] and 0 < y < S[3]


def test_clicks_fix_wrong_calibration():
    """Калибровали одного человека (или в очках), работает другой: клики чинят."""
    rng = np.random.default_rng(3)
    cal = _calibrate(rng)
    other = synth.TRUE + np.array([0.0, 0.05, 0.0, 0.04, 0.0, 0.0, 0.0])  # глаза смотрят иначе на ~3°

    def miss():
        return np.median([_miss(cal, p, theta=other) for p in [(900, 500), (1700, 1100), (500, 1200)]])

    before = miss()
    assert before > 80
    for _ in range(40):
        p = (float(rng.uniform(100, S[2] - 100)), float(rng.uniform(100, S[3] - 100)))
        cal = cal.with_click(p, synth.feat(p, theta=other, noise=0.01, rng=rng)) or cal
    assert miss() < before / 2 and len(cal.clicks) == 40
    assert cal.click_error_px is not None and np.median(cal.click_errors[-10:]) < np.median(cal.click_errors[:10])


def test_clicks_alone_calibrate_from_default():
    rng = np.random.default_rng(4)
    cal = Calibration.default(S, MM)
    before = _miss(cal, (1800, 400))
    for _ in range(30):
        p = (float(rng.uniform(100, S[2] - 100)), float(rng.uniform(100, S[3] - 100)))
        cal = cal.with_click(p, synth.feat(p, noise=0.01, rng=rng)) or cal
    assert cal.calibrated and _miss(cal, (1800, 400)) < min(before, 150)


def test_click_far_from_prediction_is_rejected():
    cal = _calibrate(np.random.default_rng(5))
    # смотрел в левый верхний угол, а кликнул в правый нижний — клик не про взгляд
    assert cal.with_click((2500, 1550), synth.feat((60, 60))) is None
    assert cal.with_click((80, 70), synth.feat((60, 60))) is not None


def test_click_features_take_frames_just_before_click():
    hist = [(t / 30, {**synth.feat((100, 100)), "gy": float(t)}) for t in range(60)]  # gy = номер кадра
    f = click_features(hist, 1.0)  # кадры с 0.65 до 0.95 с → номера 20..28
    assert 20 <= f["gy"] <= 28
    assert click_features(hist[:5], 1.0) is None  # кадров перед кликом нет


def test_calibration_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    cal = _calibrate(rng, head_move=False)
    cal = cal.with_click((640, 400), synth.feat((640, 400))) or cal
    cal.check_px = 42.0
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json", S)
    assert len(again.clicks) == len(cal.clicks) and len(again.points) == len(cal.points)
    assert again.check_px == 42.0
    f = synth.feat((300, 200))
    assert np.allclose(cal.predict(f), again.predict(f), atol=0.5)
    assert Calibration.load(tmp_path / "c.json", (0, 0, 1920, 1080)) is None  # другой экран
    assert Calibration.load(tmp_path / "missing.json") is None
    (tmp_path / "old.json").write_text('{"screen": [0, 0, 1, 1], "models": {}}', encoding="utf-8")
    assert Calibration.load(tmp_path / "old.json") is None  # старая версия


@pytest.mark.skipif(not EYE_MODEL.exists(), reason="нет модели лица")
def test_eye_model_recovers_eye_rotation():
    """Лицо в позе → зрачок на кадре → EyeModel восстанавливает поворот глаз относительно головы."""
    from gaze_check.eyes import IRIS_RADIUS
    mesh = canonical_mesh(EYE_MODEL)
    assert mesh.shape == (468, 3)
    model = EyeModel(mesh)
    w, h = 1280, 720
    f_px = h / (2 * math.tan(math.radians(63) / 2))
    def px(p):
        return w / 2 + f_px * p[0] / -p[2], h / 2 - f_px * p[1] / -p[2]

    for head, yaw, pitch in [(np.eye(3), 0.1, -0.2), (synth.rot_x(20) @ synth.rot_y(-15), -0.15, 0.05)]:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = head, (2.0, -6.0, -45.0)
        pts = np.zeros((478, 2))
        hdir = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
        for name, (c1, c2, iris) in EYES.items():
            for corner in (c1, c2):
                pts[corner] = px(head @ mesh[corner] + matrix[:3, 3])
            c = head @ (model.mids[name] + np.array([0, 0, -eyes.EYE_DEPTH])) + matrix[:3, 3]
            pts[iris] = px(c + IRIS_RADIUS * (head @ hdir))
        out, _, _ = model(pts, matrix, w, h)
        assert abs(out["gy"] - yaw) < 0.01 and abs(out["gp"] - pitch) < 0.01, (out["gy"], out["gp"])
        # из сырых данных кадра — те же признаки (для переобучения без новой калибровки)
        again = model.from_raw(EyeModel.raw(pts, matrix, w, h))
        assert abs(again["gy"] - out["gy"]) < 1e-3 and abs(again["gp"] - out["gp"]) < 1e-3


def test_fixations_order_and_gap():
    fx = Fixations(dispersion=60, min_ms=100)
    t = 0.0
    for x, y, n in ((100, 100, 10), (900, 500, 20), (905, 505, 3)):
        for _ in range(n):
            fx.add(t, x, y)
            t += 0.033
    fx.gap()
    for _ in range(2):  # 66 мс — не фиксация
        fx.add(t, 300, 300)
        t += 0.033
    found = fx.finish()
    assert [f["order"] for f in found] == [1, 2]
    assert (found[0]["x"], found[1]["x"]) == (100, 901)
    assert found[1]["duration_ms"] > found[0]["duration_ms"]


def test_one_euro_smooths_jitter_but_follows_jump():
    f = OneEuro()
    rng = np.random.default_rng(2)
    out = [f(i / 30, (500 + rng.normal(0, 20), 500)) for i in range(60)]
    assert np.std([o[0] for o in out[30:]]) < 12
    for i in range(60, 90):
        last = f(i / 30, (1500, 500))
    assert abs(last[0] - 1500) < 50


# ---------- смотрит ли в экран ----------

def test_screen_gaze_by_point():
    g = ScreenGaze(window=0.0)
    f = synth.feat((0, 0))
    assert g.update(0.0, f, True, (900, 500), S)[0] == "looking at screen"
    assert g.update(0.1, f, True, (S[2] + 100, 500), S)[0] == "looking at screen"  # запас у края
    assert g.update(0.2, f, True, (S[2] + 800, 500), S)[0] == "looking away"
    assert g.update(0.3, f, True, (900, S[3] + 600), S)[0] == "looking away"  # вниз, на телефон
    assert g.update(0.4, None, False)[0] == "face not visible"


def test_screen_gaze_without_point_uses_usual_pose():
    g = ScreenGaze(window=0.0)
    rng = np.random.default_rng(8)
    f = synth.feat((1280, 800))
    t = 0.0
    for _ in range(40):  # обычная работа: голова почти прямо
        t += 0.5
        assert g.update(t, {**f, "yaw": rng.normal(0, 0.03)}, True)[0] == "looking at screen"
    assert g.update(t + 0.5, {**f, "yaw": 0.8}, True)[0] == "looking away"  # ~46° в сторону
    assert g.update(t + 1.0, {**f, "pitch": -0.7}, True)[0] == "looking away"  # голову вниз
    assert g.update(t + 1.5, f, True)[0] == "looking at screen"


def test_screen_gaze_blink_keeps_label_long_closed_is_away():
    g = ScreenGaze(window=0.0)
    f = synth.feat((0, 0))
    assert g.update(0.0, f, True, (500, 500), S)[0] == "looking at screen"
    assert g.update(0.2, None, True)[0] == "looking at screen"  # моргнул
    assert g.update(1.0, None, True)[0] == "looking away"  # глаз не видно почти секунду


def test_screen_gaze_majority_smooths_single_frames():
    g = ScreenGaze(window=0.8)
    f = synth.feat((0, 0))
    t = 0.0
    for i in range(24):  # 0.8 с отвёрнут, среди кадров один «в экран»
        t += 1 / 30
        g.update(t, f, True, (500, 500) if i == 12 else (6000, 500), S)
    label, share = g.update(t + 1 / 30, f, True, (500, 500), S)
    assert label == "looking away" and share > 0.8


def test_lead_eye_angles_and_relead(tmp_path):
    from gaze_check.eyes import lead_angles
    f = {"gy": 0.0, "gp": 0.0, "gy_l": 0.2, "gp_l": 0.1, "gy_r": -0.1, "gp_r": 0.3}
    assert lead_angles(f, "left") == (0.2, 0.1) and lead_angles(f, "right") == (-0.1, 0.3)
    assert np.allclose(lead_angles(f, "both"), (0.05, 0.2))
    rng = np.random.default_rng(12)
    # правый глаз отклонён и «гуляет»; левый — ровно куда смотрит
    samples = []
    for p in calibration_points(S[2], S[3]):
        for _ in range(15):
            g = synth.feat(p, noise=0.005, rng=rng)
            g.update(gy_l=g["gy"], gp_l=g["gp"], gy_r=g["gy"] + 0.28 + rng.normal(0, 0.05), gp_r=g["gp"])
            g["gy"], g["gp"] = lead_angles(g, "both")
            samples.append((p, g))
    cal = Calibration.fit(S, samples, MM)
    left = cal.relead("left")
    assert left.lead == "left" and all(f["gy"] == f["gy_l"] for _, f in left.points)
    assert left.error_px < cal.error_px  # по ровному глазу точнее
    left.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json", S).lead == "left"


def test_open_eyes_works_with_lead_eye_alone():
    from gaze_check.eyes import open_eyes
    both, right_shut, left_shut = {"left": 0.3, "right": 0.3}, {"left": 0.3, "right": 0.05}, {"left": 0.04, "right": 0.3}
    assert open_eyes(both, "left") == "both" and open_eyes(both, "both") == "both"
    assert open_eyes(right_shut, "left") == "left"  # второй закрыт — работаем по ведущему
    assert open_eyes(right_shut, "both") == "left"  # ведущий не выбран — по тому, что открыт
    assert open_eyes(right_shut, "right") is None  # закрыт сам ведущий — моргание
    assert open_eyes(left_shut, "left") is None and open_eyes(left_shut, "right") == "right"
    assert open_eyes({"left": 0.04, "right": 0.05}, "both") is None
    assert open_eyes({"left": 0.35, "right": 0.14}, "left") == "left"  # второй прищурен/заслонён наполовину


def test_open_eyes_by_each_eyes_own_level():
    """Глаза разного разреза (0.14 против 0.26): узкий открытый глаз — не «закрытый»."""
    from gaze_check.eyes import open_eyes
    typical = {"left": 0.27, "right": 0.15}
    assert open_eyes({"left": 0.26, "right": 0.12}, "left", typical) == "both"
    assert open_eyes({"left": 0.26, "right": 0.03}, "left", typical) == "left"  # правый закрыли
    assert open_eyes({"left": 0.05, "right": 0.14}, "left", typical) is None  # моргнул ведущим
    # смотрит вниз: веки опустились у обоих — это не закрытие
    assert open_eyes({"left": 0.17, "right": 0.085}, "left", typical) == "both"
    assert open_eyes({"left": 0.26, "right": 0.07}, "left", typical) == "left"  # правый прищурен/заслонён
