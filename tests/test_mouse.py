import math

import numpy as np

from gaze_check.mouse import JOY_SPEED_DEFAULT, JOY_SPEEDS, Cursor, Fixation, GazeMouse, Gestures

FPS = 30


def run(gestures, pattern, t0=0.0):
    """pattern: [(секунд, закрыты ли)] → события по порядку."""
    events, t = [], t0
    for secs, closed in pattern:
        for _ in range(max(1, round(secs * FPS))):
            t += 1 / FPS
            events += gestures.update(t, 0.9 if closed else 0.05)
    return [e for e in events if e not in ("closed", "opened")], t


def test_double_blink_is_left_click():
    ev, _ = run(Gestures(), [(0.5, False), (0.12, True), (0.25, False), (0.12, True), (0.8, False)])
    assert ev == ["left"]


def test_triple_blink_opens_menu_without_click():
    ev, _ = run(Gestures(), [(0.5, False), (0.12, True), (0.25, False), (0.12, True), (0.25, False),
                             (0.12, True), (1.0, False)])
    assert ev == ["menu"]


def test_single_and_slow_double_blinks_do_nothing():
    assert run(Gestures(), [(0.5, False), (0.12, True), (1.0, False)])[0] == []
    assert run(Gestures(), [(0.5, False), (0.12, True), (1.2, False), (0.12, True), (0.5, False)])[0] == []


def test_long_blinks_are_not_double_blink():
    assert run(Gestures(), [(0.5, False), (0.6, True), (0.2, False), (0.6, True), (0.5, False)])[0] == []


def test_closed_two_to_three_seconds_is_right_click_with_beep():
    ev, _ = run(Gestures(), [(0.5, False), (2.4, True), (0.5, False)])
    assert ev == ["armed", "right"]


def test_closed_three_to_five_seconds_is_menu_longer_cancelled():
    ev, _ = run(Gestures(), [(0.5, False), (4.0, True), (0.5, False)])
    assert ev == ["armed", "menu_armed", "menu"]
    ev, _ = run(Gestures(), [(0.5, False), (6.0, True), (0.5, False)])
    assert ev == ["armed", "menu_armed", "cancel"]


def test_cooldown_after_click_and_face_lost_resets():
    g = Gestures()
    ev, t = run(g, [(0.5, False), (0.12, True), (0.25, False), (0.12, True), (0.6, False)])
    assert ev == ["left"]
    ev, t = run(g, [(0.1, True), (0.2, False), (0.1, True), (0.5, False)], t)  # сразу после клика — тихо
    assert ev == []
    assert g.update(t + 0.1, 0.9) == ["closed"] and g.update(t + 1.0, None) == []  # лицо пропало
    assert g.update(t + 3.0, 0.05) == []  # не «открылись после 3 с»


def test_fixation_ignores_jitter_and_single_outliers_but_follows_saccade():
    rng = np.random.default_rng(0)
    fix = Fixation(radius_px=110)
    t = 0.0
    for _ in range(30):  # смотрит в (500, 500), дрожь ±40 px
        t += 1 / FPS
        target = fix.update(t, (500 + rng.normal(0, 40), 500 + rng.normal(0, 40)))
    assert math.hypot(target[0] - 500, target[1] - 500) < 25  # центр фиксации, не последняя точка
    t += 1 / FPS
    assert math.hypot(*(np.array(fix.update(t, (1500, 500))) - target)) < 5  # один выброс — стоим
    for _ in range(4):  # перевёл взгляд и держит
        t += 1 / FPS
        target = fix.update(t, (1500 + rng.normal(0, 30), 800 + rng.normal(0, 30)))
    assert math.hypot(target[0] - 1500, target[1] - 800) < 40


class FakeMouse:
    def __init__(self):
        self.pos = (0, 0)
        self.t = 0.0

    def get(self):
        return self.pos

    def set(self, x, y):
        self.pos = (x, y)


def test_cursor_glides_smoothly_and_yields_to_hand():
    m = FakeMouse()
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, mode="direct")
    c.target = (1000, 0)
    xs = []
    for _ in range(10):  # 0.1 с
        m.t += 0.01
        c.step(0.01)
        xs.append(m.pos[0])
    assert all(b >= a for a, b in zip(xs, xs[1:])) and 500 < xs[-1] < 1000  # плавно, не прыжком
    for _ in range(50):
        m.t += 0.01
        c.step(0.01)
    assert abs(m.pos[0] - 1000) <= 1
    m.pos = (300, 300)  # рука сдвинула мышь
    m.t += 0.01
    c.step(0.01)
    assert m.pos == (300, 300)
    c.target = (2000, 1000)
    for _ in range(900):  # 9 с после руки взгляд молчит
        m.t += 0.01
        c.step(0.01)
    assert m.pos == (300, 300)
    for _ in range(200):  # 10 с прошло — снова за взглядом
        m.t += 0.01
        c.step(0.01)
    assert abs(m.pos[0] - 2000) <= 2


def test_gaze_mouse_clicks_where_cursor_was_before_blinks():
    m = FakeMouse()
    clicks = []
    cursor = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, mode="direct")
    gm = GazeMouse((0, 0, 2560, 1600), 74.0, click=lambda kind: clicks.append((kind, m.pos)), cursor=cursor)
    t = 0.0

    def frames(secs, point, blink):
        nonlocal t
        for _ in range(round(secs * FPS)):
            t += 1 / FPS
            m.t = t
            gm.update(t, point, blink)
            for _ in range(3):
                cursor.step(1 / 90)

    frames(1.0, (800, 600), 0.05)
    assert math.hypot(m.pos[0] - 800, m.pos[1] - 600) < 5
    # моргает дважды; пока веки закрываются, точка взгляда уезжает вниз — курсор не должен
    frames(0.1, (800, 900), 0.9)
    frames(0.2, (800, 600), 0.05)
    frames(0.1, (800, 900), 0.9)
    frames(0.8, (800, 600), 0.05)  # клик — через 0.5 с, если третьего моргания не было
    assert [k for k, _ in clicks] == ["left"]
    assert math.hypot(clicks[0][1][0] - 800, clicks[0][1][1] - 600) < 5


def test_thresholds_follow_persons_open_eye_level():
    """Открытые глаза бывают на уровне ~0.28–0.44: это не должно считаться закрытыми."""
    g = Gestures()
    rng = np.random.default_rng(1)
    t, events = 0.0, []
    for _ in range(300):  # 10 с с открытыми глазами, уровень повыше обычного
        t += 1 / FPS
        events += g.update(t, float(np.clip(rng.normal(0.36, 0.05), 0, 1)))
    gestures = {"left", "right", "menu", "armed", "menu_armed", "cancel"}
    assert not gestures & set(events) and 0.6 <= g.closed_level <= 0.72  # шум — не жест
    for secs, level in ((0.12, 0.85), (0.25, 0.36), (0.12, 0.85), (0.8, 0.36)):
        for _ in range(round(secs * FPS)):
            t += 1 / FPS
            events += g.update(t, level)
    assert "left" in events


def _gaze_mouse():
    m = FakeMouse()
    cursor = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, mode="direct")
    return GazeMouse((0, 0, 2560, 1600), 74.0, click=lambda kind: None, cursor=cursor), m


def feed(gm, m, t, pattern, rng=None):
    """pattern: [(секунд, точка, моргание)] → (события GazeMouse, t). rng — дрожь точки ±30 px."""
    out = []
    for secs, point, blink in pattern:
        for _ in range(round(secs * FPS)):
            t += 1 / FPS
            m.t = t
            p = point if rng is None else (point[0] + rng.normal(0, 30), point[1] + rng.normal(0, 30))
            out += gm.update(t, p, blink)
    return out, t


def test_steady_gaze_after_triple_blink_opens_task_view():
    """Три моргания, затем взгляд на месте 3 с — "hold" (Win+Tab), один раз; обычное моргание не мешает."""
    gm, m = _gaze_mouse()
    rng = np.random.default_rng(2)
    _, t = feed(gm, m, 0.0, [(1.0, (800, 600), 0.05)], rng)
    gm.arm_hold(t)
    ev, t = feed(gm, m, t, [(1.5, (800, 600), 0.05), (0.12, (800, 900), 0.9), (1.3, (800, 600), 0.05)], rng)
    assert ev == []  # 2,9 с — рано
    ev, t = feed(gm, m, t, [(0.4, (800, 600), 0.05)], rng)
    assert ev == ["hold"]
    ev, t = feed(gm, m, t, [(5.0, (800, 600), 0.05)], rng)
    assert ev == []


def test_task_view_gives_time_to_settle_gaze_after_blinks():
    gm, m = _gaze_mouse()
    _, t = feed(gm, m, 0.0, [(1.0, (800, 600), 0.05)])
    gm.arm_hold(t)
    ev, _ = feed(gm, m, t, [(0.8, (1500, 900), 0.05), (3.3, (800, 600), 0.05)])  # перевёл взгляд и держит
    assert ev == ["hold"]


def test_task_view_cancelled_when_gaze_wanders():
    gm, m = _gaze_mouse()
    _, t = feed(gm, m, 0.0, [(1.0, (800, 600), 0.05)])
    gm.arm_hold(t)
    ev, _ = feed(gm, m, t, [(1.0, (800 + 600 * (i % 2), 600), 0.05) for i in range(6)])
    assert ev == ["hold_lost"]


def test_click_cancels_task_view_wait():
    gm, m = _gaze_mouse()
    _, t = feed(gm, m, 0.0, [(1.0, (800, 600), 0.05)])
    gm.arm_hold(t)
    ev, _ = feed(gm, m, t, [(0.12, (800, 600), 0.9), (0.25, (800, 600), 0.05), (0.12, (800, 600), 0.9),
                            (4.0, (800, 600), 0.05)])
    assert ev == []  # двойное моргание — клик, Win+Tab уже не ждём


def test_half_closed_lids_do_not_move_cursor():
    """Веки прикрываются (коэффициент выше обычного уровня человека, но ниже «открыты») — точка взгляда
    уезжает вниз на 5–15 см (замер tools/blink_drift.py); курсор стоять не должен."""
    gm, m = _gaze_mouse()
    rng = np.random.default_rng(3)
    _, t = feed(gm, m, 0.0, [(3.0, (800, 600), 0.11)], rng)  # обычный уровень человека ~0.11
    start = gm.cursor.target
    for blink, y in ((0.22, 1000), (0.28, 1050), (0.4, 1100), (0.3, 1040), (0.24, 1010), (0.19, 900)):
        t += 1 / FPS
        m.t = t
        gm.update(t, (800, y), blink)
        assert math.hypot(gm.cursor.target[0] - start[0], gm.cursor.target[1] - start[1]) < 40
    _, t = feed(gm, m, t, [(1.0, (800, 600), 0.11)], rng)
    assert math.hypot(gm.cursor.target[0] - start[0], gm.cursor.target[1] - start[1]) < 40


def test_cursor_still_follows_gaze_with_high_open_eye_level():
    """Уровень открытых глаз бывает ~0.36 ± 0.05 — это не «веки прикрыты», курсор идёт за взглядом.
    Жесты считаются с запуска программы, к включению мыши уровень человека уже известен (10 с)."""
    gm, m = _gaze_mouse()
    rng = np.random.default_rng(4)
    t = 0.0
    for secs, point in ((10.0, (800, 600)), (1.0, (1800, 1000))):
        for _ in range(round(secs * FPS)):
            t += 1 / FPS
            m.t = t
            gm.update(t, point, float(np.clip(rng.normal(0.36, 0.05), 0, 1)))
    assert math.hypot(gm.cursor.target[0] - 1800, gm.cursor.target[1] - 1000) < 40



def _run_levels(g, pattern, fps=13.6, t=0.0):
    """pattern: [(секунд, уровень)] → события (без closed/opened)."""
    ev = []
    for secs, level in pattern:
        for _ in range(max(1, round(secs * fps))):
            t += 1 / fps
            ev += [e for e in g.update(t, level) if e not in ("closed", "opened")]
    return ev, t


def test_weak_quick_blinks_of_this_person_make_double_blink():
    """Камера 13.6 к/с, моргание — 1 кадр с коэффициентом 0.3–0.5 при уровне 0.11 (запись blink_*.json):
    раньше порог 0.5 не ловил ни одного двойного."""
    g = Gestures()
    ev, t = _run_levels(g, [(10.0, 0.11)])  # уровень человека
    assert ev == [] and g.blink_level < 0.3
    ev, t = _run_levels(g, [(0.07, 0.37), (0.3, 0.11), (0.07, 0.33), (1.0, 0.11)], t=t)
    assert ev == ["left"]


def test_looking_down_is_not_a_gesture():
    """Взгляд вниз: веки опускаются умеренно и надолго (~0.35) — ни правого клика, ни меню, ни писков."""
    g = Gestures()
    ev, t = _run_levels(g, [(10.0, 0.11), (4.0, 0.35), (1.0, 0.11)])
    assert ev == []
    ev, t = _run_levels(g, [(2.4, 0.9), (1.0, 0.11)], t=t)  # по-настоящему закрыл — правый клик
    assert ev == ["armed", "right"]


def test_double_blink_merged_into_one_closure():
    """Два моргания слились в одно закрытие (глаз приоткрылся на кадр, но не до «открыт») — это двойное."""
    g = Gestures()
    ev, t = _run_levels(g, [(10.0, 0.11)])
    ev, t = _run_levels(g, [(0.07, 0.27), (0.07, 0.26), (0.07, 0.21), (0.07, 0.52), (0.07, 0.46),
                            (0.07, 0.21), (1.0, 0.11)], t=t)
    assert ev == ["left"]
    ev, t = _run_levels(g, [(3.0, 0.11), (0.07, 0.45), (0.07, 0.40), (1.0, 0.11)], t=t)  # одиночное — ничего
    assert ev == []


def test_blinks_right_after_long_closure_are_not_a_click():
    """Закрыл глаза на 2,8 с (правый клик) — через 2 с глаза сами моргнули дважды: это не левый клик."""
    g = Gestures()
    ev, t = _run_levels(g, [(10.0, 0.11), (2.8, 0.9), (2.0, 0.11)])
    assert ev == ["armed", "right"]
    ev, t = _run_levels(g, [(0.07, 0.4), (0.3, 0.11), (0.07, 0.4), (1.5, 0.11)], t=t)
    assert ev == []
    ev, t = _run_levels(g, [(0.07, 0.4), (0.3, 0.11), (0.07, 0.4), (1.5, 0.11)], t=t)  # позже — клик
    assert ev == ["left"]



def _joy(m, c, target, secs, dt=0.01):
    c.target = target
    for _ in range(round(secs / dt)):
        m.t += dt
        c.step(dt)
    return m.pos


def test_joystick_ignores_gaze_jitter_near_cursor():
    m = FakeMouse()
    m.pos = (1280, 800)
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, px_per_cm=74.0)
    rng = np.random.default_rng(0)
    for _ in range(200):  # 2 с взгляд дрожит в пределах ~2 см вокруг курсора
        _joy(m, c, (1280 + rng.normal(0, 60), 800 + rng.normal(0, 60)), 0.01)
    assert m.pos == (1280, 800)


def test_joystick_crosses_screen_in_about_five_seconds_and_stops():
    m = FakeMouse()
    m.pos = (2400, 150)  # справа вверху
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, px_per_cm=74.0)
    target = (150, 1450)  # смотрит в левый нижний угол, ~38 см
    p1 = _joy(m, c, target, 1.0)
    moved1 = math.hypot(p1[0] - 2400, p1[1] - 150) / 74
    assert 0.5 < moved1 < 6  # в первую секунду — медленно
    p5 = _joy(m, c, target, 4.5)
    assert math.hypot(p5[0] - target[0], p5[1] - target[1]) / 74 < 2.0  # за ~5 с доехал
    p6 = _joy(m, c, target, 1.0)
    assert math.hypot(p6[0] - p5[0], p6[1] - p5[1]) / 74 < 1.0  # остановился, не проскочил


def test_joystick_turn_restarts_acceleration():
    m = FakeMouse()
    m.pos = (1280, 800)
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, px_per_cm=74.0)
    _joy(m, c, (2500, 800), 2.0)  # разогнался вправо
    before = m.pos
    p = _joy(m, c, (m.pos[0], 100), 0.3)  # резко посмотрел вверх — снова медленно
    assert abs(p[0] - before[0]) < 20 and before[1] - p[1] < 0.3 * 74 * 4


def _time_to_reach(speed, dist_cm=10.0, px=74.0):
    m = FakeMouse()
    m.pos = (400, 800)
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, px_per_cm=px, speed=speed)
    target, t, far = (400 + dist_cm * px, 800), 0.0, 0.0
    while math.hypot(m.pos[0] - target[0], m.pos[1] - target[1]) / px > 2.0 and t < 10:
        _joy(m, c, target, 0.05)
        t += 0.05
        far = max(far, m.pos[0])
    return t, (far - target[0]) / px


def test_joystick_speed_setting():
    """Скорость джойстика из меню: быстрее — раньше доезжает, но не проскакивает цель."""
    times = {k: _time_to_reach(k) for k in JOY_SPEEDS}
    order = [times[k][0] for k in sorted(JOY_SPEEDS)]
    assert order == sorted(order, reverse=True) and len(set(order)) == len(order)
    assert times[1.0][0] > 1.8 and times[JOY_SPEED_DEFAULT][0] < 1.6 and times[4.5][0] < 1.0
    assert all(over <= 0.01 for _, over in times.values())  # не проскочил


def test_joystick_speed_changes_on_the_fly_and_ignores_jitter():
    m = FakeMouse()
    m.pos = (1280, 800)
    c = Cursor(get_pos=m.get, set_pos=m.set, clock=lambda: m.t, px_per_cm=74.0, speed=4.5)
    rng = np.random.default_rng(1)
    for _ in range(200):  # самая большая скорость — дрожание взгляда всё равно не двигает
        _joy(m, c, (1280 + rng.normal(0, 60), 800 + rng.normal(0, 60)), 0.01)
    assert m.pos == (1280, 800)
    c.speed = 0.5
    p = _joy(m, c, (2400, 800), 1.0)
    assert (p[0] - 1280) / 74 < 2.5  # сменили на «очень медленно» — едет медленно
