"""Калибровка взгляда: полноэкранные точки по очереди, потом точки проверки.

9 точек — личные поправки 3D-модели; центр с покачиванием головы — чтобы модель отличала
поворот головы от поворота глаз (без этого после смены позы точка уезжает); 5 точек проверки
между точками калибровки — честная точность, которой нет в самой калибровке."""
import time

import cv2
import numpy as np

from .app import _patch, _put
from .eyes import Calibration, calibration_points, check_points, edge_points

WINDOW = "Gaze Check calibration"
SETTLE, COLLECT = 0.9, 1.3  # секунд: перевести взгляд, потом собирать
HEAD_MOVE = 5.0  # секунд смотреть в точку, покачивая головой
MIN_SAMPLES = 8
MAX_GOOD_CM = 4.0  # хуже на точках проверки — калибровку не сохраняем


def primary_monitor():
    """(left, top, width, height) основного монитора в физических пикселях."""
    import mss
    with mss.MSS() as sct:
        m = sct.monitors[1]
        return m["left"], m["top"], m["width"], m["height"]


def screen_mm(monitor=None):
    """Физический размер основного экрана, мм (из EDID через Windows); если неизвестен —
    по типичной плотности ноутбука (~150 точек на дюйм)."""
    try:
        import ctypes
        hdc = ctypes.windll.user32.GetDC(0)
        w, h = ctypes.windll.gdi32.GetDeviceCaps(hdc, 4), ctypes.windll.gdi32.GetDeviceCaps(hdc, 6)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if w > 100 and h > 60:
            return float(w), float(h)
    except (AttributeError, OSError):
        pass
    _, _, pw, ph = monitor or primary_monitor()
    return pw * 25.4 / 150, ph * 25.4 / 150


def pose_hint(eye):
    """(хорошо ли, текст) — где лицо в кадре камеры. Угол крышки сам по себе не важен (3D-модель
    его учитывает), но у края кадра точки лица хуже, а при сильном ракурсе снизу веки
    закрывают зрачок."""
    if not eye.get("face"):
        return False, "Не вижу лица — сядьте напротив камеры, включите свет"
    marks, frame = eye.get("marks") or [], eye.get("frame")
    if not marks or not frame:
        return bool(eye.get("valid")), "Глаза видны" if eye.get("valid") else "Не вижу зрачков"
    x, y = np.mean(marks, axis=0)
    fx, fy = x / frame[0], y / frame[1]
    if fy < 0.25:
        return False, "Глаза у верхнего края кадра — откиньте крышку чуть назад"
    if fy > 0.7:
        return False, "Глаза низко в кадре — наклоните крышку к себе"
    if not 0.25 <= fx <= 0.75:
        return False, "Сядьте ближе к середине экрана"
    if not eye.get("valid"):
        return False, "Не вижу зрачков — не щурьтесь, уберите блики с очков"
    return True, "Положение хорошее"


class Screen:
    def __init__(self, monitor):
        self.left, self.top, self.w, self.h = monitor
        self.font = max(18, self.h // 36)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW, self.left, self.top)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def canvas(self):
        return np.full((self.h, self.w, 3), 24, np.uint8)

    def text(self, img, lines, y=None, color=(235, 235, 235)):
        y = y if y is not None else self.h // 3
        for line in lines:
            if not line:
                continue
            patch = _patch(line, self.font, False, (24, 24, 24), color)
            _put(img, patch, (self.w - patch.shape[1]) // 2, y)
            y += patch.shape[0] + self.font // 2

    def show(self, img, wait=15):
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(wait) & 0xFF
        closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        return 27 if closed else key

    def close(self):
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass
        cv2.waitKey(1)


def _target(img, p, progress, color=(40, 40, 230)):
    r = int(28 - 18 * progress)
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def run_calibration(engine):
    """Показывает точки, собирает признаки глаз из движка, строит калибровку.
    Возвращает Calibration (уже сохранённую движком) или None при отмене.

    Этапы: как обычно (9 точек) → качать головой (центр и угол) → ближе (5) → дальше (5) →
    как обычно (5 точек проверки). Разные позы и расстояния — чтобы точка не уезжала, когда
    человек двигается и отсаживается (на записи эксперимента: 2 см вместо 3–5)."""
    monitor = primary_monitor()
    size = screen_mm(monitor)
    scr = Screen(monitor)
    w, h = scr.w, scr.h
    points = calibration_points(w, h)
    checks = check_points(w, h)
    center = points[0]
    five = [center] + edge_points(w, h)  # ближе и дальше — тоже до самых углов
    heads = [center, (round(0.2 * w), round(0.25 * h))]
    try:
        while True:
            lead = {"left": "левым", "right": "правым"}.get(getattr(engine, "lead", "both"))
            if not _prompt(scr, engine, [
                    "Калибровка взгляда: 5 коротких этапов, около полутора минут",
                    "Сядьте как обычно работаете. Смотрите на красную точку, пока она не сменится.",
                    "Голову держите спокойно, двигаются только глаза.",
                    ("Можно держать второй глаз закрытым — калибровка по ведущему (%s) глазу "
                     "добавится к двухглазой." % lead) if lead else ""]):
                return None
            samples = _collect_all(scr, engine, points)
            if samples is None:
                return None
            moving = []
            for p in heads:  # смотреть в точку и покачивать головой: глаза поворачиваются навстречу голове
                got = _collect_point(scr, engine, p, collect=HEAD_MOVE, lines=[
                    "Смотрите на точку и медленно покачайте головой:", "влево-вправо, потом вверх-вниз"])
                if got is None:
                    return None
                moving += [(p, f) for f in got]
            for text in ("Придвиньтесь ближе к экрану — примерно на ладонь.",
                         "Теперь отодвиньтесь или откиньтесь — на ладонь дальше обычного."):
                if not _prompt(scr, engine, [text, "Смотрите на точки так же, 5 точек."]):
                    return None
                got = _collect_all(scr, engine, five)
                if got is None:
                    return None
                samples += got
            if len({s[0] for s in samples}) < 6:
                img = scr.canvas()
                scr.text(img, ["Глаза почти не было видно — калибровка не получилась.",
                               "R — ещё раз, Esc — выход"])
                key = _wait_keys(scr, img, (ord("r"), 27))
                if key == 27:
                    return None
                continue
            old = engine.calibration
            same = old is not None and tuple(old.screen) == tuple(monitor)
            clicks = old.clicks if same else []
            if same and one_eye_session(samples) and old.netmap is not None:
                # с закрытым вторым глазом сеть не считается: такие кадры — 3D-модели по ведущему
                # глазу, а двухглазую калибровку (сеть) не выбрасываем — добавляем к ней
                samples = (list(old.points) + samples)[-MAX_POINT_FRAMES:]
                moving = list(old.moving) + moving
            cal = Calibration.fit(monitor, samples, size, clicks=clicks, moving=moving)
            if not _prompt(scr, engine, ["Сядьте как обычно. Последние 5 точек — проверка точности."]):
                return None
            raw_checks = []
            for p in checks:
                got = _collect_twice(scr, engine, p)
                if got is None:
                    return None
                raw_checks.append((p, got))
            cal.check(raw_checks)
            checked = [(p, np.median(cal.predict_many(got), axis=0)) for p, got in raw_checks if got]
            key = _review(scr, cal, points, samples, checked)
            if key == 13 and _good(cal):
                engine.set_calibration(cal)
                return cal
            if key == 27:
                return None
    finally:
        engine.collect = None
        scr.close()


MAX_POINT_FRAMES = 3000  # кадров точек калибровки храним не больше (старые позы вытесняются)


def run_quick(engine):
    """Быстрая подстройка (~8 с): центр и 4 угла в той позе, в которой человек сейчас; кадры
    добавляются к калибровке — модель узнаёт ещё одну позу. None — отмена."""
    monitor = primary_monitor()
    old = engine.calibration
    if old is None or not old.points or tuple(old.screen) != tuple(monitor):
        return run_calibration(engine)  # подстраивать нечего — полная калибровка
    scr = Screen(monitor)
    w, h = scr.w, scr.h
    points = [(w // 2, h // 2)] + edge_points(w, h)
    try:
        if not _prompt(scr, engine, ["Быстрая подстройка под то, как вы сейчас сидите: 5 точек, ~8 секунд.",
                                     "Смотрите на красную точку, пока она не сменится."]):
            return None
        got = _collect_all(scr, engine, points)
        if got is None:
            return None
        if len({p for p, _ in got}) < 4:
            return None  # глаз почти не было видно — не портим калибровку
        samples = (list(old.points) + got)[-MAX_POINT_FRAMES:]
        cal = Calibration.fit(monitor, samples, old.size_mm, clicks=old.clicks, moving=old.moving)
        cal.check_px, cal.checks, cal.jitter_px = old.check_px, old.checks, old.jitter_px
        engine.set_calibration(cal)
        return cal
    finally:
        engine.collect = None
        scr.close()


def one_eye_session(samples):
    """Калибровку проходили с закрытым вторым глазом: у большинства кадров нет признаков сети."""
    return sum(f.get("net") is not None for _, f in samples) < 0.3 * max(1, len(samples))


def _prompt(scr, engine, lines):
    """Экран-подсказка перед этапом: пробел — дальше (если глаза видны), Esc — отмена (False)."""
    while True:
        img = scr.canvas()
        eye = engine.state.snapshot()["eye"]
        ok, hint = pose_hint(eye)
        if eye.get("dist"):
            hint += " · до лица ~%d см" % eye["dist"]
        scr.text(img, list(lines) + ["Пробел — дальше, Esc — отмена"])
        scr.text(img, [hint], y=scr.h * 2 // 3, color=(90, 220, 90) if ok else (80, 80, 240))
        key = scr.show(img)
        if key == 27:
            return False
        if key == 32 and eye.get("valid"):
            return True


def _collect_all(scr, engine, points):
    """Точки по очереди → [(точка, признаки)] или None (отмена)."""
    out = []
    for p in points:
        got = _collect_twice(scr, engine, p)
        if got is None:
            return None
        out += [(p, f) for f in got]
    return out


def _collect_twice(scr, engine, p):
    """Точка; если глаз почти не было видно (моргал) — ещё раз. None — отмена."""
    got = []
    for _attempt in range(2):
        got = _collect_point(scr, engine, p)
        if got is None or len(got) >= MIN_SAMPLES:
            return got
    return []


def _collect_point(scr, engine, p, collect=None, lines=()):
    collect = COLLECT if collect is None else collect
    engine.collect = None
    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t >= SETTLE and engine.collect is None:
            engine.collect = []
        if t >= SETTLE + collect:
            got, engine.collect = engine.collect or [], None
            return [f for _, f in got if "gy" in f]
        img = scr.canvas()
        if lines:
            scr.text(img, lines, y=p[1] + scr.font * 3)
        _target(img, p, min(1.0, t / SETTLE))
        if scr.show(img, 10) == 27:
            return None


def _good(cal):
    """Калибровку хуже MAX_GOOD_CM на точках проверки не сохраняем: с ней хуже, чем без неё."""
    return cal.check_px is None or cal.check_px / cal.px_per_cm() <= MAX_GOOD_CM


def _review(scr, cal, points, samples, checked):
    """Цели и куда по калибровке попадает взгляд; Enter — сохранить (если калибровка годная)."""
    img = scr.canvas()
    cm = cal.px_per_cm()
    for p in points:
        preds = cal.predict_many([f for q, f in samples if q == p]) if any(q == p for q, _ in samples) else []
        cv2.circle(img, p, 10, (40, 40, 230), -1, cv2.LINE_AA)
        if len(preds):
            m = tuple(int(v) for v in np.median(preds, 0))
            cv2.line(img, p, m, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(img, m, 7, (90, 220, 90), -1, cv2.LINE_AA)
    for p, m in checked:
        m = (int(m[0]), int(m[1]))
        cv2.rectangle(img, (p[0] - 9, p[1] - 9), (p[0] + 9, p[1] + 9), (230, 160, 40), -1)
        cv2.line(img, p, m, (230, 160, 40), 1, cv2.LINE_AA)
        cv2.circle(img, m, 7, (90, 220, 90), -1, cv2.LINE_AA)
    lines = ["Красные — точки калибровки, синие квадраты — проверка, зелёные — куда попал взгляд."]
    if cal.check_px is not None:
        lines.append("Точность на проверке: ~%.1f см (%d px)" % (cal.check_px / cm, cal.check_px))
    if cal.jitter_px is not None:
        lines.append("Дрожание (точка гуляет, пока смотрите в одно место): ~%.1f см" % (cal.jitter_px / cm))
    if cal.error_px is not None:
        lines.append("На точках калибровки: ~%.1f см" % (cal.error_px / cm))
    if _good(cal):
        lines.append("Enter — сохранить, R — заново, Esc — отмена")
        scr.text(img, lines, y=scr.h * 3 // 5)
        return _wait_keys(scr, img, (13, ord("r"), 27))
    lines += ["Слишком неточно — не сохраняю. Смотрите на каждую точку, пока она не сменится,",
              "уберите телефон и руки от лица. R — ещё раз, Esc — отмена"]
    scr.text(img, lines, y=scr.h * 3 // 5, color=(120, 120, 250))
    return _wait_keys(scr, img, (ord("r"), 27))


def run_gaze_test(engine):
    """Проверка взгляда: на весь экран сетка зон и точка, куда сейчас смотрите.
    Esc — выход; C — калибровка (возвращает "calibrate")."""
    from .zones import COLS, ROWS, zone_cell, zone_of
    scr = Screen(primary_monitor())
    base = scr.canvas()
    for i in (1, 2):
        cv2.line(base, (scr.w * i // 3, 0), (scr.w * i // 3, scr.h), (70, 70, 70), 2)
        cv2.line(base, (0, scr.h * i // 3), (scr.w, scr.h * i // 3), (70, 70, 70), 2)
    for r, row in enumerate(ROWS):
        for c, col in enumerate(COLS):
            cx, cy = scr.w * (2 * c + 1) // 6, scr.h * (2 * r + 1) // 6
            cv2.circle(base, (cx, cy), 6, (110, 110, 110), -1, cv2.LINE_AA)
    trail = []
    try:
        while True:
            s = engine.state.snapshot()
            eye = s["eye"] or {}
            img = base.copy()
            point = eye.get("point")
            if point is not None:
                zone = zone_of(point, scr.w, scr.h)
                cell = zone_cell(zone)
                if cell:
                    r, c = cell
                    x0, y0 = scr.w * c // 3, scr.h * r // 3
                    cv2.rectangle(img, (x0 + 3, y0 + 3), (x0 + scr.w // 3 - 3, y0 + scr.h // 3 - 3), (60, 140, 60), 6)
                p = (int(min(max(point[0], 0), scr.w - 1)), int(min(max(point[1], 0), scr.h - 1)))
                trail = (trail + [p])[-20:]
                for a, b in zip(trail, trail[1:]):
                    cv2.line(img, a, b, (0, 170, 230), 2, cv2.LINE_AA)
                cv2.circle(img, p, 26, (40, 40, 230), 4, cv2.LINE_AA)
                cv2.circle(img, p, 6, (40, 40, 230), -1, cv2.LINE_AA)
                status = zone
            else:
                trail = []
                status = pose_hint(eye)[1]
            scr.text(img, [status, "%s · Esc — выход, C — калибровка" % (s["accuracy"] or "")],
                     y=scr.h - scr.font * 4)
            key = scr.show(img, 15)
            if key == 27:
                return None
            if key in (ord("c"), ord("C")):
                return "calibrate"
    finally:
        scr.close()


def _wait_keys(scr, img, keys):
    while True:
        key = scr.show(img, 30)
        if key in keys:
            return key
