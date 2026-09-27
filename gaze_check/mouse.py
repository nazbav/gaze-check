"""Управление мышью взглядом.

Курсор — в центр текущей фиксации взгляда (точки в пределах FIX_CM усредняются, так он не
дрожит), новая фиксация — когда взгляд ушёл дальше FIX_CM несколько кадров подряд; сам курсор
едет к цели плавно, ~100 раз в секунду. Двигаешь мышь рукой — взгляд отдаёт ей управление.

Жесты глазами (оба глаза, по коэффициентам моргания MediaPipe) — ловит Gestures:
  два коротких моргания подряд — левый клик (в меню взглядом — выбор плитки);
  три коротких моргания подряд — открыть/закрыть меню взглядом;
  глаза закрыты 2–3 с (на 2 с — писк) — правый клик;
  3–5 с (на 3 с — двойной писк) — открыть/закрыть меню взглядом;
  дольше 5 с — отмена (низкий писк).
Клик ставится туда, где курсор был до того, как закрылись глаза.

Три моргания, когда они не заняты голосом или меню, и затем взгляд на месте 3 с — Win+Tab (все окна):
программа зовёт arm_hold, GazeMouse отвечает событием "hold" или "hold_lost" (не удержал).
"""
import ctypes
import math
import threading
import time
from collections import deque

FIX_CM = 1.5  # радиус фиксации: взгляд ближе — это та же фиксация, курсор стоит
SACCADE_FRAMES = 3  # столько кадров подряд дальше FIX_CM — новая фиксация (одиночный выброс не двигает)
FIX_KEEP_S = 0.6  # точки фиксации за последние столько секунд усредняются
GLIDE_S = 0.07  # постоянная времени плавного движения курсора
MANUAL_PAUSE_S = 10.0  # рука двигала мышь — взгляд не перехватывает столько после последнего движения

# коэффициент моргания: закрыты выше, открыты ниже (гистерезис) — от обычного уровня человека
# с открытыми глазами (бывает ~0.3, а в 1% кадров до 0.45), но не ниже CLOSED/OPENED
CLOSED, OPENED = 0.5, 0.35
CLOSED_ABOVE, OPENED_ABOVE = 0.3, 0.15
BLINK_MIN_S, BLINK_MAX_S = 0.04, 0.6  # короткое моргание (при 13–15 к/с оно растягивается до ~0.5 с)
# два моргания подряд при 13–15 к/с часто сливаются в одно закрытие: глаз приоткрывается между
# ними на кадр, но не до «открыт» (запись: 0.27 → 0.21 → 0.52). Провал на такую долю высоты пика и
# новый подъём — второе моргание внутри того же закрытия.
DIP_SHARE = 0.3
# после долгого закрытия глаз они часто моргают пару раз сами — это не двойное моргание (запись:
# ложный клик через 2 с после закрытия на 2,8 с). Отсечка по глубине моргания не годится: когда
# человек моргает нарочно подряд, «обычная» глубина растёт вместе с нарочной (тройные: 0 из 3).
AFTER_LONG_S, LONG_CLOSURE_S = 2.5, 1.0
DOUBLE_GAP_S = 0.7  # между концом первого и началом второго моргания
TRIPLE_WAIT_S = 0.5  # после второго моргания ждём третье столько — чтобы отличить клик от меню
RIGHT_FROM_S, RIGHT_TO_S = 2.0, 3.0  # закрыты столько — правый клик
MENU_TO_S = 5.0  # от RIGHT_TO_S до стольких — меню взглядом; дольше — отмена
COOLDOWN_S = 0.8  # после клика жесты не ловятся
SETTLE_S = 0.15  # после открытия глаз взгляд ещё «плывёт» — не двигаем курсор
# веки прикрыты: коэффициент выше обычного уровня человека на столько — точка взгляда уезжает вниз
# на 5–15 см, при переходе на один глаз — до 45 см (замер tools/blink_drift.py); такие кадры курсор
# не берёт, и ещё LID_SETTLE_S после них. Моргание камера ~14 к/с часто ловит только полузакрытым.
LID_ABOVE = 0.07
LID_SPREAD = 2.5  # у кого уровень открытых глаз шумный — порог выше: столько размахов (p75 − медиана)
LID_SETTLE_S = 0.2
HOLD_S = 3.0  # три моргания, затем взгляд на месте столько — Win+Tab
HOLD_GRACE_S = 1.5  # сверх HOLD_S — чтобы остановить взгляд после морганий; не удержал — отмена


# Моргание — короткий ВСПЛЕСК над уровнем человека, а не порог 0.5: камера в полутьме даёт 15 к/с,
# моргание (100–150 мс) попадает в 1–2 кадра на середине, и коэффициент на замере доходит лишь
# до 0.27–0.57 (запись blink_*.json: порог 0.5 поймал 0 из 4 двойных, всплеск — 4 из 4).
BLINK_ABOVE, BLINK_NOISE_K = 0.12, 5.0  # начало моргания: уровень + max(0.12, 5·шум)
RELEASE_ABOVE, RELEASE_NOISE_K = 0.06, 2.5  # глаза снова открыты: уровень + max(0.06, 2.5·шум)
DELIBERATE_SHARE = 0.6  # долгий жест — если веки сомкнуты (выше closed_level) на такой доле кадров


class Gestures:
    """Коэффициент закрытия глаз по времени → события: "left", "right", "menu", "armed" (2 с — можно
    открывать для правого клика), "menu_armed" (3 с — для меню), "cancel" (5 с — отмена),
    "closed" (глаза закрылись), "opened".

    Короткое закрытие ловится по всплеску над уровнем человека; долгие жесты (правый клик, меню)
    засчитываются, только если веки действительно сомкнуты — взгляд вниз поднимает коэффициент
    умеренно и надолго, это не жест."""

    def __init__(self):
        self.closed_at = None
        self.last_blink_end = None
        self.blinks = 0  # коротких морганий подряд
        self.pending = None  # когда объявить двойное моргание, если третьего не будет
        self.armed = self.menu_armed = self.cancelled = False
        self.quiet_until = -1e9
        self.open_levels = deque(maxlen=300)  # коэффициент при открытых глазах, ~10 с
        self.closed_level, self.opened_level = CLOSED, OPENED
        self.blink_level, self.release_level = CLOSED, OPENED  # всплеск-моргание и «снова открыты»
        self.lid_level = OPENED  # выше — веки прикрыты, точка взгляда ненадёжна (от уровня человека)
        self.hi = self.n = 0  # кадров закрытия всего и с по-настоящему сомкнутыми веками
        self.base = 0.0  # уровень человека с открытыми глазами
        self.peak, self.dip, self.peaks = 0.0, None, 0  # пики внутри одного закрытия
        self.no_blinks_until = -1e9  # после долгого закрытия моргания в жесты не идут

    def _levels(self):
        if len(self.open_levels) >= 30:
            levels = sorted(self.open_levels)
            base, p75 = levels[len(levels) // 2], levels[len(levels) * 3 // 4]
            noise = sorted(abs(v - base) for v in levels)[len(levels) // 2]
            self.base = base
            self.closed_level = max(CLOSED, base + CLOSED_ABOVE)
            self.opened_level = max(OPENED, base + OPENED_ABOVE)
            self.blink_level = min(self.closed_level, base + max(BLINK_ABOVE, BLINK_NOISE_K * noise))
            self.release_level = min(self.blink_level, base + max(RELEASE_ABOVE, RELEASE_NOISE_K * noise))
            self.lid_level = min(self.opened_level, base + max(LID_ABOVE, LID_SPREAD * (p75 - base)))

    def _subpeaks(self, score):
        """Провал на DIP_SHARE высоты пика и новый подъём — ещё одно моргание в том же закрытии."""
        height = max(self.peak - self.base, 1e-6)
        if self.dip is None:
            if score >= self.peak:
                self.peak = score
            elif self.peak - score >= DIP_SHARE * height:
                self.dip = score
        elif score < self.dip:
            self.dip = score
        elif score - self.dip >= DIP_SHARE * height:
            self.peaks += 1
            self.peak, self.dip = score, None

    def update(self, t, score):
        """score — 0..1 (закрыты ≈1) или None (лица нет: жест сбрасывается). → список событий."""
        events = []
        if self.pending is not None and self.closed_at is None and t >= self.pending:
            self.pending, self.blinks = None, 0  # третьего моргания не было — это двойное
            self.quiet_until = t + COOLDOWN_S
            events.append("left")
        if score is None:
            self.closed_at = None
            return events
        if self.closed_at is None:
            if score < self.blink_level:
                self.open_levels.append(score)
                if len(self.open_levels) % 15 == 0:
                    self._levels()
            if score >= self.blink_level and t >= self.quiet_until:
                self.closed_at = t
                self.armed = self.menu_armed = self.cancelled = False
                self.hi, self.n = int(score >= self.closed_level), 1
                self.peak, self.dip, self.peaks = score, None, 1
                events.append("closed")
            return events
        dur = t - self.closed_at
        if score > self.release_level:  # всё ещё закрыты
            self.n += 1
            self.hi += score >= self.closed_level
            self._subpeaks(score)
            deliberate = self.hi >= DELIBERATE_SHARE * self.n  # веки сомкнуты, а не взгляд вниз
            if dur >= RIGHT_FROM_S and not self.armed and deliberate:
                self.armed = True
                events.append("armed")
            if dur > RIGHT_TO_S and not self.menu_armed and deliberate:
                self.menu_armed = True
                events.append("menu_armed")
            if dur > MENU_TO_S and not self.cancelled and deliberate:
                self.cancelled = True
                events.append("cancel")
            return events
        # открылись
        start, self.closed_at = self.closed_at, None
        events.append("opened")
        if dur >= LONG_CLOSURE_S and self.hi >= DELIBERATE_SHARE * self.n:  # глаза были закрыты по-настоящему
            self.no_blinks_until = t + AFTER_LONG_S
        if BLINK_MIN_S <= dur <= BLINK_MAX_S:
            if t < self.no_blinks_until:
                self.last_blink_end, self.blinks, self.pending = None, 0, None  # глаза «отходят»
                return events
            in_row = self.last_blink_end is not None and start - self.last_blink_end <= DOUBLE_GAP_S
            self.blinks = (self.blinks if in_row else 0) + min(self.peaks, 3)
            self.last_blink_end = t
            if self.blinks >= 3:
                self.blinks, self.pending, self.last_blink_end = 0, None, None
                self.quiet_until = t + COOLDOWN_S
                events.append("menu")
            elif self.blinks == 2:
                self.pending = t + TRIPLE_WAIT_S
        elif self.hi < DELIBERATE_SHARE * self.n:
            self.last_blink_end, self.blinks, self.pending = None, 0, None  # опускал взгляд — не жест
        elif RIGHT_FROM_S <= dur <= RIGHT_TO_S:
            self.last_blink_end, self.blinks, self.pending = None, 0, None
            self.quiet_until = t + COOLDOWN_S
            events.append("right")
        elif RIGHT_TO_S < dur <= MENU_TO_S:
            self.last_blink_end, self.blinks, self.pending = None, 0, None
            self.quiet_until = t + COOLDOWN_S
            events.append("menu")
        else:
            self.last_blink_end, self.blinks, self.pending = None, 0, None
        return events


class Fixation:
    """Точка взгляда по кадрам → цель курсора: центр текущей фиксации."""

    def __init__(self, radius_px):
        self.radius = radius_px
        self.points = deque()  # (t, x, y) текущей фиксации
        self.outliers = []
        self.target = None
        self.since = None  # когда началась текущая фиксация

    def reset(self):
        self.points.clear()
        self.outliers = []

    def update(self, t, p):
        if p is None:
            return self.target
        if not self.points:
            self.points.append((t, p[0], p[1]))
            self.since = t
        else:
            cx, cy = self.center()
            if math.hypot(p[0] - cx, p[1] - cy) <= self.radius:
                self.points.append((t, p[0], p[1]))
                self.outliers = []
            else:
                self.outliers.append((t, p[0], p[1]))
                ox = [q[1] for q in self.outliers]
                oy = [q[2] for q in self.outliers]
                spread = math.hypot(max(ox) - min(ox), max(oy) - min(oy))
                if len(self.outliers) >= SACCADE_FRAMES and spread <= 2 * self.radius:
                    self.points = deque(self.outliers)  # взгляд переместился и держится там
                    self.since = self.outliers[0][0]
                    self.outliers = []
                elif spread > 2 * self.radius:
                    self.outliers = self.outliers[-1:]  # разброс — шум, а не перевод взгляда
        while self.points and t - self.points[0][0] > FIX_KEEP_S and len(self.points) > 1:
            self.points.popleft()
        self.target = self.center()
        return self.target

    def center(self):
        return (sum(q[1] for q in self.points) / len(self.points), sum(q[2] for q in self.points) / len(self.points))


# Джойстик (обычный режим мыши взглядом): курсор не прыгает туда, куда смотришь, а едет в ту
# сторону — тем быстрее, чем дольше смотришь; за ~5 с проходит весь экран. Трогается, только если
# взгляд ушёл от курсора дальше JOY_START_CM на JOY_START_S (дрожание взгляда ~2 см не двигает),
# едет, пока не подъедет ближе JOY_STOP_CM, и тормозит на подходе. Сменил направление — разгон заново.
JOY_START_CM, JOY_START_S, JOY_STOP_CM = 3.0, 0.25, 1.5
JOY_V0, JOY_ACCEL, JOY_VMAX, JOY_BRAKE = 2.0, 3.0, 40.0, 2.5  # см/с, см/с², см/с, 1/с (скорость ≤ BRAKE·расстояние)
JOY_TURN_DEG = 60.0  # направление сменилось больше чем на столько — разгон заново


class Cursor:
    """Движение системного курсора в своём потоке; рука на мыши — пауза.
    mode "joystick" — едет в сторону взгляда с разгоном; "direct" — плавно встаёт в точку взгляда."""

    def __init__(self, get_pos=None, set_pos=None, clock=time.monotonic, hz=100, mode="joystick", px_per_cm=74.0):
        self.get_pos = get_pos or _get_cursor
        self.set_pos = set_pos or _set_cursor
        self.clock = clock
        self.period = 1.0 / hz
        self.target = None
        self.pos = None  # где курсор по нашим расчётам (дробно)
        self.last_set = None
        self.manual_until = -1e9
        self.frozen = False  # глаза закрыты — стоим
        self.stop_event = threading.Event()
        self.thread = None
        self.mode, self.px_per_cm = mode, px_per_cm
        self.moving = False
        self.away_since = None  # с какого времени взгляд дальше порога «тронуться»
        self.push_since, self.push_dir = None, None  # разгон: с какого времени и в какую сторону

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True, name="gaze-cursor")
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def step(self, dt):
        """Один шаг (для потока и тестов)."""
        now = self.clock()
        real = self.get_pos()
        if self.last_set is not None and real is not None and math.hypot(real[0] - self.last_set[0],
                                                                         real[1] - self.last_set[1]) > 3:
            self.manual_until = now + MANUAL_PAUSE_S  # мышь двигали рукой
            self.last_set = None
        if now < self.manual_until or self.frozen or self.target is None:
            self.pos = real
            return
        if self.pos is None:
            self.pos = real or self.target
        if self.mode == "joystick":
            if not self._joystick(now, dt):
                self.last_set = tuple(real) if real is not None else self.last_set
                return
        else:
            k = 1 - math.exp(-dt / GLIDE_S)
            self.pos = (self.pos[0] + (self.target[0] - self.pos[0]) * k,
                        self.pos[1] + (self.target[1] - self.pos[1]) * k)
        new = (round(self.pos[0]), round(self.pos[1]))
        if real is None or new != tuple(real):
            self.set_pos(*new)
            self.last_set = new
        else:
            self.last_set = tuple(real)

    def _joystick(self, now, dt):
        """Сдвинуть self.pos в сторону взгляда. False — стоим."""
        dx, dy = self.target[0] - self.pos[0], self.target[1] - self.pos[1]
        dist_cm = math.hypot(dx, dy) / self.px_per_cm
        if not self.moving:
            if dist_cm <= JOY_START_CM:
                self.away_since = None
                return False
            if self.away_since is None:
                self.away_since = now
            if now - self.away_since < JOY_START_S:
                return False
            self.moving, self.push_since, self.push_dir = True, now, None
        if dist_cm < JOY_STOP_CM:
            self.moving, self.away_since, self.push_dir = False, None, None
            return False
        direction = (dx / (dist_cm * self.px_per_cm), dy / (dist_cm * self.px_per_cm))
        if self.push_dir is not None:
            cos = direction[0] * self.push_dir[0] + direction[1] * self.push_dir[1]
            if cos < math.cos(math.radians(JOY_TURN_DEG)):
                self.push_since = now  # взгляд ушёл в другую сторону — разгон заново
        if self.push_dir is None or self.push_since == now:
            self.push_dir = direction
        speed = min(JOY_V0 + JOY_ACCEL * (now - self.push_since), JOY_VMAX, JOY_BRAKE * dist_cm)  # см/с
        step = speed * self.px_per_cm * dt
        self.pos = (self.pos[0] + direction[0] * step, self.pos[1] + direction[1] * step)
        return True

    def _loop(self):
        last = self.clock()
        while not self.stop_event.wait(self.period):
            now = self.clock()
            try:
                self.step(now - last)
            except Exception:  # управление мышью не должно ронять программу
                import traceback
                traceback.print_exc()
            last = now


class GazeMouse:
    """Всё вместе: точка взгляда и коэффициент моргания с каждого кадра → курсор и клики."""

    def __init__(self, screen, px_per_cm, beep=None, click=None, cursor=None, gestures=None, mode="joystick"):
        self.screen = tuple(screen)  # (left, top, w, h) — точки взгляда от его левого верхнего угла
        self.fix = Fixation(FIX_CM * px_per_cm)
        self.gestures = gestures or Gestures()  # общий с программой (жесты нужны и меню)
        self.held = False  # меню взглядом открыто — курсор стоит, клики — меню
        self.beep = beep or (lambda kind: None)
        self.click = click or _click
        self.cursor = cursor or Cursor(mode=mode, px_per_cm=px_per_cm)
        self.anchor = None  # где был курсор, когда закрылись глаза
        self.settle_until = -1e9
        self.clicked = []  # (t, "left"/"right") — для журнала и чтобы не учить калибровку на них
        self.hold = None  # Fixation, пока после трёх морганий ждём, что взгляд продержится на месте
        self.hold_at = None

    def start(self):
        self.cursor.start()
        return self

    def stop(self):
        self.cursor.stop()

    def arm_hold(self, t):
        """Три моргания: продержится взгляд на месте HOLD_S — событие "hold" (Win+Tab)."""
        self.hold, self.hold_at = Fixation(self.fix.radius), t

    def update(self, t, point, blink, events=None):
        """point — точка взгляда (px от угла экрана) или None; blink — 0..1 или None (лица нет);
        events — события жестов, если их уже посчитала программа (иначе считаем сами).
        → свои события: "hold" (взгляд продержался после трёх морганий), "hold_lost" (не продержался)."""
        out = []
        if events is None:
            events = self.gestures.update(t, blink)
            for ev in events:
                if ev in ("armed", "cancel"):
                    self.beep(ev)
        for ev in events:
            if ev == "closed":
                self.anchor = self.cursor.pos
                self.cursor.frozen = True
            elif ev == "opened":
                self.cursor.frozen = False
                self.settle_until = t + SETTLE_S
                self.fix.reset()
            elif ev in ("left", "right") and not self.held:
                if self.anchor is not None:
                    spot = (round(self.anchor[0]), round(self.anchor[1]))
                    self.cursor.set_pos(*spot)
                    self.cursor.pos, self.cursor.last_set = self.anchor, spot  # не принять за руку
                self.click(ev)
                self.clicked.append((t, ev))
            if ev in ("left", "right", "armed"):
                self.hold = None  # клик или долгое закрытие глаз — Win+Tab уже не ждём
        if self.hold is not None and t - self.hold_at > HOLD_S + HOLD_GRACE_S:
            self.hold = None
            out.append("hold_lost")
        if blink is not None and blink >= self.gestures.lid_level:
            self.settle_until = max(self.settle_until, t + LID_SETTLE_S)
            return out  # веки прикрыты или глаза закрыты — точка уезжает вниз
        if t < self.settle_until or point is None or self.held:
            return out
        left, top, w, h = self.screen
        x = min(max(point[0], 0), w - 1)
        y = min(max(point[1], 0), h - 1)
        fx, fy = self.fix.update(t, (x, y))
        self.cursor.target = (left + fx, top + fy)
        if self.hold is not None:
            self.hold.update(t, (x, y))
            if t - self.hold.since >= HOLD_S:
                self.hold = None
                out.append("hold")
        return out


# ---------- Windows ----------

class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_cursor():
    pt = _Point()
    try:
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return pt.x, pt.y
    except AttributeError:
        pass
    return None


def _set_cursor(x, y):
    try:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    except AttributeError:
        pass


def _click(kind):
    """Клик в текущей позиции курсора."""
    down, up = (0x0002, 0x0004) if kind == "left" else (0x0008, 0x0010)
    try:
        ctypes.windll.user32.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.02)
        ctypes.windll.user32.mouse_event(up, 0, 0, 0, 0)
    except AttributeError:
        pass


class Hotkey:
    """Ctrl+Alt+<буква> из любой программы → callback() (опрос клавиш, без хуков).
    keys — {"G": функция, ...}."""

    MODS = (0x11, 0x12)  # Ctrl, Alt

    def __init__(self, keys, hz=60):
        self.keys = {ord(k.upper()): f for k, f in keys.items()}
        self.period = 1.0 / hz
        self.stop_event = threading.Event()

    def start(self):
        try:
            self.user32 = ctypes.windll.user32
        except AttributeError:
            return self
        threading.Thread(target=self._loop, daemon=True, name="hotkey").start()
        return self

    def _loop(self):
        was = set()
        while not self.stop_event.wait(self.period):
            down = lambda vk: self.user32.GetAsyncKeyState(vk) & 0x8000  # noqa: E731
            now = {vk for vk in self.keys if down(vk)} if all(down(m) for m in self.MODS) else set()
            for vk in now - was:
                try:
                    self.keys[vk]()
                except Exception:
                    import traceback
                    traceback.print_exc()
            was = now

    def stop(self):
        self.stop_event.set()
