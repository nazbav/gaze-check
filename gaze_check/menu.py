"""Меню взглядом: 9 плиток на весь экран (те же зоны 3×3, что при калибровке).

Для людей, которым трудно пользоваться руками: голосовой ввод, мышь взглядом, экранная
клавиатура, прокрутка, окна, браузер, звук, «позвать на помощь». Опасного нет: лупа (с ней
точка взгляда не совпадает с экраном), экранный диктор и голосовой доступ (перехватывают
управление), закрыть окно/вкладку (потеря несохранённого) — убраны. Плитка ~11×7 см — при
точности взгляда 2–3 см выбирается надёжно (в отличие от мелких кнопок).

Выбор — задержкой взгляда (DWELL_S, полоска заполняется) или двумя морганиями. Центральная
плитка — «назад/закрыть»: в центр смотрят чаще всего. «Позвать на помощь» (громкий сигнал) —
только после подтверждения. Меню не забирает фокус: нажатия уходят в окно, где работали.
"""
import ctypes
import threading
import time

DWELL_S = 1.2  # столько смотреть на плитку, чтобы выбрать
LEAVE_S = 0.25  # взгляд ушёл с плитки на меньшее время — отсчёт не сбрасывается (дрожание)
REPEAT_S = 0.8  # повторяемое действие (прокрутка, громкость) — снова, пока смотрите
CONFIRM_S = 4.0  # столько ждёт подтверждения опасное действие

# ---------- клавиши ----------

VK = {"win": 0x5B, "ctrl": 0x11, "alt": 0x12, "shift": 0x10, "tab": 0x09, "enter": 0x0D, "esc": 0x1B,
      "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "pgup": 0x21, "pgdn": 0x22, "home": 0x24,
      "end": 0x23, "f5": 0x74,
      "vol_down": 0xAE, "vol_up": 0xAF, "mute": 0xAD, "play": 0xB3, "next": 0xB0, "prev": 0xB1}
EXTENDED = {"win", "left", "up", "right", "down", "pgup", "pgdn", "home", "end",
            "vol_down", "vol_up", "mute", "play", "next", "prev"}


def _vk(key):
    return VK[key] if key in VK else ord(key.upper())


def press(*keys):
    """Сочетание клавиш: ("ctrl", "c"), ("win", "h")... — зажать по порядку, отпустить обратно."""
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return
    for k in keys:
        user32.keybd_event(_vk(k), 0, 1 if k in EXTENDED else 0, 0)
    time.sleep(0.03)
    for k in reversed(keys):
        user32.keybd_event(_vk(k), 0, (1 if k in EXTENDED else 0) | 2, 0)
    time.sleep(0.03)


def wheel(notches):
    """Колесо мыши (+ вверх) в окне под курсором."""
    try:
        ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(120 * notches), 0)
    except AttributeError:
        pass


# ---------- страницы ----------
# плитка: (надпись, действие, флаги). Действие — ("keys", ...), ("wheel", n), ("page", имя),
# ("app", имя) — делает программа (мышь, помощь, закрыть). Флаги: "stay" — меню не закрывается
# (можно повторять, пока смотрите), "confirm" — опасно, нужно выбрать дважды.
# Порядок — по зонам: верх-лево … низ-право; центр (индекс 4) — назад/закрыть.

PAGES = {
    "main": ("Меню", [
        ("Голосовой ввод", ("keys", "win", "h"), ""),
        ("Прокрутка ▸", ("page", "scroll"), ""),
        ("Окна ▸", ("page", "windows"), ""),
        ("Мышь взглядом\nвкл / выкл", ("app", "mouse"), ""),
        ("Закрыть меню", ("app", "close"), ""),
        ("Браузер ▸", ("page", "browser"), ""),
        ("Экранная\nклавиатура", ("keys", "win", "ctrl", "o"), ""),
        ("Звук ▸", ("page", "media"), ""),
        ("Ещё ▸", ("page", "more"), ""),
    ]),
    "scroll": ("Прокрутка", [
        ("В начало", ("keys", "ctrl", "home"), "stay"),
        ("▲ Вверх", ("wheel", 3), "stay"),
        ("Страница вверх", ("keys", "pgup"), "stay"),
        ("Esc", ("keys", "esc"), "stay"),
        ("◀ Назад", ("page", "main"), ""),
        ("Enter", ("keys", "enter"), "stay"),
        ("В конец", ("keys", "ctrl", "end"), "stay"),
        ("▼ Вниз", ("wheel", -3), "stay"),
        ("Страница вниз", ("keys", "pgdn"), "stay"),
    ]),
    "windows": ("Окна", [
        ("Переключить окно", ("keys", "alt", "tab"), ""),
        ("Все окна", ("keys", "win", "tab"), ""),
        ("Проводник", ("keys", "win", "e"), ""),
        ("Свернуть", ("keys", "win", "down"), "stay"),
        ("◀ Назад", ("page", "main"), ""),
        ("Развернуть", ("keys", "win", "up"), "stay"),
        ("Рабочий стол", ("keys", "win", "d"), ""),
        ("Пуск", ("keys", "win"), ""),
        ("Поиск", ("keys", "win", "s"), ""),
    ]),
    "browser": ("Браузер", [
        ("◀ Назад\n(страница)", ("keys", "alt", "left"), "stay"),
        ("⟳ Обновить", ("keys", "f5"), "stay"),
        ("Вперёд ▶", ("keys", "alt", "right"), "stay"),
        ("◀ Вкладка", ("keys", "ctrl", "shift", "tab"), "stay"),
        ("◀ Назад", ("page", "main"), ""),
        ("Вкладка ▶", ("keys", "ctrl", "tab"), "stay"),
        ("Новая вкладка", ("keys", "ctrl", "t"), ""),
        ("Адрес + голос", ("app", "address_voice"), ""),
        ("▼ Вниз", ("wheel", -3), "stay"),
    ]),
    "media": ("Звук", [
        ("Тише", ("keys", "vol_down"), "stay"),
        ("Без звука", ("keys", "mute"), "stay"),
        ("Громче", ("keys", "vol_up"), "stay"),
        ("⏮ Предыдущий", ("keys", "prev"), "stay"),
        ("◀ Назад", ("page", "main"), ""),
        ("Следующий ⏭", ("keys", "next"), "stay"),
        ("⏯ Пауза / пуск", ("keys", "play"), "stay"),
        ("Голосовой ввод", ("keys", "win", "h"), ""),
        ("", None, ""),
    ]),
    "more": ("Ещё", [
        ("Копировать", ("keys", "ctrl", "c"), ""),
        ("Вставить", ("keys", "ctrl", "v"), ""),
        ("Отменить", ("keys", "ctrl", "z"), "stay"),
        ("Выделить всё", ("keys", "ctrl", "a"), "stay"),
        ("◀ Назад", ("page", "main"), ""),
        ("Enter", ("keys", "enter"), "stay"),
        ("Позвать\nна помощь", ("app", "help"), "confirm"),
        ("Esc", ("keys", "esc"), "stay"),
        ("Tab — следующее\nполе", ("keys", "tab"), "stay"),
    ]),
}


class MenuLogic:
    """Состояние меню и выбор взглядом; без окон — чтобы проверять тестами."""

    def __init__(self, run=None, clock=time.monotonic):
        self.run = run or (lambda action: None)  # выполнить действие ("keys", ...)/("wheel", n)/("app", ...)
        self.clock = clock
        self.lock = threading.Lock()
        self.open = False
        self.page = "main"
        self.tile = None  # плитка под взглядом
        self.since = None  # с какого времени смотрят на неё
        self.away_since = None
        self.progress = 0.0  # 0..1 — заполнение полоски
        self.confirm = None  # (страница, плитка, до какого времени ждём подтверждения)
        self.flash = None  # (плитка, до какого времени подсветить срабатывание)
        self.note = ""

    def toggle(self):
        with self.lock:
            self.open = not self.open
            self.page, self.tile, self.since, self.progress, self.confirm = "main", None, None, 0.0, None
            return self.open

    def snapshot(self):
        with self.lock:
            return {"open": self.open, "page": self.page, "tile": self.tile, "progress": self.progress,
                    "confirm": self.confirm, "flash": self.flash, "note": self.note}

    @staticmethod
    def tile_of(point, screen):
        """Точка (px от угла экрана) → номер плитки 0..8 или None (вне экрана)."""
        if point is None:
            return None
        x, y = point
        w, h = screen[2], screen[3]
        if not (-0.05 * w <= x <= 1.05 * w and -0.05 * h <= y <= 1.05 * h):
            return None
        col = min(2, max(0, int(3 * x / w)))
        row = min(2, max(0, int(3 * y / h)))
        return row * 3 + col

    def update(self, point, screen):
        """Точка взгляда с каждого кадра → выбор задержкой. Возвращает выбранное действие или None."""
        now = self.clock()
        with self.lock:
            if not self.open:
                return None
            tile = self.tile_of(point, screen)
            if tile != self.tile:
                if self.away_since is None:
                    self.away_since = now
                if now - self.away_since < LEAVE_S and self.tile is not None:
                    return None  # короткий уход с плитки — дрожание, отсчёт идёт дальше
                # отсчёт на новой плитке — с момента, когда на неё перевели взгляд
                self.tile, self.since, self.away_since = tile, self.away_since, None
                self.progress = 0.0
                return None
            self.away_since = None
            if tile is None or self.since is None:
                return None
            self.progress = min(1.0, (now - self.since) / DWELL_S)
            if self.progress >= 1.0:
                return self._choose(tile, now)
        return None

    def select(self):
        """Два моргания — выбрать плитку под взглядом сразу."""
        with self.lock:
            if not self.open or self.tile is None:
                return None
            return self._choose(self.tile, self.clock())

    def _choose(self, tile, now):
        label, action, flags = PAGES[self.page][1][tile]
        self.progress = 0.0
        if action is None:  # пустая плитка
            self.since = now
            return None
        if "confirm" in flags:
            if not (self.confirm and self.confirm[:2] == (self.page, tile) and now <= self.confirm[2]):
                self.confirm = (self.page, tile, now + CONFIRM_S)
                self.since = now  # ещё столько же смотреть — подтверждение
                self.note = "«%s» — выберите ещё раз для подтверждения" % label.replace("\n", " ")
                return None
        self.confirm = None
        self.note = ""
        self.flash = (tile, now + 0.4)
        if action[0] == "page":
            self.page, self.tile, self.since = action[1], None, None
            return None
        if "stay" in flags:
            self.since = now - DWELL_S + REPEAT_S  # повтор через REPEAT_S, пока смотрят
        else:
            self.open = False
        self.run(action)
        return action


def _ours(hwnd):
    """Окно tkinter этой программы (меню, точка взгляда) — такому фокус не отдаём."""
    try:
        import os
        user32 = ctypes.windll.user32
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, buf, 64)
        return pid.value == os.getpid() and buf.value.startswith("Tk")
    except (AttributeError, OSError):
        return False


def _passive(root, exclude_capture=False):
    """Окно tkinter: поверх всех, не активируется, мышь насквозь, нет на панели задач;
    exclude_capture — не попадает в запись экрана и скриншоты (Windows 10 2004+)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x20, 0x80
        WS_EX_TOPMOST, WS_EX_LAYERED = 0x8, 0x80000
        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
                              | WS_EX_TOPMOST | WS_EX_LAYERED)
        if exclude_capture:
            user32.SetWindowDisplayAffinity(hwnd, 0x11)  # WDA_EXCLUDEFROMCAPTURE
        return hwnd
    except (AttributeError, OSError):
        return None


class FocusGuard:
    """tkinter делает свои окна активными, когда захочет (при создании, показе): запоминаем окно, где
    работают, и если активным оказалось наше окно tkinter — сразу отдаём фокус обратно. Окна OpenCV
    (камера, калибровка) — не tkinter, им фокус нужен для клавиш, их не трогаем."""

    def __init__(self):
        self.work = self.foreground()

    @staticmethod
    def foreground():
        try:
            return ctypes.windll.user32.GetForegroundWindow()
        except AttributeError:
            return None

    def __call__(self):
        now = self.foreground()
        if not now:
            return
        if not _ours(now):
            self.work = now
        elif self.work:
            try:
                ctypes.windll.user32.SetForegroundWindow(self.work)
            except (AttributeError, OSError):
                pass


DOT_SIZE, DOT_DEAD_CM, DOT_TAU_S = 1, 0.8, 0.45  # точка — один пиксель (кружок раздражал)
DOT_COLOR_EVERY_S = 0.15  # как часто смотреть на фон под точкой


def contrast(rgb):
    """Цвет точки, заметный на фоне rgb: обратный; если обратный той же яркости (серый фон) —
    чёрный или белый."""
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if abs(2 * lum - 255) < 96:
        inv = (0, 0, 0) if lum >= 128 else (255, 255, 255)
    else:
        inv = (255 - r, 255 - g, 255 - b)
    return "#%02x%02x%02x" % inv


_GDI = None


def screen_pixel(x, y):
    """Цвет экрана в точке (физические px) → (r, g, b) или None. Своё окно точки сюда не попадает:
    оно исключено из снимков экрана."""
    global _GDI
    try:
        if _GDI is None:  # свои экземпляры DLL — чтобы не менять прототипы у общих ctypes.windll
            user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
            user32.GetDC.restype, user32.GetDC.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
            user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            gdi32.GetPixel.restype = ctypes.c_uint32
            gdi32.GetPixel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            _GDI = (user32, gdi32)
        user32, gdi32 = _GDI
        hdc = user32.GetDC(None)
        try:
            c = gdi32.GetPixel(hdc, int(x), int(y))
        finally:
            user32.ReleaseDC(None, hdc)
    except (AttributeError, OSError):
        return None
    if c == 0xFFFFFFFF:  # CLR_INVALID — за экраном
        return None
    return c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF


def _tk(tries=3):
    """tk.Tk() с повтором: изредка Tcl не читает init.tcl («couldn't read file … No error») —
    ловилось в тестах после MediaPipe в том же процессе, причина не найдена; со второго раза выходит."""
    import tkinter as tk
    for i in range(tries):
        try:
            return tk.Tk()
        except tk.TclError:
            if i == tries - 1:
                raise
            time.sleep(0.1)


class GazeDot:
    """Точка взгляда поверх всех окон: один пиксель цвета, контрастного фону под ним; клики сквозь,
    фокус не берёт, в запись экрана и скриншоты не попадает. Двигается медленно, чтобы не мельтешить:
    дрожание меньше DOT_DEAD_CM не трогает, к новому месту подплывает с постоянной времени DOT_TAU_S."""

    def __init__(self, px_per_cm, color="#ffffff", alpha=1.0):
        self.guard = FocusGuard()  # до tk.Tk(): он сразу делает своё окно активным
        self.dead = DOT_DEAD_CM * px_per_cm
        self.alpha = alpha
        self.root = _tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.configure(bg=color)
        self.root.geometry("%dx%d+-100+-100" % (DOT_SIZE, DOT_SIZE))
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.root.update_idletasks()
        _passive(self.root, exclude_capture=True)
        self.root.deiconify()  # единственный показ — невидимым
        self.root.update()
        self.guard()
        self.pos, self.last, self.shown = None, None, False
        self.color, self.color_at = color, -1e9

    def tick(self, point):
        """point — точка взгляда в абсолютных пикселях экрана или None (спрятать)."""
        self.guard()
        now = time.monotonic()
        if point is None:
            if self.shown:
                self.root.attributes("-alpha", 0.0)
                self.shown = False
        else:
            if self.pos is None:
                self.pos = point
            dx, dy = point[0] - self.pos[0], point[1] - self.pos[1]
            if (dx * dx + dy * dy) ** 0.5 > self.dead:
                k = 1 - pow(2.718281828, -min(0.2, now - (self.last or now)) / DOT_TAU_S)
                self.pos = (self.pos[0] + dx * k, self.pos[1] + dy * k)
            x, y = int(self.pos[0]) - DOT_SIZE // 2, int(self.pos[1]) - DOT_SIZE // 2
            self.root.geometry("+%d+%d" % (x, y))
            if now - self.color_at >= DOT_COLOR_EVERY_S:
                self.color_at = now
                under = screen_pixel(int(self.pos[0]), int(self.pos[1]))
                color = contrast(under) if under else self.color
                if color != self.color:
                    self.color = color
                    self.root.configure(bg=color)
            if not self.shown:
                self.root.attributes("-alpha", self.alpha)
                self.shown = True
        self.last = now
        self.root.update()
        self.guard()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


class Overlay:
    """Полупрозрачное окно поверх всех, не забирает фокус и не ловит мышь (tkinter + WinAPI).
    Живёт в главном потоке: tick() — перерисовать и обработать события окна.

    Окно создаётся один раз и дальше не прячется/не показывается (tkinter при показе делает окно
    активным — фокус ушёл бы из окна, где работают, и Win+H печатал бы не туда): закрытое меню —
    полностью прозрачное и пропускает мышь насквозь, открытое — меняет только прозрачность."""

    COLORS = {"bg": "#101418", "tile": "#1d2630", "hot": "#2b5a7a", "fill": "#3aa0e0", "confirm": "#a03030",
              "text": "#f2f4f6", "dim": "#9aa4ae", "flash": "#40c070"}

    def __init__(self, screen, alpha=0.82):
        import tkinter as tk
        self.tk = tk
        self.screen = screen
        self.alpha = alpha
        self.hwnd = None
        # окно, где работают, — запомнить ДО tkinter: tk.Tk() сразу делает своё окно активным
        self._guard = FocusGuard()
        self.root = _tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        left, top, w, h = screen
        self.root.geometry("%dx%d+%d+%d" % (w, h, left, top))
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg=self.COLORS["bg"], highlightthickness=0)
        self.canvas.pack()
        self.shown = False
        self.last = None
        self.root.update_idletasks()
        self.hwnd = _passive(self.root)
        self.root.deiconify()  # единственный показ — невидимым
        self.root.update()
        self._guard()

    def tick(self, snap):
        self._guard()
        if snap["open"] != self.shown:
            self.shown = snap["open"]
            self.root.attributes("-alpha", self.alpha if self.shown else 0.0)
            self.last = None
        if self.shown:
            key = (snap["page"], snap["tile"], round(snap["progress"], 2), snap["confirm"], snap["flash"],
                   snap["note"])
            if key != self.last:
                self.last = key
                self._draw(snap)
        self.root.update()
        self._guard()

    def _draw(self, snap):
        c, col = self.canvas, self.COLORS
        c.delete("all")
        _, _, w, h = self.screen
        title, tiles = PAGES[snap["page"]]
        now = time.monotonic()
        gap = max(8, w // 200)
        font = max(18, h // 30)
        for i, (label, _, flags) in enumerate(tiles):
            r, k = divmod(i, 3)
            x0, y0 = k * w / 3 + gap, r * h / 3 + gap
            x1, y1 = (k + 1) * w / 3 - gap, (r + 1) * h / 3 - gap
            hot = snap["tile"] == i
            confirm = snap["confirm"] and snap["confirm"][:2] == (snap["page"], i)
            flash = snap["flash"] and snap["flash"][0] == i and now <= snap["flash"][1]
            fill = col["flash"] if flash else col["confirm"] if confirm else col["hot"] if hot else col["tile"]
            c.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#3a4652", width=2)
            if hot and snap["progress"] > 0:
                c.create_rectangle(x0, y1 - gap * 2, x0 + (x1 - x0) * snap["progress"], y1, fill=col["fill"], width=0)
            c.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label, fill=col["text"],
                          font=("Segoe UI", font, "bold"), justify="center")
            if "confirm" in flags:
                c.create_text((x0 + x1) / 2, y1 - gap * 4, text="нужно подтвердить", fill=col["dim"],
                              font=("Segoe UI", font // 2))
        note = snap["note"] or ("%s · смотрите на плитку — выбор; два моргания — сразу; "
                                "три моргания — закрыть меню" % title)
        c.create_text(w / 2, gap * 3, text=note, fill=col["dim"], font=("Segoe UI", font // 2))

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass
