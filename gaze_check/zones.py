"""Зоны внимания: часть экрана (сетка 3×3) и окно/программа под точкой взгляда,
отрезки времени «куда смотрел» и итоги по зонам, программам и окнам."""
import functools
import math
import os
import sys
import time
from collections import Counter, defaultdict
from types import SimpleNamespace

import cv2
import numpy as np

ROWS = ("верх", "центр", "низ")
COLS = ("лево", "центр", "право")
GRID = tuple(tuple("центр" if (r, c) == ("центр", "центр") else "%s-%s" % (r, c) for c in COLS)
             for r in ROWS)
GRID_ZONES = tuple(z for row in GRID for z in row)
OFF_SCREEN = "вне экрана"
NO_DATA = "нет данных"
ALL_ZONES = GRID_ZONES + (OFF_SCREEN, NO_DATA)
MIN_SEGMENT = 0.3  # секунд: более короткий скачок взгляда не рвёт отрезок


def zone_of(point, screen_w, screen_h, margin=0.1):
    """Точка (пиксели от левого верхнего угла монитора) → зона сетки 3×3.
    Дальше margin размера экрана за краем — «вне экрана», нет точки — «нет данных»."""
    if point is None:
        return NO_DATA
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)) or screen_w <= 0 or screen_h <= 0:
        return NO_DATA
    if (x < -margin * screen_w or x > screen_w * (1 + margin)
            or y < -margin * screen_h or y > screen_h * (1 + margin)):
        return OFF_SCREEN
    col = min(2, max(0, math.floor(3 * x / screen_w)))
    row = min(2, max(0, math.floor(3 * y / screen_h)))
    return GRID[row][col]


def zone_cell(zone):
    """(строка, столбец) зоны сетки или None."""
    for r, row in enumerate(GRID):
        if zone in row:
            return r, row.index(zone)
    return None


# ---------- окно под взглядом ----------

def _win_api():
    """Свои экземпляры user32/kernel32: argtypes не задевают ctypes.windll в других модулях."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes as wt
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # физические координаты при любой DPI-осведомлённости процесса
        find = getattr(user32, "WindowFromPhysicalPoint", None) or user32.WindowFromPoint
        find.argtypes, find.restype = [wt.POINT], wt.HWND
        user32.GetAncestor.argtypes, user32.GetAncestor.restype = [wt.HWND, wt.UINT], wt.HWND
        user32.GetWindowTextLengthW.argtypes, user32.GetWindowTextLengthW.restype = [wt.HWND], ctypes.c_int
        user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        user32.GetWindowThreadProcessId.restype = wt.DWORD
        kernel32.OpenProcess.argtypes, kernel32.OpenProcess.restype = [wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
        kernel32.QueryFullProcessImageNameW.restype = wt.BOOL
        kernel32.CloseHandle.argtypes, kernel32.CloseHandle.restype = [wt.HANDLE], wt.BOOL
        return SimpleNamespace(ctypes=ctypes, wt=wt, user32=user32, kernel32=kernel32, find=find)
    except (OSError, AttributeError):
        return None


class WindowProbe:
    """(программа, заголовок окна) под точкой взгляда, например ("Code.exe", "app.py - …").
    Спрашивает Windows не чаще раза в interval секунд, между вызовами отдаёт прошлый ответ.
    На другой ОС и при любой ошибке — ("", "")."""

    GA_ROOT = 2
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self, screen, interval=0.1, clock=time.monotonic):
        self.left, self.top, self.width, self.height = (int(v) for v in screen)
        self.interval, self.clock = interval, clock
        self.last_t, self.last = None, ("", "")
        self.names = {}  # pid → имя exe
        self.api = _win_api()

    def __call__(self, point):
        if point is None or self.api is None:
            return ("", "")
        now = self.clock()
        if self.last_t is not None and 0 <= now - self.last_t < self.interval:
            return self.last
        self.last_t = now
        try:
            self.last = self._probe(point)
        except Exception:
            self.last = ("", "")
        return self.last

    def _probe(self, point):
        api = self.api
        x = self.left + min(max(int(round(float(point[0]))), 0), self.width - 1)
        y = self.top + min(max(int(round(float(point[1]))), 0), self.height - 1)
        hwnd = api.find(api.wt.POINT(x, y))
        if not hwnd:
            return ("", "")
        root = api.user32.GetAncestor(hwnd, self.GA_ROOT) or hwnd
        n = api.user32.GetWindowTextLengthW(root)
        buf = api.ctypes.create_unicode_buffer(max(n, 0) + 1)
        api.user32.GetWindowTextW(root, buf, len(buf))
        pid = api.wt.DWORD()
        api.user32.GetWindowThreadProcessId(root, api.ctypes.byref(pid))
        return self._process_name(pid.value), buf.value

    def _process_name(self, pid):
        if pid in self.names:
            return self.names[pid]
        name = ""
        api = self.api
        handle = api.kernel32.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            try:
                buf = api.ctypes.create_unicode_buffer(1024)
                size = api.wt.DWORD(len(buf))
                if api.kernel32.QueryFullProcessImageNameW(handle, 0, buf, api.ctypes.byref(size)):
                    name = os.path.basename(buf.value)
            finally:
                api.kernel32.CloseHandle(handle)
        self.names[pid] = name
        return name


# ---------- отрезки внимания ----------

class AttentionLog:
    """Отрезки, где не меняются (зона, программа, окно). Взгляд ушёл из текущего отрезка
    меньше чем на min_s и вернулся — скачок присоединяется к отрезку. Ушёл надольше —
    новый отрезок с момента ухода, ключ — самый частый за время ухода."""

    def __init__(self, min_s=MIN_SEGMENT):
        self.min_s = min_s
        self.segments = []
        self.cur = None  # {"key", "start", "end"}
        self.pending = []  # [(t, key)] с момента ухода из текущего отрезка
        self.last_t = None

    def add(self, t, zone, app="", window=""):
        key = (zone, app or "", window or "")
        self.last_t = t
        if self.cur is None:
            self.cur = {"key": key, "start": t, "end": t}
        elif key == self.cur["key"]:
            self.cur["end"] = t
            self.pending = []
        else:
            self.pending.append((t, key))
            if t - self.pending[0][0] >= self.min_s:
                self._switch(t)

    def _switch(self, t):
        since = self.pending[0][0]
        counts = Counter(k for _, k in self.pending)
        best = max(counts.values())
        # при равенстве — более поздний
        winner = next(k for _, k in reversed(self.pending) if counts[k] == best)
        latest = self.pending[-1][1]
        self.cur["end"] = since
        self.segments.append(self.cur)
        self.cur = {"key": winner, "start": since, "end": t}
        self.pending = [] if latest == winner else [(t, latest)]

    def finish(self):
        """Все отрезки: [{start, end, duration_s, zone, app, window}]."""
        if self.cur is not None:
            self.cur["end"] = max(self.cur["end"], self.last_t)  # неразрешённый хвост — в текущий
            self.segments.append(self.cur)
            self.cur, self.pending = None, []
        out = []
        for s in self.segments:
            zone, app, window = s["key"]
            out.append({"start": round(s["start"], 3), "end": round(s["end"], 3),
                        "duration_s": round(s["end"] - s["start"], 3),
                        "zone": zone, "app": app, "window": window})
        return out


def summarize(segments, top=20):
    """Итоги: секунды и доли по зонам (все 11), по программам и топ окон по заголовку.
    Доли — от всего времени записи, поэтому у программ и окон в сумме меньше 1."""
    total = sum(s["duration_s"] for s in segments)

    def item(v):
        return {"seconds": round(v, 1), "share": round(v / total, 4) if total else 0.0}

    zones = dict.fromkeys(ALL_ZONES, 0.0)
    apps, windows = defaultdict(float), defaultdict(float)
    for s in segments:
        d = s["duration_s"]
        zones[s["zone"]] = zones.get(s["zone"], 0.0) + d
        if s["app"]:
            apps[s["app"]] += d
        if s["app"] or s["window"]:
            windows[(s["app"], s["window"])] += d
    return {
        "total_s": round(total, 1),
        "zones": {z: item(v) for z, v in zones.items()},
        "apps": {a: item(v) for a, v in sorted(apps.items(), key=lambda kv: -kv[1])},
        "windows": [{"app": a, "window": w, **item(v)}
                    for (a, w), v in sorted(windows.items(), key=lambda kv: -kv[1])[:top]],
    }


# ---------- отрисовка ----------

@functools.lru_cache(maxsize=16)
def _font(size, bold):
    from PIL import ImageFont
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@functools.lru_cache(maxsize=512)
def text_patch(text, size, bold=False, fg=(255, 255, 255), bg=(0, 0, 0)):
    """Надпись (кириллица) на плашке, BGR. Рисуется один раз и берётся из кэша."""
    from PIL import Image, ImageDraw
    font = _font(size, bold)
    pad = max(2, size // 4)
    left, top, right, bottom = font.getbbox(text or " ")
    img = Image.new("RGB", (right - left + 2 * pad, bottom - top + 2 * pad), tuple(bg[::-1]))
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=tuple(fg[::-1]))
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def put(frame, patch, x, y):
    """Кладёт плашку левым верхним углом в (x, y) с обрезкой по краям кадра."""
    ph, pw = patch.shape[:2]
    fh, fw = frame.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + pw), min(fh, y + ph)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


def draw_zone_overlay(frame, zone, app=""):
    """Тонкая сетка 3×3, рамка текущей зоны и подпись «зона · программа» слева сверху."""
    h, w = frame.shape[:2]
    for i in (1, 2):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h - 1), (170, 170, 170), 1)
        cv2.line(frame, (0, h * i // 3), (w - 1, h * i // 3), (170, 170, 170), 1)
    cell = zone_cell(zone)
    if cell:
        r, c = cell
        cv2.rectangle(frame, (w * c // 3, h * r // 3), (w * (c + 1) // 3 - 1, h * (r + 1) // 3 - 1),
                      (0, 200, 255), 2)
    put(frame, text_patch(zone + (" · " + app if app else ""), max(14, h // 40)), 8, 8)
    return frame


def zones_image(background, summary, size=(1280, 720)):
    """Снимок экрана с сеткой 3×3 и долей времени в каждой зоне; внизу — вне экрана / нет данных."""
    if background is None:
        background = np.full((size[1], size[0], 3), 40, np.uint8)
    img = (background.astype(np.float32) * 0.4).astype(np.uint8)  # затемнить, чтобы цифры читались
    h, w = img.shape[:2]
    zones = summary["zones"]
    top_share = max([zones[z]["share"] for z in GRID_ZONES] + [1e-9])
    for r, row in enumerate(GRID):
        for c, zone in enumerate(row):
            x0, y0, x1, y1 = w * c // 3, h * r // 3, w * (c + 1) // 3, h * (r + 1) // 3
            share = zones[zone]["share"]
            if share > 0:  # чем больше доля, тем гуще заливка
                alpha = 0.15 + 0.5 * share / top_share
                cell = img[y0:y1, x0:x1].astype(np.float32)
                img[y0:y1, x0:x1] = (cell * (1 - alpha) + np.array((0, 90, 230)) * alpha).astype(np.uint8)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (235, 235, 235), 2)
            big = text_patch("%d%%" % round(share * 100), max(20, h // 9), True, bg=(0, 0, 0))
            small = text_patch("%s · %s" % (zone, _clock(zones[zone]["seconds"])), max(12, h // 32))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            put(img, big, cx - big.shape[1] // 2, cy - big.shape[0] // 2 - small.shape[0] // 2)
            put(img, small, cx - small.shape[1] // 2, cy + big.shape[0] // 2 - small.shape[0] // 2 + 4)
    caption = "%s %d%% · %s %d%% · всего %s" % (
        OFF_SCREEN, round(zones[OFF_SCREEN]["share"] * 100), NO_DATA,
        round(zones[NO_DATA]["share"] * 100), _clock(summary["total_s"]))
    patch = text_patch(caption, max(14, h // 30), True, bg=(0, 0, 0))
    put(img, patch, (w - patch.shape[1]) // 2, h - patch.shape[0] - 8)
    return img


def _clock(seconds):
    seconds = int(round(seconds))
    return "%d:%02d" % divmod(seconds, 60) if seconds >= 60 else "%d с" % seconds
