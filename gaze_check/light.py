"""Отпечаток света по кадру камеры: насколько светло лицо, с какой стороны свет, тёплый он или
холодный, светлее ли фон, сколько кадров даёт камера (в темноте она удлиняет выдержку). Свет
сдвигает признаки сети: калибровка, снятая в темноте, под лампой промахивалась на 5 см вместо 3.
Поэтому калибровка помнит свет, при котором её сняли, а программа берёт калибровку под текущий
свет (calstore.py, Engine._pick_light).

В темноте главный свет — сам экран, и тёмная тема против белой страницы меняет яркость лица:
поэтому свет — медиана за минуту, а переключение — с запасом (Engine)."""
import collections
import math

import numpy as np

KEYS = ("face", "side", "warm", "back", "fps")
# на сколько признак должен измениться, чтобы свет был «заметно другим» (расстояние 1)
STEP = {"face": 30.0, "side": 0.08, "warm": 0.10, "back": 0.30, "fps": 5.0}
SAME = 1.5  # ближе — тот же свет (калибровка заменяется или дополняется)
NEW = 3.0  # дальше от всех калибровок — незнакомый свет
LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR → яркость


def signature(frame, pts, fps):
    """Кадр BGR, точки лица (N×2, пиксели) и частота камеры → отпечаток света или None (лица нет)."""
    if pts is None or len(pts) == 0:
        return None
    h, w = frame.shape[:2]
    x0, y0 = (int(v) for v in np.maximum(np.min(pts, axis=0), 0))
    x1, y1 = (int(v) for v in np.minimum(np.max(pts, axis=0), (w - 1, h - 1)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    # середина рамки лица: без волос, ушей и фона по краям
    fx0, fx1 = x0 + (x1 - x0) // 6, x1 - (x1 - x0) // 6
    fy0, fy1 = y0 + (y1 - y0) // 5, y1 - (y1 - y0) // 10
    face = frame[fy0:fy1:2, fx0:fx1:2].astype(np.float32)
    luma = face @ LUMA
    mid = luma.shape[1] // 2
    left, right = float(luma[:, :mid].mean()), float(luma[:, mid:].mean())
    bright = (left + right) / 2
    b, _, r = face.reshape(-1, 3).mean(axis=0)
    small = frame[::8, ::8].astype(np.float32) @ LUMA
    outside = np.ones(small.shape, bool)
    outside[y0 // 8:y1 // 8 + 1, x0 // 8:x1 // 8 + 1] = False
    back = float(small[outside].mean()) if outside.any() else bright
    return {"face": bright, "side": (left - right) / max(1.0, left + right),
            "warm": math.log((float(r) + 1) / (float(b) + 1)), "back": math.log((back + 1) / (bright + 1)),
            "fps": float(min(fps or 0.0, 30.0))}


def distance(a, b):
    """Расстояние между отпечатками в «заметных различиях»; нет отпечатка — бесконечность."""
    if not a or not b:
        return math.inf
    return math.sqrt(sum(((a[k] - b[k]) / STEP[k]) ** 2 for k in KEYS))


def median(sigs):
    return {k: float(np.median([s[k] for s in sigs])) for k in KEYS}


def describe(sig):
    """Отпечаток → «темно, свет справа, тёплый» — название калибровки."""
    if not sig:
        return "свет неизвестен"
    if sig["fps"] < 12 or sig["face"] < 45:
        parts = ["темно"]
    elif sig["fps"] < 20 or sig["face"] < 90:
        parts = ["полумрак"]
    else:
        parts = ["светло"]
    if abs(sig["side"]) >= 0.06:  # левая половина кадра — правая щека человека
        parts.append("свет справа" if sig["side"] > 0 else "свет слева")
    if sig["warm"] > 0.65:
        parts.append("тёплый")
    elif sig["warm"] < 0.25:
        parts.append("холодный")
    if sig["back"] > 0.4:
        parts.append("светлый фон")
    return ", ".join(parts)


class LightWatch:
    """Отпечатки раз в секунду за последние 10 минут; свет сейчас — медиана за минуту."""

    EVERY, KEEP, WINDOW, MIN = 1.0, 600.0, 60.0, 20

    def __init__(self):
        self.samples = collections.deque()  # (ts, отпечаток)
        self.at = -math.inf

    def due(self, ts):
        return ts - self.at >= self.EVERY

    def add(self, ts, sig):
        self.at = ts
        if sig is None:
            return
        self.samples.append((ts, sig))
        while self.samples and ts - self.samples[0][0] > self.KEEP:
            self.samples.popleft()

    def current(self):
        """Свет за последнюю минуту или None — отпечатков мало (лица не было)."""
        if not self.samples:
            return None
        ts = self.samples[-1][0]
        return self.between(ts - self.WINDOW, ts, self.MIN)

    def between(self, t0, t1, min_count=5):
        got = [s for t, s in list(self.samples) if t0 <= t <= t1]
        return median(got) if len(got) >= min_count else None
