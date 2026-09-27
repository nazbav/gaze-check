"""Разбор записи tools/blink_drift.py: что с точкой взгляда вокруг морганий и куда уезжает курсор.

Кадры прогоняются через настоящую мышь взглядом (GazeMouse с ненастоящим курсором) по их времени.

    .venv\\Scripts\\python tools\\blink_analyze.py [файл blink_*.json]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.models import DATA_DIR  # noqa: E402
from gaze_check.mouse import Cursor, GazeMouse  # noqa: E402


def load(path=None):
    if path is None:
        path = sorted((DATA_DIR / "experiments").glob("blink_*.json"))[-1]
    return Path(path), json.loads(Path(path).read_text(encoding="utf-8"))


def replay(frames, screen, px_cm, make_mouse=None):
    """Кадры → [(t, цель курсора или None, события)] как в программе."""
    clock = {"t": 0.0}
    pos = {"p": (0, 0)}
    cursor = Cursor(get_pos=lambda: pos["p"], set_pos=lambda x, y: pos.update(p=(x, y)), clock=lambda: clock["t"])
    gm = (make_mouse or GazeMouse)((0, 0, screen[2], screen[3]), px_cm, click=lambda k: None, cursor=cursor)
    out = []
    for f in frames:
        clock["t"] = f["t"]
        ev = gm.gestures.update(f["t"], f["blink"])  # как в программе: жесты считает она, мышь получает события
        gm.update(f["t"], f["raw"], f["blink"], ev)
        out.append((f["t"], cursor.target, ev))
    return out


def blinks(frames, level=0.5):
    """Эпизоды «глаза закрыты» по коэффициенту: [(начало, конец)]."""
    eps, start = [], None
    for f in frames:
        closed = f["blink"] is not None and f["blink"] >= level
        if closed and start is None:
            start = f["t"]
        elif not closed and start is not None:
            eps.append((start, f["t"]))
            start = None
    return eps


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # вывод в трубу — иначе cp1251 и падение на «→»
    path, data = load(sys.argv[1] if len(sys.argv) > 1 else None)
    frames, screen, px_cm = data["frames"], data["screen"], data["px_per_cm"]
    print(path.name, "кадров", len(frames), "за %.1f с" % (frames[-1]["t"] - frames[0]["t"]),
          "(%.1f к/с)" % (len(frames) / (frames[-1]["t"] - frames[0]["t"])))
    eps = blinks(frames)
    print("морганий (коэф. ≥ 0.5):", len(eps), "длительности, с:", [round(b - a, 2) for a, b in eps])

    def err(f):
        if f["raw"] is None:
            return None
        return math.hypot(f["raw"][0] - f["target"][0], f["raw"][1] - f["target"][1]) / px_cm

    def near(t, before=0.4, after=0.6):
        return any(a - before <= t <= b + after for a, b in eps)

    calm = [e for f in frames if not near(f["t"]) for e in [err(f)] if e is not None]
    around = [e for f in frames if near(f["t"]) for e in [err(f)] if e is not None]
    print("промах точки, см: вдали от морганий медиана %.1f / 90%% %.1f;  рядом с морганием медиана %.1f / 90%% %.1f"
          % (np.median(calm), np.percentile(calm, 90), np.median(around), np.percentile(around, 90)))

    # покадрово вокруг каждого моргания
    for a, b in eps[:12]:
        print("\n--- моргание %.2f–%.2f (%.2f с)" % (a, b, b - a))
        for f in frames:
            if a - 0.4 <= f["t"] <= b + 0.7:
                e = err(f)
                dx = dy = None
                if f["raw"] is not None:
                    dx = (f["raw"][0] - f["target"][0]) / px_cm
                    dy = (f["raw"][1] - f["target"][1]) / px_cm
                print("  %+.2f  blink %s  глаза %-5s net %d  откр л/п %s/%s  точка %s" % (
                    f["t"] - a, "%.2f" % f["blink"] if f["blink"] is not None else "  — ",
                    f["eyes"] or "—", f["net"],
                    "%.2f" % f["open_l"] if f["open_l"] is not None else "—",
                    "%.2f" % f["open_r"] if f["open_r"] is not None else "—",
                    "dx %+5.1f dy %+5.1f см (%.1f)" % (dx, dy, e) if e is not None else "нет"))

    # курсор: насколько цель уходит от точки на экране
    rep = replay(frames, screen, px_cm)
    dev = []
    for (t, target, ev), f in zip(rep, frames):
        if target is not None:
            dev.append((t, math.hypot(target[0] - f["target"][0], target[1] - f["target"][1]) / px_cm, ev))
    print("\nкурсор от точки, см: медиана %.1f, 90%% %.1f, макс %.1f" % (
        np.median([d for _, d, _ in dev]), np.percentile([d for _, d, _ in dev], 90), max(d for _, d, _ in dev)))
    jumps, last = [], None
    for (t, target, ev) in rep:
        if target is not None and last is not None:
            j = math.hypot(target[0] - last[0], target[1] - last[1]) / px_cm
            if j > 2.0:
                jumps.append((t, j, near(t)))
        if target is not None:
            last = target
    print("прыжков курсора > 2 см: %d, из них рядом с морганием: %d" % (len(jumps), sum(n for *_, n in jumps)))
    for t, j, n in jumps[:20]:
        print("  t=%.2f  %.1f см  %s" % (t - frames[0]["t"], j, "у моргания" if n else ""))


if __name__ == "__main__":
    main()
