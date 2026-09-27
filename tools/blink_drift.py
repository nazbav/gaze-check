"""Что делает точка взгляда при моргании (~35 с, на живом человеке).

Точка на экране; человек смотрит на неё и моргает (обычно, потом двойными). Пишется КАЖДЫЙ кадр —
и с закрытыми глазами: коэффициент моргания, раскрытие век, каким способом посчитана точка
(MGazeNet по двум глазам или 3D-модель по одному) и сама точка текущей калибровкой.
Кадры — в %LOCALAPPDATA%\\GazeCheck\\experiments\\blink_*.json; разбор — tools/blink_analyze.py.

    .venv\\Scripts\\python tools\\blink_drift.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.calib import Screen, _prompt, _target, primary_monitor  # noqa: E402
from gaze_check.eyes import Calibration, EyeTracker  # noqa: E402
from gaze_check.models import CALIBRATION, DATA_DIR, EYE_MODEL, GAZENET_MODEL, to_rgb  # noqa: E402

PLAN = (  # (доля ширины, доля высоты, секунд, подсказка)
    (0.5, 0.5, 12.0, "Смотрите на точку и моргайте как обычно, можно почаще"),
    (0.25, 0.3, 12.0, "Двойные моргания (как клик), 3–4 раза с паузами"),
    (0.5, 0.95, 8.0, "Смотрите на точку внизу, моргайте как обычно (проверка: взгляд вниз — не жест)"),
    (0.6, 0.4, 12.0, "Тройные моргания (меню/голос), 2–3 раза с паузами"),
    (0.5, 0.5, 8.0, "Закройте глаза секунды на 2–3 и откройте — один раз"),
    (0.75, 0.7, 8.0, "Снова обычные моргания"),
)


class Capture:
    def __init__(self, cal):
        self.cal = cal
        self.tracker = EyeTracker(EYE_MODEL, GAZENET_MODEL, async_net=True)  # как в программе
        self.tracker.lead = cal.lead
        self.sink, self.stop, self.lock = None, threading.Event(), threading.Lock()
        self.eye = {}
        self.state = type("S", (), {"snapshot": lambda s: {"eye": self.eye}})()

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1920)
        cap.set(4, 1080)
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            ts = time.monotonic()
            rgb = to_rgb(frame)
            feat, pts, _ = self.tracker.process(rgb, ts)
            h, w = rgb.shape[:2]
            valid = feat is not None and "gy" in feat
            self.eye = {"face": pts is not None, "valid": valid,
                        "marks": pts[[33, 133, 362, 263, 468, 473]].tolist() if pts is not None else [],
                        "frame": (w, h)}
            raw = self.cal.predict(feat) if valid else None
            rec = {"t": ts, "blink": self.tracker.blink, "face": pts is not None,
                   "eyes": feat.get("eyes") if feat else None,
                   "net": bool(feat is not None and feat.get("net") is not None),
                   "open_l": feat.get("open_l") if feat else None, "open_r": feat.get("open_r") if feat else None,
                   "raw": None if raw is None else [float(raw[0]), float(raw[1])]}
            with self.lock:
                if self.sink is not None:
                    self.sink.append(rec)
        cap.release()


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # вывод в трубу — иначе cp1251 и падение на «→»
    monitor = primary_monitor()
    cal = Calibration.load(CALIBRATION, monitor)
    if cal is None:
        sys.exit("нужна калибровка")
    cap = Capture(cal)
    threading.Thread(target=cap.run, daemon=True).start()
    scr = Screen(monitor)
    w, h = scr.w, scr.h
    records = []
    try:
        if not _prompt(scr, cap, ["Моргание и точка взгляда: 6 этапов, ~1 минута.",
                                  "Смотрите на красную точку и моргайте, как подскажет надпись."]):
            return
        for fx, fy, secs, hint in PLAN:
            p = (round(fx * w), round(fy * h))
            got, start = [], time.monotonic()
            while True:
                t = time.monotonic() - start
                if t >= 1.0 and cap.sink is None:
                    with cap.lock:
                        cap.sink = got
                if t >= 1.0 + secs:
                    with cap.lock:
                        cap.sink = None
                    break
                img = scr.canvas()
                scr.text(img, [hint], y=h // 10)
                _target(img, p, min(1.0, t))
                if scr.show(img, 10) == 27:
                    return
            records += [dict(r, target=list(p), phase=hint) for r in got]
    finally:
        cap.stop.set()
        scr.close()
    folder = DATA_DIR / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("blink_%Y-%m-%d_%H-%M-%S.json")
    path.write_text(json.dumps({"screen": monitor, "lead": cal.lead, "px_per_cm": cal.px_per_cm(),
                                "frames": records}, ensure_ascii=False), encoding="utf-8")
    print("кадров %d → %s" % (len(records), path))
    from gaze_check.mouse import Gestures
    g, out = Gestures(), {}
    for r in records:
        for e in g.update(r["t"], r["blink"]):
            if e not in ("closed", "opened"):
                out.setdefault(r["phase"][:40], []).append(e)
    span = records[-1]["t"] - records[0]["t"] if records else 1
    print("частота %.1f к/с" % (len(records) / span))
    for phase in dict.fromkeys(r["phase"][:40] for r in records):
        print("  %-40s → %s" % (phase, out.get(phase, [])))


if __name__ == "__main__":
    main()
