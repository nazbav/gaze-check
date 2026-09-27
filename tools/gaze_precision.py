"""Замер дрожания и точности вариантов обработки на живом человеке (~25 с).

Точки на весь экран; на каждом кадре признаки MGazeNet считаются в нескольких вариантах
(как сейчас / сглаженные рамки / + усреднение сдвинутых вырезок), точка — текущей калибровкой,
плюс медиана по нескольким кадрам. Кадры сохраняются в %LOCALAPPDATA%\\GazeCheck\\experiments.

    .venv\\Scripts\\python tools\\gaze_precision.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.calib import Screen, _prompt, _target, primary_monitor  # noqa: E402
from gaze_check.eyes import Calibration, EyeTracker  # noqa: E402
from gaze_check.gazenet import GazeNet, RectFilter, rects  # noqa: E402
from gaze_check.models import CALIBRATION, DATA_DIR, EYE_MODEL, GAZENET_MODEL, to_rgb  # noqa: E402

TTA = ((0.0, 0.0), (0.03, 0.03), (-0.03, -0.03))
VARIANTS = ("как сейчас", "сглаж. рамки", "сглаж. + 3 вырезки")


class Capture:
    def __init__(self, lead):
        self.tracker = EyeTracker(EYE_MODEL)
        self.tracker.lead = lead
        self.net = GazeNet(GAZENET_MODEL)
        self.filter = RectFilter()
        self.sink, self.stop, self.lock = None, threading.Event(), threading.Lock()
        self.state = type("S", (), {"snapshot": lambda s: {"eye": self.eye}})()
        self.eye = {}
        self.ms = {v: [] for v in VARIANTS}

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
            self.eye = {"face": pts is not None, "valid": feat is not None and "gy" in feat,
                        "marks": pts[[33, 133, 362, 263, 468, 473]].tolist() if pts is not None else [],
                        "frame": (w, h)}
            if pts is None or feat is None or "gy" not in feat:
                continue
            boxes = rects(pts, w, h)
            smooth = self.filter(ts, boxes)
            nets = {}
            for name, kw in zip(VARIANTS, ({}, {"boxes": smooth}, {"boxes": smooth, "shifts": TTA})):
                t0 = time.perf_counter()
                nets[name] = self.net(rgb, pts, **kw)
                self.ms[name].append(time.perf_counter() - t0)
            if any(v is None for v in nets.values()):
                continue
            with self.lock:
                if self.sink is not None:
                    self.sink.append({"t": ts, "feat": {k: float(v) for k, v in feat.items()
                                                        if isinstance(v, float)},
                                      "nets": {k: [round(float(x), 5) for x in v] for k, v in nets.items()}})
        cap.release()


def median_window(points, k):
    if k <= 1:
        return points
    return np.array([np.median(points[max(0, i - k + 1):i + 1], axis=0) for i in range(len(points))])


def main():
    monitor = primary_monitor()
    cal = Calibration.load(CALIBRATION, monitor)
    if cal is None or cal.netmap is None:
        sys.exit("нужна калибровка с MGazeNet")
    cap = Capture(cal.lead)
    threading.Thread(target=cap.run, daemon=True).start()
    scr = Screen(monitor)
    w, h = scr.w, scr.h
    plan = [((w // 2, h // 2), 6.0)] + [((round(fx * w), round(fy * h)), 4.0) for fy in (0.25, 0.75)
                                          for fx in (0.25, 0.75)]
    records = []
    try:
        if not _prompt(scr, cap, ["Замер дрожания: 5 точек, ~25 секунд.",
                                  "Смотрите на красную точку, пока она не сменится. Голову как обычно."]):
            return
        for p, secs in plan:
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
                _target(img, p, min(1.0, t))
                if scr.show(img, 10) == 27:
                    return
            records += [dict(r, target=list(p)) for r in got]
    finally:
        cap.stop.set()
        scr.close()
    folder = DATA_DIR / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("precision_%Y-%m-%d_%H-%M-%S.json")
    path.write_text(json.dumps({"screen": monitor, "lead": cal.lead, "frames": records}), encoding="utf-8")
    px_cm = cal.px_per_cm()
    print("кадров %d → %s" % (len(records), path))
    print("мс на кадр: %s" % {k: round(1000 * float(np.median(v)), 1) for k, v in cap.ms.items() if v})
    print("%-22s %6s %12s %12s" % ("вариант", "медиана", "дрожание", "промах"))
    for name in VARIANTS:
        for k in (1, 3, 5):
            jit, acc = [], []
            for p, _ in plan:
                fr = [r for r in records if tuple(r["target"]) == p]
                if len(fr) < 5:
                    continue
                pts = cal.netmap.predict_many([dict(r["feat"], net=r["nets"][name]) for r in fr])
                pts = median_window(pts, k)
                m = np.median(pts, axis=0)
                jit.append(np.median(np.hypot(*(pts - m).T)) / px_cm)
                acc.append(np.hypot(*(m - p)) / px_cm)
            print("%-22s %6s %9.2f см %9.2f см" % (name, "×%d" % k, np.mean(jit), np.mean(acc)))


if __name__ == "__main__":
    main()
