"""Эксперимент для сравнения моделей взгляда на живом человеке.

Показывает точки на весь экран в разных положениях (как обычно, ближе, дальше, сбоку, с
движением головы) и пишет на каждом кадре признаки обеих моделей: 3D-модели глаза (с сырыми
точками лица) и MGazeNet. Разбор — tools/gaze_analyze.py, без участия человека.

    .venv\\Scripts\\python tools\\gaze_experiment.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.calib import Screen, _target, pose_hint, primary_monitor, screen_mm  # noqa: E402
from gaze_check.eyes import EyeTracker  # noqa: E402
from gaze_check.gazenet import GazeNet, head_distance_cm  # noqa: E402
from gaze_check.models import DATA_DIR, EYE_MODEL, MODELS_DIR, to_rgb  # noqa: E402

SETTLE, COLLECT = 1.0, 1.4


def grid(w, h, fx, fy):
    return [(round(x * w), round(y * h)) for y in fy for x in fx]


def plan(w, h):
    """(этап, подсказка, точки, секунд на точку или None, качать ли головой)."""
    nine = grid(w, h, (0.08, 0.5, 0.92), (0.08, 0.5, 0.92))
    inner = grid(w, h, (0.3, 0.7), (0.3, 0.7)) + [(w // 2, round(0.2 * h))]
    five = [(w // 2, h // 2)] + grid(w, h, (0.2, 0.8), (0.25, 0.75))
    return [
        ("обычно", "Сядьте как обычно работаете. Смотрите на точку, пока она не сменится.", nine, None),
        ("проверка", "Так же, ещё 5 точек.", inner, None),
        ("голова", "Смотрите на точку и медленно качайте головой влево-вправо, вверх-вниз.",
         [(w // 2, h // 2)], 6.0),
        ("голова-угол", "То же для угла: смотрите на точку и качайте головой.", [(round(0.2 * w), round(0.25 * h))],
         6.0),
        ("ближе", "Придвиньтесь ближе к экрану — примерно на ладонь.", five, None),
        ("дальше", "Отодвиньтесь или откиньтесь назад — примерно на ладонь дальше обычного.", five, None),
        ("влево", "Сядьте обычно, но сдвиньтесь влево (голова напротив левой части экрана).", five, None),
        ("вправо", "Теперь сдвиньтесь вправо.", five, None),
        ("обычно-2", "Сядьте как в начале. Последние 5 точек.", inner, None),
    ]


class Capture:
    def __init__(self):
        self.tracker = EyeTracker(EYE_MODEL)
        self.net = GazeNet(MODELS_DIR / "mgazenet.mnn")
        self.lock = threading.Lock()
        self.sink = None  # список — сюда пишутся кадры
        self.eye = {}
        self.stop = threading.Event()
        self.fps = 0.0

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1920)
        cap.set(4, 1080)
        n, t_fps = 0, time.monotonic()
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            ts = time.monotonic()
            rgb = to_rgb(frame)
            feat, pts, _ = self.tracker.process(rgb, ts)
            h, w = rgb.shape[:2]
            try:
                net = self.net(rgb, pts) if pts is not None else None
            except Exception as e:  # noqa: BLE001 — один плохой кадр не должен ронять запись
                print("кадр пропущен:", e, flush=True)
                net = None
            marks = pts[[33, 133, 362, 263, 468, 473]].tolist() if pts is not None else []
            self.eye = {"face": pts is not None, "valid": feat is not None and net is not None, "marks": marks,
                        "frame": (w, h), "dist": head_distance_cm(pts, w, h) if pts is not None else None}
            with self.lock:
                if self.sink is not None and feat is not None and "gy" in feat and net is not None:
                    self.sink.append({"t": ts, "feat": {k: (v if k == "_raw" else float(v))
                                                        for k, v in feat.items()},
                                      "net": [round(float(v), 5) for v in net],
                                      "dist": self.eye["dist"]})
            n += 1
            if ts - t_fps >= 2:
                self.fps, n, t_fps = n / (ts - t_fps), 0, ts
        cap.release()


def main():
    monitor = primary_monitor()
    size = screen_mm(monitor)
    cap = Capture()
    threading.Thread(target=cap.run, daemon=True).start()
    scr = Screen(monitor)
    records = []
    try:
        for phase, hint, targets, hold in plan(scr.w, scr.h):
            while True:  # подсказка: ждём пробел, заодно видно, как сидит человек
                img = scr.canvas()
                ok, pose = pose_hint(cap.eye)
                dist = cap.eye.get("dist")
                scr.text(img, ["Этап «%s»" % phase, hint, "Пробел — начать, Esc — закончить"])
                scr.text(img, [pose + (" · до лица ~%d см" % dist if dist else "") + " · %.0f к/с" % cap.fps],
                         y=scr.h * 2 // 3, color=(90, 220, 90) if ok else (80, 80, 240))
                key = scr.show(img)
                if key == 27:
                    return save(records, monitor, size)
                if key == 32:
                    break
            for p in targets:
                start, got = time.monotonic(), []
                collect = hold or COLLECT
                while True:
                    t = time.monotonic() - start
                    if t >= SETTLE and cap.sink is None:
                        with cap.lock:
                            cap.sink = got
                    if t >= SETTLE + collect:
                        with cap.lock:
                            cap.sink = None
                        break
                    img = scr.canvas()
                    if hold:
                        scr.text(img, ["Смотрите на точку и качайте головой"], y=p[1] + scr.font * 3)
                    _target(img, p, min(1.0, t / SETTLE))
                    if scr.show(img, 10) == 27:
                        return save(records, monitor, size)
                records += [dict(r, phase=phase, target=list(p)) for r in got]
        img = scr.canvas()
        scr.text(img, ["Готово, спасибо! Записано кадров: %d" % len(records)])
        scr.show(img, 1500)
    finally:
        cap.stop.set()
        scr.close()
    return save(records, monitor, size)


def save(records, monitor, size):
    folder = DATA_DIR / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("gaze_%Y-%m-%d_%H-%M-%S.json")
    path.write_text(json.dumps({"screen": monitor, "size_mm": size, "frames": records}), encoding="utf-8")
    print("кадров:", len(records), "→", path, flush=True)
    by = {}
    for r in records:
        by[r["phase"]] = by.get(r["phase"], 0) + 1
    print(by, flush=True)
    return path


if __name__ == "__main__":
    main()
