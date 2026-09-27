"""MGazeNet из GazeFollower (Zhu et al., ETRA 2025; веса — CC BY-NC-SA 4.0, models/mgazenet-LICENSE.txt).

Нейросеть по лицу (224×224), двум глазам (112×112) и положению лица и глаз в кадре выдаёт
признаки взгляда; калибровка переводит их в точку экрана. Училась на миллионах снимков с
естественными поворотами головы и расстояниями — поэтому, в отличие от геометрии по точкам
зрачков, не разваливается, когда двигаешь головой или отсаживаешься. ~6 МБ, MNN на процессоре.

Рамки лица и глаз — как в GazeFollower (MediaPipeFaceAlignment), по тем же точкам MediaPipe.
"""
import math
import threading
import traceback

import cv2
import numpy as np


def rects(pts, width, height):
    """Точки лица (478×2, пиксели) → рамки (x, y, w, h) лица, «левого» и «правого» глаза
    или None, если глаз у края кадра. «Левый» — глаз слева на кадре (по точкам 33/133)."""
    min_x, min_y = np.maximum(pts[:468].min(0), 0)
    max_x, max_y = np.minimum(pts[:468].max(0), (width, height))
    delta = ((max_x - min_x) - (max_y - min_y)) / 4  # лицо — квадрат
    x0, y0 = max(0.0, min_x + delta), max(0.0, min_y - delta)
    x1, y1 = min(width - 1.0, max_x - delta), min(height - 1.0, max_y + delta)
    face = (int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0))
    unit = abs(pts[362][0] - pts[133][0]) / 100  # расстояние между внутренними уголками — 100 единиц
    eyes = []
    for outer, inner in ((33, 133), (362, 263)):
        xs = (pts[outer][0] - 20 * unit, pts[inner][0] + 20 * unit)
        if xs[1] - xs[0] < 8:  # голова сильно повёрнута — уголки глаза сошлись или перепутались
            return None
        h = (xs[1] - xs[0]) * 0.75
        yc = (pts[outer][1] + pts[inner][1]) / 2
        box = (int(xs[0]), int(yc - h * 0.6), int(xs[1]), int(yc + h * 0.4))
        if box[0] <= 0 or box[1] <= 0 or box[2] >= width or box[3] >= height:
            return None
        eyes.append((box[0], box[1], box[2] - box[0], box[3] - box[1]))
    if face[2] < 16 or face[3] < 16:
        return None
    return face, eyes[0], eyes[1]


def _patch(rgb, rect, size):
    x, y, w, h = rect
    crop = rgb[y:y + h, x:x + w]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, size).astype(np.float32) / 255.0


class GazeNet:
    def __init__(self, model_path, threads=2):
        import MNN
        self.MNN = MNN
        rt = MNN.nn.create_runtime_manager(({"precision": "low", "backend": 0, "numThread": threads},))
        # байтами было бы лучше, но MNN читает только файл; путь с кириллицей он открывает (проверено)
        self.module = MNN.nn.load_module_from_file(str(model_path), ["face", "left", "right", "rect"],
                                                   ["output_0"], runtime_manager=rt)
        ph = MNN.expr.placeholder
        self.vars = [ph((1, 224, 224, 3), MNN.expr.NHWC), ph((1, 112, 112, 3), MNN.expr.NHWC),
                     ph((1, 112, 112, 3), MNN.expr.NHWC), ph((1, 12))]

    def __call__(self, rgb, pts, boxes=None, shifts=((0.0, 0.0),)):
        """RGB-кадр и точки лица → вектор признаков взгляда или None (глаза у края кадра).
        boxes — готовые рамки (например, сглаженные RectFilter); shifts — сдвиги вырезок в долях
        размера рамки: признаки по нескольким чуть сдвинутым вырезкам усредняются."""
        h, w = rgb.shape[:2]
        boxes = boxes or rects(pts, w, h)
        if boxes is None:
            return None
        outs = []
        for dx, dy in shifts:
            moved = [(int(round(x + dx * bw)), int(round(y + dy * bh)), bw, bh) for x, y, bw, bh in boxes]
            if any(x < 0 or y < 0 or x + bw >= w or y + bh >= h for x, y, bw, bh in moved):
                continue
            face, left, right = moved
            grid = np.array([v for r in (face, left, right) for v in (r[2], r[3], r[0], r[1])], np.float32)
            grid /= np.array([w, h] * 6, np.float32)
            patches = [_patch(rgb, face, (224, 224)), _patch(rgb, left, (112, 112)), _patch(rgb, right, (112, 112))]
            if any(p is None for p in patches):
                continue
            inputs = patches[:2] + [cv2.flip(patches[2], 1), grid.reshape(1, 12)]
            for var, data in zip(self.vars, inputs):
                var.write(data[None] if data.ndim == 3 else data)
            out = self.module.onForward(self.vars)[0].read()
            outs.append(np.array(out, np.float32).reshape(-1))
        return np.mean(outs, axis=0) if outs else None


# три чуть сдвинутые вырезки (доли размера рамки), признаки усредняются: на замере на человеке
# промах 2.0 см вместо 2.2 при втрое большей цене (51 мс вместо 17) — tools/gaze_precision.py
TTA_SHIFTS = ((0.0, 0.0), (0.03, 0.03), (-0.03, -0.03))


class NetWorker:
    """MGazeNet в своём потоке: цикл глаз отдаёт свежий кадр (submit) и сразу берёт последний
    готовый вектор (latest). Точки лица и моргание считаются на каждом кадре камеры (~30 к/с), а
    тяжёлая сеть (~50 мс с тремя вырезками) — в своём темпе. Когда сеть стояла прямо в цикле, он
    шёл 13,6 к/с: моргание (100–150 мс) попадало в 1–2 кадра, веки на них не успевали сомкнуться,
    и моргания не ловились. MNN отпускает GIL — оба потока работают параллельно (замер: точки
    лица 45 к/с рядом с сетью 18,7 к/с)."""

    def __init__(self, net, shifts=TTA_SHIFTS):
        self.net, self.shifts = net, shifts
        self.lock = threading.Lock()
        self.wake, self.stop_event = threading.Event(), threading.Event()
        self.pending = None  # (rgb, pts, ts) — только самый свежий кадр
        self.result = None  # (ts, вектор)
        self.thread = threading.Thread(target=self._loop, daemon=True, name="gazenet")
        self.thread.start()

    def submit(self, rgb, pts, ts):
        with self.lock:
            self.pending = (rgb, pts, ts)
        self.wake.set()

    def latest(self, now, max_age=0.25):
        """Последний вектор, если он с кадра не старше max_age секунд, иначе None."""
        with self.lock:
            r = self.result
        return r[1] if r is not None and 0 <= now - r[0] <= max_age else None

    def _loop(self):
        while not self.stop_event.is_set():
            if not self.wake.wait(0.5):
                continue
            self.wake.clear()
            with self.lock:
                job, self.pending = self.pending, None
            if job is None:
                continue
            rgb, pts, ts = job
            try:
                vec = self.net(rgb, pts, shifts=self.shifts)
            except Exception:  # один плохой кадр не должен останавливать сеть
                traceback.print_exc()
                vec = None
            if vec is not None:
                with self.lock:
                    self.result = (ts, vec)

    def stop(self):
        self.stop_event.set()
        self.wake.set()


class RectFilter:
    """Сглаживание рамок лица и глаз между кадрами (One Euro, как в GazeFollower: min_cutoff 1,
    beta 0.01): рамки по точкам лица дрожат на пару пикселей, и это дрожание идёт в признаки."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        from .eyes import OneEuro
        self.filter = OneEuro(min_cutoff, beta, d_cutoff)

    def __call__(self, t, boxes):
        if boxes is None:
            return None
        v = self.filter(t, np.array([c for b in boxes for c in b], float))
        v = np.round(v).astype(int)
        return tuple(tuple(int(c) for c in v[i:i + 4]) for i in (0, 4, 8))


def head_distance_cm(pts, width, height, fov_v=math.radians(63.0), depth=1.3):
    """Грубое расстояние до лица по ширине между внешними уголками глаз (~9 см у взрослых).
    depth — поправка на настоящий угол обзора камеры (как в калибровке 3D-модели)."""
    f_px = height / (2 * math.tan(fov_v / 2))
    span = float(np.linalg.norm(pts[263] - pts[33])) or 1.0
    return 9.0 * f_px / span * depth
