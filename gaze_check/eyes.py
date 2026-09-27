"""Точка взгляда на экране по веб-камере — через 3D-модель глаза.

MediaPipe Face Landmarker даёт 478 точек лица (с зрачками) и позу головы: матрицу, которая
переводит канонический меш лица (сантиметры) в пространство камеры. По ней центры глазных
яблок ставятся за уголками глаз, луч из камеры через зрачок на кадре входит в глазное
яблоко — так получается направление взгляда в 3D. Его пересечение с плоскостью экрана
(у ноутбука камера в рамке экрана, так что экран — это плоскость камеры) и есть точка взгляда.

Поэтому точка не зависит от наклона крышки и положения головы: всё это видно в позе.
Калибровка поправляет только личное (глаз смотрит чуть мимо своей оси, у каждого по-своему),
масштаб расстояния (настоящий угол обзора камеры неизвестен) и положение камеры над экраном.
Точность веб-камеры — порядка 2–4° (на ноутбуке ~2–4 см): видно область экрана, не кнопку.
"""
import json
import math
import time
import zipfile
from collections import deque

import numpy as np

# углы глаз, зрачки: левый/правый глаз самого человека
EYES = {"left": (362, 263, 473), "right": (133, 33, 468)}
LIDS = {"left": (386, 374), "right": (159, 145)}

FOV_V = math.radians(63.0)  # вертикальный угол обзора, который MediaPipe закладывает в позу головы
EYE_DEPTH = 1.0  # см: центр глазного яблока за серединой уголков глаза (в системе лица)
IRIS_RADIUS = 1.0  # см: от центра глазного яблока до центра радужки
ROT_KEYS = tuple("R%d" % i for i in range(9))
GEO_KEYS = ("gy", "gp", "ex", "ey", "ez") + ROT_KEYS  # что нужно калибровке от кадра
EYE_KEYS = ("gy_l", "gp_l", "gy_r", "gp_r")  # углы каждого глаза — чтобы менять ведущий глаз
LEAD_EYES = ("left", "right", "both")


def lead_angles(f, lead, eyes="both"):
    """(gy, gp) по ведущему глазу: "left"/"right" — углы этого глаза, "both" — среднее
    (а если открыт только один — его: eyes — какие глаза открыты)."""
    if lead == "both" and eyes in ("left", "right"):
        lead = eyes
    if lead in ("left", "right") and "gy_" + lead[0] in f:
        return f["gy_" + lead[0]], f["gp_" + lead[0]]
    if "gy_l" in f:
        return (f["gy_l"] + f["gy_r"]) / 2, (f["gp_l"] + f["gp_r"]) / 2
    return f["gy"], f["gp"]
RAW_POINTS = [i for c1, c2, iris in ((362, 263, 473), (133, 33, 468)) for i in (iris, c1, c2)]


# ---------- признаки кадра ----------

def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return x, i


def _fields(b):
    """Поля protobuf без схемы: (номер, тип, значение)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            v, i = b[i:i + 8], i + 8
        yield num, wt, v


def canonical_mesh(task_path):
    """468 вершин канонического лица MediaPipe (см) — из geometry_pipeline_metadata внутри .task."""
    data = zipfile.ZipFile(task_path).read("geometry_pipeline_metadata_landmarks.binarypb")
    mesh = next(v for num, wt, v in _fields(data) if num == 1 and wt == 2)
    floats = b"".join(v for num, _, v in _fields(mesh) if num == 3)  # repeated float vertex_buffer
    return np.frombuffer(floats, "<f4").reshape(-1, 5)[:, :3].astype(float)  # x, y, z, u, v


class EyeModel:
    """Направление взгляда каждого глаза в 3D по точкам лица и позе головы.

    Центр глазного яблока — за серединой НАСТОЯЩИХ уголков глаза на кадре (глубина — из позы
    по каноническому лицу): если брать уголки усреднённого лица, ошибка в 1–2 мм вбок даёт
    каждому глазу 5–10° в свою сторону — «косоглазие»."""

    def __init__(self, mesh):
        self.mids = {name: (mesh[c1] + mesh[c2]) / 2 for name, (c1, c2, _) in EYES.items()}

    @staticmethod
    def raw(pts, matrix, width, height):
        """Сырые данные кадра, из которых признаки считаются заново (для переобучения модели)."""
        return ([float(width), float(height)] + [round(float(v), 2) for i in RAW_POINTS for v in pts[i]]
                + [round(float(v), 5) for v in matrix[:3, :4].ravel()])

    def from_raw(self, raw):
        width, height = raw[0], raw[1]
        pts = np.zeros((478, 2))
        for k, i in enumerate(RAW_POINTS):
            pts[i] = raw[2 + 2 * k], raw[3 + 2 * k]
        matrix = np.eye(4)
        matrix[:3, :4] = np.array(raw[2 + 2 * len(RAW_POINTS):]).reshape(3, 4)
        return self(pts, matrix, width, height)[0]

    def __call__(self, pts, matrix, width, height):
        """→ признаки: gy, gp — поворот глаз относительно головы (рад, среднее двух глаз),
        ex, ey, ez — середина между центрами глазных яблок в пространстве камеры (см),
        R0..R8 — поворот головы; плюс то же по каждому глазу и 3D-точки для лучиков."""
        f_px = height / (2 * math.tan(FOV_V / 2))
        rot = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
        shift = matrix[:3, 3]
        out, centers, ends = {}, [], []
        yaws, pitches = [], []
        for name, (c1, c2, iris) in EYES.items():
            z = float((rot @ self.mids[name] + shift)[2])  # глубина уголков — из позы
            mu, mv = (pts[c1] + pts[c2]) / 2  # середина уголков — с кадра
            mid = np.array([(mu - width / 2) / f_px * -z, -(mv - height / 2) / f_px * -z, z])
            c = mid + rot @ np.array([0.0, 0.0, -EYE_DEPTH])
            u, v = pts[iris]
            d = np.array([(u - width / 2) / f_px, -(v - height / 2) / f_px, -1.0])
            d /= np.linalg.norm(d)
            b = float(d @ c)
            disc = b * b - float(c @ c) + IRIS_RADIUS ** 2
            if disc >= 0:
                p = (b - math.sqrt(disc)) * d  # ближняя к камере точка сферы
            else:  # шум: луч прошёл мимо — ближайшая к лучу точка сферы
                q = b * d - c
                p = c + IRIS_RADIUS * q / (np.linalg.norm(q) or 1.0)
            g = (p - c) / IRIS_RADIUS
            h = rot.T @ g  # в системе головы: +z — из лица вперёд
            yaw, pitch = math.atan2(h[0], h[2]), math.asin(max(-1.0, min(1.0, h[1])))
            out["gy_" + name[0]], out["gp_" + name[0]] = yaw, pitch
            yaws.append(yaw)
            pitches.append(pitch)
            centers.append(c)
            ends.append((p, g))
        e = np.mean(centers, axis=0)
        out.update(gy=float(np.mean(yaws)), gp=float(np.mean(pitches)),
                   ex=float(e[0]), ey=float(e[1]), ez=float(e[2]))
        out.update({k: float(v) for k, v in zip(ROT_KEYS, rot.ravel())})
        return out, ends, f_px


CLOSED_EYE = 0.12  # раскрытие век (доля ширины глаза): меньше — закрыт (пока не знаем обычного уровня)
SHUT_SHARE = 0.3  # глаз раскрыт меньше этой доли своего обычного — закрыт
SQUINT_SHARE = 0.6  # меньше этой доли и вдвое уже другого глаза — прищурен или заслонён; когда смотрят
# вниз, веки опускаются у обоих глаз одинаково — это не закрытие (иначе внизу экрана более узкий
# глаз «закрывался» и сеть отключалась)


def open_eyes(opening, lead="both", typical=None):
    """{"left": раскрытие, "right": ...}, ведущий глаз → какие глаза годятся для взгляда:
    "both", "left", "right" или None (нужный глаз закрыт — моргание).

    typical — обычное раскрытие каждого глаза этого человека: глаза бывают разного разреза (один
    на кадре может быть раскрыт вдвое уже другого, 0.14 против 0.26), так что закрытым
    глаз считается относительно своего уровня. Пока уровня нет — по абсолютному порогу и
    относительно другого глаза."""
    if typical:
        share = {n: v / typical[n] for n, v in opening.items()}
        other = {"left": share["right"], "right": share["left"]}
        ok = {n: s >= SHUT_SHARE and not (s < SQUINT_SHARE and s < 0.5 * other[n]) for n, s in share.items()}
    else:
        ok = {n: v >= CLOSED_EYE and v >= 0.45 * max(opening.values()) for n, v in opening.items()}
    if lead in ("left", "right"):
        if not ok[lead]:
            return None
        return "both" if all(ok.values()) else lead
    if all(ok.values()):
        return "both"
    return next((n for n in ("left", "right") if ok[n]), None)


def eye_openings(pts):
    """Раскрытие век каждого глаза: расстояние между веками / ширина глаза."""
    out = {}
    for name, (c1, c2, _) in EYES.items():
        width = float(np.linalg.norm(pts[c2] - pts[c1]))
        top, bottom = LIDS[name]
        out[name] = float(np.linalg.norm(pts[bottom] - pts[top])) / width if width >= 1 else 0.0
    return out


def eye_features(pts, blend=None, matrix=None, lead="both", typical=None):
    """Точки лица (N×2, пиксели) → 2D-признаки взгляда (для журнала) или None (моргание:
    закрыт ведущий глаз или, если ведущий не выбран, оба). "eyes" — какие глаза открыты."""
    hx, vy, opening = {}, {}, {}
    for name, (c1, c2, iris) in EYES.items():
        a, b = pts[c1], pts[c2]
        if a[0] > b[0]:
            a, b = b, a  # a — левый по кадру угол
        axis = b - a
        w2 = float(axis @ axis)
        if w2 < 1:
            opening[name] = 0.0
            continue
        d = pts[iris] - a
        hx[name] = float(d @ axis) / w2  # 0..1 вдоль глаза слева направо по кадру
        vy[name] = float(axis[0] * d[1] - axis[1] * d[0]) / w2  # вниз от линии углов
        top, bottom = LIDS[name]
        opening[name] = float(np.linalg.norm(pts[bottom] - pts[top])) / math.sqrt(w2)
    eyes = open_eyes(opening, lead, typical)
    if eyes is None:
        return None  # моргает или прищурился — зрачок не виден
    use = list(EYES) if eyes == "both" else [eyes]
    f = {"hx": float(np.mean([hx[n] for n in use])), "vy": float(np.mean([vy[n] for n in use])),
         "open": float(np.mean([opening[n] for n in use])), "open_l": opening["left"], "open_r": opening["right"],
         "eyes": eyes,
         "look_h": 0.0, "look_v": 0.0, "yaw": 0.0, "pitch": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}
    if blend:
        f["look_h"] = (blend.get("eyeLookOutLeft", 0) + blend.get("eyeLookInRight", 0)
                       - blend.get("eyeLookInLeft", 0) - blend.get("eyeLookOutRight", 0)) / 2
        f["look_v"] = (blend.get("eyeLookUpLeft", 0) + blend.get("eyeLookUpRight", 0)
                       - blend.get("eyeLookDownLeft", 0) - blend.get("eyeLookDownRight", 0)) / 2
    if matrix is not None:
        r = matrix[:3, :3]
        f["pitch"] = math.atan2(r[2, 1], r[2, 2])
        f["yaw"] = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
        f["tx"], f["ty"], f["tz"] = (float(v) for v in matrix[:3, 3])
    return f


RAY_CM = 4.0  # длина лучика взгляда на кадре, см в пространстве камеры


def gaze_rays(pts, ends, f_px, width, height):
    """Лучики взгляда на кадре: общее 3D-направление глаз (глаза двигаются вместе; по одному
    глазу направление шумнее), спроецированное на кадр из каждого зрачка.
    Смотрит прямо в камеру — лучик короткий (он направлен на зрителя).
    ends — {глаз: (3D-точка зрачка, направление)} только открытых глаз."""
    g = np.mean([g for _, g in ends.values()], axis=0)
    g /= np.linalg.norm(g) or 1.0
    rays = []
    for name, (p, _) in ends.items():
        iris = EYES[name][2]
        q = p + g * RAY_CM
        if q[2] >= -1.0:
            continue
        end = (width / 2 + f_px * q[0] / -q[2], height / 2 - f_px * q[1] / -q[2])
        rays.append(((float(pts[iris][0]), float(pts[iris][1])), (float(end[0]), float(end[1]))))
    return rays


# ---------- калибровка ----------

# θ: усиление и сдвиг поворота глаз по горизонтали и вертикали, масштаб расстояния (настоящий
# угол обзора камеры / 63° MediaPipe), сдвиг камеры от середины экрана и её высота над экраном (см)
PARAMS = ("gain_h", "bias_h", "gain_v", "bias_v", "depth", "cam_x", "cam_top")
# По вертикали зрачок на кадре ходит в 2–3 раза меньше, чем поворачивается глаз (веки идут
# вместе с глазом) — отсюда gain_v ~2; проверено на живой калибровке.
PRIOR = np.array([1.2, 0.0, 2.0, 0.0, 1.3, 0.0, 1.0])
SPREAD = np.array([0.5, 0.2, 1.0, 0.3, 0.5, 2.0, 2.0])  # насколько далеко от PRIOR обычно
LOW = np.array([0.4, -0.5, 0.5, -0.6, 0.6, -10.0, -4.0])  # физически возможные пределы
HIGH = np.array([3.0, 0.5, 5.0, 0.6, 3.0, 10.0, 8.0])
NOISE_CM = 2.0  # разброс точки взгляда на одном кадре — вес данных против PRIOR
ROBUST_CM = 4.0  # кадр дальше от своей точки — скорее всего, смотрели мимо: вес падает
HEAD_WEIGHT = 6.0  # кадры «качаю головой» весят как 6 точек: иначе поворот головы плохо отделяется
CLICK_WEIGHT = 0.5  # клик весит как полточки калибровки: смотрят на курсор, но не всегда точно
MAX_CLICKS = 400  # последние клики для подстройки (старые вытесняются)


def _pack(f):
    """Признаки кадра для файла калибровки; сырые точки лица — чтобы пересчитать признаки,
    если модель глаза поменяется, без новой калибровки."""
    out = {k: round(float(f[k]), 6) for k in GEO_KEYS + EYE_KEYS if k in f}
    if f.get("_raw"):
        out["_raw"] = list(f["_raw"])
    if f.get("net") is not None:
        out["net"] = [round(float(v), 4) for v in f["net"]]
    return out


def _stack(feats):
    a = np.array([[f[k] for k in GEO_KEYS] for f in feats], float).reshape(-1, len(GEO_KEYS))
    return {"gy": a[:, 0], "gp": a[:, 1], "e": a[:, 2:5], "R": a[:, 5:14].reshape(-1, 3, 3)}


def project(theta, F, screen, size_mm):
    """Признаки (_stack) → точки на экране в пикселях (N×2)."""
    gain_h, bias_h, gain_v, bias_v, depth, cam_x, cam_top = theta
    yaw = gain_h * F["gy"] + bias_h
    pitch = gain_v * F["gp"] + bias_v
    h = np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=1)
    g = np.einsum("nij,nj->ni", F["R"], h)
    e = F["e"]
    t = depth * -e[:, 2] / np.maximum(g[:, 2], 0.05)  # до плоскости экрана z = 0
    x_cm, y_cm = e[:, 0] + t * g[:, 0], e[:, 1] + t * g[:, 1]
    sx, sy = screen[2] / (size_mm[0] / 10), screen[3] / (size_mm[1] / 10)  # пикселей на см
    # камера смотрит на человека: её +x — это ЕГО левая сторона, то есть левый край экрана
    return np.stack([sx * (size_mm[0] / 20 - x_cm + cam_x), sy * (-y_cm - cam_top)], axis=1)


def _fit(samples, weights, screen, size_mm, start=None, iters=40, robust=3):
    """Левенберг–Марквардт в пределах LOW..HIGH: промахи в см (с весами) плюс притяжение к
    PRIOR. Потом `robust` раз: кадры дальше ROBUST_CM от своей точки теряют вес (моргнул,
    глянул в телефон) — и подбор заново."""
    F = _stack([f for _, f in samples])
    target = np.array([p for p, _ in samples], float).reshape(-1, 2)
    w0 = np.asarray(weights, float)
    w0 = w0 / w0.sum() * max(1.0, len(set(tuple(p) for p, _ in samples)))
    px_per_cm = np.array([screen[2] / (size_mm[0] / 10), screen[3] / (size_mm[1] / 10)])
    th = np.clip(np.array(PRIOR if start is None else start, float), LOW, HIGH)
    trust = np.ones(len(target))
    for _round in range(robust + 1):
        w = np.sqrt(w0 * trust)

        def residuals(t):
            miss = (project(t, F, screen, size_mm) - target) / px_per_cm / NOISE_CM
            return np.concatenate([(miss * w[:, None]).ravel(), (t - PRIOR) / SPREAD])

        r = residuals(th)
        cost, lam = float(r @ r), 1e-3
        for _ in range(iters):
            jac = np.empty((len(r), len(th)))
            for j in range(len(th)):
                step = np.zeros_like(th)
                step[j] = 1e-5
                jac[:, j] = (residuals(th + step) - r) / 1e-5
            jtj = jac.T @ jac
            delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-9), -jac.T @ r)
            new = np.clip(th + delta, LOW, HIGH)
            r_new = residuals(new)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                moved = np.abs(new - th).max()
                th, r, cost, lam = new, r_new, cost_new, lam / 3
                if moved < 1e-6:
                    break
            else:
                lam *= 5
                if lam > 1e8:
                    break
        if _round < robust:
            miss_cm = np.hypot(*((project(th, F, screen, size_mm) - target) / px_per_cm).T)
            trust = np.minimum(1.0, (ROBUST_CM / np.maximum(miss_cm, 1e-6)) ** 2)
    return th


class Calibration:
    """Калибровка взгляда для одного экрана.

    Точку даёт MGazeNet (gaze_check.netmap), когда есть калибровка и признаки сети; иначе —
    3D-модель глаза: без калибровки на средних значениях (PRIOR). Учится на точках калибровки
    (много кадров на точку, на разных расстояниях), на кадрах «смотрю в точку и качаю головой»
    (без них после смены позы точка уезжает) и на кликах мыши: в момент клика почти всегда
    смотрят на курсор. Хранит всё это (и сырые точки лица), чтобы учиться дальше."""

    VERSION = 4

    def __init__(self, screen, size_mm, theta=None, error_px=None, created=None, points=None, clicks=None,
                 click_errors=None, check_px=None, moving=None, checks=None, lead="both"):
        self.screen = tuple(screen)  # (left, top, width, height) основного монитора, пиксели
        self.size_mm = tuple(size_mm)  # физический размер экрана
        self.theta = np.array(PRIOR if theta is None else theta, float)
        self.error_px = error_px  # перекрёстная проверка «без одной точки»
        self.check_px = check_px  # проверка на отдельных точках после калибровки
        self.jitter_px = None  # дрожание точки на точках проверки (медиана 5 кадров, как в программе)
        self.created = created
        self.points = points or []  # [(точка, признаки)] кадры калибровки по точкам
        self.moving = moving or []  # [(точка, признаки)] смотрит в точку, голова двигается
        self.checks = checks or []  # [(точка, признаки)] кадры точек проверки (в подбор не идут)
        self.clicks = clicks or []  # [(точка, признаки)] клики
        self.click_errors = click_errors or []  # промах предсказания на последних кликах, px
        self.netmap = None  # MGazeNet → экран; строится из тех же данных
        self.lead = lead  # ведущий глаз, по которому посчитаны gy/gp в хранимых кадрах

    def relead(self, lead):
        """Та же калибровка по другому ведущему глазу: углы в хранимых кадрах пересчитываются
        (углы каждого глаза хранятся; у старых кадров — из сырых точек лица) и всё подбирается заново."""
        def redo(items):
            out = []
            for p, f in items:
                f = dict(f)
                if "gy_l" not in f and f.get("_raw"):
                    f.update({k: v for k, v in _eye_model().from_raw(f["_raw"]).items() if k in EYE_KEYS})
                f["gy"], f["gp"] = lead_angles(f, lead)
                out.append((p, f))
            return out
        cal = Calibration.fit(self.screen, redo(self.points), self.size_mm, clicks=redo(self.clicks),
                              moving=redo(self.moving))
        cal.created, cal.click_errors, cal.lead = self.created, self.click_errors, lead
        checks = {}
        for p, f in redo(self.checks):
            checks.setdefault(tuple(p), []).append(f)
        cal.check(list(checks.items()))
        return cal

    def _fit_net(self):
        from .netmap import NetMap
        self.netmap = NetMap().fit(self.points, self.moving, self.clicks) if self.points else None
        return self

    @property
    def calibrated(self):
        return bool(self.points or len(self.clicks) >= 5)

    @classmethod
    def default(cls, screen, size_mm):
        return cls(screen, size_mm)

    @staticmethod
    def _data(points, moving, clicks):
        """Выборка и веса: кадры точки калибровки вместе весят 1, кадры с движением головы —
        HEAD_WEIGHT, клик — CLICK_WEIGHT."""
        per_point = {}
        for p, _ in points:
            per_point[tuple(p)] = per_point.get(tuple(p), 0) + 1
        weights = ([1.0 / per_point[tuple(p)] for p, _ in points]
                   + [HEAD_WEIGHT / max(1, len(moving))] * len(moving) + [CLICK_WEIGHT] * len(clicks))
        return list(points) + list(moving) + list(clicks), weights

    @classmethod
    def fit(cls, screen, samples, size_mm, clicks=(), moving=(), loo=True):
        """samples: [(точка (x, y) в пикселях экрана, признаки)] → калибровка + ошибка
        по перекрёстной проверке «без одной точки»."""
        samples, clicks, moving = list(samples), list(clicks), list(moving)
        theta = _fit(*cls._data(samples, moving, clicks), screen, size_mm)
        cal = cls(screen, size_mm, theta, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  points=samples, clicks=clicks, moving=moving)._fit_net()
        points = sorted({tuple(p) for p, _ in samples})
        errs = []
        if loo and len(points) >= 5:
            from .netmap import NetMap
            for hold in points:
                train = [s for s in samples if tuple(s[0]) != hold]
                test = [f for p, f in samples if tuple(p) == hold]
                net = NetMap().fit(train, moving, clicks) if cal.netmap is not None else None
                with_net = [f for f in test if f.get("net") is not None]
                if net is not None and with_net:
                    px = np.median(net.predict_many(with_net), axis=0)
                else:
                    th = _fit(*cls._data(train, moving, clicks), screen, size_mm, start=theta, robust=1)
                    px = np.median(project(th, _stack(test), screen, size_mm), axis=0)
                errs.append(float(np.hypot(px[0] - hold[0], px[1] - hold[1])))
        cal.error_px = float(np.median(errs)) if errs else None
        return cal

    def check(self, checks):
        """Точки проверки [(точка, [признаки кадров по порядку])] → медиана промаха, px; заодно
        дрожание (jitter_px): насколько точка гуляет, пока смотрят в одно место."""
        for p, feats in checks:
            self.checks += [(tuple(p), f) for f in feats]
        self.check_px, self.jitter_px = self._score(checks)
        return self.check_px

    def _score(self, checks):
        """→ (медиана промаха, медиана дрожания), px."""
        errs, jit = [], []
        for p, feats in checks:
            if feats:
                pred = median_track(self.predict_many(feats))
                m = np.median(pred, axis=0)
                errs.append(float(np.hypot(m[0] - p[0], m[1] - p[1])))
                jit.append(float(np.median(np.hypot(*(pred - m).T))))
        return (float(np.median(errs)) if errs else None), (float(np.median(jit)) if jit else None)

    def with_click(self, point, feat, reject=0.3):
        """Новая калибровка с учётом клика; None — клик отброшен (смотрели явно не на курсор).
        Промах до подстройки копится в click_errors — это честная точность на живой работе."""
        diag = float(np.hypot(self.screen[2], self.screen[3]))
        px, py = self.predict(feat)
        err = float(np.hypot(px - point[0], py - point[1]))
        if err > (reject if self.calibrated else 0.5) * diag:
            return None
        clicks = (self.clicks + [(tuple(point), _pack(feat))])[-MAX_CLICKS:]
        theta = _fit(*self._data(self.points, self.moving, clicks), self.screen, self.size_mm,
                     start=self.theta, iters=10, robust=1)
        return Calibration(self.screen, self.size_mm, theta, self.error_px, self.created, self.points, clicks,
                           (self.click_errors + [err])[-50:], self.check_px, self.moving, self.checks,
                           self.lead)._fit_net()

    @property
    def click_error_px(self):
        """Медиана промаха на последних кликах (до подстройки) или None."""
        return float(np.median(self.click_errors)) if len(self.click_errors) >= 5 else None

    def px_per_cm(self):
        return self.screen[2] / (self.size_mm[0] / 10)

    def predict(self, f):
        x, y = self.predict_many([f])[0]
        return float(x), float(y)

    def predict_many(self, feats):
        """Кадры с признаками сети — MGazeNet, без них (один глаз, глаза у края кадра) — 3D-модель."""
        feats = list(feats)
        net = [i for i, f in enumerate(feats) if f.get("net") is not None] if self.netmap is not None else []
        if len(net) == len(feats) and feats:
            return self.netmap.predict_many(feats)
        out = project(self.theta, _stack(feats), self.screen, self.size_mm)
        if net:
            out[net] = self.netmap.predict_many([feats[i] for i in net])
        return out

    @property
    def method(self):
        return "MGazeNet" if self.netmap is not None else "3D-модель"

    def to_json(self):
        def pairs(items):
            return [[list(p), _pack(f)] for p, f in items]
        return {"version": self.VERSION, "screen": self.screen, "size_mm": self.size_mm,
                "theta": dict(zip(PARAMS, (round(float(v), 5) for v in self.theta))),
                "error_px": self.error_px, "check_px": self.check_px, "jitter_px": self.jitter_px,
                "click_error_px": self.click_error_px,
                "created": self.created, "lead": self.lead, "points": pairs(self.points), "moving": pairs(self.moving),
                "checks": pairs(self.checks), "clicks": pairs(self.clicks),
                "click_errors": [round(e, 1) for e in self.click_errors]}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path, screen=None):
        """None — нет файла, старая версия или калибровка другого экрана."""
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version") != cls.VERSION or (screen is not None and tuple(d["screen"]) != tuple(screen)):
                return None

            def pairs(key):
                return [(tuple(p), f) for p, f in d.get(key, [])]
            cal = cls(d["screen"], d["size_mm"], [d["theta"][k] for k in PARAMS], d.get("error_px"),
                       d.get("created"), pairs("points"), pairs("clicks"), d.get("click_errors", []),
                       d.get("check_px"), pairs("moving"), pairs("checks"), d.get("lead", "both"))._fit_net()
            cal.jitter_px = d.get("jitter_px")
            if cal.jitter_px is None and cal.checks:  # калибровка до замера дрожания — досчитать
                groups = {}
                for p, f in cal.checks:
                    groups.setdefault(tuple(p), []).append(f)
                cal.jitter_px = cal._score(list(groups.items()))[1]
            return cal
        except (OSError, ValueError, KeyError, TypeError):
            return None


def median_track(points, k=5):
    """Скользящая медиана по k кадрам — так программа сглаживает точку взгляда."""
    points = np.asarray(points, float)
    return np.array([np.median(points[max(0, i - k + 1):i + 1], axis=0) for i in range(len(points))])


_EYE_MODEL = None


def _eye_model():
    """EyeModel для пересчёта старых кадров из сырых точек (канонический меш — из модели лица)."""
    global _EYE_MODEL
    if _EYE_MODEL is None:
        from .models import EYE_MODEL
        _EYE_MODEL = EyeModel(canonical_mesh(EYE_MODEL))
    return _EYE_MODEL


def click_features(history, t_click, before=(0.35, 0.05), min_frames=3):
    """Признаки глаз в момент клика: медиана кадров за 350–50 мс до нажатия (взгляд уже
    на цели, рука ещё не дёрнулась). history — [(ts, признаки)]; None — глаз не было видно."""
    got = [f for t, f in history if t_click - before[0] <= t <= t_click - before[1] and "gy" in f]
    if len(got) < min_frames:
        return None
    out = {k: float(np.median([f[k] for f in got])) for k in GEO_KEYS + EYE_KEYS if all(k in f for f in got)}
    if all(f.get("net") is not None for f in got):
        out["net"] = np.median(np.array([f["net"] for f in got], float), axis=0)
    return out


EDGE_PX = 30  # край калибровки: мишень (радиус до 28 px) целиком на экране, но у самой рамки


def edge_points(width, height, edge=EDGE_PX):
    """Углы экрана у самой рамки: левый верхний, правый верхний, правый нижний, левый нижний."""
    return [(edge, edge), (width - 1 - edge, edge), (width - 1 - edge, height - 1 - edge), (edge, height - 1 - edge)]


def calibration_points(width, height, edge=EDGE_PX):
    """9 точек: центр первым, затем углы и середины краёв по часовой стрелке — у самой рамки экрана
    (было 8% от края: точка взгляда у краёв недотягивала — туда калибровка не заглядывала)."""
    xs = (edge, width // 2, width - 1 - edge)
    ys = (edge, height // 2, height - 1 - edge)
    order = [(1, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    return [(xs[i], ys[j]) for i, j in order]


def check_points(width, height):
    """Точки проверки — между точками калибровки, чтобы честно видеть точность."""
    return [(round(fx * width), round(fy * height)) for fx, fy in
            ((0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.5, 0.2))]


# ---------- сглаживание, фиксации ----------

class OneEuro:
    """Сглаживание точки взгляда: на месте почти не дрожит, при быстром переводе не отстаёт."""

    def __init__(self, min_cutoff=0.8, beta=0.004, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        x = np.asarray(x, float)
        if self.prev is None or t - self.prev[0] > 1.0:
            self.prev = (t, x, np.zeros_like(x))
            return x
        t0, x0, dx0 = self.prev
        dt = max(t - t0, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * (x - x0) / dt + (1 - a_d) * dx0
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(dx)), dt)
        out = a * x + (1 - a) * x0
        self.prev = (t, out, dx)
        return out


class Fixations:
    """Фиксации по дисперсии (I-DT): взгляд держится в пятне ≤ dispersion px не меньше min_ms."""

    def __init__(self, dispersion, min_ms=120):
        self.dispersion = dispersion
        self.min_s = min_ms / 1000
        self.window = []
        self.found = []

    def _close(self):
        if self.window and self.window[-1][0] - self.window[0][0] >= self.min_s:
            pts = np.array([(x, y) for _, x, y in self.window])
            self.found.append({"order": len(self.found) + 1, "start": self.window[0][0],
                               "end": self.window[-1][0],
                               "duration_ms": round((self.window[-1][0] - self.window[0][0]) * 1000),
                               "x": round(float(pts[:, 0].mean())), "y": round(float(pts[:, 1].mean())),
                               "samples": len(pts)})
        self.window = []

    def add(self, t, x, y):
        """Возвращает номер текущей фиксации или None (саккада / ещё не набралась)."""
        cand = self.window + [(t, x, y)]
        xs, ys = [p[1] for p in cand], [p[2] for p in cand]
        if (max(xs) - min(xs)) + (max(ys) - min(ys)) <= self.dispersion:
            self.window = cand
        else:
            self._close()
            self.window = [(t, x, y)]
        if self.window[-1][0] - self.window[0][0] >= self.min_s:
            return len(self.found) + 1
        return None

    def gap(self):
        """Нет данных (моргание, отвернулся) — текущая фиксация закончилась."""
        self._close()

    def finish(self):
        self._close()
        return self.found


class Trail:
    """Последние точки взгляда для отрисовки следа."""

    def __init__(self, seconds=1.5):
        self.seconds = seconds
        self.points = deque()

    def add(self, t, x, y):
        self.points.append((t, x, y))
        while self.points and t - self.points[0][0] > self.seconds:
            self.points.popleft()


# ---------- смотрит ли в экран ----------

AWAY_YAW, AWAY_PITCH = 30.0, 25.0  # градусов от обычного положения головы (если нет позы)
AWAY_LOOK = 0.6  # eyeLook*: глаза сильно в сторону или вниз (если нет позы)
SCREEN_MARGIN = (0.12, 0.15)  # запас за краем экрана на погрешность (доли ширины/высоты)
LOOSE_MARGIN = (0.25, 0.3)  # то же без калибровки: там точка грубее
CLOSED_AFTER = 0.7  # глаза не видны дольше — «не смотрит» (опустил взгляд, закрыл глаза)


class ScreenGaze:
    """Смотрит ли человек в экран → метка "looking at screen" / "looking away" / "face not visible".

    По точке взгляда: внутри экрана с запасом на погрешность (без калибровки запас больше).
    Если точки нет — поворот головы и глаз не дальше порогов от обычного положения: медиана
    за последние 10 минут. Итог сглаживается большинством за `window` секунд, чтобы одиночный
    кадр не рвал счёт времени."""

    def __init__(self, window=0.8, baseline_s=600.0):
        self.window = window
        self.baseline_s = baseline_s
        self.recent = deque()  # (t, метка)
        self.base = deque()  # (t, yaw, pitch, look_h, look_v) — раз в полсекунды
        self.eyes_seen = -1e9

    def _baseline(self, ts, feat):
        if not self.base or ts - self.base[-1][0] >= 0.5:
            self.base.append((ts, feat["yaw"], feat["pitch"], feat["look_h"], feat["look_v"]))
            while ts - self.base[0][0] > self.baseline_s:
                self.base.popleft()
        return np.median(np.array([b[1:] for b in self.base]), axis=0)

    def raw(self, ts, feat, face, point=None, screen=None, margin=SCREEN_MARGIN):
        if not face:
            return "face not visible"
        if feat is None:
            # моргание — прошлое решение; глаз не видно дольше — опустил взгляд или закрыл глаза
            return None if ts - self.eyes_seen < CLOSED_AFTER else "looking away"
        self.eyes_seen = ts
        if point is not None and screen is not None:
            w, h = screen[2], screen[3]
            mx, my = margin[0] * w, margin[1] * h
            inside = -mx <= point[0] <= w + mx and -my <= point[1] <= h + my
            return "looking at screen" if inside else "looking away"
        base = self._baseline(ts, feat)
        yaw, pitch, look_h, look_v = feat["yaw"], feat["pitch"], feat["look_h"], feat["look_v"]
        away = (abs(math.degrees(yaw - base[0])) > AWAY_YAW or abs(math.degrees(pitch - base[1])) > AWAY_PITCH
                or abs(look_h - base[2]) > AWAY_LOOK or abs(look_v - base[3]) > AWAY_LOOK)
        return "looking away" if away else "looking at screen"

    def update(self, ts, feat, face, point=None, screen=None, margin=SCREEN_MARGIN):
        """→ (метка, доля кадров за окно с этой меткой)."""
        label = self.raw(ts, feat, face, point, screen, margin)
        if label is None:
            label = self.recent[-1][1] if self.recent else "looking at screen"
        self.recent.append((ts, label))
        while ts - self.recent[0][0] > self.window:
            self.recent.popleft()
        counts = {}
        for _, lab in self.recent:
            counts[lab] = counts.get(lab, 0) + 1
        top = max(counts, key=lambda k: (counts[k], k == label))
        return top, counts[top] / len(self.recent)


# ---------- трекер ----------

class EyeTracker:
    """MediaPipe Face Landmarker в режиме видео + 3D-модель глаза + признаки MGazeNet."""

    def __init__(self, model_path, net_path=None, async_net=False):
        from .mp import mediapipe
        self.mp, vision, BaseOptions = mediapipe()
        opts = vision.FaceLandmarkerOptions(
            # байтами, а не путём: MediaPipe не открывает пути с кириллицей
            base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.model = EyeModel(canonical_mesh(model_path))
        self.net = self.worker = None
        if net_path is not None and net_path.exists():
            from .gazenet import GazeNet, NetWorker
            self.net = GazeNet(net_path)
            # в программе сеть — в своём потоке, чтобы цикл глаз шёл с частотой камеры (моргания)
            self.worker = NetWorker(self.net) if async_net else None
        self.last_ms = -1
        self.blink = None  # 0..1 — насколько закрыты оба глаза (None — лица нет); для жестов
        self.lead = "both"  # ведущий глаз: чьи углы идут в gy/gp
        self.openings = {n: deque(maxlen=1800) for n in EYES}  # раскрытие век за ~минуту
        self.typical = None  # обычное раскрытие каждого глаза этого человека
        self.frames = 0

    def process(self, rgb, ts):
        """RGB-кадр → (признаки или None, точки лица N×2 или None, лучики взгляда)."""
        mp = self.mp
        ms = max(int(ts * 1000), self.last_ms + 1)  # MediaPipe требует строго растущее время
        self.last_ms = ms
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.face_landmarks:
            self.blink = None
            return None, None, []
        h, w = rgb.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], float)
        blend = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else None
        matrix = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        # закрыты ли глаза: по коэффициентам моргания (оба глаза — берём меньший, чтобы
        # прищур одним глазом не считался морганием); без них — по раскрытию век
        if blend:
            self.blink = min(blend.get("eyeBlinkLeft", 0.0), blend.get("eyeBlinkRight", 0.0))
        else:
            self.blink = None
        opening = eye_openings(pts)
        for n, v in opening.items():
            self.openings[n].append(v)
        self.frames += 1
        if self.frames % 15 == 0 and len(self.openings["left"]) >= 60:
            # 90-й процентиль: моргания короткие и в него не попадают; не ниже 0.08 — чтобы глаз,
            # который долго держат закрытым, не стал «обычно таким»
            self.typical = {n: max(0.08, float(np.percentile(v, 90))) for n, v in self.openings.items()}
        feat = eye_features(pts, blend, matrix, self.lead, self.typical)
        if feat is None or matrix is None:
            return feat, pts, []
        geo, ends, f_px = self.model(pts, matrix, w, h)
        feat.update(geo)
        feat["gy"], feat["gp"] = lead_angles(geo, self.lead, feat["eyes"])
        feat["_raw"] = EyeModel.raw(pts, matrix, w, h)
        if self.net is not None and feat["eyes"] == "both":
            # сеть учили на лицах с двумя открытыми глазами; с одним — 3D-модель по открытому глазу
            if self.worker is not None:
                self.worker.submit(rgb, pts, ts)
                feat["net"] = self.worker.latest(ts)  # None — сеть ещё не успела / глаза у края
            else:
                from .gazenet import TTA_SHIFTS
                feat["net"] = self.net(rgb, pts, shifts=TTA_SHIFTS)  # None — глаза у края кадра
        ends = dict(zip(EYES, ends))
        if feat["eyes"] != "both":
            ends = {feat["eyes"]: ends[feat["eyes"]]}
        return feat, pts, gaze_rays(pts, ends, f_px, w, h)
