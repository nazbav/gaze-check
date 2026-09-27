"""Синтетические признаки 3D-модели: «человек» с заданными личными параметрами смотрит в точку."""
import math

import numpy as np

from gaze_check.eyes import ROT_KEYS

SCREEN = (0, 0, 2560, 1600)
SIZE_MM = (344.0, 215.0)
# «настоящие» параметры человека: усиления, сдвиги глаз, масштаб расстояния, камера
TRUE = np.array([1.12, 0.06, 0.88, -0.04, 1.45, 0.6, 1.3])


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def feat(point, head=None, eye=(0.0, -8.0, -40.0), theta=TRUE, noise=0.0, rng=None,
         screen=SCREEN, size_mm=SIZE_MM):
    """Признаки кадра, когда человек с параметрами theta смотрит в point (пиксели экрана).
    head — поворот головы в пространстве камеры, eye — середина глаз (см, как у MediaPipe)."""
    gain_h, bias_h, gain_v, bias_v, depth, cam_x, cam_top = theta
    head = np.eye(3) if head is None else head
    sx, sy = screen[2] / (size_mm[0] / 10), screen[3] / (size_mm[1] / 10)
    x_cm = size_mm[0] / 20 + cam_x - point[0] / sx  # +x камеры — левый край экрана
    y_cm = -(point[1] / sy + cam_top)
    e = np.array(eye, float)
    g = np.array([x_cm - e[0], y_cm - e[1], -depth * e[2]])
    h = head.T @ (g / np.linalg.norm(g))
    yaw, pitch = math.atan2(h[0], h[2]), math.asin(h[1])
    f = {"gy": (yaw - bias_h) / gain_h, "gp": (pitch - bias_v) / gain_v,
         "ex": e[0], "ey": e[1], "ez": e[2]}
    if noise and rng is not None:
        f["gy"] += rng.normal(0, noise)
        f["gp"] += rng.normal(0, noise)
    f.update({k: float(v) for k, v in zip(ROT_KEYS, head.ravel())})
    # 2D-признаки для журнала и запасного решения «смотрит ли в экран»
    f.update(hx=0.5, vy=0.0, open=0.3, look_h=0.0, look_v=0.0, yaw=0.0, pitch=0.0, tx=0.0, ty=0.0, tz=e[2])
    return f
