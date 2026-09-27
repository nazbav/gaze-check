"""Программа ничего не передаёт: у MediaPipe отрезана отправка статистики (gaze_check/offline.py)."""
import time

import cv2


def test_mediapipe_cannot_send_and_closes_fast():
    from gaze_check import mp as gmp
    from gaze_check.detector import Detector
    from gaze_check.models import DETECTOR_MODEL, to_rgb
    gmp.mediapipe()
    assert set(gmp._offline()) == {"HttpSendRequestA", "InternetOpenA", "InternetReadFile"}
    rgb = to_rgb(cv2.imread("tests/data/person.jpg"))
    t = time.perf_counter()
    Detector(DETECTOR_MODEL).detect(rgb)  # создать, найти, закрыть (временный объект)
    assert time.perf_counter() - t < 10  # с отправкой закрытие ждало сеть ~42 с
