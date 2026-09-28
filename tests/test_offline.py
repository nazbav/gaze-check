"""Программа ничего не передаёт: у MediaPipe отрезана отправка статистики (gaze_check/offline.py)."""
import time

import cv2


def test_mediapipe_cannot_send_and_closes_fast():
    from gaze_check import mp as gmp
    from gaze_check.eyes import EyeTracker
    from gaze_check.models import EYE_MODEL, to_rgb
    gmp.mediapipe()
    assert set(gmp._offline()) == {"HttpSendRequestA", "InternetOpenA", "InternetReadFile"}
    rgb = to_rgb(cv2.imread("tests/data/person.jpg"))
    t = time.perf_counter()
    EyeTracker(EYE_MODEL).process(rgb, 0.0)  # создать, найти лицо, закрыть (временный объект)
    assert time.perf_counter() - t < 10  # с отправкой закрытие ждало сеть ~42 с


def test_offline_patches_once_even_from_many_threads(monkeypatch):
    """Поток глаз и поток телефона грузят MediaPipe одновременно: подмена таблицы импорта — один раз.
    Две подмены разом снимали друг другу защиту страницы — вылет 0xc0000005 при запуске (28.09)."""
    import threading

    from gaze_check import mp as gmp
    from gaze_check import offline
    calls, inside, most = [], [0], [0]

    def slow(handle, dll="WININET.dll"):
        inside[0] += 1
        most[0] = max(most[0], inside[0])
        time.sleep(0.05)
        calls.append(handle)
        inside[0] -= 1
        return ["HttpSendRequestA"]

    monkeypatch.setattr(gmp, "_blocked", None)
    monkeypatch.setattr(offline, "block_network", slow)
    threads = [threading.Thread(target=gmp._offline) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and most[0] == 1 and gmp._blocked == ["HttpSendRequestA"]
