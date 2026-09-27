import numpy as np

from gaze_check.zones import (ALL_ZONES, NO_DATA, OFF_SCREEN, AttentionLog, WindowProbe, draw_zone_overlay,
                              summarize, zone_of, zones_image)

W, H = 1920, 1080


def test_zone_grid_and_borders():
    assert zone_of(None, W, H) == NO_DATA
    assert zone_of((float("nan"), 10), W, H) == NO_DATA
    assert zone_of((0, 0), W, H) == "верх-лево"
    assert zone_of((W / 3 - 0.1, 0), W, H) == "верх-лево"
    assert zone_of((W / 3, 0), W, H) == "верх-центр"
    assert zone_of((W / 2, H / 2), W, H) == "центр"
    assert zone_of((W - 1, H / 2), W, H) == "центр-право"
    assert zone_of((W - 1, H - 1), W, H) == "низ-право"
    assert zone_of((W / 2, H), W, H) == "низ-центр"  # ровно на краю — ещё экран


def test_zone_margin_outside_screen():
    assert zone_of((-0.05 * W, H / 2), W, H) == "центр-лево"  # чуть за краем — погрешность, ещё экран
    assert zone_of((1.08 * W, 0), W, H) == "верх-право"
    assert zone_of((-0.2 * W, H / 2), W, H) == OFF_SCREEN
    assert zone_of((W / 2, 1.15 * H), W, H) == OFF_SCREEN
    assert zone_of((W / 2, 1.15 * H), W, H, margin=0.2) == "низ-центр"


def feed(log, t0, seconds, key, fps=30):
    n = round(seconds * fps)
    for i in range(n):
        log.add(t0 + i / fps, *key)
    return t0 + n / fps


def test_short_jump_does_not_break_segment():
    log = AttentionLog()
    a, b = ("центр", "Code.exe", "app.py"), ("верх-право", "chrome.exe", "Site")
    t = feed(log, 0.0, 1.0, a)
    t = feed(log, t, 0.1, b)  # взгляд дёрнулся на 0.1 с
    feed(log, t, 1.0, a)
    segs = log.finish()
    assert len(segs) == 1 and segs[0]["zone"] == "центр" and segs[0]["app"] == "Code.exe"
    assert abs(segs[0]["duration_s"] - 2.1) < 0.05


def test_real_switch_starts_at_leave_time():
    log = AttentionLog()
    t = feed(log, 0.0, 1.0, ("центр", "Code.exe", "app.py"))
    feed(log, t, 1.0, ("низ-лево", "chrome.exe", "Site"))
    segs = log.finish()
    assert [s["zone"] for s in segs] == ["центр", "низ-лево"]
    assert abs(segs[0]["end"] - 1.0) < 1e-6 and segs[0]["end"] == segs[1]["start"]
    assert segs[1]["window"] == "Site"


def test_jitter_away_switches_to_most_frequent():
    log = AttentionLog()
    t = feed(log, 0.0, 1.0, ("центр", "", ""))
    b, c = ("верх-лево", "", ""), ("верх-центр", "", "")
    for i in range(30):  # 1 с дрожит на границе двух зон, чаще — в b
        log.add(t + i / 30, *(c if i % 3 == 0 else b))
    segs = log.finish()
    assert segs[0]["zone"] == "центр" and segs[1]["zone"] == "верх-лево"
    total = sum(s["duration_s"] for s in segs)
    assert abs(total - (segs[-1]["end"] - segs[0]["start"])) < 1e-6


def test_summary_zones_apps_windows():
    segs = [
        {"start": 0, "end": 6, "duration_s": 6.0, "zone": "центр", "app": "Code.exe", "window": "app.py"},
        {"start": 6, "end": 9, "duration_s": 3.0, "zone": "верх-право", "app": "chrome.exe", "window": "Site"},
        {"start": 9, "end": 10, "duration_s": 1.0, "zone": NO_DATA, "app": "", "window": ""},
    ]
    s = summarize(segs)
    assert s["total_s"] == 10.0
    assert set(s["zones"]) == set(ALL_ZONES)
    assert s["zones"]["центр"] == {"seconds": 6.0, "share": 0.6}
    assert s["zones"][NO_DATA]["share"] == 0.1 and s["zones"]["низ-лево"]["share"] == 0.0
    assert list(s["apps"]) == ["Code.exe", "chrome.exe"]
    assert s["windows"][0] == {"app": "Code.exe", "window": "app.py", "seconds": 6.0, "share": 0.6}
    assert len(s["windows"]) == 2
    img = zones_image(None, s)
    assert img.shape == (720, 1280, 3)


def test_overlay_draws_in_place():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_zone_overlay(frame, "центр", "Code.exe")
    assert out is frame and frame.any()


def test_window_probe_returns_strings_and_throttles():
    clock = [0.0]
    probe = WindowProbe((0, 0, 1920, 1080), clock=lambda: clock[0])
    first = probe((960, 540))
    assert isinstance(first, tuple) and len(first) == 2 and all(isinstance(v, str) for v in first)
    assert probe((10, 10)) is first  # раньше interval — прошлый ответ
    assert probe(None) == ("", "")
    clock[0] = 1.0
    again = probe((960, 540))
    assert all(isinstance(v, str) for v in again)
