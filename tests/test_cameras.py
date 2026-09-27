import threading
import types

from gaze_check.cameras import list_cameras


def test_list_cameras_numbers_in_order():
    cams = list_cameras()  # на машине без камер — пусто, но не падает
    assert [i for i, _ in cams] == list(range(len(cams)))
    assert all(isinstance(name, str) and name for _, name in cams)


def test_set_camera_asks_reader_to_reopen():
    from gaze_check.app import Engine
    eng = types.SimpleNamespace(is_file=False, args=types.SimpleNamespace(source="0"), reopen=threading.Event())
    assert Engine.set_camera(eng, "0") is False and not eng.reopen.is_set()  # та же — ничего
    assert Engine.set_camera(eng, "1") is True
    assert eng.args.source == "1" and eng.reopen.is_set()
    video = types.SimpleNamespace(is_file=True, args=types.SimpleNamespace(source="a.mp4"), reopen=threading.Event())
    assert Engine.set_camera(video, "1") is False and video.args.source == "a.mp4"


def test_list_modes_sorted_and_large_enough():
    from gaze_check.cameras import MIN_MODE_HEIGHT, list_modes
    for index, _ in list_cameras():
        modes = list_modes(index)  # [(ширина, высота, частота)], крупные первыми
        assert [m[0] * m[1] for m in modes] == sorted((m[0] * m[1] for m in modes), reverse=True)
        assert all(h >= MIN_MODE_HEIGHT for _, h, _ in modes) or all(h < MIN_MODE_HEIGHT for _, h, _ in modes)
    assert list_modes(99) == []


def test_set_size_asks_reader_to_reopen():
    from gaze_check.app import Engine
    eng = types.SimpleNamespace(is_file=False, args=types.SimpleNamespace(size="1920x1080"), reopen=threading.Event())
    assert Engine.set_size(eng, "1920x1080") is False and not eng.reopen.is_set()
    assert Engine.set_size(eng, "1280x720") is True and eng.args.size == "1280x720" and eng.reopen.is_set()


def test_frame_gate_skips_repeats_and_limits_rate():
    import numpy as np
    from gaze_check.app import FrameGate
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(40)]
    gate = FrameGate()
    kept = [gate(f, i / 30) for i, f in enumerate(frames)]
    assert all(kept) and abs(gate.fps - 30) < 1
    gate = FrameGate()  # камера повторяет каждый кадр трижды (так делал Media Foundation): новых 10 в секунду
    kept = [gate(frames[i // 3], i / 30) for i in range(90)]
    assert sum(kept) == 30 and abs(gate.fps - 10) < 1
    gate = FrameGate(limit=10)  # 30 новых в секунду, предел 10
    kept = [gate(frames[i % 40], i / 30) for i in range(90)]
    assert 28 <= sum(kept) <= 31 and abs(gate.fps - 30) < 1
