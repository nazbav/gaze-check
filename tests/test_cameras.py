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
