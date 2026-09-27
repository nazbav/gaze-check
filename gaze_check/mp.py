"""Импорт MediaPipe без matplotlib и sounddevice.

mediapipe при импорте тянет vision/drawing_utils (`import matplotlib.pyplot`) и audio/audio_record
(`import sounddevice`). Нам не нужно ни то ни другое, а стоят они ~0.5 с на запуск; в exe
matplotlib ещё и строит кэш шрифтов при каждом старте. Поэтому до импорта подкладываем пустые
модули-заглушки (только если настоящие ещё не загружены).
"""
import sys
import types

_STUBS = ("matplotlib", "matplotlib.pyplot", "sounddevice")


def mediapipe():
    """→ (mp, vision, BaseOptions)."""
    for name in _STUBS:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__gaze_check_stub__ = True
            sys.modules[name] = stub
    if not hasattr(sys.modules["matplotlib"], "pyplot"):
        sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    _offline()
    return mp, vision, BaseOptions


_blocked = None


def _offline():
    """Загрузить библиотеку MediaPipe и сразу отрезать ей сеть (см. offline.py) — до первой модели."""
    global _blocked
    if _blocked is None:
        from mediapipe.tasks.python.core import mediapipe_c_bindings
        from .offline import block_network
        _blocked = block_network(mediapipe_c_bindings.load_raw_library()._handle)
    return _blocked
