"""Клики левой кнопкой мыши по всей системе — для подстройки калибровки на ходу.

Опрос GetAsyncKeyState ~100 раз в секунду: без хуков и сторонних библиотек, почти не грузит
процессор. Координаты курсора — физические пиксели (процесс DPI-aware)."""
import ctypes
import threading
import time

VK_LBUTTON = 0x01


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class ClickWatcher:
    """callback(monotonic-время нажатия, x, y) в своём потоке на каждое нажатие левой кнопки."""

    def __init__(self, callback, hz=100):
        self.callback = callback
        self.period = 1.0 / hz
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        try:
            self.user32 = ctypes.windll.user32
        except AttributeError:
            return self  # не Windows — подстройки по кликам нет
        self.thread = threading.Thread(target=self._loop, daemon=True, name="clicks")
        self.thread.start()
        return self

    def _loop(self):
        down = False
        pt = _Point()
        while not self.stop_event.wait(self.period):
            now = bool(self.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
            if now and not down and self.user32.GetCursorPos(ctypes.byref(pt)):
                try:
                    self.callback(time.monotonic(), pt.x, pt.y)
                except Exception:  # подстройка не должна ронять программу
                    import traceback
                    traceback.print_exc()
            down = now

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
