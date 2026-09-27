"""Меню трея строится целиком: pystray сразу зовёт checked/text у каждого пункта — ошибка там роняет exe."""
import pytest


class Built(Exception):
    pass


class FakeIcon:
    made = []

    def __init__(self, *args):
        self.menu = None
        FakeIcon.made.append(self)

    def run_detached(self):
        raise Built  # меню уже собрано — дальше камера и модели, в тесте не нужны

    def notify(self, *args):
        pass


def walk(menu):
    for item in menu.items:
        yield item
        if item.submenu:
            yield from walk(item.submenu)


@pytest.mark.parametrize("show_dot", [False, True])
def test_tray_menu_builds(monkeypatch, show_dot):
    import gaze_check.tray as tray  # не при сборе тестов: pystray и прочее грузятся только здесь
    from gaze_check.app import _main
    settings = dict(tray.DEFAULTS, show_dot=show_dot)
    monkeypatch.setattr(tray.pystray, "Icon", FakeIcon)
    monkeypatch.setattr(tray, "single_instance", lambda: True)
    monkeypatch.setattr(tray, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(tray, "save_settings", lambda s: None)
    FakeIcon.made.clear()
    with pytest.raises(Built):
        _main([])
    items = list(walk(FakeIcon.made[-1].menu))
    texts = [item.text for item in items]
    for item in items:
        item.checked  # noqa: B018 — ради вызова лямбды
    assert "Показывать точку взгляда" in texts and "Лёгкий журнал взгляда (без видео)" in texts
    assert "Проигрыватель журнала взгляда" in texts
    dot = next(i for i in items if i.text == "Показывать точку взгляда")
    assert dot.checked is show_dot
