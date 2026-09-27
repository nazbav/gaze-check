from gaze_check.menu import DWELL_S, PAGES, REPEAT_S, MenuLogic, Overlay

SCREEN = (0, 0, 2560, 1600)
CENTER = {i: ((i % 3 + 0.5) * 2560 / 3, (i // 3 + 0.5) * 1600 / 3) for i in range(9)}


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def make():
    clock, done = Clock(), []
    menu = MenuLogic(run=done.append, clock=clock)
    menu.toggle()
    return menu, clock, done


def look(menu, clock, tile, secs, fps=15):
    out = []
    for _ in range(round(secs * fps)):
        clock.t += 1 / fps
        a = menu.update(CENTER[tile] if tile is not None else None, SCREEN)
        if a:
            out.append(a)
    return out


def test_tiles_follow_screen_zones():
    assert [MenuLogic.tile_of(CENTER[i], SCREEN) for i in range(9)] == list(range(9))
    assert MenuLogic.tile_of((5000, 800), SCREEN) is None and MenuLogic.tile_of(None, SCREEN) is None
    assert all(len(tiles) == 9 for _, tiles in PAGES.values())
    assert all(tiles[4][1][0] in ("app", "page") for _, tiles in PAGES.values())  # центр — назад/закрыть


def test_dwell_selects_and_closes():
    menu, clock, done = make()
    assert look(menu, clock, 0, DWELL_S * 0.7) == []  # рано
    assert look(menu, clock, 0, DWELL_S * 0.5) == [("keys", "win", "h")]  # голосовой ввод
    assert not menu.open and done == [("keys", "win", "h")]


def test_short_glance_away_keeps_progress_long_resets():
    menu, clock, done = make()
    look(menu, clock, 0, DWELL_S * 0.6)
    look(menu, clock, 1, 0.13)  # дрожание на соседнюю плитку
    assert look(menu, clock, 0, DWELL_S * 0.5)  # дозрело
    menu, clock, done = make()
    look(menu, clock, 0, DWELL_S * 0.6)
    look(menu, clock, 1, 0.5)  # ушёл по-настоящему
    assert look(menu, clock, 0, DWELL_S * 0.6) == []


def test_pages_and_repeat_while_looking():
    menu, clock, done = make()
    look(menu, clock, 1, DWELL_S + 0.1)  # «Прокрутка ▸»
    assert menu.page == "scroll" and menu.open
    acts = look(menu, clock, 7, DWELL_S + 3 * REPEAT_S + 0.05)  # «▼ Вниз» — листает, пока смотрят
    assert acts and all(a == ("wheel", -3) for a in acts) and len(acts) >= 3 and menu.open
    look(menu, clock, 4, DWELL_S + 0.1)  # центр — назад
    assert menu.page == "main"


def test_no_dangerous_actions_and_help_needs_confirmation():
    dangerous = {("keys", "alt", "f4"), ("keys", "ctrl", "w"), ("keys", "win", "plus"), ("keys", "win", "esc"),
                 ("keys", "win", "ctrl", "enter"), ("keys", "win", "ctrl", "s")}
    assert not {a for _, tiles in PAGES.values() for _, a, _ in tiles if a} & dangerous
    menu, clock, done = make()
    look(menu, clock, 8, DWELL_S + 0.1)  # «Ещё ▸»
    assert look(menu, clock, 6, DWELL_S + 0.1) == []  # «Позвать на помощь» — только просит подтвердить
    assert "подтвержд" in menu.snapshot()["note"] and done == []
    assert look(menu, clock, 6, DWELL_S + 0.1) == [("app", "help")]


def test_empty_tile_does_nothing():
    menu, clock, done = make()
    look(menu, clock, 7, DWELL_S + 0.1)  # «Звук ▸»
    assert look(menu, clock, 8, DWELL_S * 3) == [] and menu.open and done == []


def test_double_blink_selects_immediately():
    menu, clock, done = make()
    look(menu, clock, 3, 0.3)
    assert menu.select() == ("app", "mouse") and not menu.open


def test_overlay_draws_and_hides():
    ov = Overlay((0, 0, 800, 500))
    try:
        ov.tick({"open": True, "page": "main", "tile": 1, "progress": 0.5, "confirm": None, "flash": None,
                 "note": ""})
        assert ov.shown and len(ov.canvas.find_all()) > 18
        ov.tick({"open": False, "page": "main", "tile": None, "progress": 0, "confirm": None, "flash": None,
                 "note": ""})
        assert not ov.shown
    finally:
        ov.close()


def test_menu_gesture_respects_setting():
    """Выключили «открывать морганием» — три моргания меню не открывают (было: открывали)."""
    import types
    from gaze_check.app import Engine
    opened, sounds = [], []

    def fake(menu_gestures, events, open_=False):
        eng = types.SimpleNamespace(
            gestures=types.SimpleNamespace(update=lambda t, b: events), mouse=None, closure_used=False,
            menu=types.SimpleNamespace(open=open_, select=lambda: None), sound=sounds.append,
            menu_gestures=menu_gestures, addons=[], toggle_menu=lambda: opened.append(True))
        Engine._gestures(eng, 0.0, 0.1)
        return eng

    fake(False, ["menu_armed", "menu"])
    assert opened == [] and sounds == []  # ни меню, ни писка
    fake(True, ["menu_armed", "menu"])
    assert opened == [True] and sounds == ["menu_armed"]
    fake(False, ["menu"], open_=True)  # уже открытое — закрыть можно всегда
    assert opened == [True, True]


class Addon:
    """Дополнение для тестов: забирает события из takes."""
    listening = busy = False
    status = ""

    def __init__(self, takes=()):
        self.takes, self.taken = takes, []

    def gesture(self, ev, ts):
        if ev in self.takes:
            self.taken.append(ev)
            return True
        return False


def test_triple_blink_with_gaze_mouse_waits_for_steady_gaze():
    """Мышь взглядом: три моргания (меню морганием выключено) — ждать взгляда для Win+Tab;
    если три моргания забрало дополнение — не ждать."""
    import types
    from gaze_check.app import Engine
    armed, sounds = [], []

    def fake(addon):
        eng = types.SimpleNamespace(
            gestures=types.SimpleNamespace(update=lambda t, b: ["menu"]),
            mouse=types.SimpleNamespace(arm_hold=armed.append), closure_used=False,
            menu=types.SimpleNamespace(open=False, select=lambda: None), sound=sounds.append,
            menu_gestures=False, addons=[addon], toggle_menu=lambda: None)
        Engine._gestures(eng, 5.0, 0.1)

    busy = Addon(takes=("menu",))
    fake(busy)
    assert busy.taken == ["menu"] and armed == []
    fake(Addon())
    assert armed == [5.0] and sounds == ["hold_armed"]


def test_addon_takes_long_closure():
    """Дополнение забрало «глаза закрыты 2 с» — правого клика и меню из того же закрытия уже нет."""
    import types
    from gaze_check.app import Engine
    from gaze_check.mouse import Gestures
    opened, addon = [], Addon(takes=("armed",))
    eng = types.SimpleNamespace(gestures=Gestures(), addons=[addon], mouse=None, closure_used=False,
                                menu=types.SimpleNamespace(open=False, select=lambda: None), sound=lambda k: None,
                                menu_gestures=True, toggle_menu=lambda: opened.append(True))
    fps, events = 15, []
    for i, score in enumerate([0.05] * 45 + [0.9] * 60 + [0.05] * 45):  # глаза закрыты 4 с
        events += Engine._gestures(eng, i / fps, score)
    assert addon.taken == ["armed"] and opened == []
    assert not {"armed", "right", "menu", "menu_armed"} & set(events)


def test_mute_silences_gesture_beeps(monkeypatch):
    """«Без звука» глушит и писк жестов глазами, не только тревогу."""
    import types

    import gaze_check.app as app
    heard = []
    monkeypatch.setattr(app, "beep", lambda *a: heard.append(a))
    engine = types.SimpleNamespace(journal=types.SimpleNamespace(sound=False), SOUNDS=app.Engine.SOUNDS,
                                   SAY=app.Engine.SAY)
    for kind in ("left", "armed", "menu"):
        app.Engine.sound(engine, kind)
    assert heard == []
    engine.journal.sound = True
    app.Engine.sound(engine, "left")
    assert heard == [app.Engine.SOUNDS["left"]]


def test_task_view_events_press_win_tab(monkeypatch):
    import types
    import gaze_check.menu as menu
    from gaze_check.app import Engine
    pressed, sounds = [], []
    monkeypatch.setattr(menu, "press", lambda *keys: pressed.append(keys))
    eng = types.SimpleNamespace(sound=sounds.append)
    Engine._mouse_events(eng, ["hold"])
    Engine._mouse_events(eng, ["hold_lost"])
    assert pressed == [("win", "tab")] and sounds == ["task_view", "hold_lost"]


def test_gaze_dot_moves_slowly_and_ignores_jitter():
    from gaze_check.menu import GazeDot
    dot = GazeDot(74.0)
    try:
        dot.tick((500, 500))
        assert dot.pos == (500, 500) and dot.shown
        assert (dot.root.winfo_width(), dot.root.winfo_height()) == (1, 1)  # один пиксель, не кружок
        assert dot.color.startswith("#") and len(dot.color) == 7
        dot.tick((530, 510))  # дрожание < 0,8 см — стоит
        assert dot.pos == (500, 500)
        import time
        time.sleep(0.05)
        dot.tick((900, 500))  # ушёл далеко — подплывает, а не прыгает
        assert 500 < dot.pos[0] < 700
        dot.tick(None)
        assert not dot.shown
    finally:
        dot.close()


def test_dot_color_contrasts_background():
    from gaze_check.menu import contrast
    assert contrast((255, 255, 255)) == "#000000"  # белая страница — чёрная точка
    assert contrast((30, 30, 30)) == "#e1e1e1"  # тёмная тема — светлая
    assert contrast((128, 128, 128)) in ("#000000", "#ffffff")  # серый: обратный сольётся
    assert contrast((0, 90, 200)) == "#ffa537"  # синий — оранжевая
