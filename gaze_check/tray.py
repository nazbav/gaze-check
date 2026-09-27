"""Фоновый режим: значок в трее, писк и уведомления; окно с камерой — по клику на значок."""
import ctypes
import json
import os
import sys
import threading
import time
import traceback

import cv2
import pystray
from PIL import Image, ImageDraw

from .app import GRAY, GREEN, ORANGE, RED, Engine, Journal, Viewer, human_seconds
from .models import DATA_DIR, FROZEN, MODELS_DIR, ROOT, models_ready

APP = "GazeCheck"
SETTINGS = DATA_DIR / "settings.json"
DEFAULTS = {"away_after": 60, "sound": True, "phone": True, "clicks": True, "lead_eye": "both",
            "menu_gestures": False, "camera": "0", "mouse_mode": "joystick", "show_dot": False,
            "gaze_log": False}
LEAD_NAMES = {"left": "Левый", "right": "Правый", "both": "Оба"}
AWAY_CHOICES = (30, 60, 120, 300)
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def load_settings():
    try:
        return {**DEFAULTS, **json.loads(SETTINGS.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(settings):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(settings, ensure_ascii=False, indent=1), encoding="utf-8")


def icon_image(color):
    """Глаз: радужка цвета состояния."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2, 14, 62, 50), fill=(255, 255, 255, 255), outline=(30, 30, 30, 255), width=4)
    d.ellipse((19, 19, 45, 45), fill=color + (255,))
    d.ellipse((27, 27, 37, 37), fill=(20, 20, 20, 255))
    return img


def autostart_command():
    if FROZEN:
        return '"%s"' % sys.executable
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return '"%s" "%s"' % (pythonw, ROOT / "launcher.py")


def autostart_enabled():
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP)
            return True
    except OSError:
        return False


def set_autostart(on):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if on:
            winreg.SetValueEx(key, APP, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, APP)
            except OSError:
                pass


_mutex = None


def single_instance():
    """Второй экземпляр (автозапуск + ручной запуск) не нужен: камера одна."""
    global _mutex
    _mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\GazeCheckSingleInstance")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def message_box(text):
    ctypes.windll.user32.MessageBoxW(None, text, "Gaze Check", 0x40)


def status_of(engine):
    """(цвет, подсказка) для значка."""
    s = engine.state.snapshot()
    st, ph = s["status"], s["phone"]
    if s["gaze_error"] or s["phone_error"]:
        return RED, "ошибка — см. журнал"
    if engine.paused.is_set():
        return GRAY, "пауза"
    if s["camera_error"]:
        return RED, s["camera_error"]
    if not s["gaze_ready"]:
        return GRAY, "загрузка моделей…"
    rec = ("".join(" · " + a.status for a in engine.addons if a.status) + (" · мышь взглядом" if s["mouse"] else "")
           + (" · запись" if s["recording"] else "") + (" · журнал" if s["logging"] else "")
           + ("" if engine.journal.sound else " · без звука")
           + (" · " + s["accuracy"] if s["accuracy"] else " · без калибровки"))
    if ph.get("phone_alarm"):
        return RED, "телефон в кадре" + rec
    if st.get("prolonged_away"):
        return RED, "не смотришь в экран %d с" % st["away_seconds"] + rec
    if st.get("status") in ("brief_away", "face_not_visible"):
        return ORANGE, "не смотришь в экран %d с" % st["away_seconds"] + rec
    if st.get("status") == "no_person":
        return GRAY, "никого нет — не пищу" + rec
    if st.get("status") == "looking_at_screen":
        return GREEN, "смотришь в экран" + rec
    return GRAY, "жду первый кадр…"


def run_tray(args):
    if not single_instance():
        message_box("Gaze Check уже запущен — значок в трее (возле часов).")
        return
    settings = load_settings()
    if args.source is None:
        args.source = str(settings["camera"])
    if args.away_after is None:
        args.away_after = float(settings["away_after"])
    args.no_phone = args.no_phone or not settings["phone"]
    args.no_clicks = args.no_clicks or not settings["clicks"]
    args.lead_eye = args.lead_eye or settings["lead_eye"]
    args.mouse_mode = args.mouse_mode or settings["mouse_mode"]
    args.menu_gestures = args.menu_gestures or settings["menu_gestures"]

    icon = pystray.Icon(APP, icon_image(GRAY), "Gaze Check — запуск…")
    journal = Journal(args.log, sound=settings["sound"] and not args.quiet,
                      notify=lambda text: icon.notify(text, "Gaze Check"))
    engine = Engine(args, journal)
    ui = {"window": False, "quit": False, "calibrate": bool(args.calibrate), "gaze_test": bool(args.gaze_test),
          "dot": bool(settings["show_dot"])}  # всё, что читает меню, — до создания меню: pystray сразу зовёт checked

    def save(**kw):
        settings.update(kw)
        save_settings(settings)

    def toggle_window(_icon, _item):
        ui["window"] = not ui["window"]

    def toggle_pause(_icon, _item):
        engine.pause(not engine.paused.is_set())

    def set_away(sec):
        def action(_icon, _item):
            engine.away_after = float(sec)
            save(away_after=sec)
        return action

    def toggle_phone(_icon, _item):
        engine.phone_on = not engine.phone_on
        save(phone=engine.phone_on)

    def toggle_clicks(_icon, _item):
        engine.clicks_on = not engine.clicks_on
        save(clicks=engine.clicks_on)
        if engine.clicks_on and engine.watcher is None:
            from .clicks import ClickWatcher
            engine.watcher = ClickWatcher(lambda t, x, y: engine.clicks.append((t, x, y))).start()

    def toggle_sound(_icon, _item):
        journal.sound = not journal.sound
        save(sound=journal.sound)

    def toggle_autostart(_icon, _item):
        set_autostart(not autostart_enabled())

    def open_log(_icon, _item):
        log = DATA_DIR / "gaze-check.log"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(log if log.exists() else DATA_DIR)

    def quit_app(_icon, _item):
        ui["quit"] = True

    def calibrate(_icon, _item):
        ui["calibrate"] = True

    def gaze_test(_icon, _item):
        ui["gaze_test"] = True

    def toggle_mouse(_icon, _item):
        engine.toggle_mouse()

    def toggle_dot(_icon, _item):
        ui["dot"] = not ui["dot"]
        save(show_dot=ui["dot"])

    def set_mouse_mode(mode):
        def action(_icon, _item):
            engine.mouse_mode = mode
            save(mouse_mode=mode)
            if engine.mouse is not None:  # включена — пересоздать в новом режиме
                engine.set_mouse(False)
                engine.set_mouse(True)
        return action

    def toggle_menu(_icon, _item):
        engine.toggle_menu()

    def toggle_menu_gestures(_icon, _item):
        engine.menu_gestures = not engine.menu_gestures
        save(menu_gestures=engine.menu_gestures)

    def set_lead(lead):
        def action(_icon, _item):
            engine.set_lead_eye(lead)
            save(lead_eye=lead)
        return action

    from .cameras import list_cameras
    cameras = list_cameras() or [(int(args.source), "Камера %s" % args.source)]  # перечислить не вышло

    def set_camera(index, name):
        def action(_icon, _item):
            if engine.set_camera(str(index)):
                save(camera=str(index))
                icon.notify("Камера: %s. Калибровка делалась на прежней — лучше откалибровать заново." % name,
                            "Gaze Check")
        return action

    def quick(_icon, _item):
        engine.requests.add("quick")

    def toggle_record(_icon, _item):
        if engine.recorder is not None:
            folder = engine.stop_recording()
            icon.notify("Запись сохранена: %s" % folder, "Gaze Check")
            return
        engine.start_recording()
        hint = "" if engine.calibration.calibrated else " Без калибровки точка грубее — лучше её сделать."
        icon.notify("Пишу экран, камеру и взгляд. Остановить — в меню.%s" % hint, "Gaze Check")

    def toggle_gazelog(_icon, _item):
        path = engine.set_log(engine.gazelog is None)
        save(gaze_log=engine.gazelog is not None)
        if path:
            icon.notify("Журнал взгляда сохранён: %s" % path.name, "Gaze Check")

    def open_player(_icon, _item):
        import subprocess
        cmd = [sys.executable, "--play"] if FROZEN else [sys.executable, "-m", "gaze_check", "--play"]
        subprocess.Popen(cmd, cwd=str(ROOT), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def open_logs(_icon, _item):
        from .gazelog import logs_dir
        logs_dir().mkdir(parents=True, exist_ok=True)
        os.startfile(logs_dir())

    def open_records(_icon, _item):
        from .recorder import records_dir
        records_dir().mkdir(parents=True, exist_ok=True)
        os.startfile(records_dir())

    item, menu = pystray.MenuItem, pystray.Menu
    icon.menu = menu(
        item("Показать камеру", toggle_window, default=True, checked=lambda i: ui["window"]),
        item("Без звука", toggle_sound, checked=lambda i: not journal.sound),
        item("Пауза (камера выключена)", toggle_pause, checked=lambda i: engine.paused.is_set()),
        item("Камера", menu(*[
            item(name, set_camera(index, name), radio=True,
                 checked=lambda i, index=index: engine.args.source == str(index))
            for index, name in cameras])),
        menu.SEPARATOR,
        item("Калибровка взгляда…", calibrate),
        item("Быстрая подстройка (Ctrl+Alt+C)", quick),
        item("Проверка взгляда…", gaze_test),
        item("Показывать точку взгляда", toggle_dot, checked=lambda i: ui["dot"]),
        item("Мышь — взглядом (Ctrl+Alt+G)", toggle_mouse, checked=lambda i: engine.mouse is not None),
        item("Мышь взглядом: как двигать", menu(
            item("Джойстик — едет в сторону взгляда", set_mouse_mode("joystick"), radio=True,
                 checked=lambda i: engine.mouse_mode == "joystick"),
            item("Прямо за взглядом", set_mouse_mode("direct"), radio=True,
                 checked=lambda i: engine.mouse_mode == "direct"))),
        item("Меню взглядом: открывать морганием (3 раза)", toggle_menu_gestures,
             checked=lambda i: engine.menu_gestures),
        item("Открыть меню взглядом (Ctrl+Alt+M)", toggle_menu),
        *[x for addon in engine.addons for x in addon.menu_items(item, settings, save)],
        item("Подстройка по кликам", toggle_clicks, checked=lambda i: engine.clicks_on),
        item("Ведущий глаз", menu(*[
            item(LEAD_NAMES[lead], set_lead(lead), radio=True, checked=lambda i, lead=lead: engine.lead == lead)
            for lead in ("left", "right", "both")])),
        item("Запись экрана и взгляда", toggle_record, checked=lambda i: engine.recorder is not None),
        item("Открыть записи", open_records),
        item("Лёгкий журнал взгляда (без видео)", toggle_gazelog, checked=lambda i: engine.gazelog is not None),
        item("Проигрыватель журнала взгляда", open_player),
        item("Открыть журналы взгляда", open_logs),
        menu.SEPARATOR,
        item("Пищать, если не смотрю в экран", menu(*[
            item(human_seconds(sec), set_away(sec), radio=True,
                 checked=lambda i, sec=sec: engine.away_after == sec)
            for sec in AWAY_CHOICES])),
        item("Пищать при телефоне в кадре", toggle_phone, checked=lambda i: engine.phone_on),
        item("Запускать вместе с Windows", toggle_autostart, checked=lambda i: autostart_enabled()),
        menu.SEPARATOR,
        item("Открыть журнал", open_log),
        item("Выход", quit_app),
    )
    icon.run_detached()

    def prepare():
        if not models_ready():
            engine.state.set(gaze_error="нет моделей в %s" % MODELS_DIR, notice="Нет моделей")
            icon.notify("Не нашёл модели в %s" % MODELS_DIR, "Gaze Check")
            return
        engine.start()
        if settings["gaze_log"] or getattr(args, "gaze_log", False):
            engine.set_log(True)
        hint = ("" if engine.calibration.calibrated
                else " Для точной точки взгляда — меню → «Калибровка взгляда».")
        icon.notify("Работаю в фоне. Пищу, если %s не смотришь в экран или в кадре телефон.%s"
                    % (human_seconds(engine.away_after), hint), "Gaze Check")

    threading.Thread(target=prepare, daemon=True).start()
    viewer = Viewer(engine)
    shown = None
    overlay = None  # окно меню взглядом (tkinter; живёт в этом, главном, потоке)
    dot = None  # точка взгляда поверх экрана
    try:
        while not ui["quit"]:
            if ui["dot"] or dot is not None:
                try:
                    if dot is None:
                        from .menu import GazeDot
                        dot = GazeDot(engine.calibration.px_per_cm())
                    eye = engine.state.snapshot()["eye"] or {}
                    p, scr = eye.get("point"), engine.calibration.screen
                    dot.tick((scr[0] + p[0], scr[1] + p[1]) if ui["dot"] and p else None)
                except Exception:
                    traceback.print_exc()  # точка не должна ронять программу
                    ui["dot"], dot = False, None
            menu = engine.menu.snapshot()
            if menu["open"] or overlay is not None:
                try:
                    if overlay is None:
                        from .menu import Overlay
                        overlay = Overlay(engine.calibration.screen)
                    overlay.tick(menu)
                except Exception:
                    traceback.print_exc()  # меню не должно ронять программу
                    engine.menu.open = False
                    overlay = None
            if ui["calibrate"] and engine.threads:
                ui["calibrate"] = False
                viewer.close()
                from .calib import run_calibration
                try:
                    cal = run_calibration(engine)
                except Exception as e:
                    traceback.print_exc()
                    icon.notify("Калибровка не удалась: %s. Подробности — в журнале." % e, "Gaze Check")
                    cal = None
                if cal is not None:
                    icon.notify("Калибровка сохранена. %s" % engine.state.snapshot()["accuracy"], "Gaze Check")
                    ui["gaze_test"] = True  # сразу видно, как попадает
                continue
            if "quick" in engine.requests and engine.threads:
                engine.requests.discard("quick")
                viewer.close()
                from .calib import run_quick
                try:
                    cal = run_quick(engine)
                except Exception as e:
                    traceback.print_exc()
                    icon.notify("Подстройка не удалась: %s. Подробности — в журнале." % e, "Gaze Check")
                    cal = None
                if cal is not None:
                    icon.notify("Подстроено под текущую позу. %s" % engine.state.snapshot()["accuracy"], "Gaze Check")
                    ui["gaze_test"] = True  # сразу видно, достаёт ли до углов
                continue
            if ui["gaze_test"] and engine.threads:
                ui["gaze_test"] = False
                viewer.close()
                from .calib import run_gaze_test
                try:
                    ui["calibrate"] = run_gaze_test(engine) == "calibrate"
                except Exception:
                    traceback.print_exc()
                continue
            if ui["window"]:
                try:
                    shown_ok = viewer.tick()
                except cv2.error:
                    traceback.print_exc()  # сбой окна не должен останавливать фоновую работу
                    shown_ok = False
                    viewer.open = False
                if not shown_ok:
                    ui["window"] = False
                    icon.update_menu()
            else:
                viewer.close()
                time.sleep(0.03 if menu["open"] or ui["dot"] else 0.2)  # меню или точка — плавно
            color, text = status_of(engine)
            if (color, text) != shown:
                shown = (color, text)
                if icon.icon is None or color != getattr(icon, "_gc_color", None):
                    icon.icon = icon_image(color)
                    icon._gc_color = color
                icon.title = ("Gaze Check — " + text)[:120]
    finally:
        viewer.close()
        if overlay is not None:
            overlay.close()
        if dot is not None:
            dot.close()
        engine.close()
        icon.stop()
