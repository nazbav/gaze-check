"""Проверка взгляда с веб-камеры — локально, кадры никуда не уходят."""
import argparse
import collections
import datetime
import functools
import json
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .workflow import (GAZE_CLASSES, PROLONGED_AFTER, GazeTimer, Hold, person_seen, phone_outputs, presence,
                       top_class)

TITLE = "Gaze Check"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GAZE_RU = {"looking at screen": "смотрит на экран", "looking away": "смотрит в сторону",
           "face not visible": "лица не видно"}
GREEN, ORANGE, RED, GRAY, YELLOW = (40, 160, 70), (230, 130, 20), (215, 40, 40), (90, 90, 90), (250, 210, 20)


# ---------- отрисовка ----------

@functools.lru_cache(maxsize=8)
def _font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@functools.lru_cache(maxsize=128)
def _patch(text, size, bold, fill, color):
    """Надпись на плашке (BGR). Рисуется один раз: PIL на каждый кадр тормозил бы обработку."""
    font = _font(size, bold)
    pad = size // 3
    left, top, right, bottom = font.getbbox(text)
    img = Image.new("RGB", (right - left + 2 * pad, bottom - top + 2 * pad), fill)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=color)
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def _put(frame, patch, x, y, right=False, bottom=False):
    """Кладёт плашку углом в (x, y); возвращает нижний край."""
    ph, pw = patch.shape[:2]
    x, y = int(x - pw if right else x), int(y - ph if bottom else y)
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    return y + ph


def draw_trace(frame, eye):
    """Углы глаз, зрачки и тонкие лучики взгляда."""
    for i, (x, y) in enumerate(eye.get("marks", [])):
        cv2.circle(frame, (int(x), int(y)), 2 if i < 4 else 3,
                   (90, 220, 90) if i < 4 else (40, 40, 230), -1, cv2.LINE_AA)
    for (x0, y0), (x1, y1) in eye.get("rays", []):
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 230, 255), 1, cv2.LINE_AA)
    return frame


def _minimap(frame, eye, size):
    """Экран в уменьшенном виде (справа снизу) с точкой взгляда."""
    h, w = frame.shape[:2]
    _, _, sw, sh = eye["screen"]
    mw = w // 5
    mh = max(1, round(mw * sh / sw))
    x0, y0 = w - mw - size // 2, h - mh - size * 2
    cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (220, 220, 220), 1)
    if eye.get("point"):
        px = int(x0 + min(max(eye["point"][0] / sw, 0), 1) * mw)
        py = int(y0 + min(max(eye["point"][1] / sh, 0), 1) * mh)
        cv2.circle(frame, (px, py), max(3, mw // 40), (40, 40, 230), -1, cv2.LINE_AA)


class Painter:
    def draw(self, bgr, snap):
        """Кадр + состояние → кадр с разметкой (как output_image в workflow)."""
        frame = bgr.copy()
        h, w = frame.shape[:2]
        big = max(14, min(w // 40, h // 20))  # 32 px при 1280×720
        small = max(11, big * 2 // 3)
        m = big // 2
        white, black = (255, 255, 255), (0, 0, 0)

        ph = snap["phone"]
        for p in ph.get("phone_predictions", []):
            x1, y1 = int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)
            x2, y2 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW[::-1], 3)
            label = _patch("телефон %d%%" % round(p["confidence"] * 100), small, False, YELLOW, black)
            _put(frame, label, x1, y1 - 2, bottom=True)
        if snap["status"].get("prolonged_away") or ph.get("phone_alarm", ph.get("phone_present")):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED[::-1], max(6, h // 60))
        eye = snap.get("eye") or {}
        draw_trace(frame, eye)
        if eye.get("screen"):
            _minimap(frame, eye, big)
        if snap.get("muted"):
            _put(frame, _patch("без звука (M)", small, False, black, white), m, h - m - big * 2, bottom=True)
        if snap.get("recording"):
            cv2.circle(frame, (w - big, h - big), big // 3, RED[::-1], -1, cv2.LINE_AA)
            _put(frame, _patch("запись", small, True, black, white), w - big * 3 // 2, h - big, right=True,
                 bottom=False)

        # слева сверху — ответ модели взгляда
        gaze = snap["gaze"]
        if snap["gaze_error"]:
            text, fill = "Взгляд: ошибка, см. консоль", RED
        elif not snap["gaze_ready"]:
            text, fill = "Взгляд: загрузка модели…", GRAY
        elif gaze is None and snap["gaze_done"] == 0:
            text, fill = "Взгляд: первый кадр…", GRAY
        else:
            label = top_class(gaze)
            text = "Взгляд: " + GAZE_RU.get(label, label or "не разобрал ответ")
            if gaze:
                text += " %d%%" % round(gaze["confidence"] * 100)
            fill = {"looking at screen": GREEN, "looking away": ORANGE}.get(label, GRAY)
        y = _put(frame, _patch(text, big, True, fill, white), m, m)
        if snap["gaze_latency"]:
            info = "%d мс на кадр · %s" % (round(snap["gaze_latency"] * 1000), snap["accuracy"] or "без калибровки")
            if eye.get("dist"):
                info += " · до лица ~%d см" % eye["dist"]
            if eye.get("method"):
                info += " · " + eye["method"]
            if eye.get("one_eye"):
                info += " · по одному глазу (%s)" % eye["one_eye"]
            _put(frame, _patch(info, small, False, black, white), m, y + m // 2)

        # слева снизу — статус по времени
        st = snap["status"]
        if st:
            text, fill = {
                "looking_at_screen": ("На экране", GREEN),
                "brief_away": ("В сторону ~%d с" % round(st["away_seconds"]), ORANGE),
                "prolonged_away": ("Долго в сторону: %d с!" % round(st["away_seconds"]), RED),
                "face_not_visible": ("Лица не видно", GRAY),
                "no_person": ("Никого нет — не пищу", GRAY),
            }[st["status"]]
            _put(frame, _patch(text, big, True, fill, white), m, h - m, bottom=True)

        # справа сверху — телефон
        if snap["phone_error"]:
            text, fill, color = "Телефон: ошибка, см. консоль", RED, white
        elif not snap["phone_ready"]:
            text, fill, color = "Телефон: загрузка модели…", GRAY, white
        elif ph.get("phone_alarm", ph.get("phone_present")):
            text, fill, color = "Телефон в кадре — убери!", YELLOW, black
        else:
            text, fill, color = "Телефона нет", black, YELLOW
        _put(frame, _patch(text, big, True, fill, color), w - m, m, right=True)
        return frame


# ---------- состояние и журнал ----------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.gaze = None
        self.gaze_done = 0
        self.accuracy = ""  # «точность ~N px» — по калибровке и по кликам
        self.gaze_latency = 0.0
        self.gaze_ready = ""
        self.gaze_error = ""
        self.status = {}
        self.phone = {}
        self.phone_ready = False
        self.phone_error = ""
        self.camera_error = ""
        self.notice = ""  # пауза, загрузка моделей — показывается вместо кадра
        self.eye = {}  # трекер глаз: valid, точки глаз на кадре, точка взгляда на экране
        self.eye_error = ""
        self.person_at = None  # monotonic: когда RF-DETR последний раз видел человека
        self.face_at = None  # monotonic: когда MediaPipe последний раз видел лицо
        self.recording = ""  # папка текущей записи
        self.mouse = False  # мышь ведёт взгляд
        self.logging = False  # пишется лёгкий журнал взгляда

    def snapshot(self):
        with self.lock:
            return {k: v for k, v in vars(self).items() if k != "lock"}

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)


def load_addons(engine):
    """Локальные дополнения: модуль gaze_check/extras.py, если он есть (в репозиторий не входит).
    Дополнение — объект с полями listening, busy, status и методами gesture(ev, ts) → занято ли событие,
    tick(ts), menu_items(item, settings, save) → пункты трея, close()."""
    try:
        from . import extras
    except ImportError:
        return []
    try:
        return [make(engine) for make in extras.ADDONS]
    except Exception:
        traceback.print_exc()  # сломанное дополнение не должно ронять программу
        return []


def beep(freq, times=1, ms=220):
    try:
        import winsound
    except ImportError:
        return

    def run():
        for _ in range(times):
            winsound.Beep(freq, ms)
            time.sleep(0.08)
    threading.Thread(target=run, daemon=True).start()


def human_seconds(sec):
    return "%d мин" % (sec // 60) if sec >= 60 and sec % 60 == 0 else "%d с" % sec


class Journal:
    """Смены состояния в консоль/журнал, тревоги (писк + уведомление) и JSONL с полями workflow."""

    REPEAT = 10.0  # пока тревога не снята, писк повторяется

    def __init__(self, path=None, sound=True, notify=None):
        self.file = open(path, "a", encoding="utf-8") if path else None
        self.sound = sound
        self.notify = notify
        self.last_status = None
        self.last_phone = False
        self.active = {}

    def _say(self, text):
        print(time.strftime("%H:%M:%S"), text, flush=True)

    def _alarm(self, kind, on, text, freq):
        if not on:
            self.active.pop(kind, None)
            return
        now, last = time.monotonic(), self.active.get(kind)
        if last is None or now - last >= self.REPEAT:
            if self.sound:
                beep(freq, 2)
            if last is None and self.notify:
                self.notify(text)
            self.active[kind] = now

    def gaze(self, snap, away_after):
        st = snap["status"]
        if st["status"] != self.last_status:
            names = {"looking_at_screen": "на экране", "brief_away": "смотрит в сторону",
                     "prolonged_away": "НЕ СМОТРИТ В ЭКРАН %d с" % st["away_seconds"],
                     "face_not_visible": "лица не видно", "no_person": "никого нет"}
            self._say("взгляд: " + names[st["status"]])
            self.last_status = st["status"]
        self._alarm("away", st["prolonged_away"],
                    "Не смотришь в экран уже %s" % human_seconds(away_after), 700)
        if self.file:
            ph = snap["phone"]
            rec = {"time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                   "predictions": snap["gaze"], "gaze_status": st["status"],
                   "away_seconds": st["away_seconds"], "prolonged_away": st["prolonged_away"],
                   "phone_predictions": ph.get("phone_predictions", []),
                   "phone_count": ph.get("phone_count", 0), "phone_present": ph.get("phone_present", False),
                   "point": (snap["eye"] or {}).get("point")}
            self.file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.file.flush()

    def phone(self, alarm):
        if alarm != self.last_phone:
            self._say("телефон в кадре" if alarm else "телефон убран")
            self.last_phone = alarm
        self._alarm("phone", alarm, "Телефон в кадре — убери!", 1200)


# ---------- источник и фоновые модели ----------

class Latest:
    """Последний кадр источника: модели берут только свежий, старые пропускаются."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frame, self.seq, self.ts, self.ended = None, 0, 0.0, False

    def put(self, frame):
        with self.cond:
            self.frame, self.ts = frame, time.monotonic()
            self.seq += 1
            self.cond.notify_all()

    def clear(self):
        with self.cond:
            self.frame = None

    def end(self):
        with self.cond:
            self.ended = True
            self.cond.notify_all()

    def newer(self, seq, timeout=0.5):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq or self.ended, timeout)
            if self.seq > seq and self.frame is not None:
                return self.frame, self.seq, self.ts
            return None


def open_source(source, size):
    """Камера (номер) или файл; None, если не открылся."""
    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if source.isdigit() else cv2.VideoCapture(source)
    if source.isdigit() and size:
        w, h = (int(v) for v in size.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


MEDIAN_FRAMES, MEDIAN_S = 5, 0.5  # точка взгляда — медиана стольких последних кадров, не старше


class Engine:
    """Камера, глаза и детектор телефона в фоновых потоках; пауза отпускает камеру."""

    STATUS_EVERY = 0.2  # с: как часто обновлять статус, журнал и тревогу (глаза — каждый кадр)
    SAVE_EVERY = 20.0  # с: подстроенную по кликам калибровку сохранять не чаще

    def __init__(self, args, journal):
        self.args = args
        self.journal = journal
        self.is_file = not args.source.isdigit()
        self.state, self.latest = State(), Latest()
        self.stop, self.paused = threading.Event(), threading.Event()
        self.reopen = threading.Event()  # сменили камеру — поток чтения переоткроет
        self.phone_on = not args.no_phone
        self.clicks_on = not args.no_clicks and not self.is_file
        self.away_after = args.away_after
        self.epoch = 0  # растёт на паузе — таймер взгляда начинает заново
        self.threads = []
        from .calib import primary_monitor, screen_mm
        from .eyes import Calibration
        from .models import CALIBRATION
        screen = primary_monitor()
        # без калибровки — средние параметры 3D-модели; клики и калибровка их уточняют
        self.calibration = (Calibration.load(CALIBRATION, screen)
                            or Calibration.default(screen, screen_mm(screen)))
        self.collect = None  # список — калибровка собирает сюда признаки глаз
        self.recorder = None
        self.rec_lock = threading.Lock()
        self.history = collections.deque(maxlen=120)  # (ts, признаки) последних кадров — для кликов
        self.clicks = collections.deque(maxlen=50)  # (t, x, y) ещё не разобранные клики
        self.dirty, self.saved_at = False, time.monotonic()
        self.watcher = None
        self.mouse = None  # управление мышью взглядом (GazeMouse), когда включено
        self.requests = set()  # просьбы к главному потоку (окна OpenCV живут только в нём): "quick"
        self.lead = getattr(args, "lead_eye", None) or "both"  # ведущий глаз
        from .menu import MenuLogic
        from .mouse import Gestures
        self.gestures = Gestures()  # жесты глазами: клики мыши взглядом и меню
        self.menu = MenuLogic(run=self._menu_action)  # меню взглядом (9 плиток)
        self.mouse_mode = getattr(args, "mouse_mode", None) or "joystick"  # джойстик / прямо за взглядом
        from .mouse import JOY_SPEED_DEFAULT
        self.joy_speed = float(getattr(args, "joy_speed", None) or JOY_SPEED_DEFAULT)  # множитель скорости
        self.gazelog = None  # лёгкий журнал взгляда (gazelog.GazeLog), когда пишется
        self.log_probe = None
        self.log_closed_at = None
        self.menu_gestures = bool(getattr(args, "menu_gestures", False))  # открывать меню морганием
        self.closure_used = False  # текущее закрытие глаз уже сработало в дополнении
        self.tracker = None
        self.hotkey = None
        self.addons = load_addons(self)
        self._show_accuracy()

    def start(self):
        self.threads = [threading.Thread(target=self._phone_loop, daemon=True),
                        threading.Thread(target=self._eye_loop, daemon=True)]
        for t in self.threads:
            t.start()
        if not self.is_file:
            from .mouse import Hotkey
            self.hotkey = Hotkey({"G": self.toggle_mouse, "C": lambda: self.requests.add("quick"),
                                  "M": self.toggle_menu}).start()
        if self.clicks_on:
            from .clicks import ClickWatcher
            self.watcher = ClickWatcher(lambda t, x, y: self.clicks.append((t, x, y))).start()
        if self.is_file:  # файл ждёт модели, иначе его начало проиграется впустую
            while not self.models_loaded() and not self.stop.is_set():
                time.sleep(0.1)
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()
        self.threads.insert(0, reader)

    def models_loaded(self):
        s = self.state.snapshot()
        return bool((s["gaze_ready"] or s["gaze_error"]) and (s["phone_ready"] or s["phone_error"]))

    def workers_alive(self):
        return any(t.is_alive() for t in self.threads[1:])

    def pause(self, on):
        if on:
            self.paused.set()
            self.epoch += 1
            self.state.set(status={}, phone={}, gaze=None)
        else:
            self.paused.clear()

    def close(self):
        self.stop_recording()
        self.set_log(False)
        self.set_mouse(False)
        for addon in self.addons:
            addon.close()
        for w in (self.watcher, self.hotkey):
            if w is not None:
                w.stop()
        self.stop.set()
        self.latest.end()
        for t in self.threads:
            t.join(timeout=5)
        self._save_calibration()

    def present(self, now=None):
        """Человек в кадре. Пока оба детектора не готовы — считаем, что есть (как раньше)."""
        s = self.state.snapshot()
        if not s["phone_ready"] and (s["eye_error"] or not s["eye"]):
            return True
        return presence(time.monotonic() if now is None else now, s["person_at"], s["face_at"])

    def set_calibration(self, cal):
        """Новая калибровка по точкам (клики, накопленные раньше, калибровка уже учла)."""
        from .models import CALIBRATION
        cal.lead = self.lead  # признаки в ней посчитаны по текущему ведущему глазу
        cal.save(CALIBRATION)
        self.calibration = cal
        self.dirty, self.saved_at = False, time.monotonic()
        self._show_accuracy()

    def _save_calibration(self):
        """Подстроенную по кликам калибровку — на диск (раз в SAVE_EVERY и при выходе)."""
        from .models import CALIBRATION
        cal = self.calibration
        if cal is not None and self.dirty:
            try:
                cal.save(CALIBRATION)
            except OSError:
                traceback.print_exc()
            self.dirty = False
            self.saved_at = time.monotonic()

    def _show_accuracy(self):
        cal = self.calibration
        cm = cal.px_per_cm()
        parts = []
        if cal.check_px is not None:
            parts.append("точность ~%.1f см" % (cal.check_px / cm))
        elif cal.error_px is not None:
            parts.append("калибровка ~%.1f см" % (cal.error_px / cm))
        if cal.jitter_px is not None:
            parts.append("дрожание ~%.1f см" % (cal.jitter_px / cm))
        if cal.click_error_px is not None:
            parts.append("по кликам ~%.1f см" % (cal.click_error_px / cm))
        self.state.set(accuracy=", ".join(parts) if cal.calibrated else "без калибровки")

    def set_lead_eye(self, lead):
        """Ведущий глаз: "left" / "right" / "both". Калибровка пересчитывается в фоне."""
        self.lead = lead
        if self.tracker is not None:
            self.tracker.lead = lead
        self._relead()

    def _relead(self):
        cal = self.calibration
        if not cal.points or cal.lead == self.lead:
            return

        def work():
            from .models import CALIBRATION
            try:
                new = cal.relead(self.lead)
            except Exception:
                traceback.print_exc()
                return
            if self.calibration is cal:  # пока считали, калибровку не заменили
                self.calibration = new
                new.save(CALIBRATION)
                self._show_accuracy()
                print(time.strftime("%H:%M:%S"), "ведущий глаз:", self.lead, "—",
                      self.state.snapshot()["accuracy"], flush=True)
        threading.Thread(target=work, daemon=True).start()

    SOUNDS = {"armed": (1000, 1, 90), "menu_armed": (1300, 2, 70), "cancel": (400, 1, 250),
              "left": (1500, 1, 40), "right": (1200, 1, 40), "menu": (900, 1, 60), "choose": (1700, 1, 40),
              "hold_armed": (1100, 1, 60), "task_view": (1400, 2, 50), "hold_lost": (500, 1, 200)}
    SAY = {"armed": "2 с — открой для правого клика", "menu_armed": "3 с — открой для меню",
           "cancel": "отмена (дольше 5 с)", "left": "левый клик", "right": "правый клик", "menu": "меню",
           "choose": "выбор в меню",
           "hold_armed": "три моргания — держи взгляд 3 с для Win+Tab", "task_view": "Win+Tab — все окна",
           "hold_lost": "Win+Tab: взгляд не удержан — отмена"}

    def sound(self, kind):
        """Писк на жест глазами; «Без звука» глушит и его."""
        freq, times, ms = self.SOUNDS[kind]
        if self.journal.sound:
            beep(freq, times, ms)
        print(time.strftime("%H:%M:%S"), "жест глазами:", self.SAY[kind], flush=True)

    def _gestures(self, ts, blink):
        """Жесты глазами с каждого кадра → звуки, меню; события — мыши взглядом."""
        events, out = self.gestures.update(ts, blink), []
        for ev in events:
            if ev == "closed":
                self.closure_used = False  # новое закрытие глаз — жест ещё свободен
            elif self.closure_used and ev in ("armed", "menu_armed", "cancel", "right", "menu"):
                continue  # это закрытие уже сработало в дополнении — ни правого клика, ни меню
            if any(addon.gesture(ev, ts) for addon in self.addons):
                self.closure_used = ev == "armed"  # долгое закрытие занято — остальное из него не нужно
                continue
            out.append(ev)
            if ev == "armed" and self.mouse is not None and not self.menu.open:
                self.sound(ev)  # правый клик есть только у мыши взглядом
            elif ev in ("menu_armed", "cancel"):
                if self.menu_gestures or self.menu.open or any(a.listening for a in self.addons):
                    self.sound(ev)
            elif ev == "menu":
                if self.menu_gestures or self.menu.open:  # выключено — моргания меню не открывают
                    self.toggle_menu()
                elif self.mouse is not None:  # мышь взглядом: ещё и взгляд на месте 3 с — Win+Tab
                    self.mouse.arm_hold(ts)
                    self.sound("hold_armed")
            elif ev == "left" and self.menu.open:
                self.menu.select()
                self.sound("choose")
        return out

    def _mouse_events(self, events):
        """События мыши взглядом: "hold" — Win+Tab (все окна, выбрать — взглядом и двумя морганиями)."""
        from .menu import press
        for ev in events:
            if ev == "hold":
                press("win", "tab")
                self.sound("task_view")
            elif ev == "hold_lost":
                self.sound("hold_lost")

    def toggle_menu(self):
        opened = self.menu.toggle()
        self.sound("menu")
        print(time.strftime("%H:%M:%S"), "меню взглядом:", "открыто" if opened else "закрыто", flush=True)

    def _menu_action(self, action):
        """Выполнить выбранное в меню (в потоке глаз; меню не забирает фокус — нажатия уходят в окно)."""
        from .menu import press, wheel
        kind = action[0]
        print(time.strftime("%H:%M:%S"), "меню:", action, flush=True)
        if kind == "keys":
            press(*action[1:])
        elif kind == "wheel":
            wheel(action[1])
        elif action == ("app", "mouse"):
            self.toggle_mouse()
        elif action == ("app", "address_voice"):
            press("ctrl", "l")
            time.sleep(0.3)
            press("win", "h")
        elif action == ("app", "help"):
            beep(1800, 12, 300)
            if self.journal.notify:
                self.journal.notify("Зовут на помощь!")

    def set_log(self, on):
        """Лёгкий журнал взгляда: включить/выключить. → путь файла при выключении."""
        from .gazelog import GazeLog
        from .zones import WindowProbe
        path = None
        if on and self.gazelog is None:
            cal = self.calibration
            self.log_probe = WindowProbe(cal.screen)
            self.gazelog = GazeLog(cal.screen, cal.px_per_cm())
            print(time.strftime("%H:%M:%S"), "журнал взгляда:", self.gazelog.path, flush=True)
        elif not on and self.gazelog is not None:
            log, self.gazelog = self.gazelog, None
            path = log.close()
            print(time.strftime("%H:%M:%S"), "журнал взгляда сохранён:", path, flush=True)
        self.state.set(logging=self.gazelog is not None)
        return path

    def _log_frame(self, log, ts, feat, point, eye, label, events):
        """Кадр → журнал: события морганий каждый кадр, отсчёт — 10 раз в секунду."""
        from .gazelog import BLINK, BOTH_EYES, CLOSED, DOUBLE, FACE, LOOKING, PHONE, RIGHT, TRIPLE
        from .zones import ALL_ZONES, GRID_ZONES, zone_of
        for ev in events:
            if ev == "closed":
                self.log_closed_at = ts
            elif ev == "opened" and self.log_closed_at is not None:
                dur, self.log_closed_at = ts - self.log_closed_at, None
                if dur <= 0.6:
                    log.event(ts, BLINK, dur * 1000)
            elif ev in ("left", "menu", "right"):
                log.event(ts, {"left": DOUBLE, "menu": TRIPLE, "right": RIGHT}[ev])
        if ts < log.next_sample:
            return
        s = self.state.snapshot()
        flags = ((FACE if eye["face"] else 0) | (BOTH_EYES if feat and feat.get("eyes") == "both" else 0)
                 | (LOOKING if label == "looking at screen" else 0)
                 | (PHONE if s["phone"].get("phone_alarm") else 0)
                 | (CLOSED if self.gestures.closed_at is not None else 0))
        scr = self.calibration.screen
        name = zone_of(point, scr[2], scr[3])
        zone = ALL_ZONES.index(name)
        app, title = self.log_probe(point) if name in GRID_ZONES else ("", "")  # за экраном окна нет
        log.sample(ts, point, eye.get("dist"), flags, zone, ("%s — %s" % (app, title)) if app else "")

    def set_joy_speed(self, speed):
        """Скорость джойстика (множитель) — сразу, без перезапуска мыши."""
        self.joy_speed = float(speed)
        if self.mouse is not None:
            self.mouse.cursor.speed = self.joy_speed

    def set_mouse(self, on):
        """Включить/выключить управление мышью взглядом. → включено ли теперь."""
        from .mouse import GazeMouse
        if on and self.mouse is None and not self.is_file:
            cal = self.calibration

            def click(kind):
                from .mouse import _click
                _click(kind)
                self.sound(kind)
            self.mouse = GazeMouse(cal.screen, cal.px_per_cm(), beep=self.sound, click=click,
                                   gestures=self.gestures, mode=self.mouse_mode, speed=self.joy_speed).start()
        elif not on and self.mouse is not None:
            mouse, self.mouse = self.mouse, None
            mouse.stop()
        self.state.set(mouse=self.mouse is not None)
        return self.mouse is not None

    def toggle_mouse(self):
        on = self.set_mouse(self.mouse is None)
        if self.journal.notify:
            self.journal.notify("Мышь — взглядом. Два моргания — левый клик; закрыть глаза до писка (2 с) и "
                                "открыть — правый. Ctrl+Alt+G — выключить." if on else "Управление мышью взглядом выключено.")
        print(time.strftime("%H:%M:%S"), "мышь взглядом:", "вкл" if on else "выкл", flush=True)

    def _take_clicks(self, now):
        """Клики, у которых уже есть кадры после нажатия, → подстройка калибровки."""
        from .eyes import click_features
        changed = False
        while self.clicks and self.clicks[0][0] < now:
            t, x, y = self.clicks.popleft()
            if not self.clicks_on or self.paused.is_set() or self.collect is not None or self.mouse is not None:
                continue  # во время калибровки и когда мышь ведёт взгляд клики ничему не учат
            screen = self.calibration.screen
            p = (x - screen[0], y - screen[1])
            if not (0 <= p[0] < screen[2] and 0 <= p[1] < screen[3]):
                continue  # другой монитор
            feat = click_features(self.history, t)
            if feat is None:
                continue  # глаз не было видно
            new = self.calibration.with_click(p, feat)
            if new is not None:
                self.calibration, changed = new, True
        if changed:
            self._show_accuracy()
            self.dirty = True
        if self.dirty and now - self.saved_at >= self.SAVE_EVERY:
            self._save_calibration()

    def start_recording(self):
        """Папка записи или None, если уже пишем."""
        from .recorder import Recorder
        with self.rec_lock:
            if self.recorder is not None:
                return None
            self.recorder = Recorder(self.calibration)
            self.state.set(recording=str(self.recorder.folder))
            return self.recorder.folder

    def stop_recording(self):
        with self.rec_lock:
            rec, self.recorder = self.recorder, None
        if rec is None:
            return None
        self.state.set(recording="")
        return rec.stop()

    def _eye_loop(self):
        """Глаза на каждом кадре (~10 мс на процессоре): точка взгляда на экране и смотрит ли
        в экран; раз в STATUS_EVERY — счёт времени, журнал и тревога."""
        from .eyes import LOOSE_MARGIN, SCREEN_MARGIN, EyeTracker, OneEuro, ScreenGaze
        from .models import EYE_MODEL, GAZENET_MODEL, to_rgb
        try:
            tracker = EyeTracker(EYE_MODEL, GAZENET_MODEL, async_net=True)
            tracker.lead = self.lead
            self.tracker = tracker
            self._relead()  # калибровка посчитана по другому глазу — пересчитать
        except Exception as e:
            traceback.print_exc()
            self.state.set(eye_error=str(e), gaze_error="трекер глаз: %s" % e)
            return
        self.state.set(gaze_ready="MediaPipe")
        smooth, cal = OneEuro(), None
        judge, timer, epoch, status_at, latency = ScreenGaze(), None, None, -1e9, None
        recent = collections.deque(maxlen=MEDIAN_FRAMES)  # (ts, точка) — для медианы
        for frame, seq, ts in self._frames():
            t0 = time.perf_counter()
            self._take_clicks(ts)
            feat, pts, rays = tracker.process(to_rgb(frame), ts)
            if cal is not self.calibration:
                smooth, cal = OneEuro(), self.calibration  # новая калибровка — сглаживание заново
            raw = point = None
            if feat is not None and "gy" in feat:
                self.history.append((ts, feat))
                raw = cal.predict(feat)
                recent.append((ts, raw))
                while recent and ts - recent[0][0] > MEDIAN_S:
                    recent.popleft()
                # медиана последних кадров: дрожание −25% на замере (tools/gaze_precision.py)
                point = smooth(ts, np.median([r for _, r in recent], axis=0))
                if self.collect is not None:
                    self.collect.append((ts, feat))
            if pts is not None:
                self.state.set(face_at=time.monotonic())
            events = self._gestures(ts, tracker.blink)
            for addon in self.addons:
                addon.tick(ts)
            mouse = self.mouse
            if mouse is not None:  # меню или дополнение заняли взгляд — курсор стоит, не кликаем
                mouse.held = self.menu.open or any(a.busy for a in self.addons)
                self._mouse_events(mouse.update(ts, raw, tracker.blink, events))
            if self.menu.open:
                self.menu.update(None if point is None else (float(point[0]), float(point[1])), cal.screen)
            label, share = judge.update(ts, feat, pts is not None, point, cal.screen,
                                        SCREEN_MARGIN if cal.calibrated else LOOSE_MARGIN)
            log = self.gazelog
            marks = pts[[33, 133, 362, 263, 468, 473]].tolist() if pts is not None else []
            eye = {"valid": point is not None, "face": pts is not None, "marks": marks, "rays": rays,
                   "point": None if point is None else (float(point[0]), float(point[1])),
                   "screen": cal.screen, "frame": (frame.shape[1], frame.shape[0]),
                   "dist": -feat["ez"] * cal.theta[4] if feat is not None and "ez" in feat else None,
                   "method": ("MGazeNet" if cal.netmap is not None and feat is not None and feat.get("net") is not None
                              else "3D-модель"),
                   "one_eye": {"left": "левый", "right": "правый"}.get(feat.get("eyes")) if feat else None}
            if log is not None:
                try:
                    self._log_frame(log, ts, feat, point, eye, label, events)
                except Exception:
                    traceback.print_exc()  # журнал не должен останавливать слежение
            spent = time.perf_counter() - t0
            latency = spent if latency is None else 0.9 * latency + 0.1 * spent
            self.state.set(eye=eye, gaze_latency=latency)

            if epoch != self.epoch:
                timer, epoch = GazeTimer(), self.epoch  # после паузы счёт заново
            timer.prolonged_after = self.away_after
            if ts - status_at >= self.STATUS_EVERY:
                status_at = ts
                status = timer.update(label, ts, seq, present=self.present())
                preds = {"top": label, "confidence": round(share, 2),
                         "predictions": [{"class": c, "class_id": i, "confidence": round(share, 2) if c == label else 0.0}
                                         for i, c in enumerate(GAZE_CLASSES)]}
                self.state.set(gaze=preds, status=status, gaze_done=self.state.gaze_done + 1)
                self.journal.gaze(self.state.snapshot(), self.away_after)

            rec = self.recorder
            if rec is not None:
                s = self.state.snapshot()
                rec.camera_frame(frame, ts, draw_trace(frame.copy(), eye))
                rec.gaze(ts, feat, raw, point, s["status"].get("status", ""),
                         phone=s["phone"].get("phone_alarm", False), dist=eye["dist"], method=eye["method"])

    def set_camera(self, source):
        """Сменить камеру на ходу (номер DirectShow, см. cameras.py). → сменилась ли."""
        if self.is_file or source == self.args.source:
            return False
        self.args.source = source
        self.reopen.set()
        return True

    def _read_loop(self):
        cap, period, nxt = None, 0.0, 0.0
        while not self.stop.is_set():
            if self.reopen.is_set():
                self.reopen.clear()
                if cap is not None:
                    cap.release()
                    cap = None
                    self.latest.clear()
            if self.paused.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                    self.latest.clear()
                    self.state.set(notice="Пауза")
                self.stop.wait(0.2)
                continue
            if cap is None:
                cap = open_source(self.args.source, self.args.size)
                if cap is None:
                    if self.is_file:
                        self.state.set(camera_error="Не открыть файл: %s" % self.args.source)
                        break
                    self.state.set(camera_error="Камера %s не открылась — занята или отключена"
                                   % self.args.source)
                    self.stop.wait(3)
                    continue
                self.state.set(camera_error="", notice="")
                period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25) if self.is_file else 0.0
                nxt = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    break
                cap.release()  # камера отвалилась — переоткроем
                cap = None
                self.stop.wait(1)
                continue
            self.latest.put(frame)
            if period:  # файл проигрываем в реальном времени, как живой поток
                nxt += period
                self.stop.wait(max(0.0, nxt - time.monotonic()))
        if cap is not None:
            cap.release()
        self.latest.end()

    def _frames(self):
        seq = 0
        while not self.stop.is_set():
            item = self.latest.newer(seq)
            if item is None:
                if self.latest.ended:
                    return
                continue
            seq = item[1]
            yield item

    def _phone_loop(self):
        """EfficientDet-Lite0 (MediaPipe, процессор): телефон в кадре и есть ли вообще человек."""
        from .detector import Detector
        from .models import DETECTOR_MODEL, to_rgb
        try:
            model = Detector(DETECTOR_MODEL, confidence=self.args.phone_confidence)
        except Exception as e:
            traceback.print_exc()
            self.state.set(phone_error=str(e))
            return
        self.state.set(phone_ready=True)
        hold = Hold()
        for frame, _, _ in self._frames():
            t0 = time.monotonic()
            dets = model.detect(to_rgb(frame))
            if person_seen(dets):
                self.state.set(person_at=t0)
            out = phone_outputs(dets) if self.phone_on else {}
            out["phone_alarm"] = (hold.update(out.get("phone_present", False), t0)
                                  and self.phone_on and self.present(t0))
            self.state.set(phone=out)
            self.journal.phone(out["phone_alarm"])
            # чаще не нужно: телефон и человек не исчезают за полсекунды, а процессор нужен глазам
            if self.args.phone_fps > 0:
                self.stop.wait(max(0.0, 1.0 / self.args.phone_fps - (time.monotonic() - t0)))


def placeholder(snap, size=(1280, 720)):
    """Кадр-заглушка, когда камеры нет: пауза, загрузка, ошибка."""
    w, h = size
    frame = np.full((h, w, 3), 32, np.uint8)
    text = snap["camera_error"] or snap["notice"] or "Жду камеру…"
    patch = _patch(text, max(14, w // 40), True, (32, 32, 32), (230, 230, 230))
    _put(frame, patch, (w - patch.shape[1]) // 2, (h - patch.shape[0]) // 2)
    return frame


# ---------- режимы ----------

def run_image(args):
    """Одна картинка: JSON (поля как у журнала) и размеченный кадр."""
    from .detector import Detector
    from .eyes import Calibration, EyeTracker, ScreenGaze
    from .models import CALIBRATION, DETECTOR_MODEL, EYE_MODEL, GAZENET_MODEL, to_rgb
    bgr = cv2.imdecode(np.fromfile(args.source, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit("Не прочитать картинку: %s" % args.source)
    rgb = to_rgb(bgr)
    state = State()
    t0 = time.perf_counter()
    dets = Detector(DETECTOR_MODEL, confidence=args.phone_confidence).detect(rgb)
    state.phone, state.phone_ready = phone_outputs(dets), True
    feat, pts, rays = EyeTracker(EYE_MODEL, GAZENET_MODEL).process(rgb, 0.0)
    from .calib import primary_monitor, screen_mm
    screen = primary_monitor()
    cal = Calibration.load(CALIBRATION, screen) or Calibration.default(screen, screen_mm(screen))
    point = cal.predict(feat) if feat is not None and "gy" in feat else None
    label, share = ScreenGaze().update(0.0, feat, pts is not None, point, cal.screen)
    state.gaze = {"top": label, "confidence": share,
                  "predictions": [{"class": c, "class_id": i, "confidence": share if c == label else 0.0}
                                  for i, c in enumerate(GAZE_CLASSES)]}
    state.gaze_latency, state.gaze_done, state.gaze_ready = time.perf_counter() - t0, 1, "MediaPipe"
    state.status = GazeTimer().update(label, 0.0, present=pts is not None or person_seen(dets))
    state.eye = {"marks": pts[[33, 133, 362, 263, 468, 473]].tolist() if pts is not None else [], "rays": rays}
    snap = state.snapshot()
    result = json.dumps({"predictions": snap["gaze"], "gaze_status": snap["status"]["status"],
                         "away_seconds": snap["status"]["away_seconds"],
                         "prolonged_away": snap["status"]["prolonged_away"],
                         "phone_predictions": snap["phone"]["phone_predictions"],
                         "phone_count": snap["phone"]["phone_count"],
                         "phone_present": snap["phone"]["phone_present"],
                         "person": person_seen(dets), "face": pts is not None,
                         "net": feat is not None and feat.get("net") is not None,
                         "point": None if point is None else [round(point[0]), round(point[1])],
                         "eye": None if feat is None else {k: round(v, 4) for k, v in feat.items()
                                                           if isinstance(v, float)},
                         "seconds": round(snap["gaze_latency"], 3)}, ensure_ascii=False, indent=1)
    print(result)
    if args.json:
        Path(args.json).write_text(result, encoding="utf-8")
    out = Painter().draw(bgr, snap)
    if args.save:
        cv2.imencode(Path(args.save).suffix or ".jpg", out)[1].tofile(args.save)
    if not args.no_window:
        cv2.imshow(TITLE, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Viewer:
    """Окно с разметкой; живёт в главном потоке (так требует OpenCV)."""

    def __init__(self, engine):
        self.engine = engine
        self.painter = Painter()
        self.open = False
        self.shown = -1
        self.key = 255

    def tick(self):
        """Один шаг окна; False — окно закрыли (крестик, q, Esc)."""
        with self.engine.latest.cond:
            frame, seq = self.engine.latest.frame, self.engine.latest.seq
        snap = self.engine.state.snapshot()
        if not self.open:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, 960, 540)
            self.open, self.shown = True, -1
        if frame is None:
            cv2.imshow(TITLE, placeholder(snap))
        elif seq != self.shown:
            self.shown = seq
            snap["muted"] = not self.engine.journal.sound
            cv2.imshow(TITLE, self.painter.draw(frame, snap))
        key = cv2.waitKey(15) & 0xFF
        self.key = key
        if key in (27, ord("q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            return False
        return True

    def close(self):
        if self.open:
            self.open = False
            try:
                cv2.destroyWindow(TITLE)
            except cv2.error:
                pass  # окно уже закрыли крестиком — OpenCV его сам уничтожил
            cv2.waitKey(1)


def run_window(args):
    """Окно с разметкой (камера или видеофайл); с --no-window — только журнал."""
    journal = Journal(args.log, sound=not args.quiet)
    engine = Engine(args, journal)
    engine.start()
    print("Источник: %s. Окно: c — калибровка, g — проверка взгляда, r — запись, m — звук, q/Esc — выход."
          % args.source, flush=True)
    viewer = Viewer(engine)
    if args.record:
        print("Запись:", engine.start_recording(), flush=True)
    if getattr(args, "gaze_log", False):
        engine.set_log(True)
    overlay = None
    try:
        while True:
            menu = engine.menu.snapshot()
            if menu["open"] or overlay is not None:
                try:
                    if overlay is None:
                        from .menu import Overlay
                        overlay = Overlay(engine.calibration.screen)
                    overlay.tick(menu)
                except Exception:
                    traceback.print_exc()
                    engine.menu.open = False
                    overlay = None
            if args.no_window:
                time.sleep(0.03 if menu["open"] else 0.2)
            elif not viewer.tick():
                break
            if viewer.key == ord("c") or args.calibrate:
                args.calibrate = False
                from .calib import run_calibration
                viewer.close()
                cal = run_calibration(engine)
                print("Калибровка:", "отменена" if cal is None else engine.state.snapshot()["accuracy"], flush=True)
                args.gaze_test = args.gaze_test or cal is not None
            elif "quick" in engine.requests:
                engine.requests.discard("quick")
                from .calib import run_quick
                viewer.close()
                cal = run_quick(engine)
                print("Подстройка:", "отменена" if cal is None else engine.state.snapshot()["accuracy"], flush=True)
            elif viewer.key == ord("g") or args.gaze_test:
                args.gaze_test = False
                from .calib import run_gaze_test
                viewer.close()
                args.calibrate = run_gaze_test(engine) == "calibrate"
            elif viewer.key == ord("m"):
                journal.sound = not journal.sound
                print("Звук:", "вкл" if journal.sound else "выкл", flush=True)
            elif viewer.key == ord("r"):
                folder = engine.stop_recording() or engine.start_recording()
                print("Запись:", folder, flush=True)
            if engine.latest.ended and not engine.workers_alive():
                break  # файл кончился и модели дообработали последний кадр
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        if overlay is not None:
            overlay.close()
        engine.close()
    snap = engine.state.snapshot()
    for err in (snap["camera_error"], snap["gaze_error"], snap["phone_error"]):
        if err:
            print("Ошибка:", err, flush=True)


def quiet_libraries():
    """Прячет служебные сообщения MediaPipe (W0000/I0000 от glog) и предупреждения."""
    import os
    import warnings
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings("ignore")


def redirect_output():
    """У exe без консоли нет stdout/stderr — пишем их в журнал в %LOCALAPPDATA%\\GazeCheck."""
    from .models import DATA_DIR
    if sys.stdout is not None and sys.stderr is not None:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = DATA_DIR / "gaze-check.log"
    if log.exists() and log.stat().st_size > 5_000_000:
        log.replace(log.with_suffix(".log.old"))
    stream = open(log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def main(argv=None):
    """Точка входа. У exe без консоли необработанная ошибка иначе висит окном PyInstaller."""
    try:
        _main(argv)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:
        traceback.print_exc()
        from .models import DATA_DIR, FROZEN
        interactive = not any(a in (argv or sys.argv) for a in ("--no-window", "--json"))
        if FROZEN and interactive:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, "Gaze Check остановился из-за ошибки.\nПодробности: %s" % (DATA_DIR / "gaze-check.log"),
                "Gaze Check", 0x10)
        sys.exit(1)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="GazeCheck", description=__doc__)
    ap.add_argument("--source", default=None,
                    help="номер камеры, видеофайл или картинка (по умолчанию — из настроек, иначе 0)")
    ap.add_argument("--window", action="store_true", help="сразу окно вместо значка в трее")
    ap.add_argument("--size", default="1920x1080", help="разрешение камеры (больше — точнее зрачок)")
    ap.add_argument("--away-after", type=float, default=None,
                    help="через сколько секунд без взгляда в экран пищать (по умолчанию 60)")
    ap.add_argument("--phone-confidence", type=float, default=0.3)
    ap.add_argument("--phone-fps", type=float, default=2,
                    help="сколько раз в секунду искать телефон и человека (0 — каждый кадр)")
    ap.add_argument("--no-phone", action="store_true", help="не пищать при телефоне в кадре")
    ap.add_argument("--no-clicks", action="store_true", help="не подстраивать калибровку по кликам мыши")
    ap.add_argument("--menu-gestures", action="store_true",
                    help="три моргания или глаза закрыты 3–5 с — открыть меню взглядом")
    ap.add_argument("--mouse-mode", choices=["joystick", "direct"], default=None,
                    help="мышь взглядом: джойстик (едет в сторону взгляда) или прямо за взглядом")
    ap.add_argument("--joy-speed", type=float, default=None,
                    help="скорость джойстика: множитель (0.5 — очень медленно … 4.5 — очень быстро; по умолчанию 2)")
    ap.add_argument("--lead-eye", choices=["left", "right", "both"], default=None,
                    help="ведущий глаз: по нему считается взгляд (по умолчанию оба)")
    ap.add_argument("--log", help="писать результаты в JSONL")
    ap.add_argument("--save", help="для картинки: сохранить размеченный кадр")
    ap.add_argument("--json", help="для картинки: записать результат в файл")
    ap.add_argument("--quiet", action="store_true", help="без звука")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="сразу калибровка взгляда")
    ap.add_argument("--gaze-test", action="store_true", help="сразу проверка взгляда (точка на экране)")
    ap.add_argument("--record", action="store_true", help="сразу начать запись")
    ap.add_argument("--gaze-log", action="store_true", help="сразу вести лёгкий журнал взгляда (.gazelog)")
    ap.add_argument("--play", nargs="?", const="", metavar="ФАЙЛ",
                    help="проигрыватель журнала взгляда (без файла — последний)")
    args = ap.parse_args(argv)
    try:  # пиксели экрана, окна и снимки экрана — в одних физических единицах
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass
    redirect_output()
    if args.play is not None:
        from .player import main as play
        return play(args.play or None)
    quiet_libraries()
    if (args.source is None or args.source.isdigit()) and not (args.window or args.no_window):
        from .tray import run_tray
        return run_tray(args)  # камера по умолчанию — из настроек трея
    if args.source is None:  # окно — та же камера, что выбрана в трее
        from .tray import load_settings
        args.source = str(load_settings()["camera"])
    if args.away_after is None:
        args.away_after = PROLONGED_AFTER
    from .models import MODELS_DIR, models_ready
    if not models_ready():
        sys.exit("Нет моделей в %s — см. README, раздел «Для разработки»." % MODELS_DIR)
    if Path(args.source).suffix.lower() in IMAGE_EXT:
        run_image(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
