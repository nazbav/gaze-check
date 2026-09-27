"""Запись сеанса для анализа интерфейса: видео камеры, видео экрана со следом взгляда,
точки взгляда и фиксации (CSV), тепловая карта с порядком первых фиксаций, зоны внимания
(часть экрана, программа и окно под взглядом по времени), описание сеанса."""
import csv
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .eyes import Fixations, Trail
from .zones import GRID_ZONES, NO_DATA, AttentionLog, WindowProbe, draw_zone_overlay, summarize, zone_of, zones_image

SCREEN_FPS = 10
SCREEN_WIDTH = 1280  # экран в видео ужимается до этой ширины
# новые колонки — только в конец, чтобы старые разборщики не сломались
GAZE_COLUMNS = ("t", "time", "x", "y", "raw_x", "raw_y", "valid", "fixation", "hx", "vy", "look_h", "look_v",
                "yaw", "pitch", "gaze_status", "phone", "zone", "app", "window", "dist_cm", "method")
ATTENTION_COLUMNS = ("start", "end", "time", "duration_s", "zone", "app", "window")


def records_dir():
    from .models import DATA_DIR
    videos = Path.home() / "Videos"
    return (videos if videos.exists() else DATA_DIR) / "GazeCheck"


class TimedWriter:
    """Видео с постоянной частотой кадров из кадров с неровным временем: недостающие
    кадры повторяются, лишние пропускаются — длительность совпадает с реальной."""

    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.writer = None
        self.t0 = None
        self.count = 0
        self.last = None

    def add(self, frame, t):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t0 = t
        due = int((t - self.t0) * self.fps) + 1
        while self.count < due:
            self.writer.write(frame)
            self.count += 1
        self.last = frame

    def close(self):
        if self.writer is not None:
            self.writer.release()


def grab_screen(sct, mon):
    shot = np.asarray(sct.grab(mon))[:, :, :3]
    scale = SCREEN_WIDTH / shot.shape[1]
    return cv2.resize(shot, (SCREEN_WIDTH, round(shot.shape[0] * scale)), interpolation=cv2.INTER_AREA), scale


def draw_gaze(frame, scale, trail, point, fixation):
    """След за последние полторы секунды, текущая точка и номер фиксации."""
    pts = [(int(x * scale), int(y * scale)) for _, x, y in trail.points]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 200, 255), 2, cv2.LINE_AA)
    if point is not None:
        c = (int(point[0] * scale), int(point[1] * scale))
        cv2.circle(frame, c, 18, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, c, 4, (0, 0, 255), -1, cv2.LINE_AA)
        if fixation:
            cv2.putText(frame, str(fixation), (c[0] + 22, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def heatmap(background, fixations, scale, first=10):
    """Тепловая карта фиксаций (вес — длительность) и номера первых фиксаций."""
    h, w = background.shape[:2]
    heat = np.zeros((h, w), np.float32)
    for f in fixations:
        x, y = int(f["x"] * scale), int(f["y"] * scale)
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += f["duration_ms"]
    out = background.copy()
    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=w / 40)
        heat /= heat.max()
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = (np.clip(heat, 0, 1) * 0.6)[:, :, None]
        out = (out * (1 - alpha) + color * alpha).astype(np.uint8)
    for f in fixations[:first]:
        c = (int(f["x"] * scale), int(f["y"] * scale))
        cv2.circle(out, c, 16, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, c, 16, (0, 0, 0), 2, cv2.LINE_AA)
        text = str(f["order"])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, text, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                    cv2.LINE_AA)
    return out


class Recorder:
    def __init__(self, calibration, grab=None, folder=None, probe=None):
        self.folder = folder or records_dir() / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cal = calibration
        self.screen = tuple(calibration.screen) if calibration else None
        self.grab = grab  # для тестов: вместо снимка экрана
        self.started = time.time()
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.camera = TimedWriter(self.folder / "camera.mp4", 30)
        self.camera_trace = TimedWriter(self.folder / "camera_trace.mp4", 30)
        self.screen_video = TimedWriter(self.folder / "screen_gaze.mp4", SCREEN_FPS)
        self.csv_file = open(self.folder / "gaze.csv", "w", newline="", encoding="utf-8")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(GAZE_COLUMNS)
        dispersion = 0.05 * self.screen[2] if self.screen else 60
        self.fix = Fixations(dispersion=dispersion)
        self.trail = Trail()
        self.point, self.fixation = None, None
        # probe: точка → (программа, окно); для тестов подменяется
        self.probe = probe if probe is not None else (WindowProbe(self.screen) if self.screen else None)
        self.attention = AttentionLog()
        self.zone, self.app = NO_DATA, ""
        self.last_screen, self.scale = None, 1.0
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._screen_loop, daemon=True)
        self.thread.start()

    def camera_frame(self, frame, ts, traced=None):
        with self.lock:
            self.camera.add(frame, ts - self.t0)
            if traced is not None:
                self.camera_trace.add(traced, ts - self.t0)

    def gaze(self, ts, feat, raw, smooth, status="", phone=False, dist=None, method=""):
        t = ts - self.t0
        zone = zone_of(smooth, self.screen[2], self.screen[3]) if self.screen else NO_DATA
        # окно под взглядом — вне блокировки; WindowProbe сам спрашивает Windows не чаще 10 раз в секунду
        app, window = self.probe(smooth) if self.probe is not None and zone in GRID_ZONES else ("", "")
        with self.lock:
            fixation = None
            if smooth is not None:
                fixation = self.fix.add(t, float(smooth[0]), float(smooth[1]))
                self.trail.add(t, float(smooth[0]), float(smooth[1]))
            else:
                self.fix.gap()
            self.point, self.fixation = smooth, fixation
            f = feat or {}
            self.csv.writerow([
                "%.3f" % t, time.strftime("%H:%M:%S", time.localtime(self.started + t)),
                *(("%.1f" % v if v is not None else "") for v in (
                    smooth[0] if smooth is not None else None, smooth[1] if smooth is not None else None,
                    raw[0] if raw is not None else None, raw[1] if raw is not None else None)),
                int(smooth is not None), fixation or "",
                *("%.4f" % f[k] if k in f else "" for k in ("hx", "vy", "look_h", "look_v", "yaw", "pitch")),
                status, int(bool(phone)), zone, app, window, "%.0f" % dist if dist else "", method])
            self.attention.add(t, zone, app, window)
            self.zone, self.app = zone, app
            self.samples += 1

    def _screen_loop(self):
        if self.screen is None and self.grab is None:
            return
        import mss
        with mss.MSS() as sct:
            left, top, width, height = self.screen
            mon = {"left": left, "top": top, "width": width, "height": height}
            while not self.stop_event.is_set():
                t = time.monotonic() - self.t0
                frame, scale = self.grab() if self.grab else grab_screen(sct, mon)
                with self.lock:
                    self.last_screen, self.scale = frame.copy(), scale
                    shown = draw_gaze(frame, scale, self.trail, self.point, self.fixation)
                    draw_zone_overlay(shown, self.zone, self.app)
                    self.screen_video.add(shown, t)
                self.stop_event.wait(max(0.0, 1 / SCREEN_FPS - (time.monotonic() - self.t0 - t)))

    def stop(self):
        """Закрывает файлы, пишет фиксации, тепловую карту и описание; возвращает папку."""
        self.stop_event.set()
        self.thread.join(timeout=5)
        with self.lock:
            fixations = self.fix.finish()
            segments = self.attention.finish()
            self.camera.close()
            self.camera_trace.close()
            self.screen_video.close()
            self.csv_file.close()
        with open(self.folder / "fixations.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ("order", "start", "end", "duration_ms", "x", "y", "samples"))
            w.writeheader()
            for fx in fixations:
                w.writerow({**fx, "start": "%.3f" % fx["start"], "end": "%.3f" % fx["end"]})
        with open(self.folder / "attention.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ATTENTION_COLUMNS)
            w.writeheader()
            for s in segments:
                w.writerow({**s, "start": "%.3f" % s["start"], "end": "%.3f" % s["end"],
                            "duration_s": "%.3f" % s["duration_s"],
                            "time": time.strftime("%H:%M:%S", time.localtime(self.started + s["start"]))})
        attention = summarize(segments)
        if self.last_screen is not None:
            cv2.imencode(".png", heatmap(self.last_screen, fixations, self.scale))[1].tofile(
                str(self.folder / "heatmap.png"))
        cv2.imencode(".png", zones_image(self.last_screen, attention))[1].tofile(str(self.folder / "zones.png"))
        summary = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "duration_s": round(time.monotonic() - self.t0, 1),
            "screen": self.screen, "calibration": self.cal.to_json() if self.cal else None,
            "gaze_samples": self.samples, "fixations": len(fixations),
            "first_fixations": fixations[:10],
            "zones": attention["zones"], "apps": attention["apps"], "windows": attention["windows"],
            "files": {"camera.mp4": "камера как есть", "camera_trace.mp4": "камера с лучиками взгляда",
                      "screen_gaze.mp4": "экран со следом взгляда, сеткой зон 3×3 и текущей зоной",
                      "gaze.csv": "точка взгляда на каждом кадре (пиксели экрана), зона, программа и окно",
                      "fixations.csv": "фиксации по порядку", "heatmap.png": "тепловая карта и порядок",
                      "attention.csv": "отрезки внимания по времени: зона экрана, программа, окно, длительность",
                      "zones.png": "доля времени в каждой зоне экрана 3×3"}}
        (self.folder / "session.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
        return self.folder
