"""Набор калибровок: для каждой камеры и света — своя (свет — light.py). Файл на калибровку в
%LOCALAPPDATA%\\GazeCheck\\calibrations: формат Calibration.to_json, в meta — id, камера, свет,
когда последний раз выбрана. Прежний calibration.json при первом запуске становится первой
калибровкой (свет неизвестен) и не трогается — чтобы можно было вернуться на старую версию."""
import math
import time
import uuid
from pathlib import Path

from .eyes import Calibration
from .light import SAME, describe, distance


def title(cal):
    """«темно, тёплый · 28.09 00:30 · ~2,2 см» — для меню и журнала."""
    light = cal.meta.get("light")
    parts = [describe(light) if light else "прежняя, свет неизвестен"]
    created = cal.created or ""
    if len(created) >= 16:
        parts.append("%s.%s %s" % (created[8:10], created[5:7], created[11:16]))
    px = cal.check_px if cal.check_px is not None else cal.error_px
    if px is not None:
        parts.append(("~%.1f см" % (px / cal.px_per_cm())).replace(".", ","))
    return " · ".join(parts)


class CalStore:
    def __init__(self, folder, screen, legacy=None, camera=None):
        self.folder, self.screen = Path(folder), tuple(screen)
        self.items = []  # Calibration с meta["id"]
        fresh = not self.folder.is_dir()
        if not fresh:
            for p in sorted(self.folder.glob("*.json")):
                cal = Calibration.load(p, self.screen)  # другой экран или версия — пропускаем
                if cal is not None and cal.meta.get("id") == p.stem:
                    self.items.append(cal)
        if fresh and legacy is not None and camera is not None:  # переезд — один раз, пока папки нет
            old = Calibration.load(Path(legacy), self.screen)
            if old is not None and old.calibrated:
                self.put(old, camera, None)

    def for_camera(self, camera):
        return [c for c in self.items if c.meta.get("camera") == camera]

    def nearest(self, camera, sig):
        """→ (калибровка, расстояние по свету): ближайшая среди калибровок камеры со своим светом;
        таких нет или свет ещё не понят — последняя выбранная, расстояние бесконечность."""
        cams = self.for_camera(camera)
        known = [c for c in cams if c.meta.get("light")]
        if sig is not None and known:
            best = min(known, key=lambda c: distance(sig, c.meta["light"]))
            return best, distance(sig, best.meta["light"])
        return (max(cams, key=lambda c: c.meta.get("used", 0)) if cams else None), math.inf

    def same_light(self, camera, sig):
        """Калибровка этой камеры с тем же светом или None."""
        best, d = self.nearest(camera, sig)
        return best if d <= SAME else None

    def put(self, cal, camera, light, replace=None):
        """Калибровку — в набор: новая или вместо replace (тот же id, тот же файл)."""
        cal.meta = {"id": replace.meta["id"] if replace is not None else uuid.uuid4().hex[:12],
                    "camera": camera, "light": light, "used": time.time()}
        self.update(cal)
        return cal

    def update(self, cal):
        """Та же калибровка после кликов, пересчёта под ведущий глаз и т. п. — заменить и сохранить."""
        if not cal.meta.get("id"):
            return
        for i, c in enumerate(self.items):
            if c.meta.get("id") == cal.meta["id"]:
                self.items[i] = cal
                break
        else:
            self.items.append(cal)
        self.save(cal)

    def save(self, cal):
        cal.save(self.folder / ("%s.json" % cal.meta["id"]))
