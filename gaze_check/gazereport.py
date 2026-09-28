"""Отчёт по лёгкому журналу взгляда — те же итоги, что у записи сеанса (recorder.py), только без видео:
фиксации, отрезки внимания, доли по зонам, программам и окнам, тепловая карта, картинка зон, и то, что
есть только в журнале: моргания, «смотрит в экран», лицо видно, расстояние до экрана.
Пишется при остановке журнала в папку рядом с ним (то же имя без .gazelog); для старых журналов —
GazeCheck.exe --report [файл.gazelog]."""
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .eyes import Fixations
from .gazelog import BLINK, CLOSED, DOUBLE, FACE, LOOKING, RIGHT, TRIPLE, logs_dir, read_log, to_csv
from .recorder import ATTENTION_COLUMNS, SCREEN_WIDTH, heatmap
from .zones import ALL_ZONES, NO_DATA, AttentionLog, summarize, zones_image

FILES = {"session.json": "итоги: зоны, программы, окна, фиксации, моргания, доля «смотрит в экран»",
         "gaze.csv": "отсчёты журнала (10 в секунду): точка, флаги, зона, программа; внизу — события",
         "fixations.csv": "фиксации по порядку", "heatmap.png": "тепловая карта фиксаций и порядок первых",
         "attention.csv": "отрезки внимания по времени: зона экрана, программа, окно, длительность",
         "zones.png": "доля времени в каждой зоне экрана 3×3"}


def report_dir(path):
    """Папка отчёта: рядом с журналом, то же имя без расширения."""
    return Path(path).with_suffix("")


def latest_log():
    logs = sorted(logs_dir().glob("*.gazelog"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _screen(w, h):
    """Снимка экрана в журнале нет — тёмный экран с сеткой зон 3×3 в масштабе видео записи."""
    scale = SCREEN_WIDTH / max(1, w)
    img = np.full((max(1, round(h * scale)), SCREEN_WIDTH, 3), 34, np.uint8)
    for i in (1, 2):
        cv2.line(img, (SCREEN_WIDTH * i // 3, 0), (SCREEN_WIDTH * i // 3, img.shape[0]), (80, 80, 80), 1)
        cv2.line(img, (0, img.shape[0] * i // 3), (SCREEN_WIDTH, img.shape[0] * i // 3), (80, 80, 80), 1)
    return img, scale


def build(path, out=None):
    """Журнал → папка отчёта (как у записи сеанса). → путь папки."""
    log = read_log(path)
    out = Path(out) if out else report_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    (w, h), samples, events = log["screen"], log["samples"], log["events"]
    fix, attention_log = Fixations(dispersion=0.05 * w), AttentionLog()
    for t, x, y, _cm, _flags, zone, app in samples:
        if x is None:
            fix.gap()
        else:
            fix.add(t, float(x), float(y))
        prog, _, window = app.partition(" — ")  # в журнале — «программа — заголовок окна»
        attention_log.add(t, ALL_ZONES[zone] if zone < len(ALL_ZONES) else NO_DATA, prog, window)
    fixations, segments = fix.finish(), attention_log.finish()
    attention = summarize(segments)

    def clock(t):
        return time.strftime("%H:%M:%S", time.localtime(log["start"] + t))

    with open(out / "fixations.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ("order", "start", "end", "duration_ms", "x", "y", "samples"))
        wr.writeheader()
        for fx in fixations:
            wr.writerow({**fx, "start": "%.3f" % fx["start"], "end": "%.3f" % fx["end"]})
    with open(out / "attention.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ATTENTION_COLUMNS)
        wr.writeheader()
        for s in segments:
            wr.writerow({**s, "start": "%.3f" % s["start"], "end": "%.3f" % s["end"],
                         "duration_s": "%.3f" % s["duration_s"], "time": clock(s["start"])})
    screen, scale = _screen(w, h)
    cv2.imencode(".png", heatmap(screen, fixations, scale))[1].tofile(str(out / "heatmap.png"))
    cv2.imencode(".png", zones_image(screen, attention))[1].tofile(str(out / "zones.png"))
    to_csv(log, out / "gaze.csv")

    n = len(samples) or 1
    dur = samples[-1][0] - samples[0][0] if samples else 0.0
    blinks = [v for _, kind, v in events if kind == BLINK]
    dists = [s[3] for s in samples if s[3]]

    def share(flag):
        return round(sum(1 for s in samples if s[4] & flag) / n, 4)

    def count(kind):
        return sum(1 for _, k, _ in events if k == kind)

    summary = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log["start"])),
        "duration_s": round(dur, 1), "screen": [w, h], "source": Path(path).name,
        "samples": len(samples), "gaze_samples": sum(1 for s in samples if s[1] is not None),
        "fixations": len(fixations), "first_fixations": fixations[:10],
        "zones": attention["zones"], "apps": attention["apps"], "windows": attention["windows"],
        "looking_share": share(LOOKING), "face_share": share(FACE),
        "eyes_closed_share": share(CLOSED),
        "blinks": len(blinks), "blinks_per_min": round(len(blinks) * 60 / dur, 1) if dur else 0.0,
        "blink_ms_median": int(np.median(blinks)) if blinks else None,
        "double_blinks": count(DOUBLE), "triple_blinks": count(TRIPLE), "eyes_closed_2_3s": count(RIGHT),
        "distance_cm_median": int(np.median(dists)) if dists else None,
        "files": FILES,
    }
    (out / "session.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
