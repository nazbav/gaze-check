"""Лёгкий журнал взгляда: куда смотрю, моргания, зона, программа — без видео, ~0,5 МБ в час.

Бинарный файл .gazelog: заголовок, потом записи трёх видов (первый байт — вид):
  1 — отсчёт (10 раз в секунду): мс от начала, точка x/y (px экрана; −32768 — нет), расстояние
      до лица (см), флаги, зона (номер в zones.ALL_ZONES), программа (номер строки);
  2 — событие: мс, вид (BLINK, DOUBLE, TRIPLE, RIGHT), значение (длительность моргания, мс);
  3 — строка: номер и текст «программа — заголовок окна» (пишется один раз, дальше по номеру).
JSON был бы раз в 10 больше; файл пишется потоком и читается даже недописанным.
"""
import struct
import time
from pathlib import Path

MAGIC, VERSION = b"GZLG", 1
HEADER = struct.Struct("<4sHdHHf")  # магия, версия, начало (unix), ширина, высота экрана, px на см
SAMPLE = struct.Struct("<IhhBBBH")  # мс, x, y, см, флаги, зона, программа
EVENT = struct.Struct("<IBH")  # мс, вид, значение
STRING = struct.Struct("<HH")  # номер, длина байт
NO_POINT = -32768
SAMPLE_EVERY = 0.1  # с

# флаги отсчёта
FACE, BOTH_EYES, LOOKING, PHONE, CLOSED = 1, 2, 4, 8, 16
# события
BLINK, DOUBLE, TRIPLE, RIGHT = 1, 2, 3, 4
EVENT_NAMES = {BLINK: "моргание", DOUBLE: "двойное моргание", TRIPLE: "тройное моргание",
               RIGHT: "глаза закрыты 2–3 с"}


def logs_dir():
    return Path.home() / "Documents" / "GazeCheck" / "logs"


class GazeLog:
    """Пишет журнал. sample() можно звать с каждого кадра — в файл идёт не чаще SAMPLE_EVERY."""

    def __init__(self, screen, px_per_cm, path=None, clock=time.monotonic):
        self.path = Path(path) if path else logs_dir() / time.strftime("gaze_%Y-%m-%d_%H-%M-%S.gazelog")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "wb")
        self.file.write(HEADER.pack(MAGIC, VERSION, time.time(), int(screen[2]), int(screen[3]), float(px_per_cm)))
        self.clock = clock
        self.t0 = clock()
        self.next_sample = -1.0
        self.strings = {}
        self.flushed = self.t0

    def _ms(self, t):
        return max(0, min(0xFFFFFFFF, int((t - self.t0) * 1000)))

    def _string(self, text):
        text = (text or "")[:300]
        sid = self.strings.get(text)
        if sid is None:
            sid = self.strings[text] = min(len(self.strings), 0xFFFF)
            data = text.encode("utf-8")
            self.file.write(b"\x03" + STRING.pack(sid, len(data)) + data)
        return sid

    def sample(self, t, point, dist_cm=None, flags=0, zone=0, app=""):
        if t < self.next_sample:
            return False
        if self.next_sample < t - SAMPLE_EVERY:  # был перерыв — не догонять
            self.next_sample = t
        self.next_sample += SAMPLE_EVERY  # по сетке, а не от кадра: при 15 к/с выйдет 10 в с, не 7,5
        x, y = (NO_POINT, NO_POINT) if point is None else (
            max(-32767, min(32767, int(point[0]))), max(-32767, min(32767, int(point[1]))))
        sid = self._string(app)
        self.file.write(b"\x01" + SAMPLE.pack(self._ms(t), x, y, max(0, min(255, int(dist_cm or 0))),
                                              flags & 0xFF, zone & 0xFF, sid))
        if t - self.flushed > 2.0:  # раз в 2 с на диск — вылет программы теряет не больше 2 с
            self.file.flush()
            self.flushed = t
        return True

    def event(self, t, kind, value=0):
        self.file.write(b"\x02" + EVENT.pack(self._ms(t), kind, max(0, min(0xFFFF, int(value)))))

    def close(self):
        try:
            self.file.close()
        except OSError:
            pass
        return self.path


def read_log(path):
    """→ {"start", "screen": (w, h), "px_per_cm", "samples": [(с, x|None, y|None, см, флаги, зона, программа)],
    "events": [(с, вид, значение)]}. Недописанный хвост (вылет программы) пропускается."""
    data = Path(path).read_bytes()
    magic, version, start, w, h, px_cm = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError("не журнал взгляда Gaze Check: %s" % path)
    pos, strings, samples, events = HEADER.size, {}, [], []
    while pos < len(data):
        kind = data[pos]
        pos += 1
        try:
            if kind == 1:
                ms, x, y, cm, flags, zone, sid = SAMPLE.unpack_from(data, pos)
                pos += SAMPLE.size
                samples.append((ms / 1000, None if x == NO_POINT else x, None if y == NO_POINT else y,
                                cm or None, flags, zone, strings.get(sid, "")))
            elif kind == 2:
                ms, ev, value = EVENT.unpack_from(data, pos)
                pos += EVENT.size
                events.append((ms / 1000, ev, value))
            elif kind == 3:
                sid, n = STRING.unpack_from(data, pos)
                pos += STRING.size
                strings[sid] = data[pos:pos + n].decode("utf-8", "replace")
                pos += n
            else:
                break  # мусор — дальше не читаем
        except struct.error:
            break  # недописанная запись в конце
    return {"start": start, "screen": (w, h), "px_per_cm": px_cm, "samples": samples, "events": events}


def blink_rate(events, t, window=60.0):
    """Морганий в минуту за последние window секунд до t (из журнала)."""
    n = sum(1 for te, kind, _ in events if kind == BLINK and t - window < te <= t)
    return n * 60.0 / window


def summary(log):
    """Итоги: длительность, моргания, доли времени по зонам и программам."""
    s = log["samples"]
    if not s:
        return {"duration_s": 0, "blinks": 0, "blinks_per_min": 0, "zones": {}, "apps": {}}
    from .zones import ALL_ZONES
    dur = s[-1][0] - s[0][0] or 1e-6
    zones, apps = {}, {}
    for _, x, _, _, flags, zone, app in s:
        if x is not None:
            name = ALL_ZONES[zone] if zone < len(ALL_ZONES) else "?"
            zones[name] = zones.get(name, 0) + 1
            if app:
                apps[app] = apps.get(app, 0) + 1
    total = len(s)
    blinks = sum(1 for _, kind, _ in log["events"] if kind == BLINK)
    top = lambda d: dict(sorted(((k, round(v / total, 3)) for k, v in d.items()), key=lambda kv: -kv[1])[:10])  # noqa: E731
    return {"duration_s": round(dur, 1), "blinks": blinks, "blinks_per_min": round(blinks * 60 / dur, 1),
            "zones": top(zones), "apps": top(apps)}


def to_csv(log, path):
    """Экспорт отсчётов и событий в CSV — для Excel/pandas."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "dist_cm", "face", "both_eyes", "looking", "phone", "closed", "zone", "app"])
        from .zones import ALL_ZONES
        for t, x, y, cm, flags, zone, app in log["samples"]:
            w.writerow(["%.2f" % t, x, y, cm, int(bool(flags & FACE)), int(bool(flags & BOTH_EYES)),
                        int(bool(flags & LOOKING)), int(bool(flags & PHONE)), int(bool(flags & CLOSED)),
                        ALL_ZONES[zone] if zone < len(ALL_ZONES) else "", app])
        w.writerow([])
        w.writerow(["event_t", "event", "value_ms"])
        for t, kind, value in log["events"]:
            w.writerow(["%.2f" % t, EVENT_NAMES.get(kind, kind), value])
    return path
