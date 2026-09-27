from gaze_check.gazelog import (BLINK, CLOSED, DOUBLE, FACE, LOOKING, SAMPLE_EVERY, GazeLog, blink_rate,
                                read_log, summary, to_csv)
from gaze_check.player import PlayerModel
from gaze_check.zones import ALL_ZONES, zone_of

SCREEN = (0, 0, 2560, 1600)


def write(path, secs=10.0, fps=15):
    """Сеанс: взгляд слева направо, моргание каждые 2 с, одно двойное."""
    log = GazeLog(SCREEN, 60.0, path=path, clock=lambda: 0.0)
    for i in range(round(secs * fps)):
        t = i / fps
        point = None if 4.0 <= t < 5.0 else (t / secs * 2560, 800)
        zone = ALL_ZONES.index(zone_of(point, 2560, 1600))
        log.sample(t, point, 60, FACE | LOOKING, zone, "Code.exe — app.py" if t < 6 else "chrome.exe — сайт")
        if i % (2 * fps) == fps:
            log.event(t, BLINK, 150)
    log.event(7.0, DOUBLE)
    return log.close()


def test_roundtrip(tmp_path):
    log = read_log(write(tmp_path / "a.gazelog"))
    assert log["screen"] == (2560, 1600) and abs(log["px_per_cm"] - 60) < 1e-6
    s = log["samples"]
    assert abs(len(s) - 10 / SAMPLE_EVERY) <= 2  # 15 к/с в 10 отсчётов в секунду
    t, x, y, cm, flags, zone, app = s[0]
    assert (x, y, cm, app) == (0, 800, 60, "Code.exe — app.py") and flags & LOOKING
    assert any(x is None for _, x, *_ in s)  # «нет точки» записано и прочитано
    assert s[-1][6] == "chrome.exe — сайт"
    assert [k for _, k, _ in log["events"]].count(BLINK) == 5
    assert (7.0, DOUBLE, 0) in log["events"]
    assert (tmp_path / "a.gazelog").stat().st_size < 3000  # ~10 байт на отсчёт


def test_truncated_tail_is_readable(tmp_path):
    path = write(tmp_path / "b.gazelog")
    whole = read_log(path)
    data = path.read_bytes()
    path.write_bytes(data[:-5])  # вылет посреди записи
    cut = read_log(path)
    assert len(cut["samples"]) + len(cut["events"]) == len(whole["samples"]) + len(whole["events"]) - 1


def test_blink_rate_and_summary(tmp_path):
    log = read_log(write(tmp_path / "c.gazelog"))
    assert blink_rate(log["events"], 10.0) == 5.0  # 5 за последнюю минуту
    s = summary(log)
    assert s["blinks"] == 5 and 25 <= s["blinks_per_min"] <= 35
    assert set(s["zones"]) <= set(ALL_ZONES) and s["apps"]["Code.exe — app.py"] > s["apps"]["chrome.exe — сайт"]
    csv = to_csv(log, tmp_path / "c.csv")
    text = csv.read_text(encoding="utf-8")
    assert text.startswith("t,x,y") and "моргание" in text


def test_player_state(tmp_path):
    model = PlayerModel(read_log(write(tmp_path / "d.gazelog")))
    assert model.state_at(-1) is None
    st = model.state_at(3.2)
    assert st["point"][0] > 700 and len(st["trail"]) > 10 and st["blinks"] == 2
    assert not st["closed"] and model.state_at(0.95)["closed"]  # моргание 0,85–1,0 с
    assert model.state_at(4.5)["point"] is None
    assert model.state_at(9.0)["app"] == "chrome.exe — сайт"


def test_closed_flag(tmp_path):
    log = GazeLog(SCREEN, 60.0, path=tmp_path / "e.gazelog", clock=lambda: 0.0)
    log.sample(0.0, (10, 10), 60, FACE | CLOSED, 0, "")
    model = PlayerModel(read_log(log.close()))
    assert model.state_at(0.05)["closed"]


def test_player_heat_and_timeline(tmp_path):
    model = PlayerModel(read_log(write(tmp_path / "f.gazelog")))
    heat = model.heat(8, 5)
    assert max(max(r) for r in heat) == 1.0 and max(heat[0]) == 0  # взгляд шёл по середине, верх пуст
    line = model.timeline(10)
    assert len(line) == 10 and sum(bl for *_, bl in line) == 5
    assert all(face == 1.0 for _, face, _ in line) and line[0][0] == 1.0
