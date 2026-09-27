from gaze_check.workflow import GazeTimer, Hold, person_seen, phone_outputs, presence, top_class


def test_timer_counts_away_and_goes_prolonged():
    t = GazeTimer(prolonged_after=10.0, face_counts=False)
    assert t.update("looking at screen", 0.0)["status"] == "looking_at_screen"
    r = t.update("looking away", 1.0)
    assert r["status"] == "brief_away" and r["away_seconds"] == 0.0
    for now in range(2, 11):
        r = t.update("looking away", float(now))
    assert r["away_seconds"] == 9.0 and not r["prolonged_away"]
    r = t.update("looking away", 11.0)
    assert r["status"] == "prolonged_away" and r["prolonged_away"]
    assert r["display_text"] == "Looking away ~10s"


def test_timer_resets_on_look_back_and_on_gap():
    t = GazeTimer(face_counts=False)
    t.update("looking away", 0.0)
    t.update("looking away", 5.0)
    assert t.update("looking at screen", 6.0)["away_seconds"] == 0.0
    assert t.update("looking away", 7.0)["away_seconds"] == 0.0
    # разрыв больше 8 с — серия начинается заново
    assert t.update("looking away", 16.0)["away_seconds"] == 0.0
    assert t.update("looking away", 20.0)["away_seconds"] == 4.0


def test_top_class():
    assert top_class({"top": "looking away"}) == "looking away" and top_class(None) is None


def test_timer_unknown_label_is_face_not_visible():
    r = GazeTimer().update(None, 0.0)
    assert r["status"] == "face_not_visible" and r["display_text"] == "Face not visible"


def test_phone_filter():
    dets = [{"class": "person"}, {"class": "cell phone"}, {"class": "cell phone"}]
    out = phone_outputs(dets)
    assert out["phone_count"] == 2 and out["phone_present"]
    assert phone_outputs([{"class": "person"}])["phone_message"] == "No phone detected"


def test_hold_needs_two_hits_and_bridges_gaps():
    h = Hold(hold=1.5, hits=2)
    assert not h.update(True, 0.0)  # одиночная рамка — ещё не тревога
    assert not h.update(False, 0.25)
    assert not h.update(True, 0.5)
    assert h.update(True, 0.75)
    assert h.update(False, 1.0) and h.update(False, 2.25)  # пропуски внутри 1.5 с — держится
    assert not h.update(False, 2.5)


def test_timer_default_minute_and_face_counts():
    t = GazeTimer()
    t.update("looking at screen", 0.0)
    t.update("looking away", 1.0)
    # поворот головы: метка скачет между away и face — счёт не сбрасывается
    for now in range(3, 61, 2):
        r = t.update("face not visible" if now % 4 == 1 else "looking away", float(now))
    assert r["away_seconds"] == 58.0 and not r["prolonged_away"]
    r = t.update("face not visible", 61.0)
    assert r["status"] == "prolonged_away" and r["away_seconds"] == 60.0
    assert t.update("looking at screen", 62.0)["away_seconds"] == 0.0


def test_timer_cloud_mode_face_resets():
    t = GazeTimer(prolonged_after=10.0, face_counts=False)
    t.update("looking away", 0.0)
    assert t.update("face not visible", 5.0)["away_seconds"] == 0.0
    assert t.update("looking away", 6.0)["away_seconds"] == 0.0


def test_no_person_resets_and_never_alarms():
    t = GazeTimer(prolonged_after=10.0)
    for now in range(0, 9):
        t.update("looking away", float(now))
    r = t.update("face not visible", 9.0, present=False)
    assert r["status"] == "no_person" and not r["prolonged_away"] and r["away_seconds"] == 0.0
    for now in range(10, 40):  # человек ушёл надолго — тишина
        assert not t.update("face not visible", float(now), present=False)["prolonged_away"]
    assert t.update("looking away", 40.0)["away_seconds"] == 0.0  # вернулся — счёт с нуля


def test_presence_from_person_or_face():
    assert not presence(10.0)
    assert presence(10.0, person_at=8.0) and presence(10.0, face_at=7.5)
    assert not presence(10.0, person_at=6.0, face_at=6.9)
    assert person_seen([{"class": "person", "confidence": 0.8}])
    assert not person_seen([{"class": "person", "confidence": 0.3}, {"class": "cell phone", "confidence": 0.9}])
