"""Логика проверки взгляда без моделей: метки взгляда, счёт времени без взгляда в экран
(как блок Gaze_Duration_Status из Roboflow-workflow), фильтр «cell phone» и присутствие человека.
"""

GAZE_CLASSES = ("looking at screen", "looking away", "face not visible")
PHONE_CLASS = "cell phone"
PROLONGED_AFTER = 60.0  # секунд без взгляда в экран до prolonged_away (в облаке было 10)
STALE_AFTER = 8.0  # разрыв между кадрами, после которого серия начинается заново


def top_class(predictions):
    return predictions.get("top") if predictions else None


class GazeTimer:
    """Блок Gaze_Duration_Status для одного видеопотока.

    Отличие от облака: по умолчанию «лица не видно» тоже считается «не смотрит в экран»,
    иначе при повороте головы метка скачет между away и face и счёт сбрасывается.
    face_counts=False — поведение облачного блока один в один.
    """

    def __init__(self, prolonged_after=PROLONGED_AFTER, stale_after=STALE_AFTER, face_counts=True):
        self.prolonged_after = prolonged_after
        self.stale_after = stale_after
        self.face_counts = face_counts
        self.state = None

    def update(self, gaze_class, now, frame=None, present=True):
        if not present:  # в кадре никого — отвлечения нет, отсчёт сначала
            self.state = None
            return {"status": "no_person", "display_text": "No person", "away_seconds": 0.0,
                    "prolonged_away": False}
        label = str(gaze_class or "").strip().lower()
        if label not in GAZE_CLASSES:
            label = "face not visible"
        away = label == "looking away" or (self.face_counts and label == "face not visible")
        prev = self.state
        if (prev is None
                or (frame is not None and prev["frame"] is not None and frame <= prev["frame"])
                or now < prev["last"] or now - prev["last"] > self.stale_after):
            prev = {"key": None, "since": now, "last": now, "frame": frame}
        key = "away" if away else label
        if key != prev["key"]:
            prev["since"] = now
        elapsed = max(0.0, now - prev["since"]) if away else 0.0
        prev.update(key=key, last=now, frame=frame)
        self.state = prev
        prolonged = away and elapsed >= self.prolonged_after
        if prolonged:
            status = "prolonged_away"
        elif label == "face not visible":
            status = "face_not_visible"
        elif label == "looking away":
            status = "brief_away"
        else:
            status = "looking_at_screen"
        if label == "face not visible" and not prolonged:
            text = "Face not visible"
        elif away:
            text = "Looking away ~%ds" % round(elapsed)
        else:
            text = "Looking at screen"
        return {"status": status, "display_text": text,
                "away_seconds": float(round(elapsed, 1)), "prolonged_away": prolonged}


def phone_outputs(detections):
    """Ветка телефона: detections_filter по «cell phone», счётчик и сообщение."""
    phones = [d for d in detections if d.get("class") == PHONE_CLASS]
    return {"phone_predictions": phones, "phone_count": len(phones),
            "phone_present": len(phones) > 0,
            "phone_message": "Phone visible - please put it away" if phones else "No phone detected"}


class Hold:
    """Тревога по телефону без мигания: включается после hits находок подряд
    и держится ещё hold секунд после последней.

    У порога детектор то находит телефон, то теряет, и без этого предупреждение мигает
    и пищит заново, а одиночная ложная рамка сразу даёт тревогу. Выходы workflow
    (phone_present) при этом не меняются.
    """

    def __init__(self, hold=1.5, hits=2):
        self.hold = hold
        self.hits = hits
        self.streak = 0
        self.last = None

    def update(self, on, now):
        self.streak = self.streak + 1 if on else 0
        if self.streak >= self.hits:
            self.last = now
        return self.last is not None and now - self.last <= self.hold


PERSON_CLASS = "person"
PERSON_CONFIDENCE = 0.5
PRESENCE_HOLD = 3.0  # секунд после последней находки человек ещё считается в кадре


def person_seen(detections):
    return any(d.get("class") == PERSON_CLASS and d.get("confidence", 0) >= PERSON_CONFIDENCE
               for d in detections)


def presence(now, person_at=None, face_at=None, hold=PRESENCE_HOLD):
    """Есть ли человек: RF-DETR видел person или MediaPipe — лицо не позже hold секунд назад."""
    return any(t is not None and 0 <= now - t <= hold for t in (person_at, face_at))
