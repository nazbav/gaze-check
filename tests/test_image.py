import json
from pathlib import Path

import pytest

from gaze_check.models import models_ready

PERSON = Path(__file__).parent / "data" / "person.jpg"


@pytest.mark.skipif(not models_ready(), reason="нет моделей")
def test_image_mode_writes_json(tmp_path, monkeypatch):
    """Режим одной картинки (им же build.py проверяет собранный exe)."""
    import gaze_check.models as models
    monkeypatch.setattr(models, "CALIBRATION", tmp_path / "none.json")
    from gaze_check.app import main
    out = tmp_path / "r.json"
    main(["--source", str(PERSON), "--no-window", "--json", str(out), "--save", str(tmp_path / "r.jpg")])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["face"] and res["person"] and res["eye"]["gy"] is not None and res["net"]
    assert res["predictions"]["top"] in ("looking at screen", "looking away")
    assert (tmp_path / "r.jpg").exists()
