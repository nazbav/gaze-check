import types

import cv2
import numpy as np

from gaze_check.app import TITLE, Latest, State, Viewer


def test_window_closed_by_cross_does_not_crash():
    """Крестик уничтожает окно раньше нас — повторный destroyWindow ронял программу."""
    engine = types.SimpleNamespace(latest=Latest(), state=State(), journal=types.SimpleNamespace(sound=True))
    engine.latest.put(np.zeros((72, 128, 3), np.uint8))
    viewer = Viewer(engine)
    assert viewer.tick()
    cv2.destroyWindow(TITLE)
    cv2.waitKey(1)
    assert not viewer.tick() and not viewer.open
    viewer.close()
    assert viewer.tick()
    viewer.close()
