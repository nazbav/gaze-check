"""Пути к моделям и данным. Модели маленькие и едут вместе с программой, ничего не качается:
MediaPipe Face Landmarker (глаза, ~4 МБ) и MGazeNet (взгляд). Детектор телефона и человека убран 28.09."""
import os
import sys
from pathlib import Path

import numpy as np

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "GazeCheck"  # настройки, журнал


def _models_dir():
    """GAZECHECK_MODELS; у exe — models внутри сборки; иначе models/ в корне проекта."""
    if os.environ.get("GAZECHECK_MODELS"):
        return Path(os.environ["GAZECHECK_MODELS"])
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", ROOT)) / "models"
    return ROOT / "models"


MODELS_DIR = _models_dir()
EYE_MODEL = MODELS_DIR / "face_landmarker.task"
GAZENET_MODEL = MODELS_DIR / "mgazenet.mnn"  # из GazeFollower, CC BY-NC-SA 4.0
CALIBRATION = DATA_DIR / "calibration.json"  # прежняя одна калибровка — переезжает в набор (calstore.py)
CALIBRATIONS = DATA_DIR / "calibrations"  # набор: у каждой камеры и света своя
GAZEFOLLOWER_COMMIT = "13806edabefe76fc6b964c4c5bdcd62846c1ac1d"
# откуда брать модели для разработки (в exe они уже внутри)
MODEL_URLS = {
    EYE_MODEL.name: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
                    "float16/latest/face_landmarker.task",
    # MGazeNet из GazeFollower (base.mnn), закреплён на коммите — веса CC BY-NC-SA 4.0
    GAZENET_MODEL.name: "https://github.com/GanchengZhu/GazeFollower/raw/%s/"
                        "gazefollower/res/model_weights/base.mnn" % GAZEFOLLOWER_COMMIT,
    "mgazenet-LICENSE.txt": "https://github.com/GanchengZhu/GazeFollower/raw/%s/LICENSE-CC-BY-NC-SA"
                            % GAZEFOLLOWER_COMMIT,
}


def models_ready():
    return EYE_MODEL.exists()


def to_rgb(bgr):
    return np.ascontiguousarray(bgr[:, :, ::-1])
