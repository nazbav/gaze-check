"""Скачивает две модели MediaPipe в models/ (для разработки; в exe они уже внутри)."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.models import MODEL_URLS, MODELS_DIR  # noqa: E402


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODELS_DIR / name
        if path.exists():
            print("есть:", path)
            continue
        print("качаю:", url)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    print("Готово:", MODELS_DIR)


if __name__ == "__main__":
    main()
