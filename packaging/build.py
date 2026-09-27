"""Сборка GazeCheck.exe: тесты, PyInstaller (папка dist/GazeCheck), прогон exe на фото, zip.

    .venv\\Scripts\\python packaging\\build.py            exe + zip (модели внутри)
    .venv\\Scripts\\python packaging\\build.py --no-zip   только exe
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = "GazeCheck"
DIST = ROOT / "dist"
WORK = ROOT / "build"
EXE_DIR = DIST / APP
# ничего из этого программе не нужно; matplotlib и sounddevice импортирует MediaPipe,
# но gaze_check/mp.py подменяет их заглушками
EXCLUDE = ["matplotlib", "sounddevice", "torch", "torchvision", "transformers", "rfdetr",
           "onnxruntime", "onnx", "scipy", "pandas", "jax", "jaxlib", "IPython", "pytest", "sympy",
           "huggingface_hub", "accelerate", "bitsandbytes", "supervision"]


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def make_icon():
    from gaze_check.app import GREEN
    from gaze_check.tray import icon_image
    WORK.mkdir(exist_ok=True)
    ico = WORK / "icon.ico"
    icon_image(GREEN).resize((256, 256), Image.LANCZOS).save(
        ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    return ico


def pyinstaller(ico):
    from gaze_check.models import DETECTOR_MODEL, EYE_MODEL, GAZENET_MODEL
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
           "--name", APP, "--icon", ico, "--distpath", DIST, "--workpath", WORK / "pyi",
           "--specpath", WORK, "--additional-hooks-dir", ROOT / "packaging" / "hooks",
           "--hidden-import", "mss", "--collect-all", "MNN"]
    for model in (EYE_MODEL, DETECTOR_MODEL, GAZENET_MODEL, GAZENET_MODEL.parent / "mgazenet-LICENSE.txt"):
        cmd += ["--add-data", "%s;models" % model]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    run(cmd + [ROOT / "launcher.py"])
    shutil.copy2(ROOT / "README.md", EXE_DIR / "README.md")


def smoke():
    """exe на фото с человеком: находит лицо, глаза и человека — значит, всё собралось."""
    img = ROOT / "tests" / "data" / "person.jpg"
    out = WORK / "smoke.json"
    out.unlink(missing_ok=True)
    t = time.time()
    subprocess.run([EXE_DIR / f"{APP}.exe", "--source", img, "--no-window", "--json", out],
                   timeout=300, check=True)  # ошибка → код 1, без окна
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["face"] and res["eye"], "трекер глаз не отработал"
    assert res["person"], "детектор не нашёл человека"
    assert res["net"], "MGazeNet не отработал (MNN или модель не попали в сборку)"
    print("smoke: %.1f с от запуска до ответа, взгляд=%s, точка=%s" % (
        time.time() - t, res["predictions"]["top"], res["point"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    if not args.skip_tests:
        run([sys.executable, "-m", "pyflakes", "gaze_check", "tests", "launcher.py", "packaging"])
        run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
    shutil.rmtree(EXE_DIR, ignore_errors=True)
    pyinstaller(make_icon())
    smoke()
    size = sum(p.stat().st_size for p in EXE_DIR.rglob("*") if p.is_file())
    print("dist\\%s: %.0f МБ" % (APP, size / 2**20), flush=True)
    if not args.no_zip:
        zip_path = DIST / (APP + "-win64")
        shutil.make_archive(str(zip_path), "zip", DIST, APP)
        print("zip: %s.zip, %.0f МБ" % (zip_path, Path(str(zip_path) + ".zip").stat().st_size / 2**20))


if __name__ == "__main__":
    main()
