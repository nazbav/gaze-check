# mediapipe 1.x грузит tasks/c/libmediapipe.dll через ctypes по пути внутри пакета
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

binaries = [(src, "mediapipe/tasks/c") for src, _ in collect_dynamic_libs("mediapipe")]
datas = collect_data_files("mediapipe")
hiddenimports = collect_submodules("mediapipe.tasks.python", filter=lambda n: ".audio" not in n and "test" not in n)
