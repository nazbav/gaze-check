"""Достаёт ли модель до углов: на двух записях — «без одного угла», новые позы, качание головой.

    .venv\\Scripts\\python tools\\gaze_corners.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_analyze import HEAD, Geo, load, metrics  # noqa: E402
from gaze_variants import PCAModel  # noqa: E402

from gaze_check.eyes import Calibration  # noqa: E402


class LinRBF(PCAModel):
    """Ядро «линейное + RBF»: вдали от точек калибровки продолжает прямую, а не тянет к центру."""

    def __init__(self, k=16, gamma=0.02, lam=0.05, lin=1.0, **kw):
        super().__init__(k=k, kind="rbf", gamma=gamma, lam=lam, **kw)
        self.lin = lin
        self.name = "PCA%d lin+rbf g=%g l=%g" % (k, gamma, lin)

    def _kernel(self, x):
        d = ((x[:, None, :] - self.x_train[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d) + self.lin * (x @ self.x_train.T) / x.shape[1]


def datasets():
    cal = Calibration.load(Path(os.path.expandvars(r"%LOCALAPPDATA%\GazeCheck\calibration.json")))

    def conv(items, phase):
        return [{"phase": phase, "target": list(p), "feat": f, "net": f["net"]} for p, f in items
                if f.get("net") is not None]
    b = conv(cal.points, "обычно") + conv(cal.moving, "голова") + conv(cal.checks, "проверка")
    a = load()["frames"]
    return cal.screen, cal.size_mm, {"A": (a, ("обычно", "ближе", "дальше") + HEAD), "B": (b, ("обычно",) + HEAD)}


def corner_loo(make, frames, train, screen, size, px_cm):
    tr = [f for f in frames if f["phase"] in train]
    w, h = screen[2], screen[3]
    corners = {tuple(f["target"]) for f in tr if f["phase"] == "обычно"
               and (f["target"][0] < 0.15 * w or f["target"][0] > 0.85 * w)
               and (f["target"][1] < 0.15 * h or f["target"][1] > 0.85 * h)}
    errs = []
    for c in corners:
        m = make().fit([f for f in tr if tuple(f["target"]) != c], screen, size)
        test = [f for f in tr if tuple(f["target"]) == c]
        p = np.median(m.predict(test), 0)
        errs.append(np.hypot(*(p - c)) / px_cm)
    return float(np.mean(errs))


def main():
    screen, size, sets = datasets()
    px_cm = screen[2] / (size[0] / 10)
    cands = {
        "3D-модель": lambda: Geo(),
        "PCA16 rbf .02 (сейчас)": lambda: PCAModel(k=16, kind="rbf", gamma=0.02, lam=0.05),
        "PCA8 ridge a=3": lambda: PCAModel(k=8, kind="ridge", alpha=3.0),
        "PCA16 ridge a=3": lambda: PCAModel(k=16, kind="ridge", alpha=3.0),
        "PCA16 ridge a=10": lambda: PCAModel(k=16, kind="ridge", alpha=10.0),
        "PCA16 ridge+3D a=3": lambda: PCAModel(k=16, kind="ridge", alpha=3.0, geo=True),
        "PCA16 ridge+3D a=10": lambda: PCAModel(k=16, kind="ridge", alpha=10.0, geo=True),
        "PCA16 lin+rbf .02": lambda: LinRBF(k=16, gamma=0.02),
        "PCA16 lin+rbf .01": lambda: LinRBF(k=16, gamma=0.01),
        "PCA8 lin+rbf .02": lambda: LinRBF(k=8, gamma=0.02),
        "PCA16 lin+rbf+3D .02": lambda: LinRBF(k=16, gamma=0.02, geo=True),
    }
    print("%-24s %8s %8s %8s | %8s %8s %8s" % ("модель", "A углы", "A новые", "A гол±", "B углы", "B пров.", "B гол±"))
    for name, make in cands.items():
        row = []
        for frames, train in sets.values():
            row.append(corner_loo(make, frames, train, screen, size, px_cm))
            m = make().fit([f for f in frames if f["phase"] in train], screen, size)
            r = metrics(m, frames, px_cm)
            test = [p for p in r if p not in train]
            row += [np.mean([r[p][0] for p in test]), np.mean([r[p][1] for p in HEAD if p in r])]
        print("%-24s %6.1f см %6.1f см %6.1f см | %6.1f см %6.1f см %6.1f см" % (name, *row))


if __name__ == "__main__":
    main()
