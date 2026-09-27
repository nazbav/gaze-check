"""Разбор записи tools/gaze_experiment.py: какая модель взгляда точнее в каких положениях.

    .venv\\Scripts\\python tools\\gaze_analyze.py [файл.json]   (по умолчанию — последний)

Модели учатся на этапах из TRAIN и проверяются на остальных. Метрики, см на экране:
«точность» — медиана промаха медианной точки по целям этапа, «дрожь» — разброс кадров
вокруг своей медианы (у этапов «голова» — разброс при покачивании, это и есть «разлетается»).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_check.eyes import Calibration  # noqa: E402
from gaze_check.models import DATA_DIR  # noqa: E402

HEAD = ("голова", "голова-угол")


def load(path=None):
    if path is None:
        path = max((DATA_DIR / "experiments").glob("gaze_*.json"))
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print("запись:", path, "кадров:", len(d["frames"]))
    return d


class Geo:
    """3D-модель глаза (gaze_check.eyes)."""
    name = "3D-модель"

    def fit(self, frames, screen, size):
        pts = [(tuple(f["target"]), f["feat"]) for f in frames if f["phase"] not in HEAD]
        mov = [(tuple(f["target"]), f["feat"]) for f in frames if f["phase"] in HEAD]
        self.cal = Calibration.fit(screen, pts, size, moving=mov, loo=False)
        return self

    def predict(self, frames):
        return self.cal.predict_many([f["feat"] for f in frames])


class NetRidge:
    """MGazeNet: гребневая регрессия по всем признакам (альфа — по «без одной цели»)."""
    name = "MGazeNet+ridge"

    def __init__(self, alphas=(0.1, 1, 10, 100, 1000)):
        self.alphas = alphas

    @staticmethod
    def _solve(x, y, w, alpha):
        z = np.hstack([x, np.ones((len(x), 1))])
        reg = alpha * np.eye(z.shape[1])
        reg[-1, -1] = 0
        zw = z * w[:, None]
        return np.linalg.solve(z.T @ zw + reg, zw.T @ y)

    def fit(self, frames, screen, size):
        x = np.array([f["net"] for f in frames], float)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        x = (x - self.mean) / self.std
        y = np.array([f["target"] for f in frames], float)
        keys = [tuple(f["target"]) + (f["phase"],) for f in frames]
        count = {k: keys.count(k) for k in set(keys)}
        w = np.array([1.0 / count[k] for k in keys])
        best = None
        for a in self.alphas:
            errs = []
            for k in set(keys):
                m = np.array([kk != k for kk in keys])
                wt = self._solve(x[m], y[m], w[m], a)
                p = np.median(np.hstack([x[~m], np.ones((int((~m).sum()), 1))]) @ wt, 0)
                errs.append(np.hypot(*(p - np.array(k[:2]))))
            if best is None or np.median(errs) < best[0]:
                best = (np.median(errs), a)
        self.alpha = best[1]
        self.w = self._solve(x, y, w, self.alpha)
        return self

    def predict(self, frames):
        x = (np.array([f["net"] for f in frames], float) - self.mean) / self.std
        return np.hstack([x, np.ones((len(x), 1))]) @ self.w


class NetKernel:
    """MGazeNet: ядерная гребневая регрессия с RBF (γ=0.005, как у SVR в GazeFollower; в opencv 5
    нет cv2.ml). Кадры цели усредняются — ядро по сотням почти одинаковых кадров не нужно."""
    name = "MGazeNet+RBF"

    def fit(self, frames, screen, size, lam=1e-3, gamma=0.005):
        groups = {}
        for f in frames:
            groups.setdefault((tuple(f["target"]), f["phase"]), []).append(f["net"])
        self.x = np.array([np.median(v, 0) for v in groups.values()], float)
        y = np.array([k[0] for k in groups], float)
        self.gamma = gamma
        k = self._kernel(self.x)
        self.mean = y.mean(0)
        self.a = np.linalg.solve(k + lam * np.eye(len(k)), y - self.mean)
        return self

    def _kernel(self, x):
        d = ((x[:, None, :] - self.x[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d)

    def predict(self, frames):
        return self._kernel(np.array([f["net"] for f in frames], float)) @ self.a + self.mean


class NetAffine:
    """MGazeNet: только «сырая» точка (первые 2 выхода) + линейная поправка."""
    name = "MGazeNet+аффинно"

    def fit(self, frames, screen, size):
        x = np.array([f["net"][:2] for f in frames], float)
        y = np.array([f["target"] for f in frames], float)
        z = np.hstack([x, np.ones((len(x), 1))])
        self.w = np.linalg.lstsq(z, y, rcond=None)[0]
        return self

    def predict(self, frames):
        x = np.array([f["net"][:2] for f in frames], float)
        return np.hstack([x, np.ones((len(x), 1))]) @ self.w


def metrics(model, frames, px_cm):
    """→ {этап: (точность см, дрожь см)}."""
    out = {}
    pred = model.predict(frames)
    phases = sorted({f["phase"] for f in frames}, key=[f["phase"] for f in frames].index)
    for ph in phases:
        errs, jit = [], []
        for t in {tuple(f["target"]) for f in frames if f["phase"] == ph}:
            idx = [i for i, f in enumerate(frames) if f["phase"] == ph and tuple(f["target"]) == t]
            p = pred[idx]
            m = np.median(p, 0)
            errs.append(np.hypot(*(m - t)) / px_cm)
            jit.append(np.median(np.hypot(*(p - m).T)) / px_cm)
        out[ph] = (float(np.median(errs)), float(np.median(jit)))
    return out


def main():
    d = load(sys.argv[1] if len(sys.argv) > 1 else None)
    frames, screen, size = d["frames"], tuple(d["screen"]), tuple(d["size_mm"])
    px_cm = screen[2] / (size[0] / 10)
    dist = {}
    for f in frames:
        if f.get("dist"):
            dist.setdefault(f["phase"], []).append(f["dist"])
    print("расстояние до лица по этапам, см (грубо):",
          {k: round(float(np.median(v))) for k, v in dist.items()})
    trains = {"калибровка: обычно + голова": ("обычно",) + HEAD,
              "калибровка: + ближе и дальше": ("обычно", "ближе", "дальше") + HEAD}
    for tname, train in trains.items():
        print("\n=== %s ===" % tname)
        tr = [f for f in frames if f["phase"] in train]
        rows = []
        for model in (Geo(), NetRidge(), NetKernel(), NetAffine()):
            try:
                model.fit(tr, screen, size)
            except Exception as e:  # noqa: BLE001 — модель не смогла, так и пишем
                print(model.name, "не обучилась:", e)
                continue
            rows.append((model.name, metrics(model, frames, px_cm)))
        phases = list(rows[0][1])
        print("%-18s" % "этап (не в калибровке — *)" + "".join("%20s" % n for n, _ in rows))
        for ph in phases:
            mark = "" if ph in train else " *"
            print("%-18s" % (ph + mark) + "".join("%11.1f ±%4.1f см" % r[ph] for _, r in rows))
        test = [ph for ph in phases if ph not in train]
        print("%-18s" % "среднее по *" + "".join("%11.1f        " % np.mean([r[ph][0] for ph in test]) for _, r in rows))


if __name__ == "__main__":
    main()
