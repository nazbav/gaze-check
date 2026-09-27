"""Варианты калибровки MGazeNet (и гибрид с 3D-моделью) на записи эксперимента.

    .venv\\Scripts\\python tools\\gaze_variants.py [файл.json]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaze_analyze import HEAD, Geo, load, metrics  # noqa: E402


class PCAModel:
    """Главные компоненты признаков сети (+ по желанию точка 3D-модели) → ridge или RBF по кадрам.
    Кадры «качаю головой» идут в обучение как есть: так модель учится, что голова двигается,
    а точка та же."""

    def __init__(self, k=16, kind="ridge", geo=False, alpha=None, gamma=None, lam=None, every=3, head_w=3.0):
        self.k, self.kind, self.geo, self.every, self.head_w = k, kind, geo, every, head_w
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.name = "PCA%d+%s%s" % (k, kind, "+3D" if geo else "")

    def _x(self, frames):
        x = (np.array([f["net"] for f in frames], float) - self.mu) @ self.pc
        if self.geo:
            g = self.geo_model.predict(frames) / 1000.0
            x = np.hstack([x, g])
        return x

    def fit(self, frames, screen, size):
        frames = frames[::self.every]
        net = np.array([f["net"] for f in frames], float)
        self.mu = net.mean(0)
        _, _, vt = np.linalg.svd(net - self.mu, full_matrices=False)
        self.pc = vt[:self.k].T
        if self.geo:
            self.geo_model = Geo().fit(frames, screen, size)
        x = self._x(frames)
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-9
        x = (x - self.xm) / self.xs
        y = np.array([f["target"] for f in frames], float)
        keys = [(tuple(f["target"]), f["phase"]) for f in frames]
        cnt = {k: keys.count(k) for k in set(keys)}
        w = np.array([(self.head_w if k[1] in HEAD else 1.0) / cnt[k] for k in keys])
        self.x_train, self.y_mean = x, y.mean(0)
        if self.kind == "ridge":
            z = np.hstack([x, np.ones((len(x), 1))])
            reg = (self.alpha or 1.0) * np.eye(z.shape[1])
            reg[-1, -1] = 0
            zw = z * w[:, None]
            self.w = np.linalg.solve(z.T @ zw + reg, zw.T @ y)
        else:
            kmat = self._kernel(x)
            wd = np.diag(w / w.mean())
            self.a = np.linalg.solve(wd @ kmat + (self.lam or 1e-2) * np.eye(len(x)), wd @ (y - self.y_mean))
        return self

    def _kernel(self, x):
        d = ((x[:, None, :] - self.x_train[None, :, :]) ** 2).sum(-1)
        return np.exp(-(self.gamma or 0.05) * d)

    def predict(self, frames):
        x = (self._x(frames) - self.xm) / self.xs
        if self.kind == "ridge":
            return np.hstack([x, np.ones((len(x), 1))]) @ self.w
        return self._kernel(x) @ self.a + self.y_mean


def main():
    d = load(sys.argv[1] if len(sys.argv) > 1 else None)
    frames, screen, size = d["frames"], tuple(d["screen"]), tuple(d["size_mm"])
    px_cm = screen[2] / (size[0] / 10)
    train_sets = {"обычно+голова": ("обычно",) + HEAD, "+ближе/дальше": ("обычно", "ближе", "дальше") + HEAD}
    models = [lambda: Geo()]
    for k in (8, 16, 32):
        for a in (1.0, 10.0):
            models.append(lambda k=k, a=a: PCAModel(k=k, kind="ridge", alpha=a))
    for k in (8, 16):
        for g in (0.02, 0.1):
            models.append(lambda k=k, g=g: PCAModel(k=k, kind="rbf", gamma=g, lam=0.05))
    models.append(lambda: PCAModel(k=16, kind="ridge", alpha=10.0, geo=True))
    models.append(lambda: PCAModel(k=8, kind="rbf", gamma=0.05, lam=0.05, geo=True))
    for tname, train in train_sets.items():
        tr = [f for f in frames if f["phase"] in train]
        print("\n=== калибровка: %s ===" % tname)
        print("%-26s %8s %8s %9s %9s" % ("модель", "новые", "худшее", "голова±", "обычно-2"))
        for make in models:
            m = make()
            m.fit(tr, screen, size)
            r = metrics(m, frames, px_cm)
            test = [ph for ph in r if ph not in train]
            label = m.name + ("" if not hasattr(m, "alpha") or m.kind != "ridge" else " a=%g" % m.alpha)
            if getattr(m, "kind", "") == "rbf":
                label += " g=%g" % m.gamma
            print("%-26s %6.1f см %6.1f см %7.1f см %7.1f см" % (
                label, np.mean([r[p][0] for p in test]), max(r[p][0] for p in test),
                np.mean([r[p][1] for p in HEAD]), r["обычно-2"][0]))


if __name__ == "__main__":
    main()
