"""Калибровка MGazeNet: признаки сети → точка на экране.

Главные компоненты (16) признаков сети → ядерная гребневая регрессия по отдельным кадрам,
включая кадры «смотрю в точку и качаю головой» (с весом 3): так отображение учится не сдвигать
точку, когда двигается голова. Точки калибровки на разных расстояниях учат тому же для расстояния.

Ядро — «линейное + RBF». Чистый RBF хорошо ведёт себя рядом с точками калибровки, но в другой
позе (признаки отошли от калибровочных) сползает к центру экрана — углы экрана не
доставались. Линейная часть продолжает зависимость по прямой.

На записях (tools/gaze_corners.py): новые позы 2,0 см, качание головой ±1,7 см, угол, которого
не было в калибровке, 5,3–6,7 см (чистый RBF 6,0–7,1; 3D-модель 4,8 / ±5,4 / 5,6–7,3).
"""
import numpy as np

K, GAMMA, LAM = 16, 0.02, 0.05
LINEAR = 1.0  # вес линейной части ядра
HEAD_W = 3.0  # кадры «качаю головой» вместе весят как 3 точки
CLICK_W = 0.5  # клик — как полточки
EVERY = 3  # из соседних кадров точки берём каждый третий: они почти одинаковые
# углы ведущего глаза и поворот головы — ещё признаки рядом с признаками сети: сеть учили на людях
# с параллельными глазами, а при косоглазии глаза смотрят в разные стороны; с ними угол вне
# калибровки 3,6–4,3 см вместо 5,3–6,2 (tools/gaze_eyes.py)
EXTRA = ("gy", "gp", "R2", "R5")


class NetMap:
    def fit(self, points, moving=(), clicks=()):
        """points/moving/clicks — [(точка (x, y), признаки с "net")]. None — мало данных."""
        rows, weights = [], []
        for group, total in ((points, 1.0), (moving, HEAD_W)):
            by = {}
            for p, f in group:
                if f.get("net") is not None:
                    by.setdefault(tuple(p), []).append(f)
            for p, feats in by.items():
                feats = feats[::EVERY] or feats
                rows += [(p, f) for f in feats]
                weights += [total / len(feats)] * len(feats)
        for p, f in clicks:
            if f.get("net") is not None:
                rows.append((tuple(p), f))
                weights.append(CLICK_W)
        if len({p for p, _ in rows}) < 5:
            return None
        feats = [f for _, f in rows]
        self.extra = all(k in f for f in feats for k in EXTRA)
        net = np.array([f["net"] for f in feats], float)
        y = np.array([p for p, _ in rows], float)
        self.mu = net.mean(0)
        _, _, vt = np.linalg.svd(net - self.mu, full_matrices=False)
        self.pc = vt[:K].T
        x = self._raw_x(feats)
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-9
        self.x = (x - self.xm) / self.xs
        w = np.array(weights)
        w = w / w.mean()
        self.y_mean = y.mean(0)
        self.a = np.linalg.solve(w[:, None] * self._kernel(self.x) + LAM * np.eye(len(y)),
                                 w[:, None] * (y - self.y_mean))
        return self

    def _kernel(self, x):
        d = ((x[:, None, :] - self.x[None, :, :]) ** 2).sum(-1)
        return np.exp(-GAMMA * d) + LINEAR * (x @ self.x.T) / x.shape[1]

    def _raw_x(self, feats):
        net = np.array([f["net"] for f in feats], float).reshape(len(feats), -1)
        x = (net - self.mu) @ self.pc
        if self.extra:
            x = np.hstack([x, np.array([[f[k] for k in EXTRA] for f in feats], float)])
        return x

    def predict_many(self, feats):
        x = (self._raw_x(feats) - self.xm) / self.xs
        return self._kernel(x) @ self.a + self.y_mean
