"""Какой глаз точнее: 3D-модель по левому, по правому, по обоим; MGazeNet и гибрид с левым глазом.

    .venv\\Scripts\\python tools\\gaze_eyes.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaze_analyze import HEAD, Geo, metrics  # noqa: E402
from gaze_corners import LinRBF, corner_loo, datasets  # noqa: E402

from gaze_check.eyes import EyeModel, canonical_mesh  # noqa: E402
from gaze_check.models import EYE_MODEL  # noqa: E402

MODEL = EyeModel(canonical_mesh(EYE_MODEL))


def per_eye(frames):
    """Дописывает в признаки gy_l/gp_l/gy_r/gp_r (из сырых точек, если их нет)."""
    for f in frames:
        feat = f["feat"]
        if "gy_l" not in feat and feat.get("_raw"):
            g = MODEL.from_raw(feat["_raw"])
            feat.update({k: g[k] for k in ("gy_l", "gp_l", "gy_r", "gp_r")})
    return [f for f in frames if "gy_l" in f["feat"]]


def one_eye(frames, eye):
    out = []
    for f in frames:
        feat = dict(f["feat"])
        feat["gy"], feat["gp"] = feat["gy_" + eye], feat["gp_" + eye]
        out.append(dict(f, feat=feat))
    return out


class Hybrid(LinRBF):
    """MGazeNet (PCA) + углы ведущего глаза и поворот головы как ещё признаки."""

    def __init__(self, eye="l", **kw):
        super().__init__(**kw)
        self.eye = eye
        self.name = "MGazeNet + %s глаз" % eye

    def _x(self, frames):
        x = super()._x(frames)
        extra = np.array([[f["feat"]["gy_" + self.eye], f["feat"]["gp_" + self.eye],
                           f["feat"]["R2"], f["feat"]["R5"]] for f in frames]) * 10
        return np.hstack([x, extra])


def main():
    screen, size, sets = datasets()
    px_cm = screen[2] / (size[0] / 10)
    sets = {k: (per_eye(fr), tr) for k, (fr, tr) in sets.items()}
    for k, (fr, _) in sets.items():
        d = np.degrees([f["feat"]["gy_l"] - f["feat"]["gy_r"] for f in fr])
        dv = np.degrees([f["feat"]["gp_l"] - f["feat"]["gp_r"] for f in fr])
        print("%s: расхождение глаз по горизонтали %.1f° ±%.1f°, по вертикали %.1f° ±%.1f°" % (
            k, np.median(d), np.std(d), np.median(dv), np.std(dv)))
    cands = [("3D: оба глаза", lambda: Geo(), None), ("3D: левый глаз", lambda: Geo(), "l"),
             ("3D: правый глаз", lambda: Geo(), "r"), ("MGazeNet (сейчас)", lambda: LinRBF(k=16, gamma=0.02), None),
             ("MGazeNet + левый", lambda: Hybrid("l", k=16, gamma=0.02), None),
             ("MGazeNet + правый", lambda: Hybrid("r", k=16, gamma=0.02), None)]
    print("\n%-20s %8s %8s %8s | %8s %8s %8s" % ("модель", "A углы", "A новые", "A гол±", "B углы", "B пров.", "B гол±"))
    for name, make, eye in cands:
        row = []
        for frames, train in sets.values():
            fr = one_eye(frames, eye) if eye else frames
            row.append(corner_loo(make, fr, train, screen, size, px_cm))
            m = make().fit([f for f in fr if f["phase"] in train], screen, size)
            r = metrics(m, fr, px_cm)
            test = [p for p in r if p not in train]
            row += [np.mean([r[p][0] for p in test]), np.mean([r[p][1] for p in HEAD if p in r])]
        print("%-20s %6.1f см %6.1f см %6.1f см | %6.1f см %6.1f см %6.1f см" % (name, *row))


if __name__ == "__main__":
    main()
