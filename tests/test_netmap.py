import numpy as np

from gaze_check.eyes import Calibration
from gaze_check.netmap import NetMap

W, H = 2560, 1600
rng = np.random.default_rng(0)
MIX = rng.normal(0, 1, (40, 5))  # признаки сети — смесь точки взгляда и «помех» (голова, расстояние)


def net(p, head=0.0, dist=0.0, noise=0.02):
    z = np.array([p[0] / W, p[1] / H, head, dist, head * dist])
    return MIX @ z + rng.normal(0, noise, 40)


def grid(n=3):
    return [(round(fx * W), round(fy * H)) for fy in np.linspace(0.08, 0.92, n) for fx in np.linspace(0.08, 0.92, n)]


def frames(points, **kw):
    return [(p, {"net": net(p, **kw)}) for p in points for _ in range(12)]


def test_netmap_learns_head_and_distance_invariance():
    points = frames(grid()) + frames(grid(), dist=1.0) + frames(grid(), dist=-1.0)
    center = (W // 2, H // 2)
    moving = [(center, {"net": net(center, head=h)}) for h in np.linspace(-1, 1, 60)]
    m = NetMap().fit(points, moving)
    for p, kw in [((700, 500), {}), ((1900, 1200), {"dist": 0.7}), ((W // 2, H // 2), {"head": 0.8}),
                  ((900, 1000), {"head": -0.5})]:
        pred = np.median(m.predict_many([{"net": net(p, **kw)} for _ in range(10)]), axis=0)
        assert np.hypot(*(pred - p)) < 150, (p, kw, pred)


def test_netmap_needs_several_points():
    assert NetMap().fit(frames([(100, 100), (200, 200)])) is None


def test_calibration_uses_netmap_and_survives_save(tmp_path):
    import synth
    pts = [(p, {**synth.feat(p), "net": net(p)}) for p in grid() for _ in range(9)]
    cal = Calibration.fit(synth.SCREEN, pts, synth.SIZE_MM)
    assert cal.method == "MGazeNet"
    f = {**synth.feat((800, 600)), "net": net((800, 600), noise=0)}
    x, y = cal.predict(f)
    assert abs(x - 800) < 150 and abs(y - 600) < 150
    cal = cal.with_click((800, 600), f) or cal
    cal.save(tmp_path / "c.json")
    again = Calibration.load(tmp_path / "c.json", synth.SCREEN)
    assert again.method == "MGazeNet" and np.allclose(again.predict(f), cal.predict(f), atol=2)
    # кадр без признаков сети (глаза у края) — 3D-модель, а не падение
    assert len(cal.predict(synth.feat((800, 600)))) == 2


def test_calibration_with_some_frames_without_net():
    """Часть кадров без признаков сети (моргнул, один глаз) — калибровка не падает (было: KeyError)."""
    import synth
    pts = []
    for i, p in enumerate(grid()):
        for j in range(9):
            f = synth.feat(p)
            if (i + j) % 4:
                f["net"] = net(p)
            pts.append((p, f))
    cal = Calibration.fit(synth.SCREEN, pts, synth.SIZE_MM)
    assert cal.method == "MGazeNet" and cal.error_px is not None
    mixed = [{**synth.feat((800, 600)), "net": net((800, 600), noise=0)}, synth.feat((800, 600))]
    out = cal.predict_many(mixed)
    assert out.shape == (2, 2) and np.all(np.isfinite(out))
    cal.check([((800, 600), mixed)])
    assert cal.check_px is not None


def test_net_worker_runs_aside_and_gives_fresh_result():
    """Сеть в своём потоке: submit не ждёт расчёта, latest отдаёт свежий вектор, старый — нет."""
    import time
    from gaze_check.gazenet import NetWorker

    def slow_net(rgb, pts, shifts=None):
        time.sleep(0.05)
        return np.array([float(pts)])

    w = NetWorker(slow_net)
    try:
        t0 = time.perf_counter()
        w.submit(None, 1, 10.0)
        assert time.perf_counter() - t0 < 0.01  # не ждём сеть
        assert w.latest(10.0) is None  # ещё считает
        time.sleep(0.12)
        assert w.latest(10.1)[0] == 1.0  # готово, кадр свежий
        assert w.latest(10.5) is None  # кадр старше 0.25 с — не годится
        for i in range(5):  # поток кадров быстрее сети — считается самый свежий
            w.submit(None, i, 11.0 + i * 0.01)
        time.sleep(0.2)
        assert w.latest(11.1)[0] == 4.0
    finally:
        w.stop()
