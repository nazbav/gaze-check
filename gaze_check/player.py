"""Проигрыватель лёгкого журнала взгляда (.gazelog): лицо-схема с морганиями, экран с точкой и следом,
шкала времени, частота морганий, итоги по зонам и программам, экспорт в CSV.

    GazeCheck.exe --play [файл.gazelog]      (без файла — последний журнал)
"""
import bisect
import sys
from pathlib import Path

from .gazelog import BLINK, CLOSED, FACE, LOOKING, blink_rate, logs_dir, read_log, summary, to_csv
from .zones import ALL_ZONES

TRAIL_S = 3.0


class PlayerModel:
    """Что показать в момент t — без окон, чтобы проверять тестами."""

    def __init__(self, log):
        self.log = log
        self.samples = log["samples"]
        self.times = [s[0] for s in self.samples]
        self.blinks = [(t - v / 1000.0, t) for t, kind, v in log["events"] if kind == BLINK]
        self.duration = self.times[-1] if self.times else 0.0

    def state_at(self, t):
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return None
        _, x, y, cm, flags, zone, app = self.samples[i]
        lo = bisect.bisect_left(self.times, t - TRAIL_S)
        trail = [(s[1], s[2]) for s in self.samples[lo:i + 1] if s[1] is not None]
        closed = bool(flags & CLOSED) or any(a <= t <= b for a, b in self.blinks)
        return {"t": t, "point": None if x is None else (x, y), "trail": trail, "dist": cm, "closed": closed,
                "looking": bool(flags & LOOKING), "zone": ALL_ZONES[zone] if zone < len(ALL_ZONES) else "",
                "app": app, "blinks_per_min": blink_rate(self.log["events"], t, window=max(10.0, min(60.0, t))),
                "blinks": sum(1 for _, b in self.blinks if b <= t)}

    def heat(self, cols, rows):
        """Сколько времени взгляд провёл в каждой клетке экрана → [[0..1]] (1 — самая «горячая»)."""
        sw, sh = self.log["screen"]
        grid = [[0] * cols for _ in range(rows)]
        for _, x, y, *_ in self.samples:
            if x is not None and 0 <= x < sw and 0 <= y < sh:
                grid[int(y * rows / sh)][int(x * cols / sw)] += 1
        top = max(max(r) for r in grid) or 1
        return [[v / top for v in r] for r in grid]

    def timeline(self, n):
        """Шкала времени по n столбцам: (доля «смотрит в экран», доля «лицо видно», морганий) в каждом."""
        out = [[0, 0, 0, 0] for _ in range(n)]
        span = self.duration or 1.0
        for t, _, _, _, flags, _, _ in self.samples:
            c = out[min(n - 1, int(t / span * n))]
            c[0] += bool(flags & LOOKING)
            c[1] += bool(flags & FACE)
            c[3] += 1
        for _, b in self.blinks:
            out[min(n - 1, int(b / span * n))][2] += 1
        res, last = [], (0.0, 0.0)
        for lo, face, bl, k in out:
            if k:  # столбцов больше, чем отсчётов, — пустой берёт соседа слева
                last = (lo / k, face / k)
            res.append(last + (bl,))
        return res


def _blend(a, b, share):
    """Цвет между a и b ("#rrggbb"), share 0..1."""
    ca, cb = (int(a[i:i + 2], 16) for i in (1, 3, 5)), (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * share) for x, y in zip(ca, cb))


def _clock(s):
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60) if s >= 3600 else "%d:%02d" % (s // 60, s % 60)


class Player:

    def __init__(self, path):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.path = Path(path)
        self.model = PlayerModel(read_log(self.path))
        self.sw, self.sh = self.model.log["screen"]
        self.t, self.playing, self.speed = 0.0, False, 1.0
        self.root = tk.Tk()
        k = self.k = max(1.0, self.root.winfo_fpixels("1i") / 96)  # экран с масштабом 150% — всё крупнее
        self.FACE = int(220 * k)
        self.root.title("Журнал взгляда — %s" % self.path.name)
        self.root.configure(bg="#15191e")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        top = tk.Frame(self.root, bg="#15191e")
        top.pack(fill="both", expand=True, padx=10, pady=10)
        left = tk.Frame(top, bg="#15191e")
        left.pack(side="left", fill="y")
        self.face = tk.Canvas(left, width=self.FACE, height=self.FACE, bg="#15191e", highlightthickness=0)
        self.face.pack()
        self.info = tk.Label(left, justify="left", anchor="w", bg="#15191e", fg="#dfe5ec", font=("Segoe UI", 10))
        self.info.pack(fill="x", pady=8)
        self.W = int(620 * k)
        self.H = int(self.W * self.sh / max(1, self.sw))
        self.scr = tk.Canvas(top, width=self.W, height=self.H, bg="#0c0f12", highlightthickness=1,
                             highlightbackground="#3a4652")
        self.scr.pack(side="left", padx=(12, 0))
        self._heat()
        self.strip = tk.Canvas(self.root, height=int(26 * k), bg="#0c0f12", highlightthickness=0)
        self.strip.pack(fill="x", padx=10, pady=(0, 4))
        self.strip.bind("<Configure>", lambda e: self._strip())
        self.strip.bind("<Button-1>", self._strip_seek)
        self.strip.bind("<B1-Motion>", self._strip_seek)
        bar = tk.Frame(self.root, bg="#15191e")
        bar.pack(fill="x", padx=10, pady=(0, 6))
        self.play_btn = ttk.Button(bar, text="▶ Пуск", width=9, command=self.toggle)
        self.play_btn.pack(side="left")
        for sp in (1, 4, 16):
            ttk.Button(bar, text="×%d" % sp, width=4, command=lambda sp=sp: self.set_speed(sp)).pack(side="left")
        self.scale = ttk.Scale(bar, from_=0, to=max(0.1, self.model.duration), orient="horizontal",
                               command=self.seek)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_lbl = tk.Label(bar, bg="#15191e", fg="#dfe5ec", font=("Consolas", 10))
        self.time_lbl.pack(side="left")
        ttk.Button(bar, text="CSV", command=self.export).pack(side="left", padx=(8, 0))
        s = summary(self.model.log)
        zones = ", ".join("%s %d%%" % (k, v * 100) for k, v in list(s["zones"].items())[:5])
        apps = ", ".join("%s %d%%" % (k.split(" — ")[0], v * 100) for k, v in list(s["apps"].items())[:4])
        tk.Label(self.root, justify="left", anchor="w", bg="#15191e", fg="#9aa4ae", font=("Segoe UI", 9),
                 text="Всего %s · морганий %d (%.1f в минуту)\nЗоны: %s\nПрограммы: %s" % (
                     _clock(s["duration_s"]), s["blinks"], s["blinks_per_min"], zones or "—", apps or "—")
                 ).pack(fill="x", padx=10, pady=(0, 10))
        self.root.bind("<space>", lambda e: self.toggle())
        self.draw()
        self.root.after(33, self.tick)

    def toggle(self):
        self.playing = not self.playing
        if self.playing and self.t >= self.model.duration:
            self.t = 0.0
        self.play_btn.configure(text="⏸ Пауза" if self.playing else "▶ Пуск")

    def set_speed(self, sp):
        self.speed = sp

    def seek(self, value):
        self.t = float(value)
        if not self.playing:
            self.draw()

    def _strip_seek(self, e):
        w = max(1, self.strip.winfo_width())
        self.scale.set(max(0.0, min(1.0, e.x / w)) * self.model.duration)

    def _heat(self):
        """Тепловая карта всего журнала — подложка экрана (не перерисовывается)."""
        cols, rows = 48, max(1, round(48 * self.sh / max(1, self.sw)))
        cw, ch = self.W / cols, self.H / rows
        for r, row in enumerate(self.model.heat(cols, rows)):
            for c, v in enumerate(row):
                if v > 0.02:
                    self.scr.create_rectangle(c * cw, r * ch, (c + 1) * cw, (r + 1) * ch, outline="",
                                              fill=_blend("#0c0f12", "#2f7fd0", min(1.0, v ** 0.5)), tags="heat")

    def _strip(self):
        """Полоса под экраном: зелёное — смотрел в экран, серое — лицо есть, но не в экран, тёмное — нет лица;
        белые штрихи — моргания; жёлтая черта — сейчас."""
        c = self.strip
        c.delete("all")
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        for i, (look, face, bl) in enumerate(self.model.timeline(w)):
            color = _blend(_blend("#0c0f12", "#4a5563", face), "#2e9e5b", look)
            c.create_line(i, 0, i, h, fill=color)
            if bl:
                c.create_line(i, 0, i, h * 0.45, fill="#e8eef4")
        self._strip_cursor()

    def _strip_cursor(self):
        c = self.strip
        c.delete("cursor")
        x = self.t / (self.model.duration or 1.0) * max(1, c.winfo_width())
        c.create_line(x, 0, x, c.winfo_height(), fill="#ffd60a", width=2 * self.k, tags="cursor")

    def export(self):
        path = to_csv(self.model.log, self.path.with_suffix(".csv"))
        self.info.configure(text="CSV: %s" % path)

    def tick(self):
        if self.playing:
            self.t = min(self.model.duration, self.t + 0.033 * self.speed)
            self.scale.set(self.t)
            if self.t >= self.model.duration:
                self.toggle()
            self.draw()
        self.root.after(33, self.tick)

    def draw(self):
        st = self.model.state_at(self.t) or {"point": None, "trail": [], "closed": False, "dist": None,
                                             "looking": False, "zone": "", "app": "", "blinks_per_min": 0,
                                             "blinks": 0}
        self._face(st)
        self._screen(st)
        self._strip_cursor()
        self.time_lbl.configure(text="%s / %s" % (_clock(self.t), _clock(self.model.duration)))
        self.info.configure(text="Морганий: %d (%.0f в минуту)\nДо экрана: %s\n%s\nЗона: %s\n%s" % (
            st["blinks"], st["blinks_per_min"], ("%d см" % st["dist"]) if st["dist"] else "—",
            "смотрит в экран" if st["looking"] else "не смотрит в экран", st["zone"] or "—",
            (st["app"] or "")[:40]))

    def _face(self, st):
        """Лицо как в зеркале: зрачки сдвинуты в ту сторону экрана, куда смотрел; моргнул — глаза чертой."""
        c, s, u = self.face, self.FACE, self.k
        c.delete("all")
        c.create_oval(20 * u, 15 * u, s - 20 * u, s - 5 * u, fill="#e9c8a8", outline="#b8977a", width=2 * u)
        px, py = 0.0, 0.0
        if st["point"]:
            px = max(-1, min(1, (st["point"][0] / max(1, self.sw) - 0.5) * 2))
            py = max(-1, min(1, (st["point"][1] / max(1, self.sh) - 0.5) * 2))
        for ex in (s * 0.34, s * 0.66):
            ey = s * 0.42
            if st["closed"]:
                c.create_line(ex - 22 * u, ey, ex + 22 * u, ey, fill="#4a3526", width=4 * u, capstyle="round")
            else:
                c.create_oval(ex - 24 * u, ey - 14 * u, ex + 24 * u, ey + 14 * u, fill="white", outline="#4a3526",
                              width=2 * u)
                cx, cy = ex + px * 12 * u, ey + py * 6 * u
                c.create_oval(cx - 8 * u, cy - 8 * u, cx + 8 * u, cy + 8 * u, fill="#3b6ea5", outline="")
                c.create_oval(cx - 4 * u, cy - 4 * u, cx + 4 * u, cy + 4 * u, fill="#111", outline="")
        c.create_arc(s * 0.36, s * 0.58, s * 0.64, s * 0.78, start=200, extent=140, style="arc",
                     outline="#8a5a44", width=3 * u)

    def _screen(self, st):
        c, W, H, u = self.scr, self.W, self.H, self.k
        c.delete("live")
        for i in (1, 2):
            c.create_line(W * i / 3, 0, W * i / 3, H, fill="#26303a", tags="live")
            c.create_line(0, H * i / 3, W, H * i / 3, fill="#26303a", tags="live")
        kx, ky = W / max(1, self.sw), H / max(1, self.sh)
        pts = [(x * kx, y * ky) for x, y in st["trail"]]
        for a, b in zip(pts, pts[1:]):
            c.create_line(*a, *b, fill="#f0a030", width=2 * u, tags="live")
        if st["point"]:
            x, y = st["point"][0] * kx, st["point"][1] * ky
            r = (8 if not st["closed"] else 4) * u
            c.create_oval(x - r, y - r, x + r, y + r, fill="#ff453a", outline="white", width=2 * u, tags="live")

    def run(self):
        self.root.mainloop()


def latest_log():
    logs = sorted(logs_dir().glob("*.gazelog"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main(path=None):
    path = path or latest_log()
    if path is None:
        import tkinter as tk
        from tkinter import messagebox
        tk.Tk().withdraw()
        messagebox.showinfo("Журнал взгляда", "Журналов пока нет — включите «Лёгкий журнал взгляда» в меню.")
        return
    Player(path).run()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
