import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import time
import csv
import os
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[AVERTISMENT] pip install matplotlib")

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    print("[AVERTISMENT] pip install mss")

mp_face_mesh = mp.solutions.face_mesh
facemesh_model = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTH_FACTOR       = 8
EAR_THRESHOLD       = 0.20
EXIT_THRESHOLD      = 3.0
HEATMAP_BLUR        = 51
HEATMAP_ALPHA       = 0.45
GRID_N              = 3
HEATMAP_W           = 320
HEATMAP_H           = 180
FIXATION_RADIUS_PX  = 60
FIXATION_MIN_MS     = 100

ZONE_NAMES = [
    "Stanga-Sus",    "Centru-Sus",    "Dreapta-Sus",
    "Stanga-Mijloc", "Centru",        "Dreapta-Mijloc",
    "Stanga-Jos",    "Centru-Jos",    "Dreapta-Jos",
]

LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_BOX      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_BOX     = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
NOSE_TIP          = 1
LEFT_EYE_EAR_PTS  = [362, 385, 386, 263, 374, 380]
RIGHT_EYE_EAR_PTS = [33,  159, 158, 133, 153, 145]

CALIB_POINTS_NORM = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def run_calibration(cap) -> dict:
    print("\n[CALIBRARE] Priveste fiecare punct galben timp de 2 secunde.")
    print("            Apasa SPACE pentru a incepe calibrarea.\n")

    calib_win = "CALIBRARE - Apasa SPACE"
    blank = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv2.putText(blank, "Apasa SPACE pentru calibrare",
                (SCREEN_W//2 - 300, SCREEN_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    cv2.namedWindow(calib_win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(calib_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(calib_win, blank)
    while True:
        if cv2.waitKey(30) & 0xFF == ord(' '):
            break

    collected_x = []
    collected_y = []

    for idx, (tx_norm, ty_norm) in enumerate(CALIB_POINTS_NORM):
        px = int(tx_norm * SCREEN_W)
        py = int(ty_norm * SCREEN_H)
        iris_xs, iris_ys = [], []
        deadline = time.time() + 2.5

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = facemesh_model.process(rgb)

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            for i, (fx, fy) in enumerate(CALIB_POINTS_NORM):
                col = (80, 80, 80) if i != idx else (0, 255, 255)
                cv2.circle(canvas, (int(fx*SCREEN_W), int(fy*SCREEN_H)), 18, col, -1)
                cv2.circle(canvas, (int(fx*SCREEN_W), int(fy*SCREEN_H)), 22, col, 2)

            elapsed = 2.5 - (deadline - time.time())
            pct = int(elapsed / 2.5 * 44)
            cv2.rectangle(canvas, (px - 22, py + 30), (px - 22 + pct, py + 36), (0, 255, 255), -1)
            cv2.putText(canvas, f"Punct {idx+1}/9 - priveste punctul galben",
                        (SCREEN_W//2 - 300, SCREEN_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 1)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                mesh = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in lm])
                (l_cx, _), _ = cv2.minEnclosingCircle(mesh[LEFT_IRIS])
                (r_cx, _), _ = cv2.minEnclosingCircle(mesh[RIGHT_IRIS])
                (_, l_cy), _ = cv2.minEnclosingCircle(mesh[LEFT_IRIS])
                (_, r_cy), _ = cv2.minEnclosingCircle(mesh[RIGHT_IRIS])
                nx = (l_cx/w + r_cx/w) / 2
                ny = (l_cy/h + r_cy/h) / 2
                if elapsed > 0.5:
                    iris_xs.append(nx)
                    iris_ys.append(ny)

            cv2.imshow(calib_win, canvas)
            cv2.waitKey(1)

        if iris_xs:
            collected_x.append(np.median(iris_xs))
            collected_y.append(np.median(iris_ys))

    cv2.destroyWindow(calib_win)

    if len(collected_x) < 4:
        print("[CALIBRARE] Prea putine date - folosesc valori implicite.")
        return {"x_range": [0.41, 0.59], "y_range": [0.41, 0.59]}

    x_min = float(np.percentile(collected_x, 10))
    x_max = float(np.percentile(collected_x, 90))
    y_min = float(np.percentile(collected_y, 10))
    y_max = float(np.percentile(collected_y, 90))

    if x_max - x_min < 0.05:
        x_min, x_max = 0.41, 0.59
    if y_max - y_min < 0.05:
        y_min, y_max = 0.41, 0.59

    print(f"[CALIBRARE] X interval: [{x_min:.3f}, {x_max:.3f}]")
    print(f"[CALIBRARE] Y interval: [{y_min:.3f}, {y_max:.3f}]")
    return {"x_range": [x_min, x_max], "y_range": [y_min, y_max]}


class Fixation:
    def __init__(self, x, y, ts_start):
        self.x        = x
        self.y        = y
        self.ts_start = ts_start
        self.ts_end   = ts_start
        self.points   = [(x, y)]

    def update(self, x, y, ts):
        self.points.append((x, y))
        self.ts_end = ts
        self.x = float(np.mean([p[0] for p in self.points]))
        self.y = float(np.mean([p[1] for p in self.points]))

    @property
    def duration_ms(self):
        return (self.ts_end - self.ts_start) * 1000

    @property
    def zone_idx(self):
        col = min(int(self.x * GRID_N), GRID_N - 1)
        row = min(int(self.y * GRID_N), GRID_N - 1)
        return row * GRID_N + col


class SessionData:
    def __init__(self):
        self.gaze_points        = []
        self.zone_dwell         = defaultdict(float)
        self.heatmap_acc        = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
        self.start_ts           = time.time()
        self.last_ts            = time.time()
        self.fixations          = []
        self._current_fixation  = None
        self.saccade_lengths    = []
        self.ttff_per_zone      = {}
        self._prev_gaze_px      = None

    def record(self, norm_x: float, norm_y: float):
        now = time.time()
        dt  = now - self.last_ts
        self.last_ts = now

        if len(self.gaze_points) < 100_000:
            self.gaze_points.append((norm_x, norm_y, now))

        px = int(np.clip(norm_x, 0, 1) * (HEATMAP_W - 1))
        py = int(np.clip(norm_y, 0, 1) * (HEATMAP_H - 1))
        self.heatmap_acc[py, px] += 1.0

        col = min(int(norm_x * GRID_N), GRID_N - 1)
        row = min(int(norm_y * GRID_N), GRID_N - 1)
        zone_idx = row * GRID_N + col
        self.zone_dwell[zone_idx] += dt

        if zone_idx not in self.ttff_per_zone:
            self.ttff_per_zone[zone_idx] = now - self.start_ts

        gaze_px = (norm_x * SCREEN_W, norm_y * SCREEN_H)
        self._update_fixation(norm_x, norm_y, gaze_px, now)

        if self._prev_gaze_px is not None:
            dist = np.hypot(gaze_px[0] - self._prev_gaze_px[0],
                            gaze_px[1] - self._prev_gaze_px[1])
            if dist > FIXATION_RADIUS_PX * 2:
                self.saccade_lengths.append(dist)
        self._prev_gaze_px = gaze_px

    def _update_fixation(self, nx, ny, px, ts):
        if self._current_fixation is None:
            self._current_fixation = Fixation(nx, ny, ts)
        else:
            dist = np.hypot(
                px[0] - self._current_fixation.x * SCREEN_W,
                px[1] - self._current_fixation.y * SCREEN_H
            )
            if dist <= FIXATION_RADIUS_PX:
                self._current_fixation.update(nx, ny, ts)
            else:
                if self._current_fixation.duration_ms >= FIXATION_MIN_MS:
                    self.fixations.append(self._current_fixation)
                self._current_fixation = Fixation(nx, ny, ts)

    def finalize(self):
        if self._current_fixation and self._current_fixation.duration_ms >= FIXATION_MIN_MS:
            self.fixations.append(self._current_fixation)

    def get_heatmap_overlay(self, frame_w, frame_h):
        blurred = cv2.GaussianBlur(self.heatmap_acc, (HEATMAP_BLUR, HEATMAP_BLUR), 0)
        norm = blurred / blurred.max() if blurred.max() > 0 else blurred
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.resize(colored, (frame_w, frame_h)), norm

    def top_zones(self, n=3):
        sorted_z = sorted(self.zone_dwell.items(), key=lambda x: x[1], reverse=True)
        return [(ZONE_NAMES[i], s) for i, s in sorted_z[:n] if s > 0]

    def total_duration(self):
        return time.time() - self.start_ts

    def academic_metrics(self) -> dict:
        dur = max(self.total_duration(), 0.001)
        fix = self.fixations
        fix_durations = [f.duration_ms for f in fix]
        fix_per_zone  = defaultdict(list)
        for f in fix:
            fix_per_zone[f.zone_idx].append(f)
        ttff = {ZONE_NAMES[k]: round(v, 2) for k, v in self.ttff_per_zone.items()}
        return {
            "durata_sesiune_s":         round(dur, 1),
            "nr_fixatii":               len(fix),
            "fixatii_pe_minut":         round(len(fix) / (dur / 60), 1) if dur > 5 else 0,
            "durata_medie_fixatie_ms":  round(np.mean(fix_durations), 1) if fix_durations else 0,
            "durata_max_fixatie_ms":    round(max(fix_durations), 1) if fix_durations else 0,
            "nr_saccade":               len(self.saccade_lengths),
            "lungime_medie_saccada_px": round(np.mean(self.saccade_lengths), 1) if self.saccade_lengths else 0,
            "ttff_per_zona_s":          ttff,
            "fixatii_per_zona":         {ZONE_NAMES[k]: len(v) for k, v in fix_per_zone.items()},
            "zona_dominanta":           ZONE_NAMES[max(self.zone_dwell, key=self.zone_dwell.get)] if self.zone_dwell else "N/A",
        }


def save_screen_heatmap(session: SessionData, out_dir: str, ts_str: str):
    if not HAS_MSS:
        return None
    try:
        with mss.mss() as sct:
            monitor    = sct.monitors[1]
            shot       = sct.grab(monitor)
            screen_img = np.array(shot)[:, :, :3]
            screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)

        sw, sh    = screen_img.shape[1], screen_img.shape[0]
        blurred   = cv2.GaussianBlur(session.heatmap_acc, (HEATMAP_BLUR, HEATMAP_BLUR), 0)
        norm      = blurred / blurred.max() if blurred.max() > 0 else blurred
        heat_full = cv2.resize((norm * 255).astype(np.uint8), (sw, sh))
        colored   = cv2.applyColorMap(heat_full, cv2.COLORMAP_JET)
        mask      = (heat_full > 15).astype(np.float32)
        mask_3ch  = np.stack([mask, mask, mask], axis=-1)
        result    = screen_img.copy().astype(np.float32)
        result    = result * (1 - mask_3ch * 0.55) + colored.astype(np.float32) * mask_3ch * 0.55
        result    = np.clip(result, 0, 255).astype(np.uint8)

        if len(session.gaze_points) > 2:
            pts = session.gaze_points[::max(1, len(session.gaze_points)//500)]
            for i in range(1, len(pts)):
                x1       = int(pts[i-1][0] * sw)
                y1       = int(pts[i-1][1] * sh)
                x2       = int(pts[i][0]   * sw)
                y2       = int(pts[i][1]   * sh)
                progress = i / len(pts)
                color    = (int(255*(1-progress)), 50, int(255*progress))
                cv2.line(result, (x1,y1), (x2,y2), color, 1, cv2.LINE_AA)

        for gi in range(1, GRID_N):
            cv2.line(result, (sw*gi//GRID_N, 0),  (sw*gi//GRID_N, sh), (255,255,255), 1)
            cv2.line(result, (0, sh*gi//GRID_N),  (sw, sh*gi//GRID_N), (255,255,255), 1)

        path = os.path.join(out_dir, f"screen_heatmap_{ts_str}.jpg")
        cv2.imwrite(path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[RAPORT] Screen heatmap salvat: {path}")
        return path
    except Exception as e:
        print(f"[AVERTISMENT] Screen heatmap esuat: {e}")
        return None


def save_report(session: SessionData, out_dir="eye_report"):
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(out_dir, f"gaze_{ts_str}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["norm_x", "norm_y", "timestamp"])
        for gx, gy, gt in session.gaze_points:
            w.writerow([f"{gx:.4f}", f"{gy:.4f}", f"{gt:.3f}"])

    fix_csv = os.path.join(out_dir, f"fixations_{ts_str}.csv")
    with open(fix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center_x_norm", "center_y_norm", "duration_ms", "ts_start", "zone"])
        for fx in session.fixations:
            w.writerow([f"{fx.x:.4f}", f"{fx.y:.4f}",
                        f"{fx.duration_ms:.1f}", f"{fx.ts_start:.3f}",
                        ZONE_NAMES[fx.zone_idx]])
    print(f"[RAPORT] CSVuri salvate in {out_dir}/")

    save_screen_heatmap(session, out_dir, ts_str)

    if not HAS_MPL:
        return

    metrics = session.academic_metrics()

    BG    = "#0d0d14"
    CARD  = "#13131e"
    ACC   = "#00e5ff"
    ACC2  = "#ff5ca8"
    TEXT  = "#e8e8f0"
    MUTED = "#5a5a7a"

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle("Eye Interest Report  v2", color=TEXT,
                 fontsize=20, fontweight="bold", y=0.97,
                 fontfamily="monospace")
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.05, right=0.97, top=0.92, bottom=0.06)

    def style_ax(ax, title=""):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_color("#2a2a3a")
            sp.set_linewidth(0.5)
        ax.tick_params(colors=MUTED, labelsize=8)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10, pad=6, fontfamily="monospace")
        return ax

    ax1 = style_ax(fig.add_subplot(gs[0, 0]), "Heatmap privire")
    blurred   = cv2.GaussianBlur(session.heatmap_acc, (HEATMAP_BLUR, HEATMAP_BLUR), 0)
    heat_norm = blurred / blurred.max() if blurred.max() > 0 else blurred
    im = ax1.imshow(heat_norm, cmap="inferno", origin="upper", aspect="auto")
    for gi in range(1, GRID_N):
        ax1.axvline(HEATMAP_W * gi / GRID_N, color="white", lw=0.4, alpha=0.3)
        ax1.axhline(HEATMAP_H * gi / GRID_N, color="white", lw=0.4, alpha=0.3)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=MUTED)
    ax1.set_xlabel("X ecran", color=MUTED, fontsize=8)
    ax1.set_ylabel("Y ecran", color=MUTED, fontsize=8)
    for fx in session.fixations:
        ax1.add_patch(plt.Circle((fx.x * HEATMAP_W, fx.y * HEATMAP_H),
                                 max(1, fx.duration_ms / 500),
                                 color=ACC, fill=False, linewidth=0.6, alpha=0.7))

    ax2 = style_ax(fig.add_subplot(gs[0, 1]), "Distributie zone")
    zone_times = [session.zone_dwell.get(i, 0.0) for i in range(GRID_N * GRID_N)]
    total_t    = max(sum(zone_times), 0.001)
    pcts       = [t / total_t * 100 for t in zone_times]
    cmap_bar   = plt.cm.plasma
    bar_colors = [cmap_bar(p / max(pcts)) for p in pcts]
    bars = ax2.barh(range(len(ZONE_NAMES)), pcts, color=bar_colors,
                    edgecolor="#2a2a3a", linewidth=0.5)
    ax2.set_yticks(range(len(ZONE_NAMES)))
    ax2.set_yticklabels(ZONE_NAMES, fontsize=7.5, color=TEXT)
    ax2.set_xlabel("% timp atentie", color=MUTED, fontsize=8)
    for bar, pct in zip(bars, pcts):
        if pct > 1.5:
            ax2.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{pct:.1f}%", va="center", color=TEXT, fontsize=7)

    ax3 = style_ax(fig.add_subplot(gs[0, 2]), "Traiectorie privire")
    if session.gaze_points:
        xs = [p[0] for p in session.gaze_points]
        ys = [p[1] for p in session.gaze_points]
        ts = [p[2] - session.start_ts for p in session.gaze_points]
        sc = ax3.scatter(xs, ys, c=ts, cmap="plasma", s=1.5, alpha=0.5)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(1, 0)
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04,
                     label="Timp (s)").ax.yaxis.set_tick_params(color=MUTED)
    ax3.set_xlabel("X normalizat", color=MUTED, fontsize=8)
    ax3.set_ylabel("Y normalizat", color=MUTED, fontsize=8)
    for gi in range(1, GRID_N):
        ax3.axvline(gi/GRID_N, color=MUTED, lw=0.3, alpha=0.4)
        ax3.axhline(gi/GRID_N, color=MUTED, lw=0.3, alpha=0.4)

    ax4 = style_ax(fig.add_subplot(gs[0, 3]), "Metrici sesiune")
    ax4.axis("off")
    lines = [
        ("Durata sesiune",      f"{metrics['durata_sesiune_s']} s"),
        ("Nr. fixatii",         f"{metrics['nr_fixatii']}"),
        ("Fixatii / minut",     f"{metrics['fixatii_pe_minut']}"),
        ("Fixatie medie",       f"{metrics['durata_medie_fixatie_ms']} ms"),
        ("Fixatie maxima",      f"{metrics['durata_max_fixatie_ms']} ms"),
        ("Nr. saccade",         f"{metrics['nr_saccade']}"),
        ("Saccada medie",       f"{metrics['lungime_medie_saccada_px']} px"),
        ("Zona dominanta",      metrics['zona_dominanta']),
    ]
    for i, (label, val) in enumerate(lines):
        y = 0.95 - i * 0.115
        ax4.text(0.02, y, label, transform=ax4.transAxes,
                 color=MUTED, fontsize=8.5, fontfamily="monospace")
        ax4.text(0.98, y, val, transform=ax4.transAxes,
                 color=ACC, fontsize=9, fontfamily="monospace",
                 ha="right", fontweight="bold")
        ax4.plot([0.02, 0.98], [y - 0.04, y - 0.04],
                 color="#2a2a3a", linewidth=0.4, transform=ax4.transAxes, clip_on=False)

    ax5 = style_ax(fig.add_subplot(gs[1, 0]), "Distributie durata fixatii")
    if session.fixations:
        durs = [f.duration_ms for f in session.fixations]
        ax5.hist(durs, bins=20, color=ACC2, edgecolor=BG, alpha=0.85)
        ax5.axvline(np.mean(durs), color=ACC, lw=1.5, linestyle="--",
                    label=f"medie {np.mean(durs):.0f}ms")
        ax5.legend(fontsize=7.5, labelcolor=TEXT, facecolor=CARD, edgecolor=MUTED)
    ax5.set_xlabel("Durata (ms)", color=MUTED, fontsize=8)
    ax5.set_ylabel("Nr. fixatii", color=MUTED, fontsize=8)

    ax6 = style_ax(fig.add_subplot(gs[1, 1]), "Fixatii per zona")
    fix_counts = [len([f for f in session.fixations if f.zone_idx == i])
                  for i in range(GRID_N * GRID_N)]
    ax6.bar(range(len(ZONE_NAMES)), fix_counts,
            color=[cmap_bar(c / max(fix_counts + [1])) for c in fix_counts],
            edgecolor=BG, linewidth=0.5)
    ax6.set_xticks(range(len(ZONE_NAMES)))
    ax6.set_xticklabels([n.replace("-", "\n") for n in ZONE_NAMES],
                        fontsize=6, color=TEXT, rotation=0)
    ax6.set_ylabel("Nr. fixatii", color=MUTED, fontsize=8)

    ax7 = style_ax(fig.add_subplot(gs[1, 2]), "TTFF (Time to First Fixation)")
    ttff_data = [(ZONE_NAMES[k], v) for k, v in session.ttff_per_zone.items()]
    ttff_data.sort(key=lambda x: x[1])
    if ttff_data:
        labels_t = [d[0] for d in ttff_data]
        vals_t   = [d[1] for d in ttff_data]
        bars_t   = ax7.barh(range(len(labels_t)), vals_t, color=ACC, edgecolor=BG, alpha=0.8)
        ax7.set_yticks(range(len(labels_t)))
        ax7.set_yticklabels(labels_t, fontsize=7.5, color=TEXT)
        ax7.set_xlabel("Timp (s)", color=MUTED, fontsize=8)
        for b, v in zip(bars_t, vals_t):
            ax7.text(v + 0.1, b.get_y() + b.get_height()/2,
                     f"{v:.1f}s", va="center", color=TEXT, fontsize=7)

    ax8 = style_ax(fig.add_subplot(gs[1, 3]), "Lungime saccade (timeline)")
    if session.saccade_lengths:
        ax8.plot(session.saccade_lengths, color=ACC2, linewidth=0.6, alpha=0.7)
        ax8.fill_between(range(len(session.saccade_lengths)),
                         session.saccade_lengths, alpha=0.2, color=ACC2)
        ax8.axhline(np.mean(session.saccade_lengths), color=ACC, lw=1, linestyle="--", alpha=0.8)
    ax8.set_xlabel("Index saccada", color=MUTED, fontsize=8)
    ax8.set_ylabel("Lungime (px)", color=MUTED, fontsize=8)

    fig_path = os.path.join(out_dir, f"report_{ts_str}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[RAPORT] Grafic complet salvat: {fig_path}")

    import json
    json_path = os.path.join(out_dir, f"metrics_{ts_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[RAPORT] Metrici JSON salvate: {json_path}")


def get_ear(landmarks, points):
    v1 = np.hypot(landmarks[points[1]].x - landmarks[points[5]].x,
                  landmarks[points[1]].y - landmarks[points[5]].y)
    v2 = np.hypot(landmarks[points[2]].x - landmarks[points[4]].x,
                  landmarks[points[2]].y - landmarks[points[4]].y)
    h  = np.hypot(landmarks[points[0]].x - landmarks[points[3]].x,
                  landmarks[points[0]].y - landmarks[points[3]].y)
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


def open_camera():
    for idx in [1, 2, 0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return None


def draw_zone_grid(frame, session: SessionData):
    h, w  = frame.shape[:2]
    total = max(sum(session.zone_dwell.values()), 0.001)
    for row in range(GRID_N):
        for col in range(GRID_N):
            idx       = row * GRID_N + col
            dwell     = session.zone_dwell.get(idx, 0.0)
            pct       = dwell / total * 100
            x1, y1    = int(col*w/GRID_N), int(row*h/GRID_N)
            x2, y2    = int((col+1)*w/GRID_N), int((row+1)*h/GRID_N)
            intensity = int(min(pct / 30.0, 1.0) * 100)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, intensity, intensity), 1)
            if pct > 0.5:
                cv2.putText(frame, f"{pct:.0f}%",
                            ((x1+x2)//2 - 18, (y1+y2)//2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 170), 1)


def draw_fixations_overlay(frame, session: SessionData):
    h, w   = frame.shape[:2]
    recent = session.fixations[-20:] if len(session.fixations) > 20 else session.fixations
    for fx in recent:
        radius = max(4, int(fx.duration_ms / 50))
        cv2.circle(frame, (int(fx.x * w), int(fx.y * h)), radius, (0, 230, 255), 1)


def main():
    history_x, history_y   = [], []
    eyes_closed_start_time = None
    show_heatmap           = False
    show_fixations         = True
    IS_TRACKING            = True

    cap = open_camera()
    if cap is None:
        print("[EROARE] Camera nu a putut fi deschisa.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nDoresti sa rulezi calibrarea? (recomandat)")
    print("  C = calibrare  |  ENTER = skip (foloseste valori implicite)")
    cv2.namedWindow("EYE INTEREST TRACKER v2")
    calib_params = {"x_range": [0.41, 0.59], "y_range": [0.41, 0.59]}

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "C = Calibrare   ENTER = Skip",
                    (50, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,255), 2)
        cv2.imshow("EYE INTEREST TRACKER v2", frame)
    key = cv2.waitKey(5000) & 0xFF
    if key == ord('c'):
        calib_params = run_calibration(cap)

    session = SessionData()

    print("\n[START] Tracker pornit.")
    print("  ESC=pauza | S=start | H=heatmap | F=fixatii | Q=iesire+raport")
    print("  Tine ochii inchisi 3s = iesire\n")

    while cap.isOpened():
        try:
            ok, frame = cap.read()
            if not ok:
                continue

            if keyboard.is_pressed('esc'): IS_TRACKING = False
            if keyboard.is_pressed('s'):   IS_TRACKING = True
            if keyboard.is_pressed('h'):
                show_heatmap = not show_heatmap
                time.sleep(0.2)
            if keyboard.is_pressed('f'):
                show_fixations = not show_fixations
                time.sleep(0.2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = facemesh_model.process(rgb)

            if res.multi_face_landmarks:
                lm_raw   = res.multi_face_landmarks[0].landmark
                mesh_pts = np.array([
                    np.multiply([p.x, p.y], [w, h]).astype(int) for p in lm_raw
                ])

                (l_cx, l_cy), l_rad = cv2.minEnclosingCircle(mesh_pts[LEFT_IRIS])
                (r_cx, r_cy), r_rad = cv2.minEnclosingCircle(mesh_pts[RIGHT_IRIS])
                cl = np.array([l_cx, l_cy], dtype=np.int32)
                cr = np.array([r_cx, r_cy], dtype=np.int32)

                norm_x = (l_cx/w + r_cx/w) / 2
                norm_y = (l_cy/h + r_cy/h) / 2

                for eye_pts in [LEFT_EYE_BOX, RIGHT_EYE_BOX]:
                    rx, ry, ew, eh = cv2.boundingRect(mesh_pts[eye_pts])
                    cv2.rectangle(frame, (rx,ry), (rx+ew,ry+eh), (0,255,255), 1)

                if IS_TRACKING:
                    tx = np.interp(norm_x, calib_params["x_range"], [0, SCREEN_W])
                    ty = np.interp(norm_y, calib_params["y_range"], [0, SCREEN_H])
                    history_x.append(tx)
                    history_y.append(ty)
                    if len(history_x) > SMOOTH_FACTOR:
                        history_x.pop(0)
                        history_y.pop(0)
                    pyautogui.moveTo(int(np.mean(history_x)), int(np.mean(history_y)))

                    snx = float(np.clip(np.interp(norm_x, calib_params["x_range"], [0,1]), 0, 1))
                    sny = float(np.clip(np.interp(norm_y, calib_params["y_range"], [0,1]), 0, 1))
                    session.record(snx, sny)

                ear_l  = get_ear(lm_raw, LEFT_EYE_EAR_PTS)
                ear_r  = get_ear(lm_raw, RIGHT_EYE_EAR_PTS)
                closed = ear_l < EAR_THRESHOLD and ear_r < EAR_THRESHOLD

                if closed:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    dur = time.time() - eyes_closed_start_time
                    rem = int(EXIT_THRESHOLD - dur) + 1
                    cv2.putText(frame, f"IESIRE in: {rem}",
                                (w//2-120, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                    if dur >= EXIT_THRESHOLD:
                        break
                else:
                    if eyes_closed_start_time is not None:
                        dur = time.time() - eyes_closed_start_time
                        if dur < 0.5:
                            pyautogui.click()
                            cv2.putText(frame, "CLICK!",
                                        (w//2-50, 100),
                                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2)
                        eyes_closed_start_time = None

                cv2.circle(frame, cl, int(l_rad), (255,0,255), 1)
                cv2.circle(frame, cr, int(r_rad), (255,0,255), 1)

                m = session.academic_metrics()
                info = [
                    f"EAR S: {ear_l:.3f}   D: {ear_r:.3f}",
                    f"Fixatii: {m['nr_fixatii']}  ({m['fixatii_pe_minut']}/min)",
                    f"Fix medie: {m['durata_medie_fixatie_ms']}ms",
                    f"Saccade: {m['nr_saccade']}  medie: {m['lungime_medie_saccada_px']}px",
                ]
                for li, txt in enumerate(info):
                    cv2.putText(frame, txt, (12, 70 + li*26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,220,255), 1)

                top3    = session.top_zones(3)
                dur_tot = session.total_duration()
                y_off   = h - 15
                for rank, (zn, zt) in enumerate(reversed(top3), 1):
                    cv2.putText(frame,
                                f"#{len(top3)-rank+1} {zn}: {zt:.1f}s "
                                f"({zt/max(dur_tot,1)*100:.0f}%)",
                                (w - 340, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,255,200), 1)
                    y_off -= 20

            if show_heatmap and len(session.gaze_points) > 10:
                ov, _ = session.get_heatmap_overlay(w, h)
                frame  = cv2.addWeighted(frame, 1-HEATMAP_ALPHA, ov, HEATMAP_ALPHA, 0)
                cv2.putText(frame, "HEATMAP", (w-130, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,120,255), 2)

            if show_fixations:
                draw_fixations_overlay(frame, session)

            draw_zone_grid(frame, session)

            sc  = (0,255,0) if IS_TRACKING else (0,0,255)
            st  = "ACTIV" if IS_TRACKING else "PAUZA"
            ctg = "CALIBRAT" if calib_params["x_range"] != [0.41, 0.59] else "DEFAULT"
            cv2.putText(frame,
                        f"{st} | {ctg} | Pts:{len(session.gaze_points)} | "
                        f"Dur:{session.total_duration():.0f}s | H=heat F=fix Q=raport",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, sc, 1)

            cv2.imshow("EYE INTEREST TRACKER v2", frame)

        except Exception as e:
            print(f"[EROARE] {e}")
            import traceback; traceback.print_exc()


            break

    session.finalize()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("  Sesiune incheiata. Se genereaza raportul complet...")
    save_report(session)
    print("  Raportul este in folderul 'eye_report/'")
    print("="*55)


if __name__ == "__main__":
    main()