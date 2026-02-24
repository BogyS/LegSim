"""
playback_csv_gui_record_validate_v2.py

GUI CSV playback (no command-line arguments):
- Click "LOAD CSV" to select a CSV/TSV file (Excel/ANSYS style supported).
- Expects rows with at least 6 numeric values per row:
    hip_time, hip_omega, knee_time, knee_omega, ankle_time, ankle_omega
  Separators can be comma, tab, or semicolon. Extra header lines are ignored.

Features:
- 2D leg + torso animation moving left->right (fixed world axes).
- Angle plot shows 3 curves, each centered around its own zero (mean removed),
  separated by three dashed baselines.
- Buttons aligned to the LEFT: LOAD, PAUSE, RESET, REC, STOP.
- Sign controls to flip joint directions for HIP/KNEE/ANKLE so you can handle "inverted" datasets.

Ankle validation:
- If ankle "velocity" looks invalid (e.g., equals time, or unrealistically large),
  it will be replaced with zeros and a warning is shown.

Recording (REC/STOP):
- Requires: pip install mss opencv-python
- Records the entire Matplotlib window to MP4.
- Output MP4 name is based on the loaded CSV filename (stem + timestamp).

Offsets:
- HIP_OFFSET is -90° (vertical down) by default.
- IMPORTANT: Sign flip is applied to the integrated *relative* angle, then the offset is added:
    q = offset + sign * theta_rel
"""

import re
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path

# Optional deps for recording
try:
    import mss
    import cv2
    _REC_AVAILABLE = True
except Exception:
    _REC_AVAILABLE = False

# -------------------------
# Geometry
# -------------------------
L1 = 0.45   # thigh [m]
L2 = 0.43   # shank [m]
FOOT_TOTAL = 0.265      # EU42-ish heel-to-toe [m]
HEEL_BACK = 0.06        # ankle->heel [m]
TOE_FWD = FOOT_TOTAL - HEEL_BACK
TORSO_LEN = 0.55

# -------------------------
# Offsets (rad) — default start: vertical down
# -------------------------
HIP_OFFSET = -np.pi / 2
KNEE_OFFSET = 0.0
ANKLE_OFFSET = np.pi / 2

# Playback rendering
PLAYBACK_FPS = 60
SPEED = 1.0

# Visual travel distance (world frame)
TRAVEL_M = 1.2

def _numeric_tokens(line: str):
    parts = re.split(r"[,\t;]+", line.strip())
    nums = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", p):
            nums.append(float(p))
    return nums

def load_ansys_style_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            nums = _numeric_tokens(ln)
            if len(nums) >= 6:
                rows.append(nums[:6])
    if not rows:
        raise ValueError("No numeric rows found with >= 6 numeric values.")
    arr = np.array(rows, dtype=float)
    return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]

def dedupe_and_sort(t, y):
    order = np.argsort(t)
    t2 = t[order]
    y2 = y[order]
    keep = np.concatenate(([True], np.diff(t2) != 0))
    return t2[keep], y2[keep]

def resample_to_common_time(hip_t, hip_w, knee_t, knee_w, ankle_t, ankle_w):
    hip_t, hip_w = dedupe_and_sort(hip_t, hip_w)
    knee_t, knee_w = dedupe_and_sort(knee_t, knee_w)
    ankle_t, ankle_w = dedupe_and_sort(ankle_t, ankle_w)

    t = hip_t.copy()
    knee_w_i = np.interp(t, knee_t, knee_w)
    ankle_w_i = np.interp(t, ankle_t, ankle_w)
    return t, hip_w, knee_w_i, ankle_w_i

def integrate_velocity_rel(t, w):
    """Integrate omega to a *relative* angle theta_rel starting at 0 rad."""
    theta = np.empty_like(w, dtype=float)
    theta[0] = 0.0
    for i in range(1, len(t)):
        dt = float(t[i] - t[i-1])
        theta[i] = theta[i-1] + 0.5 * (w[i] + w[i-1]) * dt
    return theta

def ankle_velocity_looks_invalid(t, ankle_w):
    """
    Heuristics:
    1) ankle_w ~= time (common export mistake): corr ~ 1 and small error
    2) magnitude too large for typical joint omega streams
    """
    if len(t) < 10:
        return False, "Not enough samples to validate ankle."

    tw = t - np.mean(t)
    aw = ankle_w - np.mean(ankle_w)
    denom = (np.linalg.norm(tw) * np.linalg.norm(aw)) + 1e-12
    corr = float(np.dot(tw, aw) / denom)

    rmse = float(np.sqrt(np.mean((ankle_w - t)**2)))
    rng = float(np.ptp(t)) + 1e-12
    nrmse = rmse / rng

    if corr > 0.999 and nrmse < 0.02:
        return True, "Ankle 'velocity' appears to match time (likely wrong column)."

    max_abs = float(np.max(np.abs(ankle_w)))
    mean_abs = float(np.mean(np.abs(ankle_w)))
    if max_abs > 15.0 or mean_abs > 6.0:
        return True, f"Ankle velocity magnitude looks unrealistic (max|ω|={max_abs:.2f} rad/s, mean|ω|={mean_abs:.2f} rad/s)."

    return False, "Ankle data looks plausible."

# -------------------------
# Recording (whole window)
# -------------------------
class WindowRecorder:
    def __init__(self, fig, fps=30):
        self.fig = fig
        self.fps = int(fps)
        self.out_path = "recording.mp4"
        self.recording = False
        self.sct = None
        self.writer = None
        self.timer = None
        self.last_t = 0.0
        self.region = None

    def _get_window_region(self):
        mgr = self.fig.canvas.manager
        win = getattr(mgr, "window", None)
        if win is not None and hasattr(win, "winfo_rootx"):
            win.update_idletasks()
            left = win.winfo_rootx()
            top = win.winfo_rooty()
            width = win.winfo_width()
            height = win.winfo_height()
            if width <= 2 or height <= 2:
                win.update()
                width = win.winfo_width()
                height = win.winfo_height()
            return {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        raise RuntimeError("Could not access OS window geometry. Try using the TkAgg backend.")

    def start(self):
        if not _REC_AVAILABLE:
            raise RuntimeError("Recording deps missing. Install: pip install mss opencv-python")
        if self.recording:
            return

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.region = self._get_window_region()

        self.sct = mss.mss()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = self.region["width"], self.region["height"]
        self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError("Could not open VideoWriter for MP4.")

        self.recording = True
        self.last_t = 0.0

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def attach_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self._capture_frame)
        self.timer.start()

    def _capture_frame(self):
        if not self.recording:
            return

        now = time.time()
        if self.last_t != 0.0 and (now - self.last_t) < (1.0 / self.fps):
            return
        self.last_t = now

        frame = np.array(self.sct.grab(self.region))  # BGRA
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.writer.write(frame_bgr)

# -------------------------
# Playback UI + state
# -------------------------
class PlaybackApp:
    def __init__(self):
        self.fig, (self.ax_walk, self.ax_ang) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [2.2, 1.0]}
        )
        plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18, wspace=0.25)

        self.csv_path = None

        self.t = None
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None

        self.q1 = None
        self.q2 = None
        self.q3 = None

        self.frame_indices = None
        self.paused = False
        self.anim = None

        self.warning_text = None

        # Sign flips (1 or -1)
        self.sign = {"hip": 1.0, "knee": 1.0, "ankle": 1.0}

        # Recorder
        self.recorder = WindowRecorder(self.fig, fps=30)
        self.recorder.attach_timer()

        self._setup_axes()
        self._setup_artists()
        self._setup_controls()

        # Prompt file selection at start
        self.load_csv_dialog()

    def _setup_axes(self):
        self.ax_walk.set_title("CSV Playback (ω integrated to angles)")
        self.ax_walk.set_aspect("equal", adjustable="box")
        self.ax_walk.set_xlabel("x [m]")
        self.ax_walk.set_ylabel("y [m]")
        self.ax_walk.axhline(0.0, linestyle=":", alpha=0.6)
        self.ax_walk.set_xlim(-0.2, TRAVEL_M + 0.6)
        self.ax_walk.set_ylim(-0.05, 0.92 + TORSO_LEN + 0.25)
        self.ax_walk.grid(True)

        self.ax_ang.set_title("Joint angles (each centered to its own 0)")
        self.ax_ang.set_xlabel("time [s]")
        self.ax_ang.set_ylabel("angle [deg], centered")
        self.ax_ang.grid(True)

    def _setup_artists(self):
        self.leg_line, = self.ax_walk.plot([], [], "-o", linewidth=3, markersize=5, label="Leg")
        self.foot_line, = self.ax_walk.plot([], [], "-", linewidth=3, label="Foot")
        self.pelvis_pt, = self.ax_walk.plot([], [], "o", markersize=7, label="Pelvis")
        self.torso_ln, = self.ax_walk.plot([], [], "-", linewidth=3, alpha=0.8, label="Torso")
        self.ax_walk.legend(loc="upper left")

        self.warning_text = self.ax_walk.text(
            0.02, 0.98, "", transform=self.ax_walk.transAxes,
            va="top", ha="left"
        )

        self.offset1, self.offset2, self.offset3 = 40.0, 0.0, -40.0
        self.base1 = self.ax_ang.axhline(self.offset1, linewidth=1, linestyle="--", alpha=0.7)
        self.base2 = self.ax_ang.axhline(self.offset2, linewidth=1, linestyle="--", alpha=0.7)
        self.base3 = self.ax_ang.axhline(self.offset3, linewidth=1, linestyle="--", alpha=0.7)

        self.l1 = None
        self.l2 = None
        self.l3 = None
        self.cursor = None

    def _setup_controls(self):
        # Left-aligned button row (no sliders)
        y = 0.06
        w = 0.11
        h = 0.07
        x0 = 0.08
        pad = 0.012

        ax_load = plt.axes([x0 + (w+pad)*0, y, w, h])
        ax_pause = plt.axes([x0 + (w+pad)*1, y, w, h])
        ax_reset = plt.axes([x0 + (w+pad)*2, y, w, h])
        ax_rec = plt.axes([x0 + (w+pad)*3, y, w, h])
        ax_stop = plt.axes([x0 + (w+pad)*4, y, w, h])

        self.b_load = Button(ax_load, "LOAD CSV")
        self.b_pause = Button(ax_pause, "PAUSE")
        self.b_reset = Button(ax_reset, "RESET")
        self.b_rec = Button(ax_rec, "REC")
        self.b_stop = Button(ax_stop, "STOP")

        self.b_load.on_clicked(lambda _e: self.load_csv_dialog())
        self.b_pause.on_clicked(lambda _e: self.toggle_pause())
        self.b_reset.on_clicked(lambda _e: self.reset_playback())
        self.b_rec.on_clicked(lambda _e: self.start_recording())
        self.b_stop.on_clicked(lambda _e: self.stop_recording())

        # CheckButtons for sign flip (left-aligned, fits under plots)
        ax_chk = plt.axes([x0 + (w+pad)*5 + 0.02, y, 0.26, h])
        labels = ["Flip HIP", "Flip KNEE", "Flip ANKLE"]
        actives = [False, False, False]
        self.chk = CheckButtons(ax_chk, labels, actives)
        ax_chk.set_title("Direction", fontsize=9)

        def on_check(label):
            if label == "Flip HIP":
                self.sign["hip"] *= -1.0
            elif label == "Flip KNEE":
                self.sign["knee"] *= -1.0
            elif label == "Flip ANKLE":
                self.sign["ankle"] *= -1.0
            self._recompute_angles_from_theta()
            self._update_angle_plot()
            self.fig.canvas.draw_idle()

        self.chk.on_clicked(on_check)

        if not _REC_AVAILABLE:
            self.b_rec.label.set_text("REC*")
            self.b_stop.label.set_text("STOP*")

    def load_csv_dialog(self):
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = askopenfilename(
            title="Select CSV/TSV file",
            filetypes=[("CSV/TSV", "*.csv *.tsv *.txt"), ("All files", "*.*")]
        )
        root.destroy()
        if not path:
            return
        self.load_csv(path)

    def load_csv(self, path: str):
        try:
            self.csv_path = path
            hip_t, hip_w, knee_t, knee_w, ankle_t, ankle_w = load_ansys_style_csv(path)
            t, hip_w, knee_w, ankle_w = resample_to_common_time(hip_t, hip_w, knee_t, knee_w, ankle_t, ankle_w)

            invalid, reason = ankle_velocity_looks_invalid(t, ankle_w)
            if invalid:
                self.warning_text.set_text(f"WARNING: {reason}\nAnkle ω set to 0.")
                ankle_w = np.zeros_like(ankle_w)
            else:
                self.warning_text.set_text("")

            # Integrate -> RELATIVE angles (start at 0)
            theta1 = integrate_velocity_rel(t, hip_w)
            theta2 = integrate_velocity_rel(t, knee_w)
            theta3 = integrate_velocity_rel(t, ankle_w)

            self.t = t
            self.theta1, self.theta2, self.theta3 = theta1, theta2, theta3

            # Apply sign + offsets
            self._recompute_angles_from_theta()

            # Frame indices for playback speed
            base_dt = float(np.median(np.diff(t))) if len(t) > 2 else 1.0 / PLAYBACK_FPS
            render_dt = 1.0 / PLAYBACK_FPS
            step = max(1, int(round((render_dt * SPEED) / max(base_dt, 1e-9))))
            self.frame_indices = np.arange(0, len(t), step, dtype=int)

            self._update_angle_plot()
            self.reset_playback()

            print(f"Loaded: {path}")
            print("Ankle validation:", reason, "->", ("replaced with zeros" if invalid else "ok"))

        except Exception as e:
            self.warning_text.set_text(f"ERROR loading file:\n{e}")
            print("ERROR:", e)

    def _recompute_angles_from_theta(self):
        if self.t is None:
            return
        self.q1 = HIP_OFFSET + self.sign["hip"] * self.theta1
        self.q2 = KNEE_OFFSET + self.sign["knee"] * self.theta2
        self.q3 = ANKLE_OFFSET + self.sign["ankle"] * self.theta3

    def _update_angle_plot(self):
        # Remove previous plot lines/cursor (keep baselines)
        for obj in (self.l1, self.l2, self.l3, self.cursor):
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass

        if self.t is None:
            self.fig.canvas.draw_idle()
            return

        t = self.t
        q1, q2, q3 = self.q1, self.q2, self.q3

        # Center each curve around its own 0 (remove mean)
        q1c = np.rad2deg(q1 - np.mean(q1))
        q2c = np.rad2deg(q2 - np.mean(q2))
        q3c = np.rad2deg(q3 - np.mean(q3))

        self.l1, = self.ax_ang.plot(t, q1c + self.offset1, linewidth=2, label="Hip (centered)")
        self.l2, = self.ax_ang.plot(t, q2c + self.offset2, linewidth=2, label="Knee (centered)")
        self.l3, = self.ax_ang.plot(t, q3c + self.offset3, linewidth=2, label="Ankle (centered)")

        self.ax_ang.set_ylim(self.offset3 - 90, self.offset1 + 90)
        self.ax_ang.legend(loc="best")

        # Cursor
        self.cursor, = self.ax_ang.plot([t[0], t[0]], self.ax_ang.get_ylim(), "--", linewidth=1)

    def toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.label.set_text("RESUME" if self.paused else "PAUSE")
        self.fig.canvas.draw_idle()

    def reset_playback(self):
        self.paused = False
        self.b_pause.label.set_text("PAUSE")

        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass

        if self.frame_indices is None or self.t is None:
            self.fig.canvas.draw_idle()
            return

        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.frame_indices),
            interval=int(1000 / PLAYBACK_FPS),
            blit=True,
            init_func=self._init_anim
        )
        self.fig.canvas.draw_idle()

    def _init_anim(self):
        for a in (self.leg_line, self.foot_line, self.pelvis_pt, self.torso_ln):
            a.set_data([], [])
        if self.cursor is not None and self.t is not None:
            self.cursor.set_data([self.t[0], self.t[0]], self.ax_ang.get_ylim())
        return self._artists()

    def _artists(self):
        arts = [self.leg_line, self.foot_line, self.pelvis_pt, self.torso_ln]
        if self.cursor is not None:
            arts.append(self.cursor)
        return tuple(arts)

    def _update_frame(self, frame_pos):
        if self.paused or self.t is None:
            return self._artists()

        i = int(self.frame_indices[int(frame_pos)])

        t = self.t
        q1, q2, q3 = self.q1, self.q2, self.q3

        duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        hx = (t[i] - t[0]) / max(duration, 1e-9) * TRAVEL_M
        hy = 0.92

        # FK
        kx = hx + L1*np.cos(q1[i])
        ky = hy + L1*np.sin(q1[i])

        axx = kx + L2*np.cos(q1[i] + q2[i])
        ayy = ky + L2*np.sin(q1[i] + q2[i])

        foot_ang = q1[i] + q2[i] + q3[i]
        tx = axx + TOE_FWD*np.cos(foot_ang)
        ty = ayy + TOE_FWD*np.sin(foot_ang)
        hx2 = axx - HEEL_BACK*np.cos(foot_ang)
        hy2 = ayy - HEEL_BACK*np.sin(foot_ang)

        self.leg_line.set_data([hx, kx, axx], [hy, ky, ayy])
        self.foot_line.set_data([hx2, tx], [hy2, ty])
        self.pelvis_pt.set_data([hx], [hy])
        self.torso_ln.set_data([hx, hx], [hy, hy + TORSO_LEN])

        if self.cursor is not None:
            self.cursor.set_data([t[i], t[i]], self.ax_ang.get_ylim())

        return self._artists()

    def _mp4_name_from_csv(self):
        if not self.csv_path:
            stem = "recording"
        else:
            stem = Path(self.csv_path).stem
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{ts}.mp4"

    def start_recording(self):
        if not _REC_AVAILABLE:
            self.warning_text.set_text("Recording unavailable. Install: pip install mss opencv-python")
            return
        try:
            self.recorder.out_path = self._mp4_name_from_csv()
            self.recorder.start()
            self.b_rec.label.set_text("REC ●")
            self.b_stop.label.set_text("STOP ■")
            self.fig.canvas.draw_idle()
            print(f"Recording started: {self.recorder.out_path}")
        except Exception as e:
            self.warning_text.set_text(f"REC ERROR: {e}")
            print("REC ERROR:", e)

    def stop_recording(self):
        if not _REC_AVAILABLE:
            return
        self.recorder.stop()
        self.b_rec.label.set_text("REC")
        self.b_stop.label.set_text("STOP")
        self.fig.canvas.draw_idle()
        print(f"Recording saved: {self.recorder.out_path}")

def main():
    PlaybackApp()
    plt.show()

if __name__ == "__main__":
    main()
