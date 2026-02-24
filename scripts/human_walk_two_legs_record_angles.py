"""human_walk_two_legs_record_angles_v3.py

Two-leg 2D walking demo with 3 DOF per leg (hip, knee, ankle) + foot segment.

Features:
- Equal x/y scale in the walking plot.
- Short sliders (one shared parameter set for both legs).
- Angle plots for ONE leg (left) shown as 3 separate axes, each centered at 0.
  (We plot angle deviation from the leg's mean over the whole run.)
- A vertical time cursor synchronized across all angle axes.
- Pause / Reset buttons.
- REC / STOP buttons to record the ENTIRE Matplotlib window (including sliders/buttons)
  to an MP4 via screen capture.

Dependencies:
- numpy, matplotlib
- For recording: mss, opencv-python
  pip install mss opencv-python

Notes:
- Recording is screen-capture: if another window overlaps the figure, it will be recorded too.
- Tested best with TkAgg or QtAgg backends.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Recording deps (optional until you press REC)
try:
    import mss  # type: ignore
    import cv2  # type: ignore
except Exception:
    mss = None
    cv2 = None

# =========================
# Human proportions (~1.8 m)
# =========================
L1 = 0.45  # thigh: hip -> knee [m]
L2 = 0.43  # shank: knee -> ankle [m]

# Foot: EU 42 ≈ 26.5 cm heel-to-toe.
# Place ankle slightly forward of heel (simple anatomical cue).
FOOT_TOTAL = 0.265
HEEL_BACK = 0.06
TOE_FWD = FOOT_TOTAL - HEEL_BACK

# Visual-only torso reference
TORSO_LEN = 0.55

# =========================
# Simulation timing
# =========================
T = 1.0
NUM_STEPS = 3
FPS = 60
TOTAL_TIME = NUM_STEPS * T
N = int(TOTAL_TIME * FPS)
T_ARR = np.linspace(0.0, TOTAL_TIME, N, endpoint=False)

DEG = np.deg2rad


def smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def leg_phase(time_s: float, offset: float) -> float:
    return ((time_s / T) + offset) % 1.0


# =========================
# Shared parameter set
# =========================
DEFAULTS = dict(
    hip_height=0.92,     # pelvis height [m]
    step_len=0.70,       # step length [m]
    clearance=0.08,      # swing ankle lift [m]
    stance_ratio=0.62,   # stance fraction [0..1]
    toe_up_deg=12.0,     # dorsiflex in swing [deg]
    push_off_deg=12.0,   # plantarflex near toe-off [deg]
    knee_branch=-1.0,    # IK branch (+1 or -1)
)
STATE = DEFAULTS.copy()


# =========================
# Trajectory: ankle position + desired foot orientation
# =========================
def build_leg_trajectory(
    offset: float,
    hip_x: np.ndarray,
    stance_ratio: float,
    step_len: float,
    clearance: float,
    toe_up: float,
    push_off: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return desired ankle (x,y) in world + desired foot angle theta in world."""
    ank_x = np.zeros(N)
    ank_y = np.zeros(N)
    theta = np.zeros(N)

    planted_x = float(hip_x[0])
    prev_in_stance: bool | None = None

    for i in range(N):
        ph = leg_phase(float(T_ARR[i]), offset)
        in_stance = ph < stance_ratio

        if prev_in_stance is None:
            prev_in_stance = in_stance

        # Touchdown: plant slightly ahead of hip
        if in_stance and not prev_in_stance:
            planted_x = float(hip_x[i] + 0.25 * step_len)

        # Toe-off: leave from slightly behind hip
        if (not in_stance) and prev_in_stance:
            planted_x = float(hip_x[i] - 0.25 * step_len)

        prev_in_stance = in_stance

        if in_stance:
            s = ph / stance_ratio  # 0..1
            ank_x[i] = planted_x
            ank_y[i] = 0.0

            # Mostly flat; push-off near end of stance
            push = smoothstep(np.array([(s - 0.75) / 0.25], dtype=float))[0]
            theta[i] = -push_off * push
        else:
            s = (ph - stance_ratio) / (1.0 - stance_ratio)  # 0..1

            x_takeoff = planted_x
            x_land = float(hip_x[i] + 0.25 * step_len)
            ank_x[i] = x_takeoff + (x_land - x_takeoff) * smoothstep(np.array([s], dtype=float))[0]
            ank_y[i] = clearance * np.sin(np.pi * s)

            theta[i] = toe_up * np.sin(np.pi * s)  # toe-up mid swing

    return ank_x, ank_y, theta


# =========================
# IK: hip+knee to ankle; ankle sets foot angle
# =========================
def solve_leg_ik(
    ank_x: np.ndarray,
    ank_y: np.ndarray,
    theta: np.ndarray,
    hip_x: np.ndarray,
    hip_y: np.ndarray,
    knee_branch: float,
) -> tuple[np.ndarray, ...]:
    x_rel = ank_x - hip_x
    y_rel = ank_y - hip_y

    r2 = x_rel * x_rel + y_rel * y_rel
    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = clamp(c2, -1.0, 1.0)
    s2 = knee_branch * np.sqrt(np.maximum(0.0, 1.0 - c2 * c2))

    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(y_rel, x_rel) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    q3 = theta - (q1 + q2)
    foot_ang = q1 + q2 + q3

    knee_x = hip_x + L1 * np.cos(q1)
    knee_y = hip_y + L1 * np.sin(q1)

    ankle_x_fk = knee_x + L2 * np.cos(q1 + q2)
    ankle_y_fk = knee_y + L2 * np.sin(q1 + q2)

    toe_x = ankle_x_fk + TOE_FWD * np.cos(foot_ang)
    toe_y = ankle_y_fk + TOE_FWD * np.sin(foot_ang)

    heel_x = ankle_x_fk - HEEL_BACK * np.cos(foot_ang)
    heel_y = ankle_y_fk - HEEL_BACK * np.sin(foot_ang)

    return (
        q1,
        q2,
        q3,
        knee_x,
        knee_y,
        ankle_x_fk,
        ankle_y_fk,
        heel_x,
        heel_y,
        toe_x,
        toe_y,
    )


DATA: dict[str, np.ndarray] = {}


def recompute_all() -> None:
    hip_height = float(STATE["hip_height"])
    step_len = float(STATE["step_len"])
    clearance = float(STATE["clearance"])
    stance_ratio = float(STATE["stance_ratio"])
    toe_up = DEG(float(STATE["toe_up_deg"]))
    push_off = DEG(float(STATE["push_off_deg"]))
    knee_branch = float(STATE["knee_branch"])

    v = step_len / T
    hip_x = v * T_ARR
    hip_y = np.full(N, hip_height)

    ankL_x, ankL_y, thetaL = build_leg_trajectory(0.0, hip_x, stance_ratio, step_len, clearance, toe_up, push_off)
    ankR_x, ankR_y, thetaR = build_leg_trajectory(0.5, hip_x, stance_ratio, step_len, clearance, toe_up, push_off)

    (
        L_q1,
        L_q2,
        L_q3,
        L_kx,
        L_ky,
        L_ax,
        L_ay,
        L_hx,
        L_hy,
        L_tx,
        L_ty,
    ) = solve_leg_ik(ankL_x, ankL_y, thetaL, hip_x, hip_y, knee_branch)

    (
        R_q1,
        R_q2,
        R_q3,
        R_kx,
        R_ky,
        R_ax,
        R_ay,
        R_hx,
        R_hy,
        R_tx,
        R_ty,
    ) = solve_leg_ik(ankR_x, ankR_y, thetaR, hip_x, hip_y, knee_branch)

    DATA.clear()
    DATA.update(
        dict(
            hip_x=hip_x,
            hip_y=hip_y,
            ankL_x=ankL_x,
            ankL_y=ankL_y,
            ankR_x=ankR_x,
            ankR_y=ankR_y,
            # Left leg
            L_q1=L_q1,
            L_q2=L_q2,
            L_q3=L_q3,
            L_kx=L_kx,
            L_ky=L_ky,
            L_ax=L_ax,
            L_ay=L_ay,
            L_hx=L_hx,
            L_hy=L_hy,
            L_tx=L_tx,
            L_ty=L_ty,
            # Right leg
            R_q1=R_q1,
            R_q2=R_q2,
            R_q3=R_q3,
            R_kx=R_kx,
            R_ky=R_ky,
            R_ax=R_ax,
            R_ay=R_ay,
            R_hx=R_hx,
            R_hy=R_hy,
            R_tx=R_tx,
            R_ty=R_ty,
        )
    )


recompute_all()

# =========================
# Figure layout
# Left: walking
# Right: three stacked angle axes
# Bottom: short sliders (left) + buttons (right)
# =========================
fig = plt.figure(figsize=(12, 6))

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[2.2, 1.0],
    height_ratios=[1.0, 1.0, 1.0],
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.30,
    wspace=0.25,
    hspace=0.15,
)

ax_walk = fig.add_subplot(gs[:, 0])
ax_q1 = fig.add_subplot(gs[0, 1])
ax_q2 = fig.add_subplot(gs[1, 1], sharex=ax_q1)
ax_q3 = fig.add_subplot(gs[2, 1], sharex=ax_q1)

fig.suptitle("Two-Leg Walking (3 DOF per leg) + Angle Plots + MP4 Recording", fontsize=12)

# ---- walking axis ----
ax_walk.set_title("Walking view (equal x/y scale)")
ax_walk.set_aspect("equal", adjustable="box")
ax_walk.set_xlabel("x [m]")
ax_walk.set_ylabel("y [m]")
ax_walk.axhline(0.0, linestyle=":")
ax_walk.grid(True)

# Scale check square (0.2m)
_sq = 0.20
ax_walk.plot([0, _sq, _sq, 0, 0], [0, 0, _sq, _sq, 0], linewidth=1, alpha=0.25)

# faint ankle paths
pathL, = ax_walk.plot(DATA["ankL_x"], DATA["ankL_y"], linewidth=1, alpha=0.18)
pathR, = ax_walk.plot(DATA["ankR_x"], DATA["ankR_y"], linewidth=1, alpha=0.18)

# legs: markers on joints
legL, = ax_walk.plot([], [], "-o", linewidth=3, markersize=5, label="Left leg")
legR, = ax_walk.plot([], [], "-o", linewidth=3, markersize=5, label="Right leg")

# feet: no markers
footL, = ax_walk.plot([], [], "-", linewidth=2, label="Left foot")
footR, = ax_walk.plot([], [], "-", linewidth=2, label="Right foot")

pelvis_pt, = ax_walk.plot([], [], "o", markersize=7, label="Pelvis")
torso_ln, = ax_walk.plot([], [], "-", linewidth=3, alpha=0.8, label="Torso")

ax_walk.legend(loc="upper left")

# IMPORTANT: fixed world window so the body moves left->right across the axis
# (No camera tracking.)
hip_x_all = DATA["hip_x"]
min_x = float(np.min(hip_x_all)) - 0.40
max_x = float(np.max(hip_x_all)) + 0.40
ax_walk.set_xlim(min_x, max_x)
ax_walk.set_ylim(-0.05, float(STATE["hip_height"]) + TORSO_LEN + 0.15)


# ---- angle axes (left leg) ----
for ax in (ax_q1, ax_q2, ax_q3):
    ax.grid(True)
    ax.set_ylabel("deg")

ax_q3.set_xlabel("time [s]")
ax_q1.set_title("Left leg angles (each centered at 0)")

# Center each angle around 0 (de-mean) so each axis has its own 0 baseline.
# This matches your request for "3 zeros" (one per angle).
L_q1_deg = np.rad2deg(DATA["L_q1"])
L_q2_deg = np.rad2deg(DATA["L_q2"])
L_q3_deg = np.rad2deg(DATA["L_q3"])

L_q1_c = L_q1_deg - float(np.mean(L_q1_deg))
L_q2_c = L_q2_deg - float(np.mean(L_q2_deg))
L_q3_c = L_q3_deg - float(np.mean(L_q3_deg))

line_q1, = ax_q1.plot(T_ARR, L_q1_c, linewidth=2, label="hip (q1) - mean")
line_q2, = ax_q2.plot(T_ARR, L_q2_c, linewidth=2, label="knee (q2) - mean")
line_q3, = ax_q3.plot(T_ARR, L_q3_c, linewidth=2, label="ankle (q3) - mean")

# Zero lines (explicit 0 baseline on each axis)
ax_q1.axhline(0.0, linewidth=1, alpha=0.6)
ax_q2.axhline(0.0, linewidth=1, alpha=0.6)
ax_q3.axhline(0.0, linewidth=1, alpha=0.6)

ax_q1.legend(loc="upper right", fontsize=8)
ax_q2.legend(loc="upper right", fontsize=8)
ax_q3.legend(loc="upper right", fontsize=8)

# Shared vertical time cursor + points
cursor_q1, = ax_q1.plot([T_ARR[0], T_ARR[0]], [0, 1], "--", linewidth=1)
cursor_q2, = ax_q2.plot([T_ARR[0], T_ARR[0]], [0, 1], "--", linewidth=1)
cursor_q3, = ax_q3.plot([T_ARR[0], T_ARR[0]], [0, 1], "--", linewidth=1)

pt_q1, = ax_q1.plot([T_ARR[0]], [L_q1_c[0]], "o", markersize=5)
pt_q2, = ax_q2.plot([T_ARR[0]], [L_q2_c[0]], "o", markersize=5)
pt_q3, = ax_q3.plot([T_ARR[0]], [L_q3_c[0]], "o", markersize=5)


def autoscale_angle_axes() -> None:
    for ax, y in ((ax_q1, L_q1_c), (ax_q2, L_q2_c), (ax_q3, L_q3_c)):
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        pad = 0.10 * (y_max - y_min + 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Update cursor y-ranges
    y1 = ax_q1.get_ylim()
    y2 = ax_q2.get_ylim()
    y3 = ax_q3.get_ylim()
    cursor_q1.set_data([T_ARR[0], T_ARR[0]], [y1[0], y1[1]])
    cursor_q2.set_data([T_ARR[0], T_ARR[0]], [y2[0], y2[1]])
    cursor_q3.set_data([T_ARR[0], T_ARR[0]], [y3[0], y3[1]])


autoscale_angle_axes()

# =========================
# Sliders (short, under walking area)
# =========================
axcolor = "lightgoldenrodyellow"

slider_left = 0.09
slider_width = 0.50
slider_h = 0.030
slider_gap = 0.012
base_y = 0.22

ax_step = fig.add_axes([slider_left, base_y, slider_width, slider_h], facecolor=axcolor)
ax_stance = fig.add_axes([slider_left, base_y - 1 * (slider_h + slider_gap), slider_width, slider_h], facecolor=axcolor)
ax_hip = fig.add_axes([slider_left, base_y - 2 * (slider_h + slider_gap), slider_width, slider_h], facecolor=axcolor)
ax_clear = fig.add_axes([slider_left, base_y - 3 * (slider_h + slider_gap), slider_width, slider_h], facecolor=axcolor)
ax_toe = fig.add_axes([slider_left, base_y - 4 * (slider_h + slider_gap), slider_width, slider_h], facecolor=axcolor)
ax_push = fig.add_axes([slider_left, base_y - 5 * (slider_h + slider_gap), slider_width, slider_h], facecolor=axcolor)

s_step = Slider(ax_step, "step_len (m)", 0.45, 0.90, valinit=DEFAULTS["step_len"], valstep=0.01)
s_stance = Slider(ax_stance, "stance_ratio", 0.50, 0.70, valinit=DEFAULTS["stance_ratio"], valstep=0.005)
s_hip = Slider(ax_hip, "hip_height (m)", 0.85, 1.00, valinit=DEFAULTS["hip_height"], valstep=0.005)
s_clear = Slider(ax_clear, "clearance (m)", 0.03, 0.14, valinit=DEFAULTS["clearance"], valstep=0.005)
s_toe = Slider(ax_toe, "toe_up (deg)", 0.0, 20.0, valinit=DEFAULTS["toe_up_deg"], valstep=0.5)
s_push = Slider(ax_push, "push_off (deg)", 0.0, 25.0, valinit=DEFAULTS["push_off_deg"], valstep=0.5)


# =========================
# Buttons: Pause / Reset / REC / STOP (grouped together)
# =========================
paused = {"value": False}

btn_w = 0.085
btn_h = 0.05
btn_y = 0.05
btn_x0 = 0.66
btn_gap = 0.012

ax_btn_pause = fig.add_axes([btn_x0 + 0 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
ax_btn_reset = fig.add_axes([btn_x0 + 1 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
ax_btn_rec = fig.add_axes([btn_x0 + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
ax_btn_stop = fig.add_axes([btn_x0 + 3 * (btn_w + btn_gap), btn_y, btn_w, btn_h])

btn_pause = Button(ax_btn_pause, "Pause")
btn_reset = Button(ax_btn_reset, "Reset")
btn_rec = Button(ax_btn_rec, "REC")
btn_stop = Button(ax_btn_stop, "STOP")


# =========================
# Recording support (screen capture of entire window)
# =========================
rec_state = {
    "recording": False,
    "writer": None,
    "sct": None,
    "region": None,
    "fps": 30,
    "last_t": 0.0,
    "out_path": "recording.mp4",
}


def _get_window_region(fig_obj) -> dict:
    """Return {'left','top','width','height'} for the figure window."""
    mgr = fig_obj.canvas.manager
    win = getattr(mgr, "window", None)
    if win is None:
        raise RuntimeError("Cannot access figure window handle (manager.window).")

    # TkAgg
    if hasattr(win, "winfo_rootx"):
        win.update_idletasks()
        left = int(win.winfo_rootx())
        top = int(win.winfo_rooty())
        width = int(win.winfo_width())
        height = int(win.winfo_height())
        return {"left": left, "top": top, "width": width, "height": height}

    # QtAgg / Qt
    if hasattr(win, "frameGeometry") and hasattr(win, "mapToGlobal"):
        geo = win.frameGeometry()
        top_left = win.mapToGlobal(geo.topLeft())
        left = int(top_left.x())
        top = int(top_left.y())
        width = int(geo.width())
        height = int(geo.height())
        return {"left": left, "top": top, "width": width, "height": height}

    # Fallback: try geometry tuple
    if hasattr(win, "geometry"):
        try:
            geo = win.geometry()
            # geometry might be like (x, y, w, h)
            left, top, width, height = map(int, geo)
            return {"left": left, "top": top, "width": width, "height": height}
        except Exception:
            pass

    raise RuntimeError("Unsupported backend window type for recording.")


def _capture_frame() -> None:
    if not rec_state["recording"]:
        return

    now = time.time()
    if rec_state["last_t"] != 0.0:
        if now - rec_state["last_t"] < 1.0 / rec_state["fps"]:
            return
    rec_state["last_t"] = now

    region = rec_state["region"]
    sct = rec_state["sct"]
    writer = rec_state["writer"]

    frame = np.array(sct.grab(region))  # BGRA
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    writer.write(frame_bgr)


capture_timer = None


def start_recording(_event=None) -> None:
    if rec_state["recording"]:
        return

    if mss is None or cv2 is None:
        print("Recording requires: pip install mss opencv-python")
        return

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    region = _get_window_region(fig)
    rec_state["region"] = region

    w, h = region["width"], region["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = rec_state["out_path"]
    writer = cv2.VideoWriter(out, fourcc, rec_state["fps"], (w, h))
    if not writer.isOpened():
        print("Could not open video writer. Try a different codec or check permissions.")
        return

    rec_state["writer"] = writer
    rec_state["sct"] = mss.mss()
    rec_state["recording"] = True
    rec_state["last_t"] = 0.0

    btn_rec.label.set_text("REC ●")
    btn_stop.label.set_text("STOP ■")
    fig.canvas.draw_idle()


def stop_recording(_event=None) -> None:
    if not rec_state["recording"]:
        return

    rec_state["recording"] = False

    try:
        if rec_state["writer"] is not None:
            rec_state["writer"].release()
    finally:
        rec_state["writer"] = None

    try:
        if rec_state["sct"] is not None:
            rec_state["sct"].close()
    finally:
        rec_state["sct"] = None

    btn_rec.label.set_text("REC")
    btn_stop.label.set_text("STOP")
    fig.canvas.draw_idle()

    print(f"Saved recording to: {rec_state['out_path']}")


# Capture timer runs always; it only writes frames while recording=True.
capture_timer = fig.canvas.new_timer(interval=10)
capture_timer.add_callback(_capture_frame)
capture_timer.start()


# =========================
# Slider callbacks
# =========================
def refresh_angle_lines() -> None:
    global L_q1_deg, L_q2_deg, L_q3_deg, L_q1_c, L_q2_c, L_q3_c

    L_q1_deg = np.rad2deg(DATA["L_q1"])
    L_q2_deg = np.rad2deg(DATA["L_q2"])
    L_q3_deg = np.rad2deg(DATA["L_q3"])

    L_q1_c = L_q1_deg - float(np.mean(L_q1_deg))
    L_q2_c = L_q2_deg - float(np.mean(L_q2_deg))
    L_q3_c = L_q3_deg - float(np.mean(L_q3_deg))

    line_q1.set_ydata(L_q1_c)
    line_q2.set_ydata(L_q2_c)
    line_q3.set_ydata(L_q3_c)

    # Reset points to first sample
    pt_q1.set_data([T_ARR[0]], [L_q1_c[0]])
    pt_q2.set_data([T_ARR[0]], [L_q2_c[0]])
    pt_q3.set_data([T_ARR[0]], [L_q3_c[0]])

    autoscale_angle_axes()


def on_slider_change(_val=None) -> None:
    STATE["step_len"] = float(s_step.val)
    STATE["stance_ratio"] = float(s_stance.val)
    STATE["hip_height"] = float(s_hip.val)
    STATE["clearance"] = float(s_clear.val)
    STATE["toe_up_deg"] = float(s_toe.val)
    STATE["push_off_deg"] = float(s_push.val)

    recompute_all()

    # Update paths
    pathL.set_data(DATA["ankL_x"], DATA["ankL_y"])
    pathR.set_data(DATA["ankR_x"], DATA["ankR_y"])

    # Update fixed world xlim
    hip_x_all = DATA["hip_x"]
    min_x = float(np.min(hip_x_all)) - 0.40
    max_x = float(np.max(hip_x_all)) + 0.40
    ax_walk.set_xlim(min_x, max_x)
    ax_walk.set_ylim(-0.05, float(STATE["hip_height"]) + TORSO_LEN + 0.15)

    refresh_angle_lines()
    fig.canvas.draw_idle()


for s in (s_step, s_stance, s_hip, s_clear, s_toe, s_push):
    s.on_changed(on_slider_change)


# =========================
# Button callbacks
# =========================
def on_pause(_event=None) -> None:
    paused["value"] = not paused["value"]
    btn_pause.label.set_text("Resume" if paused["value"] else "Pause")
    fig.canvas.draw_idle()


def on_reset(_event=None) -> None:
    paused["value"] = False
    btn_pause.label.set_text("Pause")

    # Restore defaults
    for k, v in DEFAULTS.items():
        STATE[k] = v

    # Reset sliders (triggers callbacks)
    s_step.reset()
    s_stance.reset()
    s_hip.reset()
    s_clear.reset()
    s_toe.reset()
    s_push.reset()


btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(on_reset)
btn_rec.on_clicked(start_recording)
btn_stop.on_clicked(stop_recording)


# =========================
# Animation update
# =========================
def init_artists():
    for a in (legL, legR, footL, footR, pelvis_pt, torso_ln):
        a.set_data([], [])
    return (
        legL,
        legR,
        footL,
        footR,
        pelvis_pt,
        torso_ln,
        cursor_q1,
        cursor_q2,
        cursor_q3,
        pt_q1,
        pt_q2,
        pt_q3,
    )


def update_frame(i: int):
    if paused["value"]:
        return (
            legL,
            legR,
            footL,
            footR,
            pelvis_pt,
            torso_ln,
            cursor_q1,
            cursor_q2,
            cursor_q3,
            pt_q1,
            pt_q2,
            pt_q3,
        )

    hx = float(DATA["hip_x"][i])
    hy = float(DATA["hip_y"][i])

    # Left leg
    legL.set_data([hx, float(DATA["L_kx"][i]), float(DATA["L_ax"][i])],
                  [hy, float(DATA["L_ky"][i]), float(DATA["L_ay"][i])])
    footL.set_data([float(DATA["L_hx"][i]), float(DATA["L_tx"][i])],
                   [float(DATA["L_hy"][i]), float(DATA["L_ty"][i])])

    # Right leg
    legR.set_data([hx, float(DATA["R_kx"][i]), float(DATA["R_ax"][i])],
                  [hy, float(DATA["R_ky"][i]), float(DATA["R_ay"][i])])
    footR.set_data([float(DATA["R_hx"][i]), float(DATA["R_tx"][i])],
                   [float(DATA["R_hy"][i]), float(DATA["R_ty"][i])])

    pelvis_pt.set_data([hx], [hy])
    torso_ln.set_data([hx, hx], [hy, hy + TORSO_LEN])

    # Angle cursor
    ti = float(T_ARR[i])

    y1 = ax_q1.get_ylim(); y2 = ax_q2.get_ylim(); y3 = ax_q3.get_ylim()
    cursor_q1.set_data([ti, ti], [y1[0], y1[1]])
    cursor_q2.set_data([ti, ti], [y2[0], y2[1]])
    cursor_q3.set_data([ti, ti], [y3[0], y3[1]])

    pt_q1.set_data([ti], [float(L_q1_c[i])])
    pt_q2.set_data([ti], [float(L_q2_c[i])])
    pt_q3.set_data([ti], [float(L_q3_c[i])])

    return (
        legL,
        legR,
        footL,
        footR,
        pelvis_pt,
        torso_ln,
        cursor_q1,
        cursor_q2,
        cursor_q3,
        pt_q1,
        pt_q2,
        pt_q3,
    )


ani = FuncAnimation(fig, update_frame, frames=N, init_func=init_artists, interval=int(1000 / FPS), blit=True)

plt.show()
