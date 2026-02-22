import dearpygui.dearpygui as dpg
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os
import serial
import json
from datetime import date, datetime, timedelta
import math
import threading
import queue
import psutil

try:
    arduino = serial.Serial('COM5', 9600, timeout = 1)
    time.sleep(2)
    ARDUINO = True
except Exception:
    ARDUINO = False

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
if not os.path.exists(model_path):
    print("Downloading face_landmarker.task")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete")

detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options = mp_python.BaseOptions(model_asset_path = model_path),
    output_facial_transformation_matrixes = True,
    num_faces = 1,
    min_face_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
))

LEFT_IRIS       = [474, 475, 476, 477]
RIGHT_IRIS      = [469, 470, 471, 472]
LEFT_EYE        = [362, 263]
LEFT_EYE_EAR    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR   = [362, 385, 387, 263, 373, 380]
FACE_OVAL       = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EAR_PTS    = [234, 93, 132, 58, 172, 136]
RIGHT_EAR_PTS   = [454, 323, 361, 288, 397, 365]

# Tweak These
YAW_THRESHOLD           = 15      #head tilting degrees left and right
PITCH_UP_THRESHOLD      = 8       #head tilting up angle
PITCH_DOWN_THRESHOLD    = 15      #head tilting down angle
IRIS_RATIO_MIN          = 0.35    #looking left
IRIS_RATIO_MAX          = 0.65    #looking right
EAR_THRESHOLD           = 0.18    #asians should make this lower if you have thin eyes (I'm asian dw)
COUNTDOWN_SECS          = 300.0
FLASH_SECS              = 5.0
SERVO_OPEN_SECS         = 2.0     #how long to wait for the servo to fully open
# --------------------

CAM_W, CAM_H = 640, 480
DAYS         = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DATA_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facescanner_data.json')
HOSTS_PATH   = r"C:\Windows\System32\drivers\etc\hosts"
HOSTS_MARKER = "# faceScanner block"

C_BG     = (8,   8,  16, 255)
C_PANEL  = (14,  14,  28, 255)
C_BORDER = (26,  26,  53, 255)
C_ACCENT = (91,  94, 244, 255)
C_GREEN  = (16, 185, 129, 255)
C_RED    = (244, 63,  94, 255)
C_ORANGE = (251, 146,  60, 255)
C_TEXT   = (232, 234, 240, 255)
C_MUTED  = (74,  74, 112, 255)
C_BRIGHT = (210, 215, 255, 255)
C_DIM    = (30,  30,  60, 255)
C_TITLE  = (230, 235, 255, 255)

frame_queue          = queue.Queue(maxsize = 2)
gaze_status          = "Initializing"
flash_overlay        = False
camera_running       = True

session_state        = "IDLE"
session_total        = 0.0
session_remaining    = 0.0
break_total          = 0.0
break_remaining      = 0.0
session_paused       = False
last_tick_time       = None
total_session_secs   = 0.0
total_elapsed_secs   = 0.0
cycles_completed     = 0
work_min_val         = 90.0
break_min_val        = 20.0

away_since           = None
servo_rotated        = False
last_sent_secs       = -1
session_failed       = False
today_work_sec       = 0.0
app_data             = {}
has_title_font       = False




def send(cmd):
    if ARDUINO:
        try:
            arduino.write(cmd)
        except Exception:
            pass


def load_data():
    global app_data, today_work_sec
    default = {
        "day_targets": {d: 0.0 for d in DAYS},
        "work_log": {},
        "todos": [],
        "blacklist": {"apps": [], "sites": []}
    }
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                default.update(json.load(f))
        except Exception:
            pass
    if "blacklist" not in default:
        default["blacklist"] = {"apps": [], "sites": []}
    app_data       = default
    today_work_sec = app_data["work_log"].get(str(date.today()), 0.0)


def save_data():
    app_data["work_log"][str(date.today())] = today_work_sec
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(app_data, f, indent = 2)
    except Exception:
        pass


def get_streak():
    log    = app_data.get("work_log", {})
    today  = date.today()
    streak = 0
    day    = today
    while True:
        target = app_data["day_targets"].get(DAYS[day.weekday()], 0.0) * 3600
        done   = log.get(str(day), 0.0)
        if target > 0 and done >= target:
            streak += 1
            day -= timedelta(days = 1)
        else:
            break
    return streak




def get_blacklist():
    return app_data.get("blacklist", {"apps": [], "sites": []})


def write_hosts(sites):
    try:
        with open(HOSTS_PATH, "r", newline="") as f:
            raw = f.read()
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        in_block = False
        kept = []
        for l in raw.split("\n"):
            if HOSTS_MARKER in l:
                in_block = True
                continue
            if not in_block:
                kept.append(l)
        lines = [l + "\n" for l in kept if l.strip()]
        with open(HOSTS_PATH, "w") as f:
            f.writelines(lines)
            if sites:
                f.write(f"\n{HOSTS_MARKER}\n")
                for s in sites:
                    s = s.strip().lower()
                    f.write(f"127.0.0.1 {s}\n")
                    if not s.startswith("www."):
                        f.write(f"127.0.0.1 www.{s}\n")
        os.system("ipconfig /flushdns")
    except PermissionError:
        if dpg.does_item_exist("gaze_label"):
            dpg.set_value("gaze_label", "⚠ Run as administrator for website blocking")


def clear_hosts():
    write_hosts([])


def kill_banned_apps():
    banned = [x.lower() for x in get_blacklist().get("apps", [])]
    if not banned:
        return
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() in banned:
                proc.kill()
        except Exception:
            pass


def blacklist_thread_fn():
    while camera_running:
        if session_state == "WORKING":
            kill_banned_apps()
        time.sleep(3)


def resolve_url_target(url_path):
    try:
        with open(url_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("url="):
                    url = line[4:].strip()
                    protocol = url.split("://")[0]
                    if protocol and protocol.lower() not in ("http", "https", "mailto", "ftp"):
                        return protocol + ".exe"
    except Exception:
        pass
    return None


def resolve_lnk_target(lnk_path):
    try:
        import subprocess
        lnk_path_ps = lnk_path.replace("'", "\'")
        ps_cmd = (
            f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path_ps}');"
            f"Write-Output ($s.TargetPath + '|' + $s.Arguments)"
        )
        result = subprocess.check_output(
            ["powershell", "-Command", ps_cmd],
            text = True, stderr = subprocess.DEVNULL
        ).strip()
        target, _, args = result.partition("|")
        target = target.strip()
        args   = args.strip()
        if "://" in target:
            protocol = target.split("://")[0].lower()
            if protocol not in ("http", "https", "mailto", "ftp"):
                return protocol + ".exe"
        elif "--processStart" in args:
            return args.split("--processStart")[1].strip().split()[0]
        elif target:
            return os.path.basename(target)
    except Exception:
        pass
    return None

def pick_app_from_shortcut():
    try:
        import tkinter as tk
        from tkinter import filedialog

        home = os.path.expanduser("~")
        desktop = os.path.join(home, "Desktop")
        if not os.path.exists(desktop):
            for sub in os.listdir(home):
                candidate = os.path.join(home, sub, "Desktop")
                if os.path.exists(candidate):
                    desktop = candidate
                    break

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent = root,
            initialdir = desktop if os.path.exists(desktop) else home,
            title = "Select a shortcut or executable to block",
            filetypes = [("All files", "*.*"), ("Shortcuts", "*.lnk"), ("Executables", "*.exe"), ("URL Shortcuts", "*.url")]
        )
        root.destroy()

        if not path:
            return

        if path.lower().endswith(".lnk"):
            exe = resolve_lnk_target(path)
        elif path.lower().endswith(".url"):
            exe = resolve_url_target(path)
        else:
            exe = os.path.basename(path)

        if exe:
            dpg.set_value("new_app_input", exe)
            add_banned_app()
    except Exception:
        pass

def add_banned_app(sender = None, app_data_val = None):
    name = dpg.get_value("new_app_input").strip()
    if not name:
        return
    if not name.lower().endswith(".exe"):
        name += ".exe"
    bl = get_blacklist()
    if name.lower() not in [x.lower() for x in bl["apps"]]:
        bl["apps"].append(name)
        app_data["blacklist"] = bl
        save_data()
    dpg.set_value("new_app_input", "")
    render_blacklist()


def remove_banned_app(idx):
    bl = get_blacklist()
    bl["apps"].pop(idx)
    app_data["blacklist"] = bl
    save_data()
    render_blacklist()


def add_banned_site(sender = None, app_data_val = None):
    name = dpg.get_value("new_site_input").strip().lower()
    name = name.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")
    if not name:
        return
    bl = get_blacklist()
    if name not in bl["sites"]:
        bl["sites"].append(name)
        app_data["blacklist"] = bl
        save_data()
        if session_state == "WORKING":
            write_hosts(bl["sites"])
    dpg.set_value("new_site_input", "")
    render_blacklist()


def remove_banned_site(idx):
    bl = get_blacklist()
    bl["sites"].pop(idx)
    app_data["blacklist"] = bl
    save_data()
    if session_state == "WORKING":
        write_hosts(bl["sites"])
    render_blacklist()


def render_blacklist():
    if not dpg.does_item_exist("bl_app_list"):
        return
    dpg.delete_item("bl_app_list",  children_only = True)
    dpg.delete_item("bl_site_list", children_only = True)
    bl = get_blacklist()
    for i, app in enumerate(bl["apps"]):
        with dpg.group(horizontal = True, parent = "bl_app_list"):
            dpg.add_text(app, color = list(C_TEXT))
            dpg.add_button(label = " × ", small = True,
                           callback = lambda s, a, u: remove_banned_app(u),
                           user_data = i)
            dpg.bind_item_theme(dpg.last_item(), "delete_btn")
    for i, site in enumerate(bl["sites"]):
        with dpg.group(horizontal = True, parent = "bl_site_list"):
            dpg.add_text(site, color = list(C_TEXT))
            dpg.add_button(label = " × ", small = True,
                           callback = lambda s, a, u: remove_banned_site(u),
                           user_data = i)
            dpg.bind_item_theme(dpg.last_item(), "delete_btn")




def ear_score(lm, indices, w, h):
    p  = [(lm[i].x * w, lm[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    hz = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * hz + 1e-6)


def put_metric(frame, label, value_str, x, y, col):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label,     (x, y),          font, 0.8, col, 2)
    (lw, _), _ = cv2.getTextSize(label, font, 0.8, 2)
    cv2.putText(frame, value_str, (x + lw + 4, y), font, 0.8, col, 2)


def process_frame(frame):
    global gaze_status, flash_overlay, away_since, servo_rotated, last_sent_secs, session_failed

    h, w     = frame.shape[:2]
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB,
                        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result   = detector.detect(mp_image)

    looking_away = False
    eyes_closed  = False
    status       = "Locked in and mogging"

    if result.face_landmarks:
        lm     = result.face_landmarks[0]
        matrix = np.array(result.facial_transformation_matrixes[0])
        yaw    = np.degrees(np.arctan2(matrix[0][2], matrix[2][2]))
        pitch  = np.degrees(np.arctan2(-matrix[1][2], matrix[2][2]))

        lx         = lm[LEFT_EYE[0]].x * w
        rx         = lm[LEFT_EYE[1]].x * w
        ix         = np.mean([lm[i].x * w for i in LEFT_IRIS])
        iris_ratio = (ix - lx) / (rx - lx + 1e-6)

        avg_ear = (ear_score(lm, LEFT_EYE_EAR, w, h) +
                   ear_score(lm, RIGHT_EYE_EAR, w, h)) / 2.0

        if avg_ear < EAR_THRESHOLD:
            eyes_closed = True
            status = "WAKE UP!!"
        elif (abs(yaw) > YAW_THRESHOLD
              or pitch < -PITCH_UP_THRESHOLD
              or pitch > PITCH_DOWN_THRESHOLD
              or not (IRIS_RATIO_MIN < iris_ratio < IRIS_RATIO_MAX)):
            looking_away = True
            status = "FOCUS!!"

        alert    = (looking_away or eyes_closed) and session_state == "WORKING"
        ol_color = (0, 40, 200) if alert else (0, 180, 80)

        oval = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in FACE_OVAL], np.int32)
        cv2.polylines(frame, [oval], False, ol_color, 2)
        for idx in LEFT_EAR_PTS + RIGHT_EAR_PTS:
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (160, 150, 0), -1)
        for ig in [LEFT_IRIS, RIGHT_IRIS]:
            cx_ = int(np.mean([lm[i].x * w for i in ig]))
            cy_ = int(np.mean([lm[i].y * h for i in ig]))
            cv2.circle(frame, (cx_, cy_), 4, (240, 240, 240), -1)
            cv2.circle(frame, (cx_, cy_), 4, (0, 0, 0), 1)

        yaw_col   = (0, 80, 255) if abs(yaw) > YAW_THRESHOLD                                    else (0, 200, 80)
        pitch_col = (0, 80, 255) if pitch < -PITCH_UP_THRESHOLD or pitch > PITCH_DOWN_THRESHOLD  else (0, 200, 80)
        iris_col  = (0, 80, 255) if not (IRIS_RATIO_MIN < iris_ratio < IRIS_RATIO_MAX)           else (0, 200, 80)
        ear_col   = (0, 80, 255) if avg_ear < EAR_THRESHOLD                                      else (0, 200, 80)

        put_metric(frame, "Yaw:",    f"{yaw:.1f}",        8, 26,  yaw_col)
        put_metric(frame, "Pitch:",  f"{pitch:.1f}",      8, 56,  pitch_col)
        put_metric(frame, "Iris H:", f"{iris_ratio:.2f}", 8, 86,  iris_col)
        put_metric(frame, "EAR:",    f"{avg_ear:.2f}",    8, 116, ear_col)
    else:
        looking_away = True
        status = "NO FACE DETECTED"

    status_col  = (0, 200, 80) if not (looking_away or eyes_closed) else (0, 80, 255)
    (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(frame, status, (w // 2 - sw // 2, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_col, 2)

    gaze_status = status

    if session_state == "WORKING":
        distracted = looking_away or eyes_closed
        if distracted:
            if away_since is None:
                away_since = time.time()
            elapsed   = time.time() - away_since
            remaining = max(0.0, COUNTDOWN_SECS - elapsed)
            in_flash  = remaining <= FLASH_SECS

            int_rem = int(remaining)
            if int_rem != last_sent_secs:
                send(f"T{int_rem:03d}\n".encode())
                last_sent_secs = int_rem

            flash_overlay = in_flash and (int(time.time() * 2) % 2 == 0)

            cx_, cy_ = w // 2, h // 2
            r_       = min(w, h) // 6
            frac     = max(0.0, remaining / COUNTDOWN_SECS)
            sweep    = int(359 * frac)
            cv2.ellipse(frame, (cx_, cy_), (r_, r_), -90, 0, 359,   (40, 40, 80),  10)
            if sweep > 0:
                cv2.ellipse(frame, (cx_, cy_), (r_, r_), -90, 0, sweep, (0, 40, 210), 10)
            mins_, secs_ = int(remaining) // 60, int(remaining) % 60
            txt_         = f"{mins_:02d}:{secs_:02d}"
            (tw, th), _  = cv2.getTextSize(txt_, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, txt_, (cx_ - tw // 2, cy_ + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 255), 3)

            if remaining <= 0 and not servo_rotated:
                send(b'X\n')
                send(b'1\n')
                servo_rotated = True
                flash_overlay = False
                time.sleep(SERVO_OPEN_SECS)  # give the servo time to fully open
                session_failed = True
        else:
            if away_since is not None:
                send(b'X\n')
                if servo_rotated:
                    send(b'0\n')
            away_since     = None
            servo_rotated  = False
            last_sent_secs = -1
            flash_overlay  = False
    else:
        if away_since is not None:
            send(b'X\n')
        away_since     = None
        last_sent_secs = -1
        flash_overlay  = False

    if flash_overlay:
        overlay    = np.zeros_like(frame)
        overlay[:] = (0, 0, 170)
        frame      = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

    return frame


def camera_thread():
    global camera_running
    cap = cv2.VideoCapture(0)
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame     = cv2.resize(frame, (CAM_W, CAM_H))
        processed = process_frame(frame)
        if not frame_queue.full():
            frame_queue.put(processed)
    cap.release()


def frame_to_texture(frame):
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) / 255.0).ravel()




def arc_pts(cx, cy, r, start_deg, end_deg, steps = 80):
    pts = []
    for i in range(steps + 1):
        t = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        pts.append([cx + r * math.cos(t), cy + r * math.sin(t)])
    return pts


def centered_draw_text(parent, cx, y, text, color, size):
    approx_w = len(text) * size * 0.52
    dpg.draw_text([cx - approx_w / 2, y], text, color = color, size = size, parent = parent)


def redraw_wheel():
    if not dpg.does_item_exist("wheel_draw"):
        return
    dpg.delete_item("wheel_draw", children_only = True)

    today_day  = DAYS[date.today().weekday()]
    target_sec = app_data["day_targets"].get(today_day, 6.0) * 3600
    done_sec   = today_work_sec
    remaining  = max(0.0, target_sec - done_sec)
    frac       = remaining / target_sec if target_sec > 0 else 0.0

    cx, cy, r = 150, 155, 100
    dpg.draw_circle([cx, cy], r, color = C_DIM, thickness = 14, parent = "wheel_draw")
    if frac > 0.005:
        color = C_GREEN if frac > 0.3 else (C_ORANGE if frac > 0.1 else C_RED)
        end_d = -90 + (-360 * frac)
        pts   = arc_pts(cx, cy, r, -90, end_d if frac < 1.0 else -90 - 359.9)
        if len(pts) > 1:
            dpg.draw_polyline(pts, color = color, thickness = 14, parent = "wheel_draw")

    rem_h    = int(remaining) // 3600
    rem_m    = (int(remaining) % 3600) // 60
    centered_draw_text("wheel_draw", cx, cy - 20, f"{rem_h}h {rem_m:02d}m", C_TEXT,  22)
    centered_draw_text("wheel_draw", cx, cy + 12, "remaining",               C_MUTED, 13)

    done_h = int(done_sec) // 3600
    done_m = (int(done_sec) % 3600) // 60
    tgt_h  = int(target_sec) // 3600
    if dpg.does_item_exist("wheel_sub"):
        dpg.set_value("wheel_sub", f"{done_h}h {done_m:02d}m done  ·  {tgt_h}h target today")

    streak     = get_streak()
    streak_col = C_GREEN if streak >= 7 else (C_ORANGE if streak >= 3 else C_BRIGHT)
    if dpg.does_item_exist("streak_number"):
        dpg.configure_item("streak_number", default_value = str(streak), color = list(streak_col))
        if has_title_font:
            dpg.bind_item_font("streak_number", "big_font")
    if dpg.does_item_exist("streak_sublabel"):
        dpg.configure_item("streak_sublabel", color = list(streak_col))
        if has_title_font:
            dpg.bind_item_font("streak_sublabel", "streak_font")


def redraw_session_bar():
    if not dpg.does_item_exist("sess_bar_draw"):
        return
    dpg.delete_item("sess_bar_draw", children_only = True)
    W = 480
    dpg.draw_rectangle([0, 0], [W, 5], color = (0,0,0,0), fill = C_BORDER, parent = "sess_bar_draw")
    if session_state == "WORKING" and session_total > 0:
        frac  = session_remaining / session_total
        color = C_GREEN if frac > 0.3 else (C_ORANGE if frac > 0.1 else C_RED)
        dpg.draw_rectangle([0,0],[int(W*frac),5], color = (0,0,0,0), fill = color, parent = "sess_bar_draw")
    elif session_state == "BREAK" and break_total > 0:
        frac = break_remaining / break_total
        dpg.draw_rectangle([0,0],[int(W*frac),5], color = (0,0,0,0), fill = C_ACCENT, parent = "sess_bar_draw")

    if total_session_secs > 0 and dpg.does_item_exist("total_prog_draw"):
        dpg.delete_item("total_prog_draw", children_only = True)
        frac2 = min(1.0, total_elapsed_secs / total_session_secs)
        dpg.draw_rectangle([0,0],[W,5], color = (0,0,0,0), fill = C_BORDER, parent = "total_prog_draw")
        dpg.draw_rectangle([0,0],[int(W*frac2),5], color = (0,0,0,0), fill = C_ACCENT, parent = "total_prog_draw")


def update_cycle_label():
    if not dpg.does_item_exist("cycle_label"):
        return
    if total_session_secs <= 0 or session_state == "IDLE":
        dpg.set_value("cycle_label", "")
        return
    total_rem  = max(0.0, total_session_secs - total_elapsed_secs)
    th = int(total_rem) // 3600
    tm = (int(total_rem) % 3600) // 60
    ts = int(total_rem) % 60
    cycle_len    = (work_min_val + break_min_val) * 60
    total_cycles = int(total_session_secs / cycle_len) if cycle_len > 0 else 0
    rem_str = f"{th}h {tm:02d}m remaining" if th > 0 else f"{tm:02d}m {ts:02d}s remaining"
    dpg.set_value("cycle_label", f"Cycle {cycles_completed + 1}/{total_cycles}  ·  {rem_str}")




def render_todos():
    if not dpg.does_item_exist("todo_list"):
        return
    dpg.delete_item("todo_list", children_only = True)
    today = date.today()

    for i, item in enumerate(app_data["todos"]):
        with dpg.group(horizontal = True, parent = "todo_list"):
            dpg.add_checkbox(default_value = item["done"],
                             callback = lambda s, a, u: toggle_todo(u),
                             user_data = i)
            dpg.add_text(item["text"], color = list(C_MUTED if item["done"] else C_TEXT))
            due = item.get("due", "9999-12-31")
            if due != "9999-12-31":
                try:
                    due_date  = datetime.strptime(due, "%Y-%m-%d").date()
                    days_diff = (due_date - today).days
                    if days_diff < 0:
                        badge_col, label = C_RED,           f" {abs(days_diff)}d overdue "
                    elif days_diff == 0:
                        badge_col, label = C_ORANGE,        " due today "
                    elif days_diff == 1:
                        badge_col, label = (200,170,0,255), " tomorrow "
                    else:
                        badge_col, label = C_BORDER,        f" {days_diff}d left "
                except ValueError:
                    badge_col, label = C_BORDER, f" {due} "
                dpg.add_button(label = label, small = True, callback = lambda: None)
                dpg.bind_item_theme(dpg.last_item(), make_badge_theme(badge_col))
            dpg.add_button(label = " × ", small = True,
                           callback = lambda s, a, u: delete_todo(u), user_data = i)
            dpg.bind_item_theme(dpg.last_item(), "delete_btn")


def make_badge_theme(color):
    tag = f"badge_{color[0]}_{color[1]}_{color[2]}"
    if not dpg.does_item_exist(tag):
        with dpg.theme(tag = tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        color)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  color)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   color)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  3)
    return tag


def toggle_todo(idx):
    app_data["todos"][idx]["done"] = not app_data["todos"][idx]["done"]
    app_data["todos"].sort(key = lambda t: (t["done"], t.get("due", "9999-12-31")))
    save_data()
    render_todos()


def delete_todo(idx):
    app_data["todos"].pop(idx)
    save_data()
    render_todos()


def due_date_filter(sender, app_data_val, user_data):
    raw    = dpg.get_value(sender)
    digits = ''.join(c for c in raw if c.isdigit())[:8]
    result = ""
    for i, ch in enumerate(digits):
        result += ch
        if i == 3 or i == 5:
            result += "-"
    dpg.set_value(sender, result)


def add_todo_cb():
    text = dpg.get_value("todo_text_input").strip()
    if not text:
        return
    due = dpg.get_value("todo_due_input").strip()
    try:
        datetime.strptime(due, "%Y-%m-%d")
    except ValueError:
        due = "9999-12-31"
    app_data["todos"].append({"text": text, "due": due, "done": False})
    app_data["todos"].sort(key = lambda t: (t["done"], t.get("due", "9999-12-31")))
    save_data()
    dpg.set_value("todo_text_input", "")
    dpg.set_value("todo_due_input",  "")
    render_todos()




def set_preset(work_min, break_min):
    dpg.set_value("work_min_input",  work_min)
    dpg.set_value("break_min_input", break_min)


def no_break_changed(sender, app_data_val):
    show = not app_data_val
    dpg.configure_item("break_min_input", show = show)
    dpg.configure_item("break_min_label", show = show)
    dpg.configure_item("work_min_input",  show = show)
    dpg.configure_item("work_min_label",  show = show)


def start_session():
    global session_state, session_total, session_remaining, work_min_val, break_min_val
    global break_total, break_remaining, session_paused, last_tick_time
    global total_session_secs, total_elapsed_secs, cycles_completed

    if session_state == "IDLE":
        wm                 = float(dpg.get_value("work_min_input"))
        bm                 = float(dpg.get_value("break_min_input")) if not dpg.get_value("no_break_check") else 0.0
        total_m            = float(dpg.get_value("total_mins_input"))
        work_min_val       = wm
        break_min_val      = bm
        session_total      = wm * 60
        session_remaining  = wm * 60
        break_total        = bm * 60
        break_remaining    = bm * 60
        total_session_secs = total_m * 60
        total_elapsed_secs = 0.0
        cycles_completed   = 0
        session_state      = "WORKING"
        session_paused     = False
        last_tick_time     = time.time()
        write_hosts(get_blacklist()["sites"])
    elif session_state == "BREAK":
        session_state     = "WORKING"
        session_remaining = session_total
        session_paused    = False
        last_tick_time    = time.time()
        write_hosts(get_blacklist()["sites"])

    refresh_session_ui()


def pause_session():
    global session_paused, last_tick_time
    if session_state in ("WORKING", "BREAK"):
        session_paused = not session_paused
        if not session_paused:
            last_tick_time = time.time()
        dpg.configure_item("pause_btn", label = "Resume" if session_paused else "Pause")


def stop_session():
    global session_state, session_paused, away_since, last_sent_secs, flash_overlay
    global total_session_secs, total_elapsed_secs, cycles_completed

    session_state      = "IDLE"
    session_paused     = False
    away_since         = None
    last_sent_secs     = -1
    flash_overlay      = False
    total_session_secs = 0.0
    total_elapsed_secs = 0.0
    cycles_completed   = 0
    send(b'X\n')
    send(b'0\n')
    clear_hosts()
    save_data()
    refresh_session_ui()


def refresh_session_ui():
    s      = session_state
    active = s != "IDLE"
    locked = dpg.get_value("no_terminate_check")

    dpg.configure_item("session_config_group", show = not active)
    dpg.configure_item("session_active_group", show = active)

    dpg.configure_item("no_terminate_check", enabled = not active)

    if s == "IDLE":
        dpg.configure_item("start_btn", label = "Initiate",      enabled = True)
    elif s == "BREAK":
        dpg.configure_item("start_btn", label = "Skip recovery", enabled = True)
    else:
        dpg.configure_item("start_btn", label = "Initiate",      enabled = False)

    dpg.configure_item("pause_btn", enabled = active)
    if not active:
        dpg.configure_item("pause_btn", label = "Pause")

    dpg.configure_item("stop_btn", show = not locked)

    if active:
        if s == "WORKING":
            dpg.set_value("state_label", "Active")
            dpg.configure_item("state_label", color = list(C_GREEN))
        elif s == "BREAK":
            dpg.set_value("state_label", "Recovery")
            dpg.configure_item("state_label", color = list(C_ACCENT))

    update_cycle_label()


def save_day_target(day, tag):
    try:
        val = float(dpg.get_value(tag))
        app_data["day_targets"][day] = max(0.1, val)
        save_data()
        redraw_wheel()
    except Exception:
        pass




def tick():
    global session_state, session_remaining, break_remaining
    global last_tick_time, session_paused, today_work_sec
    global total_elapsed_secs, cycles_completed, session_failed

    if session_failed:
        session_failed = False
        stop_session()
        return

    if not session_paused and session_state in ("WORKING", "BREAK"):
        now = time.time()
        dt  = now - (last_tick_time or now)
        last_tick_time = now

        if session_state == "WORKING":
            session_remaining  = max(0.0, session_remaining - dt)
            today_work_sec    += dt
            total_elapsed_secs += dt
            m, s               = divmod(int(session_remaining), 60)
            dpg.set_value("timer_label", f"{m:02d}:{s:02d}")
            dpg.configure_item("timer_label", color = list(C_GREEN))
            redraw_wheel()
            update_cycle_label()

            if session_remaining <= 0:
                save_data()
                cycles_completed += 1
                no_break   = dpg.get_value("no_break_check")
                total_done = total_session_secs > 0 and total_elapsed_secs >= total_session_secs
                if total_done or no_break:
                    session_state = "IDLE"
                else:
                    session_state   = "BREAK"
                    break_remaining = break_total
                    last_tick_time  = time.time()
                    clear_hosts()
                refresh_session_ui()

        elif session_state == "BREAK":
            break_remaining    = max(0.0, break_remaining - dt)
            total_elapsed_secs += dt
            m, s               = divmod(int(break_remaining), 60)
            dpg.set_value("timer_label", f"{m:02d}:{s:02d}")
            dpg.configure_item("timer_label", color = list(C_ACCENT))
            update_cycle_label()

            if break_remaining <= 0:
                total_done = total_session_secs > 0 and total_elapsed_secs >= total_session_secs
                if total_done:
                    session_state = "IDLE"
                else:
                    session_state     = "WORKING"
                    session_remaining = session_total
                    last_tick_time    = time.time()
                    write_hosts(get_blacklist()["sites"])
                refresh_session_ui()

    redraw_session_bar()

    try:
        frame = frame_queue.get_nowait()
        tex   = frame_to_texture(frame)
        dpg.set_value("cam_texture", tex)
        dpg.set_value("gaze_label", gaze_status)
    except queue.Empty:
        pass

    if dpg.does_item_exist("wheel_draw") and session_state == "IDLE":
        redraw_wheel()




def setup_themes():
    global has_title_font
    with dpg.font_registry():
        font_paths = [
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        loaded = False
        for fp in font_paths:
            try:
                dpg.add_font(fp, 15, tag = "ui_font")
                dpg.add_font(fp, 16, tag = "title_font")
                dpg.add_font(fp, 48, tag = "big_font")
                dpg.add_font(fp, 18, tag = "streak_font")
                loaded = True
                break
            except Exception:
                continue
        has_title_font = loaded

    with dpg.theme(tag = "global_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        C_BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         C_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (40, 40, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,          C_BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    C_BRIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_Text,            C_TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_Header,          C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     C_BG)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   C_BORDER)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   12, 12)

    with dpg.theme(tag = "green_btn"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,         C_GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (20, 220, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (10, 140,  90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,           (0, 0, 0, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)

    with dpg.theme(tag = "red_btn"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,         C_RED)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (255, 80, 110, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (180, 30,  60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,           (255, 255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)

    with dpg.theme(tag = "accent_btn"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,         C_ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (110, 115, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (70,  74, 200, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)

    with dpg.theme(tag = "delete_btn"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,         C_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  C_RED)
            dpg.add_theme_color(dpg.mvThemeCol_Text,           C_MUTED)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  4)

    dpg.bind_theme("global_theme")
    if has_title_font:
        dpg.bind_font("ui_font")


def add_title(label):
    t = dpg.add_text(label, color = list(C_TITLE))
    if has_title_font:
        dpg.bind_item_font(t, "title_font")
    dpg.add_separator()




def build_ui():
    with dpg.texture_registry():
        default_tex = np.zeros((CAM_H, CAM_W, 4), dtype = np.float32).ravel()
        dpg.add_dynamic_texture(CAM_W, CAM_H, default_tex, tag = "cam_texture")

    with dpg.window(tag = "main", no_title_bar = True, no_resize = True,
                    no_move = True, no_scrollbar = True):
        with dpg.group(horizontal = True):

            with dpg.group():
                with dpg.child_window(width = 660, height = 520, border = True):
                    add_title("Face scanner")
                    dpg.add_image("cam_texture", width = 640, height = 460)
                    dpg.add_text("", tag = "gaze_label", color = list(C_MUTED))

                with dpg.child_window(width = 660, height = 315, border = True):
                    add_title("Session")

                    with dpg.group(tag = "session_config_group"):
                        dpg.add_spacer(height = 4)
                        with dpg.group(horizontal = True):
                            p1 = dpg.add_button(label = "52/17 - DeskTime method",
                                                callback = lambda: set_preset(52, 17))
                            dpg.bind_item_theme(p1, "accent_btn")
                            p2 = dpg.add_button(label = "90/20 - Ultradian rhythm",
                                                callback = lambda: set_preset(90, 20))
                            dpg.bind_item_theme(p2, "accent_btn")
                        dpg.add_spacer(height = 8)
                        with dpg.group(horizontal = True):
                            dpg.add_text("Total min", color = list(C_MUTED))
                            dpg.add_input_float(tag = "total_mins_input", default_value = 120,
                                                width = 60, step = 0, format = "%.0f")
                            dpg.add_spacer(width = 10)
                            dpg.add_text("Work min", color = list(C_MUTED), tag = "work_min_label")
                            dpg.add_input_float(tag = "work_min_input", default_value = 90,
                                                width = 60, step = 0, format = "%.0f")
                            dpg.add_spacer(width = 10)
                            dpg.add_text("Break min", color = list(C_MUTED), tag = "break_min_label")
                            dpg.add_input_float(tag = "break_min_input", default_value = 20,
                                                width = 60, step = 0, format = "%.0f")
                            dpg.add_spacer(width = 10)
                            dpg.add_checkbox(label = "No break", tag = "no_break_check",
                                             callback = no_break_changed)

                    with dpg.group(tag = "session_active_group", show = False):
                        dpg.add_spacer(height = 4)
                        dpg.add_text("", tag = "cycle_label", color = list(C_MUTED))
                        dpg.add_text("", tag = "state_label", color = list(C_MUTED))
                        dpg.add_text("--:--", tag = "timer_label", color = list(C_TEXT))
                        dpg.add_spacer(height = 2)
                        dpg.add_text("Session", color = list(C_MUTED))
                        dpg.add_drawlist(width = 480, height = 6, tag = "sess_bar_draw")
                        dpg.add_spacer(height = 4)
                        dpg.add_text("Total", color = list(C_MUTED))
                        dpg.add_drawlist(width = 480, height = 6, tag = "total_prog_draw")

                    dpg.add_spacer(height = 10)
                    with dpg.group(horizontal = True):
                        sb = dpg.add_button(label = "Initiate", tag = "start_btn",
                                            width = 110, callback = start_session)
                        dpg.bind_item_theme(sb, "green_btn")
                        dpg.add_button(label = "Pause", tag = "pause_btn",
                                       width = 90, callback = pause_session, enabled = False)
                        tb = dpg.add_button(label = "Terminate", tag = "stop_btn",
                                            width = 110, callback = stop_session)
                        dpg.bind_item_theme(tb, "red_btn")
                        dpg.add_spacer(width = 10)
                        dpg.add_checkbox(label = "No terminate", tag = "no_terminate_check",
                                         callback = lambda s, a: refresh_session_ui())

            with dpg.group():
                with dpg.child_window(width = 500, height = 520, border = True):
                    add_title("Daily work goal")
                    dpg.add_spacer(height = 8)
                    with dpg.group(horizontal = True):
                        dpg.add_drawlist(width = 300, height = 320, tag = "wheel_draw")
                        dpg.add_spacer(width = 20)
                        with dpg.group():
                            dpg.add_spacer(height = 108)
                            dpg.add_text("0", tag = "streak_number", color = list(C_BRIGHT))
                            dpg.add_spacer(height = 4)
                            dpg.add_text("day streak", tag = "streak_sublabel", color = list(C_MUTED))
                    dpg.add_spacer(height = 6)
                    dpg.add_text("", tag = "wheel_sub", color = list(C_MUTED))
                    dpg.add_spacer(height = 8)
                    today_idx = date.today().weekday()
                    with dpg.group(horizontal = True):
                        for i, d in enumerate(DAYS):
                            with dpg.group():
                                dpg.add_text(d, color = list(C_BRIGHT if i == today_idx else C_MUTED))
                                dpg.add_input_float(
                                    tag = f"day_target_{d}",
                                    default_value = app_data["day_targets"].get(d, 6.0),
                                    width = 52, step = 0, format = "%.1f",
                                    callback = lambda s, a, u: save_day_target(u[0], u[1]),
                                    user_data = (d, f"day_target_{d}")
                                )

                with dpg.child_window(width = 500, height = 315, border = True):
                    add_title("To-do")
                    with dpg.child_window(height = 225, border = False):
                        with dpg.group(tag = "todo_list"):
                            pass
                    dpg.add_separator()
                    with dpg.group(horizontal = True):
                        dpg.add_input_text(tag = "todo_text_input", hint = "New task",
                                           width = 200, on_enter = True, callback = add_todo_cb)
                        dpg.add_input_text(tag = "todo_due_input", hint = "YYYYMMDD",
                                           width = 100, callback = due_date_filter)
                        a = dpg.add_button(label = " + ", callback = add_todo_cb)
                        dpg.bind_item_theme(a, "accent_btn")

            with dpg.group():
                with dpg.child_window(width = 220, height = 840, border = True):
                    add_title("Blocklist")
                    dpg.add_spacer(height = 4)
                    dpg.add_text("Website blocking requires", color = list(C_MUTED))
                    dpg.add_text("running as administrator.", color = list(C_MUTED))
                    dpg.add_spacer(height = 10)

                    dpg.add_text("Applications", color = list(C_BRIGHT))
                    dpg.add_separator()
                    dpg.add_spacer(height = 4)
                    with dpg.child_window(height = 250, border = False):
                        with dpg.group(tag = "bl_app_list"):
                            pass
                    with dpg.group(horizontal = True):
                        dpg.add_input_text(tag = "new_app_input", hint = "app.exe",
                                           width = 110, on_enter = True, callback = add_banned_app)
                        ba = dpg.add_button(label = " + ", callback = add_banned_app)
                        dpg.bind_item_theme(ba, "accent_btn")
                        bb = dpg.add_button(label = "Browse", callback = pick_app_from_shortcut)
                        dpg.bind_item_theme(bb, "accent_btn")

                    dpg.add_spacer(height = 14)
                    dpg.add_text("Websites", color = list(C_BRIGHT))
                    dpg.add_separator()
                    dpg.add_spacer(height = 4)
                    with dpg.child_window(height = 260, border = False):
                        with dpg.group(tag = "bl_site_list"):
                            pass
                    with dpg.group(horizontal = True):
                        dpg.add_input_text(tag = "new_site_input", hint = "website.pizza",
                                           width = 148, on_enter = True, callback = add_banned_site)
                        bs = dpg.add_button(label = " + ", callback = add_banned_site)
                        dpg.bind_item_theme(bs, "accent_btn")




if __name__ == "__main__":
    load_data()

    dpg.create_context()
    dpg.create_viewport(title = "faceScanner", width = 1450, height = 880,
                        min_width = 1100, min_height = 700)

    setup_themes()
    build_ui()
    render_todos()
    render_blacklist()
    redraw_wheel()
    refresh_session_ui()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)

    threading.Thread(target = camera_thread,       daemon = True).start()
    threading.Thread(target = blacklist_thread_fn, daemon = True).start()

    while dpg.is_dearpygui_running():
        tick()
        dpg.render_dearpygui_frame()

    camera_running = False
    time.sleep(0.2)
    clear_hosts()
    save_data()
    send(b'X\n')
    send(b'0\n')
    if ARDUINO:
        try:
            arduino.close()
        except Exception:
            pass
    dpg.destroy_context()