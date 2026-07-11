"""
vision_hsv_grab.py — camera + ToF + serial for grab / point / wave.

Finds the colored blob, centers it, reads depth off the ToF grid, asks the
Arduino for world coords, then sends GRAB or POINT. Wave path is face-only
(no ToF). HSV ranges below are for OUR lighting — retune if your room sucks.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import serial


@dataclass(frozen=True)
class HsvRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def lower_np(self) -> np.ndarray:
        return np.array(self.lower, dtype=np.uint8)

    def upper_np(self) -> np.ndarray:
        return np.array(self.upper, dtype=np.uint8)


# OpenCV HSV: H 0-179, S/V 0-255. These worked for our demo table.
HSV_TARGETS: dict[str, HsvRange] = {
    # pink pig — kinda loose on S so it still finds it under weird lights
    "pig": HsvRange(lower=(130, 35, 60), upper=(175, 255, 255)),
    # green gumby — tighter so we don't lock onto random green junk
    "gumby": HsvRange(lower=(38, 75, 60), upper=(82, 255, 255)),
    # brown otter — keep V capped or it grabs tan/orange crap
    "otter": HsvRange(lower=(10, 80, 10), upper=(18, 255, 150)),
}


 


def normalize_target_name(text: str) -> str | None:
    """Map freeform text to one of: pig, gumby, otter."""
    if not text:
        return None
    t = re.sub(r"[^\w\s]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for key in ("pig", "gumby", "otter"):
        if re.search(rf"\b{re.escape(key)}\b", t):
            return key
    return None


def _default_arduino_port() -> str:
    if os.name == "nt":
        return "COM3"
    # Prefer udev-stable name if present.
    return "/dev/mira_arm"


def _default_tof_port() -> str:
    if os.name == "nt":
        return "COM7"
    # Prefer udev-stable name if present.
    return "/dev/mira_tof"


# Remember whichever camera actually opened last time so we don't burn
# seconds probing dead indexes every grab.
_LAST_GOOD_CAMERA: tuple[int, int] | None = None


def _try_open_camera(idx: int, backend: int) -> cv2.VideoCapture | None:
    c = cv2.VideoCapture(idx, backend)
    if not c.isOpened():
        c.release()
        return None
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(cv2.CAP_PROP_FPS, 30)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(5):
        ret, frame = c.read()
        if ret and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            return c
        time.sleep(0.02)
    c.release()
    return None


def open_camera() -> cv2.VideoCapture:
    global _LAST_GOOD_CAMERA

    # Fast path: reuse whatever worked last time in this process.
    if _LAST_GOOD_CAMERA is not None:
        idx, backend = _LAST_GOOD_CAMERA
        c = _try_open_camera(idx, backend)
        if c is not None:
            return c

    # Explicit override (useful on Jetson where /dev/video0 vs /dev/video1 changes).
    cam_override = (os.environ.get("ROBOT_CAMERA_INDEX") or "").strip()
    if cam_override:
        try:
            idx = int(cam_override)
            # Prefer V4L2 on Linux.
            backends = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
            for backend in backends:
                c = _try_open_camera(idx, backend)
                if c is not None:
                    _LAST_GOOD_CAMERA = (idx, backend)
                    return c
        except Exception:
            pass

    candidates: list[tuple[int, int]] = []
    if os.name == "nt":
        # Windows prefers DSHOW/MSMF, and camera index 1 is often the real webcam.
        for idx in (1, 0, 2, 3, 4, 5):
            candidates.append((idx, cv2.CAP_DSHOW))
            candidates.append((idx, cv2.CAP_MSMF))
            candidates.append((idx, cv2.CAP_ANY))
    else:
        # Linux / Jetson: prefer V4L2 and try index 0 first.
        cap_v4l2 = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        for idx in (0, 1, 2, 3):
            candidates.append((idx, cap_v4l2))
            candidates.append((idx, cv2.CAP_ANY))

    for idx, backend in candidates:
        c = _try_open_camera(idx, backend)
        if c is not None:
            _LAST_GOOD_CAMERA = (idx, backend)
            return c

    raise RuntimeError("Could not open a camera (try closing other camera apps).")


def _sample_tof_depth_mm(grid: np.ndarray, row: int, col: int, *, radius: int = 2) -> float:
    """
    Robust ToF sampling: take the closest valid depth in a (2*radius+1)^2 patch.
    Larger patch reduces "locked but depth=0" stalls.
    """
    rr = int(max(1, radius))
    r0, r1 = max(0, row - rr), min(8, row + rr + 1)
    c0, c1 = max(0, col - rr), min(8, col + rr + 1)
    patch = grid[r0:r1, c0:c1].flatten()
    valid = patch[(patch >= 40) & (patch <= 2500)]
    if valid.size == 0:
        return 0.0
    return float(np.min(valid))


def _detect_largest_blob_center(
    frame_bgr: np.ndarray,
    hsv_range: HsvRange,
    *,
    morph_kernel: int = 7,
    min_area_px: int = 1100,
) -> tuple[bool, int, int, np.ndarray, tuple[int, int, int, int] | None]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_range.lower_np(), hsv_range.upper_np())
    k = max(3, int(morph_kernel) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if not contours:
        return False, 0, 0, mask_bgr, None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < float(min_area_px):
        return False, 0, 0, mask_bgr, None
    m = cv2.moments(best)
    if m["m00"] <= 1e-6:
        return False, 0, 0, mask_bgr, None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    x, y, bw, bh = cv2.boundingRect(best)
    return True, cx, cy, mask_bgr, (x, y, bw, bh)


_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    c = cv2.CascadeClassifier(xml)
    if c.empty():
        raise RuntimeError(f"Failed to load face cascade at {xml}")
    _FACE_CASCADE = c
    return _FACE_CASCADE


def _detect_largest_face_center(
    frame_bgr: np.ndarray,
    *,
    min_size_px: int = 60,
) -> tuple[bool, int, int, tuple[int, int, int, int] | None]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(min_size_px, min_size_px),
    )
    if len(faces) == 0:
        return False, 0, 0, None
    best_idx = int(np.argmax([w * h for (_x, _y, w, h) in faces]))
    x, y, w, h = (int(v) for v in faces[best_idx])
    return True, x + w // 2, y + h // 2, (x, y, w, h)


def _drain_serial_lines(ser: serial.Serial, max_lines: int = 64) -> list[str]:
    lines: list[str] = []
    for _ in range(max_lines):
        if ser.in_waiting <= 0:
            break
        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if raw:
            lines.append(raw)
    return lines


def track_and_grab(
    target: str,
    *,
    arduino_port: str | None = None,
    tof_port: str | None = None,
    arduino: serial.Serial | None = None,
    mode: str = "grab",
) -> bool:
    """
    Full flow:
      HSV track target → center+stabilize → GET_COORDS(depth) → GRAB/POINT x,y,z → wait for COMPLETE.

    mode="grab"  (default): sends "GRAB ..." + CLAMP gate, waits for GRAB_COMPLETE.
    mode="point"          : sends "POINT ..." (arm only aims at the object), waits for
                            POINT_COMPLETE. CLAMP is skipped since nothing is picked up,
                            and no forward nudge is applied to the Y coord.

    Serial commands to Arduino (grab mode): TRACK, GET_COORDS, GRAB, CLAMP, ABORT.
    Serial commands to Arduino (point mode): TRACK, GET_COORDS, POINT, ABORT.

    If `arduino` is provided, that persistent serial connection is used and NOT closed
    here — this avoids the ~7s Arduino boot that otherwise happens on every open()
    because Windows pulses DTR on some chipsets regardless of dtr=False.
    """
    mode = (mode or "grab").strip().lower()
    if mode not in ("grab", "point"):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'grab' or 'point'.")
    _is_point = (mode == "point")
    _complete_token = "POINT_COMPLETE" if _is_point else "GRAB_COMPLETE"
    target_key = normalize_target_name(target) or target.strip().lower()
    if target_key not in HSV_TARGETS:
        raise ValueError(f"Unknown target {target!r}. Expected one of: {', '.join(HSV_TARGETS)}")
    hsv_range = HSV_TARGETS[target_key]

    arduino_port = (arduino_port or os.environ.get("ROBOT_SERIAL_PORT") or "").strip() or _default_arduino_port()
    tof_port = (tof_port or os.environ.get("ROBOT_TOF_PORT") or "").strip() or _default_tof_port()

    # Clean up any stale OpenCV windows from previous runs
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    cap = open_camera()

    # Use the caller-provided persistent serial if we have one; otherwise open a new
    # connection (slow first-time path — triggers Arduino reset on some Windows USB chipsets).
    owns_arduino = False
    if arduino is None or not getattr(arduino, "is_open", False):
        arduino = serial.Serial()
        arduino.port = arduino_port
        arduino.baudrate = 115200
        arduino.timeout = 0.1
        try:
            arduino.dtr = False
            arduino.rts = False
        except Exception:
            pass
        arduino.open()
        time.sleep(0.15)
        owns_arduino = True
    else:
        # Persistent connection — just use the current timeout.
        try:
            arduino.timeout = 0.1
        except Exception:
            pass

    # Tell the arm to leave HOME, smooth-move to the search pose, and start sweeping.
    # Even with dtr/rts forced low above, Windows can still give the Arduino a brief
    # DTR pulse on open which resets the board. To be robust we:
    #   1) absorb anything already in the buffer
    #   2) send START_SEARCH and wait for SEARCH_READY
    #   3) retry a few times in case the Arduino was still in setup() the first time
    def _wait_for_token(token: str, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        buf = ""
        while time.time() < deadline:
            try:
                chunk = arduino.read(64)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if token in buf:
                    return True
            else:
                time.sleep(0.02)
        return False

    try:
        arduino.reset_input_buffer()
    except Exception:
        pass

    sweep_ack = False
    # Short timeouts so that if the firmware is older (no START_SEARCH handler) we fall
    # through to opening the camera instead of stalling ~10s waiting for SEARCH_READY.
    # Arduino firmware with the handler responds in well under 200ms once awake.
    for attempt in range(3):
        try:
            arduino.write(b"START_SEARCH\n")
            arduino.flush()
        except Exception:
            pass
        timeout = 0.9 if attempt == 0 else 0.4
        if _wait_for_token("SEARCH_READY", timeout):
            sweep_ack = True
            break
    if not sweep_ack:
        # Fall through: if the firmware has no START_SEARCH handler, the first TRACK we
        # send below will make the arm enter search pose / enable sweep anyway.
        pass

    tof = serial.Serial(tof_port, 115200, timeout=0.05)

    DEADZONE = 32
    LOCK_STABLE_FRAMES = 3
    TRACK_SENS = 0.18
    TRACK_MAX_ABS = 10
    TRACK_SEND_INTERVAL_SEC = 0.06
    CLAMP_DEPTH_MAX_MM = 100
    GET_COORDS_INTERVAL_SEC = 0.25
    COORDS_READ_TIMEOUT_SEC = 1.2
    # Total Arduino GRAB sequence (Phase0 + Z-descent + glide + nudge + clamp + lift + drop + return-home)
    # is now ~17s with the smoother motions. Give it real headroom so successful grabs aren't
    # falsely reported as "aborted" by the Python side.
    GRAB_TIMEOUT_SEC = 30.0
    GRAB_LOST_TIMEOUT_SEC = 10.0
    # Safety gate (prevents self-collisions on bogus COORDS)
    SAFE_MIN_R_MM = float(os.environ.get("ROBOT_SAFE_MIN_R_MM", "140"))
    SAFE_MAX_R_MM = float(os.environ.get("ROBOT_SAFE_MAX_R_MM", "650"))
    SAFE_MIN_Y_MM = float(os.environ.get("ROBOT_SAFE_MIN_Y_MM", "80"))
    SAFE_MAX_Y_MM = float(os.environ.get("ROBOT_SAFE_MAX_Y_MM", "650"))
    SAFE_MIN_Z_MM = float(os.environ.get("ROBOT_SAFE_MIN_Z_MM", "40"))
    # Default max Z must be high enough for your COORDS frame (often reports 300–400mm).
    SAFE_MAX_Z_MM = float(os.environ.get("ROBOT_SAFE_MAX_Z_MM", "450"))

    last_good_grid = np.zeros((8, 8))
    ema_ex = 0.0
    ema_ey = 0.0
    ERROR_EMA_ALPHA = 0.25
    lock_stable = 0
    last_track_send = 0.0
    last_coords_sent = 0.0
    locked_depth_hist: deque[float] = deque(maxlen=7)
    last_valid_depth = 0.0

    grabbing = False
    grab_deadline: float | None = None
    clamp_sent = False
    last_seen = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                time.sleep(0.02)
                continue
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            h, w = frame.shape[:2]
            cx0, cy0 = w // 2, h // 2

            # Read ToF grid
            while tof.in_waiting > 0:
                try:
                    raw = tof.readline().decode("utf-8", errors="ignore").strip()
                    if raw.startswith("y") and ":" in raw:
                        row_idx = int(raw[1 : raw.find(":")])
                        vals = [int(v) for v in re.findall(r"\d+", raw)]
                        if 0 <= row_idx <= 7 and len(vals) >= 8:
                            last_good_grid[row_idx] = vals[:8]
                except (ValueError, IndexError):
                    pass

            found, obj_cx, obj_cy, mask_bgr, box = _detect_largest_blob_center(
                frame,
                hsv_range,
                morph_kernel=7,
                min_area_px=1100,
            )

            target_depth = 0.0
            if found:
                last_seen = time.time()
                error_x = obj_cx - cx0
                error_y = obj_cy - cy0
                ema_ex = (1.0 - ERROR_EMA_ALPHA) * ema_ex + ERROR_EMA_ALPHA * float(error_x)
                ema_ey = (1.0 - ERROR_EMA_ALPHA) * ema_ey + ERROR_EMA_ALPHA * float(error_y)
                # ToF sampling: prefer near the lower-middle of the blob (often aligns with ToF cells better).
                aim_x = obj_cx
                aim_y = obj_cy
                if box is not None:
                    _x0, _y0, _bw, _bh = box
                    aim_y = int(_y0 + int(_bh * 0.70))
                tof_col = int(np.clip(aim_x / (w / 8), 0, 7))
                tof_row = int(np.clip(aim_y / (h / 8), 0, 7))
                target_depth = _sample_tof_depth_mm(last_good_grid, tof_row, tof_col, radius=2)
                if target_depth >= 1.0:
                    last_valid_depth = float(target_depth)

                if box is not None:
                    x0, y0, bw, bh = box
                    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 255), 2)
                cv2.circle(frame, (obj_cx, obj_cy), 8, (255, 0, 255), 2)
            else:
                error_x = 0
                error_y = 0
                lock_stable = 0
                locked_depth_hist.clear()

            # If grabbing, only gate CLAMP + wait for GRAB_COMPLETE
            if grabbing:
                if not found and (time.time() - last_seen) >= GRAB_LOST_TIMEOUT_SEC:
                    try:
                        arduino.write(b"ABORT\n")
                        arduino.flush()
                    except Exception:
                        pass
                    grabbing = False
                    return False

                while arduino.in_waiting:
                    line = arduino.readline().decode("utf-8", errors="ignore").strip()
                    if _complete_token in line:
                        grabbing = False
                        return True

                if grab_deadline is not None and time.time() >= grab_deadline:
                    grabbing = False
                    return False

                if not clamp_sent and target_depth >= 1.0 and target_depth <= float(CLAMP_DEPTH_MAX_MM):
                    arduino.write(b"CLAMP\n")
                    arduino.flush()
                    clamp_sent = True

            # Tracking state: send TRACK, then GET_COORDS, then GRAB
            if not grabbing and found:
                in_deadzone = abs(error_x) < DEADZONE and abs(error_y) < DEADZONE
                lock_stable = (lock_stable + 1) if in_deadzone else 0
                locked = in_deadzone and lock_stable >= LOCK_STABLE_FRAMES

                now = time.time()
                if not locked:
                    if now - last_track_send >= TRACK_SEND_INTERVAL_SEC:
                        base_adj = int(-ema_ex * TRACK_SENS)
                        pitch_adj = int(ema_ey * TRACK_SENS)
                        base_adj = int(np.clip(base_adj, -TRACK_MAX_ABS, TRACK_MAX_ABS))
                        pitch_adj = int(np.clip(pitch_adj, -TRACK_MAX_ABS, TRACK_MAX_ABS))
                        arduino.write(f"TRACK {base_adj},{pitch_adj}\n".encode())
                        last_track_send = now
                else:
                    if target_depth >= 1.0:
                        locked_depth_hist.append(float(target_depth))
                    elif last_valid_depth >= 1.0:
                        # Keep the last good depth while locked; avoids deadlock when ToF cell momentarily reads 0.
                        locked_depth_hist.append(float(last_valid_depth))
                    else:
                        print("LOCKED (no depth) — adjust ToF aim/lighting/stand.")

                    if locked_depth_hist and now - last_coords_sent >= GET_COORDS_INTERVAL_SEC:
                        last_coords_sent = now
                        depth_for_coords = float(np.median(np.array(locked_depth_hist, dtype=np.float32)))
                        _drain_serial_lines(arduino)
                        arduino.write(f"GET_COORDS {float(depth_for_coords)}\n".encode())
                        arduino.flush()
                        time.sleep(0.03)

                        coords_line = None
                        read_until = time.time() + COORDS_READ_TIMEOUT_SEC
                        while time.time() < read_until:
                            if arduino.in_waiting:
                                incoming = arduino.readline().decode("utf-8", errors="ignore").strip()
                                if incoming.startswith("COORDS ") or incoming.startswith("COORDS,"):
                                    coords_line = incoming
                                    break
                            else:
                                time.sleep(0.005)

                        if coords_line:
                            parts = (
                                coords_line.replace("COORDS ", "")
                                .replace("COORDS,", "")
                                .strip()
                                .split(",")
                            )
                            if len(parts) == 3:
                                obj_x = float(parts[0])
                                obj_y = float(parts[1])
                                obj_z = float(parts[2])
                                r_mm = float((obj_x * obj_x + obj_y * obj_y) ** 0.5)
                                print(f"COORDS parsed: x={obj_x:.1f} y={obj_y:.1f} z={obj_z:.1f} (r={r_mm:.1f})")

                                safe = True
                                if not (SAFE_MIN_R_MM <= r_mm <= SAFE_MAX_R_MM):
                                    print(f"COORDS rejected: r out of range [{SAFE_MIN_R_MM:.0f},{SAFE_MAX_R_MM:.0f}]")
                                    safe = False
                                if not (SAFE_MIN_Y_MM <= obj_y <= SAFE_MAX_Y_MM):
                                    print(f"COORDS rejected: y out of range [{SAFE_MIN_Y_MM:.0f},{SAFE_MAX_Y_MM:.0f}]")
                                    safe = False
                                if not (SAFE_MIN_Z_MM <= obj_z <= SAFE_MAX_Z_MM):
                                    print(f"COORDS rejected: z out of range [{SAFE_MIN_Z_MM:.0f},{SAFE_MAX_Z_MM:.0f}]")
                                    safe = False

                                if safe:
                                    if _is_point:
                                        # No forward nudge for POINT — the Arduino firmware
                                        # scales the reach itself so we must send the
                                        # object's actual (x,y,z).
                                        arduino.write(f"POINT {obj_x},{obj_y},{obj_z}\n".encode())
                                    else:
                                        # Nudge forward ~1 inch each grab (keep Arduino firmware untouched).
                                        obj_y_nudged = float(obj_y + 25.4)
                                        arduino.write(f"GRAB {obj_x},{obj_y_nudged},{obj_z}\n".encode())
                                    arduino.flush()
                                    grabbing = True
                                    # POINT has no CLAMP step — mark as already-sent so
                                    # the CLAMP gate below is short-circuited.
                                    clamp_sent = True if _is_point else False
                                    grab_deadline = time.time() + GRAB_TIMEOUT_SEC
                                    last_seen = time.time()
                                else:
                                    # Stay in tracking; do not commit the arm to a potentially colliding grab.
                                    lock_stable = 0
                                    locked_depth_hist.clear()

            # Display (press q to abort)
            fused = cv2.addWeighted(frame, 0.88, cv2.resize(mask_bgr, (w, h)), 0.12, 0)
            cv2.drawMarker(fused, (cx0, cy0), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            _label_mode = "POINT" if _is_point else "GRAB"
            cv2.putText(
                fused,
                f"{_label_mode}: {target_key.upper()} | Dist: {int(target_depth)}mm",
                (w - 420, 50),
                1,
                1.3,
                (255, 100, 255),
                2,
            )
            # Use a constant window name to prevent duplicate/black windows.
            cv2.imshow("MIRA VISION", fused)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                try:
                    arduino.write(b"ABORT\n")
                    arduino.flush()
                except Exception:
                    pass
                return False
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if owns_arduino:
            try:
                arduino.close()
            except Exception:
                pass
        try:
            tof.close()
        except Exception:
            pass


def find_face_and_wave(
    *,
    arduino_port: str | None = None,
    timeout_sec: float = 25.0,
    arduino: serial.Serial | None = None,
) -> bool:
    """
    Acknowledgment flow:
      - Enter sweep/search pose (via TRACK auto-transition).
      - Detect a human face (Haar cascade), center it on screen with TRACK commands.
      - Once centered + stable, send WAVE and wait for WAVE_COMPLETE.
    No ToF is used — this is purely a visual/servoing gesture.
    """
    arduino_port = (arduino_port or os.environ.get("ROBOT_SERIAL_PORT") or "").strip() or _default_arduino_port()

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    cap = open_camera()

    owns_arduino = False
    if arduino is None or not getattr(arduino, "is_open", False):
        arduino = serial.Serial()
        arduino.port = arduino_port
        arduino.baudrate = 115200
        arduino.timeout = 0.1
        try:
            arduino.dtr = False
            arduino.rts = False
        except Exception:
            pass
        arduino.open()
        time.sleep(0.15)
        owns_arduino = True
    else:
        try:
            arduino.timeout = 0.1
        except Exception:
            pass

    def _wait_for_token(token: str, timeout_sec_inner: float) -> bool:
        deadline = time.time() + timeout_sec_inner
        buf = ""
        while time.time() < deadline:
            try:
                chunk = arduino.read(64)
            except Exception:
                chunk = b""
            if chunk:
                try:
                    buf += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                if token in buf:
                    return True
            else:
                time.sleep(0.02)
        return False

    try:
        arduino.reset_input_buffer()
    except Exception:
        pass

    # Ask the arm to enter FACE-search pose (pitch tilts UP during sweep to look
    # for a face rather than nodding DOWN to look for objects on a table).
    for attempt in range(3):
        try:
            arduino.write(b"START_FACE_SEARCH\n")
            arduino.flush()
        except Exception:
            pass
        t = 0.9 if attempt == 0 else 0.4
        if _wait_for_token("SEARCH_READY", t):
            break
    else:
        # Fall through: if Arduino firmware doesn't have START_FACE_SEARCH, the first
        # TRACK we send below will at least move it into search pose.
        pass

    DEADZONE = 28
    LOCK_STABLE_FRAMES = 5  # face stability before we wave
    TRACK_SENS = 0.16
    TRACK_MAX_ABS = 9
    TRACK_SEND_INTERVAL_SEC = 0.07
    WAVE_COMPLETE_TIMEOUT_SEC = 18.0

    ema_ex = 0.0
    ema_ey = 0.0
    ERROR_EMA_ALPHA = 0.30
    lock_stable = 0
    last_track_send = 0.0
    last_seen = time.time()
    deadline = time.time() + timeout_sec

    wave_sent = False
    wave_deadline: float | None = None

    try:
        while True:
            if time.time() > deadline and not wave_sent:
                try:
                    arduino.write(b"ABORT\n")
                    arduino.flush()
                except Exception:
                    pass
                return False

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                time.sleep(0.02)
                continue
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            h, w = frame.shape[:2]
            cx0, cy0 = w // 2, h // 2

            found, face_cx, face_cy, box = _detect_largest_face_center(frame)
            if found:
                last_seen = time.time()
                error_x = face_cx - cx0
                error_y = face_cy - cy0
                ema_ex = (1.0 - ERROR_EMA_ALPHA) * ema_ex + ERROR_EMA_ALPHA * float(error_x)
                ema_ey = (1.0 - ERROR_EMA_ALPHA) * ema_ey + ERROR_EMA_ALPHA * float(error_y)

                if box is not None:
                    x0, y0, bw, bh = box
                    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 255), 2)
                cv2.circle(frame, (face_cx, face_cy), 8, (0, 255, 255), 2)
            else:
                ema_ex = 0.0
                ema_ey = 0.0
                lock_stable = 0

            # While waiting for WAVE_COMPLETE, just drain serial.
            if wave_sent:
                while arduino.in_waiting:
                    line = arduino.readline().decode("utf-8", errors="ignore").strip()
                    if "WAVE_COMPLETE" in line:
                        return True
                if wave_deadline is not None and time.time() >= wave_deadline:
                    return False
            else:
                if found:
                    in_deadzone = abs(ema_ex) < DEADZONE and abs(ema_ey) < DEADZONE
                    lock_stable = (lock_stable + 1) if in_deadzone else 0
                    locked = in_deadzone and lock_stable >= LOCK_STABLE_FRAMES

                    now = time.time()
                    if not locked:
                        if now - last_track_send >= TRACK_SEND_INTERVAL_SEC:
                            base_adj = int(-ema_ex * TRACK_SENS)
                            pitch_adj = int(ema_ey * TRACK_SENS)
                            base_adj = int(np.clip(base_adj, -TRACK_MAX_ABS, TRACK_MAX_ABS))
                            pitch_adj = int(np.clip(pitch_adj, -TRACK_MAX_ABS, TRACK_MAX_ABS))
                            try:
                                arduino.write(f"TRACK {base_adj},{pitch_adj}\n".encode())
                            except Exception:
                                pass
                            last_track_send = now
                    else:
                        try:
                            arduino.write(b"WAVE\n")
                            arduino.flush()
                        except Exception:
                            pass
                        wave_sent = True
                        wave_deadline = time.time() + WAVE_COMPLETE_TIMEOUT_SEC

            # HUD
            status = "WAVING..." if wave_sent else ("LOCKED" if lock_stable >= LOCK_STABLE_FRAMES else "SEEKING FACE")
            cv2.drawMarker(frame, (cx0, cy0), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                frame,
                f"FACE | {status}",
                (w - 360, 50),
                1,
                1.3,
                (0, 255, 255),
                2,
            )
            cv2.imshow("MIRA FACE WAVE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                try:
                    arduino.write(b"ABORT\n")
                    arduino.flush()
                except Exception:
                    pass
                return False
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if owns_arduino:
            try:
                arduino.close()
            except Exception:
                pass

