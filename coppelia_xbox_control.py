# =========================
# coppelia_xbox_control.py
# by Taylor Ketterling, 2026
# Simple Xbox controller -> CoppeliaSim quadcopter target control
# Uses pygame for controller input and CoppeliaSim ZMQ Remote API
#
# in CappeliaSim, ensure the ZMQ Remote API server is running (Tools -> Remote API server -> ZMQ)
#
# Requires:
#   pip install pygame
#   pip install coppeliasim-zmqremoteapi-client
# this program assumes a quadcopter model with:
#   - a "/target" object to move (e.g. a dummy)
#   - a "/Quadcopter/base" object for drone position/orientation
#   - a "/Quadcopter/light" object for light toggle
#   - a "/Quadcopter/altimeter" proximity sensor pointing down
#   - a "/Quadcopter/frontSensor" proximity sensor pointing forward
# You can use the "Quadcopter" model from CoppeliaSim's model browser
# Usage:
#   - Run CoppeliaSim with the above model and ZMQ server enabled
#   - Run this script
#   - Use left stick to move, right stick X to yaw, right stick Y to altitude
#   - A button = smooth 180 flip
#   - B button = reset target above drone (go up)
#   - X button = toggle light
#
# =========================

import time
import math
import pygame
import traceback
import os
from datetime import datetime
from dataclasses import dataclass
from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # pip install coppeliasim-zmqremoteapi-client


# =========================
# Configuration / Tuning
# =========================

@dataclass
class Tuning:
    deadzone: float = 0.12

    xy_speed_mps: float = 1.2      # Left stick translation speed (m/s)
    z_speed_mps: float = 0.9       # Altitude climb speed (m/s)
    yaw_rate_rps: float = 2.0      # Yaw rotation speed (rad/s)

    min_z: float = 0.2
    max_z: float = 5.0
    loop_sleep_s: float = 0.02     # ruducing to 50Hz for stability

    sensor_hz: float = 20.0
    
    # Sensor behavior
    ground_margin_m: float = 0.35   # don't descend if altimeter says you're below this height (meters)
    slow_dist_m: float = 0.8        # begin slowing forward motion when obstacle closer than this (meters)
    stop_dist_m: float = 0.3        # block forward motion when obstacle closer than this (meters)

    debug_period_s: float = 0.5


@dataclass
class AxisMapping:
    lx: int
    ly: int
    rx_yaw: int   # Right stick left/right -> yaw
    ry_alt: int   # Right stick up/down   -> altitude


@dataclass
class ButtonMapping:
    a_flip: int = 0      # A
    b_reset: int = 1     # B
    x_light: int = 2     # X (toggle light)


CFG = Tuning()
BTN = ButtonMapping()

# Optional: lock in known axis indices (set to None to auto-calibrate)
MANUAL_LX = None
MANUAL_LY = None
MANUAL_RX_YAW = None
MANUAL_RY_ALT = None


# =========================
# Small utility helpers
# =========================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def apply_deadzone(x: float, deadzone: float) -> float:
    return 0.0 if abs(x) < deadzone else x

def pump():
    """Make sure pygame updates controller state."""
    pygame.event.pump()

def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, +pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    """Smoothstep (0..1 -> 0..1)."""
    return t * t * (3 - 2 * t)

# some reason everything is crashing periodically randomly, putting a logger here.
def log_exception(e: Exception):
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "controller_crash.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write(datetime.now().isoformat() + "\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
    print(f"[!] Logged crash to {path}")

def sim_is_running(sim):
    try:
        return sim.getSimulationState() == sim.simulation_advancing_running
    except Exception:
        return False

def safe_get(sim, path):
    h = sim.getObject(path)
    if h == -1:
        raise RuntimeError(f"Object not found: {path}")
    return h

def connect_all():
    sim = connect_sim()
    return sim, {
        "target": safe_get(sim, "/target"),
        "base":   safe_get(sim, "/Quadcopter/base"),
        "light":  safe_get(sim, "/Quadcopter/light"),
        "alt":    safe_get(sim, "/Quadcopter/altimeter"),
        "front":  safe_get(sim, "/Quadcopter/frontSensor"),
    }

# =========================
# Axis discovery / calibration
# =========================

def first_existing_axis(joy: pygame.joystick.Joystick, candidates):
    n = joy.get_numaxes()
    for ax in candidates:
        if ax < n:
            return ax
    return None


def measure_resting_abs(joy: pygame.joystick.Joystick, axes, seconds: float = 0.6):
    """
    Measure avg(abs(value)) per axis while user is not touching sticks.
    Trigger axes often rest near +/-1 and will have large avg(abs).
    """
    end = time.perf_counter() + seconds
    sums = {ax: 0.0 for ax in axes}
    count = 0

    while time.perf_counter() < end:
        pump()
        for ax in axes:
            sums[ax] += abs(joy.get_axis(ax))
        count += 1
        time.sleep(0.01)

    count = max(count, 1)
    return {ax: sums[ax] / count for ax in axes}


def capture_peak_delta(joy: pygame.joystick.Joystick, axes, prompt: str, seconds: float = 0.9):
    """
    Ask user to flick in a direction; return the peak delta per axis.
    """
    print(prompt)
    pump()
    baseline = {ax: joy.get_axis(ax) for ax in axes}
    peaks = {ax: 0.0 for ax in axes}

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pump()
        for ax in axes:
            cur = joy.get_axis(ax)
            peaks[ax] = max(peaks[ax], abs(cur - baseline[ax]))
        time.sleep(0.01)

    return peaks


def calibrate_right_stick_axes(joy: pygame.joystick.Joystick):
    """
    Explicitly map:
      - yaw axis  = the one that responds when you flick RIGHT
      - alt axis  = the one that responds when you flick UP
    Prevents yaw being tied to up/down.
    """
    n = joy.get_numaxes()
    candidates = [ax for ax in [2, 3, 4, 5] if ax < n]
    if len(candidates) < 2:
        raise RuntimeError("Not enough axes to calibrate the right stick.")

    resting = measure_resting_abs(joy, candidates, seconds=0.6)
    rs_axes = sorted(candidates, key=lambda a: resting[a])[:2]

    print(
        f"Right-stick candidates: {rs_axes} "
        f"(resting abs: {resting[rs_axes[0]]:.2f}, {resting[rs_axes[1]]:.2f})"
    )

    peaks_right = capture_peak_delta(
        joy, rs_axes,
        "Calibration: flick RIGHT stick to the RIGHT, then release.",
        seconds=0.9
    )
    peaks_up = capture_peak_delta(
        joy, rs_axes,
        "Calibration: flick RIGHT stick UP, then release.",
        seconds=0.9
    )

    yaw_axis = max(rs_axes, key=lambda a: peaks_right[a])
    alt_axis = rs_axes[0] if rs_axes[1] == yaw_axis else rs_axes[1]

    # Sanity: altitude should respond more to UP flick
    if peaks_up[alt_axis] < peaks_up[yaw_axis]:
        alt_axis = max(rs_axes, key=lambda a: peaks_up[a])
        yaw_axis = rs_axes[0] if rs_axes[1] == alt_axis else rs_axes[1]

    print(f"Calibrated mapping: YAW axis={yaw_axis} (L/R), ALT axis={alt_axis} (U/D)")
    return yaw_axis, alt_axis

# =========================
# Resolve final axis mapping
# =========================
def resolve_axis_mapping(joy: pygame.joystick.Joystick) -> AxisMapping:
    lx = MANUAL_LX if MANUAL_LX is not None else first_existing_axis(joy, [0])
    ly = MANUAL_LY if MANUAL_LY is not None else first_existing_axis(joy, [1])
    if lx is None or ly is None:
        raise RuntimeError("Could not resolve left stick axes (expected axes 0 and 1).")

    if MANUAL_RX_YAW is not None and MANUAL_RY_ALT is not None:
        rx_yaw = MANUAL_RX_YAW
        ry_alt = MANUAL_RY_ALT
        print(f"Using MANUAL right-stick mapping -> yaw={rx_yaw}, alt={ry_alt}")
    else:
        rx_yaw, ry_alt = calibrate_right_stick_axes(joy)

    return AxisMapping(lx=lx, ly=ly, rx_yaw=rx_yaw, ry_alt=ry_alt)


# =========================
# CoppeliaSim helpers
# =========================

def connect_sim():
    client = RemoteAPIClient()
    return client.require("sim")

# =========================
# Read axes with deadzone applied
# =========================
def read_axes(joy: pygame.joystick.Joystick, mapping: AxisMapping, deadzone: float):
    lx = apply_deadzone(joy.get_axis(mapping.lx), deadzone)
    ly = apply_deadzone(joy.get_axis(mapping.ly), deadzone)
    rx = apply_deadzone(joy.get_axis(mapping.rx_yaw), deadzone)  # yaw only
    ry = apply_deadzone(joy.get_axis(mapping.ry_alt), deadzone)  # altitude only
    return lx, ly, rx, ry

# =========================
# Body to world velocity conversion
# =========================
def body_to_world_velocity(forward: float, strafe_right: float, yaw: float, xy_speed_mps: float):
    """
    forward/strafe are stick commands in body space.
    Convert to world-space velocities using current yaw.
    """
    vx_body = forward * xy_speed_mps
    vy_body = -strafe_right * xy_speed_mps

    vx_world = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
    vy_world = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
    return vx_world, vy_world

# =========================
# Target reset helper
# =========================
def reset_target_above_drone(sim, target, base):
    """Teleport /target above drone base and align yaw (reset waypoint)."""
    base_pos = sim.getObjectPosition(base, -1)
    safe_z = max(base_pos[2] + 1.0, 1.0)
    sim.setObjectPosition(target, -1, [base_pos[0], base_pos[1], safe_z])

    yaw = sim.getObjectOrientation(base, -1)[2]
    sim.setObjectOrientation(target, -1, [0.0, 0.0, yaw])
    return yaw


# =========================
# Smooth 180 rotation   (flip)
# =========================

def start_flip(sim, target, base, flip_state, duration=0.9):
    base_pos = sim.getObjectPosition(base, -1)
    tgt_pos = sim.getObjectPosition(target, -1)

    dx = tgt_pos[0] - base_pos[0]
    dy = tgt_pos[1] - base_pos[1]
    radius = math.hypot(dx, dy)
    angle0 = math.atan2(dy, dx) if radius > 1e-4 else 0.0

    yaw0 = sim.getObjectOrientation(target, -1)[2]
    yaw1 = normalize_angle(yaw0 + math.pi)

    flip_state.clear()
    flip_state.update({
        "active": True,
        "t": 0.0,
        "duration": max(0.1, float(duration)),
        "base_pos": base_pos,
        "z": tgt_pos[2],
        "radius": radius,
        "angle0": angle0,
        "yaw0": yaw0,
        "yaw1": yaw1,
    })

# =========================
# Flip update step
def update_flip(sim, target, flip_state, dt):
    if not flip_state.get("active"):
        return False, None

    flip_state["t"] += dt
    T = flip_state["duration"]
    u = min(1.0, flip_state["t"] / T)
    u = smoothstep(u)

    base_pos = flip_state["base_pos"]
    z = flip_state["z"]

    if flip_state["radius"] > 1e-4:
        ang = flip_state["angle0"] + math.pi * u
        x = base_pos[0] + flip_state["radius"] * math.cos(ang)
        y = base_pos[1] + flip_state["radius"] * math.sin(ang)
        sim.setObjectPosition(target, -1, [x, y, z])

    yaw = normalize_angle(lerp(flip_state["yaw0"], flip_state["yaw1"], u))
    sim.setObjectOrientation(target, -1, [0.0, 0.0, yaw])

    if u >= 1.0:
        flip_state["active"] = False
        return False, yaw

    return True, yaw


# =========================
# Light toggle (robust)
# =========================

def set_light_enabled(sim, light_handle, enabled: bool):
    """
    CoppeliaSim light control differs a bit by version/model.
    This tries a few common approaches.
    """
    state = 1 if enabled else 0

    # 1) Try enabling/disabling via light parameters (bit-coded state)
    #    bit0 = on/off (commonly)
    try:
        # signature may vary; we keep it minimal
        sim.setLightParameters(light_handle, state, None, None, None)
        return True
    except Exception:
        pass

    # 2) Try int param if available in your sim build
    try:
        # Many builds provide sim.lightintparam_enabled
        sim.setObjectInt32Param(light_handle, sim.lightintparam_enabled, state)
        return True
    except Exception:
        pass

    # 3) Fallback: hide/unhide object (this may NOT disable lighting, only visibility)
    try:
        sim.setObjectInt32Param(light_handle, sim.objintparam_visibility_layer, 1 if enabled else 0)
        return True
    except Exception:
        pass

    return False


# =========================
# Sensor helpers
# =========================

# Read proximity sensor; return (hit: bool, distance: float)
def read_proximity(sim, sensor_handle):
    """
    Returns: (hit: bool, dist: float)
    """
    r, d, _, _, _ = sim.readProximitySensor(sensor_handle)
    return (r > 0), float(d) if r > 0 else float("inf")


# =========================
# Front obstacle helpers
def apply_front_obstacle_brake(forward_cmd: float, front_hit: bool, front_dist: float, slow_dist: float, stop_dist: float):
    """
    Only affects forward motion (positive forward_cmd).
    """
    if not front_hit:
        return forward_cmd

    if front_dist <= stop_dist and forward_cmd > 0:
        return 0.0

    if front_dist < slow_dist and forward_cmd > 0:
        # Scale down as you approach obstacle
        scale = (front_dist - stop_dist) / max((slow_dist - stop_dist), 1e-6)  # 0..1
        scale = clamp(scale, 0.0, 1.0)
        return forward_cmd * scale

    return forward_cmd


# =========================
# Altimeter helpers
def apply_ground_margin(climb_cmd: float, alt_hit: bool, alt_dist: float, margin: float):
    """
    If too close to ground, block descending (negative climb).
    """
    if alt_hit and alt_dist < margin and climb_cmd < 0:
        return 0.0
    return climb_cmd


# =========================
# Main loop
# =========================

def main():
    # ---- Pygame / controller ----
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected by pygame.")

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print(f"Controller: {joy.get_name()}")
    print(f"Axes: {joy.get_numaxes()} | Buttons: {joy.get_numbuttons()}")

    axis_map = resolve_axis_mapping(joy)
    print(f"Final mapping -> LX:{axis_map.lx} LY:{axis_map.ly} YAW(RX):{axis_map.rx_yaw} ALT(RY):{axis_map.ry_alt}")
    print("Controls: Left stick = move | Right stick X = yaw | Right stick Y = altitude")
    print("Buttons: A = flip (smooth) | B = reset target | X = toggle light | Ctrl+C to quit")

    # ---- CoppeliaSim connect ----
    sim, H = connect_all()
    target = H["target"]
    base   = H["base"]
    light  = H["light"]
    alt    = H["alt"]
    front  = H["front"]

       
    # Disable explicit handling on sensors
    try:
        sim.setExplicitHandling(alt, 0)
        sim.setExplicitHandling(front, 0)
    except Exception:
        pass

    # Initialize yaw to current drone yaw
    yaw = sim.getObjectOrientation(base, -1)[2]

    # Flip state + edge detection
    flip_state = {"active": False}

    a_prev = 0
    b_prev = 0
    x_prev = 0

    light_enabled = True
    set_light_enabled(sim, light, light_enabled)

    last_t = time.perf_counter()
    last_debug_t = 0.0

    front_hit = False
    front_dist = float("inf")
    alt_hit = False
    alt_dist = float("inf")

    sensor_period = 1.0 / CFG.sensor_hz
    sensor_accum = 0.0

    while True:
        try:
            # ---- Check if sim is running ----
            if not sim_is_running(sim):
                time.sleep(0.1)
                continue

            now = time.perf_counter()
            dt = clamp(now - last_t, 0.0, 0.05)
            last_t = now

            # ---- Pygame event pump ----
            pump()

            # ---- Buttons (edge-triggered) ----
            a_now = 1 if joy.get_button(BTN.a_flip) else 0
            b_now = 1 if joy.get_button(BTN.b_reset) else 0
            x_now = 1 if joy.get_button(BTN.x_light) else 0

            # Toggle light - X button
            if x_now and not x_prev:
                light_enabled = not light_enabled
                ok = set_light_enabled(sim, light, light_enabled)
                print(f"Light enabled = {light_enabled} (method_ok={ok})")
            x_prev = x_now

            # Reset target above drone - B button
            if b_now and not b_prev:
                yaw = reset_target_above_drone(sim, target, base)
            b_prev = b_now

            # rotate 180* - A button
            if a_now and not a_prev and not flip_state["active"]:
                start_flip(sim, target, base, flip_state, duration=0.9)
            a_prev = a_now

            # ---- If flip is running, drive target along flip path and skip manual control ----
            active, flip_yaw = update_flip(sim, target, flip_state, dt)
            if active:
                if flip_yaw is not None:
                    yaw = flip_yaw
                time.sleep(CFG.loop_sleep_s)
                continue

            # ---- Read sensors ----
            # Updated sensor reads at fixed rate, reduces load and sim crashes
            sensor_accum += dt
            if sensor_accum >= sensor_period:
                sensor_accum = 0.0  
                alt_hit, alt_dist = read_proximity(sim, alt)
                front_hit, front_dist = read_proximity(sim, front)

            # ---- Read sticks ----
            lx, ly, rx_yaw, ry_alt = read_axes(joy, axis_map, CFG.deadzone)

            # Stick commands
            forward_cmd = -ly               # + = forward
            strafe_cmd = lx                 # + = right
            climb_cmd = -ry_alt             # + = climb (stick up is typically negative)

            # Sensor safety modifiers
            forward_cmd = apply_front_obstacle_brake(
                forward_cmd,
                front_hit, front_dist,
                CFG.slow_dist_m, CFG.stop_dist_m
            )
            climb_cmd = apply_ground_margin(
                climb_cmd,
                alt_hit, alt_dist,
                CFG.ground_margin_m
            )

            # Convert to world-space velocity
            vx, vy = body_to_world_velocity(forward_cmd, strafe_cmd, yaw, CFG.xy_speed_mps)

            # Update target position
            pos = sim.getObjectPosition(target, -1)
            pos[0] += vx * dt
            pos[1] += vy * dt
            pos[2] = clamp(pos[2] + climb_cmd * CFG.z_speed_mps * dt, CFG.min_z, CFG.max_z)
            sim.setObjectPosition(target, -1, pos)

            # Update yaw
            yaw = normalize_angle(yaw + (-rx_yaw) * CFG.yaw_rate_rps * dt)
            sim.setObjectOrientation(target, -1, [0.0, 0.0, yaw])

            # Debug output
            if now - last_debug_t > CFG.debug_period_s:
                last_debug_t = now
                print(
                    f"axes: lx={lx:+.2f} ly={ly:+.2f} yaw(rx)={rx_yaw:+.2f} alt(ry)={ry_alt:+.2f} | "
                    f"front: hit={int(front_hit)} d={front_dist if front_hit else -1:.2f} | "
                    f"alt: hit={int(alt_hit)} d={alt_dist if alt_hit else -1:.2f}"
                )
            time.sleep(CFG.loop_sleep_s)
            
        except Exception as e:
            log_exception(e)
            time.sleep(0.5)
            try:
                sim, H = connect_all()
                target = H["target"]
                base   = H["base"]
                light  = H["light"]
                alt    = H["alt"]
                front  = H["front"]

                # re-sync yaw after reconnect
                yaw = sim.getObjectOrientation(base, -1)[2]
            except Exception as e2:
                log_exception(e2)
                time.sleep(1.0)
            continue

# =========================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
