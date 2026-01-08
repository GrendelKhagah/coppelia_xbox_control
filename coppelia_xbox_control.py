# =========================
# coppelia_xbox_control.py
# by Taylor Ketterling, 2026
#
# Xbox controller -> CoppeliaSim quadcopter target control (ZMQ Remote API)
#
# How it works (high-level):
#   1) Read controller axes/buttons with pygame (left stick: translation, right stick: yaw+alt).
#   2) Convert body-frame commands (forward/strafe) into world-frame velocity using current yaw.
#   3) Apply simple safety filters from proximity sensors:
#        - Front sensor: scale/stop forward motion near obstacles (piecewise linear brake).
#        - Altimeter: block descent when too close to ground (ground margin).
#   4) Move /target (a dummy/waypoint) and set its yaw each control tick.
#   5) Optional "flip" (smooth 180° yaw) overrides manual control while active.
#   6) Optional SEARCH_ON signal disables writing /target so a sim-side script can take over.
#
# Requirements:
#   pip install pygame
#   pip install coppeliasim-zmqremoteapi-client
#
# CoppeliaSim:
#   Ensure the ZMQ Remote API server is running:
#     Tools -> Remote API server -> ZMQ
#
# Scene assumptions (default Quadcopter model commonly works):
#   - a "/target" object to move (dummy)
#   - a "/Quadcopter/base" object for drone pose
#   - a "/Quadcopter/light" object for light toggle
#   - a "/Quadcopter/altimeter" proximity sensor pointing down
#   - a "/Quadcopter/frontSensor" proximity sensor pointing forward
#
# Crash logs:
#   ~/Desktop/logs/controller_crash.log  (fallback: ./logs/controller_crash.log)
# =========================

from __future__ import annotations

import math
import platform
import signal
import sys
import time
import traceback
import logging
import faulthandler
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pygame
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from logging.handlers import RotatingFileHandler


# =========================
# Configuration / Tuning
# =========================

@dataclass(frozen=True)
class Constants:
    """Small constants that otherwise become 'magic numbers'."""
    DT_MAX_S: float = 0.05                 # clamp dt to avoid big jumps on lag spikes
    LOOP_EXCEPT_SLEEP_S: float = 0.5       # pause after a loop exception
    RECONNECT_SLEEP_S: float = 1.0         # pause after a failed reconnect attempt
    SENSOR_POLL_JITTER_EPS: float = 1e-6   # numeric safety epsilon
    FLIP_RADIUS_EPS: float = 1e-4          # consider radius ~0 if below this
    RESET_ABOVE_Z_M: float = 1.0           # reset target 1m above base
    MIN_SAFE_Z_M: float = 1.0              # never reset below this


@dataclass
class Tuning:
    """
    Runtime tuning values.
    Most operations are O(1) per loop; only calibration loops do sampling work.

    Notes:
      - loop_sleep_s controls the nominal control rate (~1/loop_sleep_s).
      - sensor_hz controls the sensor polling rate; we poll sensors less frequently
        than control updates to reduce sim load and reduce remote API call frequency.
    """
    # Controller shaping
    deadzone: float = 0.12

    # Motion
    xy_speed_mps: float = 1.2      # left stick translation speed (m/s)
    z_speed_mps: float = 0.9       # altitude climb speed (m/s)
    yaw_rate_rps: float = 2.0      # yaw rotation speed (rad/s)

    # Target constraints
    min_z: float = 0.2
    max_z: float = 5.0

    # Timing
    loop_sleep_s: float = 0.02     # ~50 Hz control loop (reduces jitter/load)
    sensor_hz: float = 20.0        # sensor polling rate (Hz)
    debug_period_s: float = 0.5

    # Sensor safety behavior
    ground_margin_m: float = 0.35  # block descent if closer than this to ground
    slow_dist_m: float = 0.8       # start slowing forward motion at this distance
    stop_dist_m: float = 0.3       # stop forward motion at/inside this distance

    # Logging/verbosity
    verbose_console: bool = True   # keep calibration prompts + periodic prints


@dataclass(frozen=True)
class AxisMapping:
    lx: int
    ly: int
    rx_yaw: int   # right stick left/right -> yaw
    ry_alt: int   # right stick up/down   -> altitude


@dataclass(frozen=True)
class ButtonMapping:
    a_flip: int = 0      # A
    b_reset: int = 1     # B
    x_light: int = 2     # X
    y_search: int = 3    # Y toggles SEARCH_ON signal


@dataclass
class PrevButtons:
    a: int = 0
    b: int = 0
    x: int = 0
    y: int = 0


@dataclass
class SensorCache:
    front_hit: bool = False
    front_dist: float = float("inf")
    alt_hit: bool = False
    alt_dist: float = float("inf")
    accum_s: float = 0.0


@dataclass
class Handles:
    target: int
    base: int
    light: int
    alt_sensor: int
    front_sensor: int


@dataclass
class FlipState:
    active: bool = False
    t: float = 0.0
    duration: float = 0.9
    base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    z: float = 0.0
    radius: float = 0.0
    angle0: float = 0.0
    yaw0: float = 0.0
    yaw1: float = 0.0


@dataclass
class LoopState:
    yaw: float = 0.0
    light_enabled: bool = True
    flip: FlipState = field(default_factory=FlipState)
    prev: PrevButtons = field(default_factory=PrevButtons)
    sensors: SensorCache = field(default_factory=SensorCache)
    last_t: float = 0.0
    last_debug_t: float = 0.0
    reconnects: int = 0


CFG = Tuning()
BTN = ButtonMapping()
K = Constants()

# Optional: lock in known axis indices (set to None to auto-calibrate)
MANUAL_LX: Optional[int] = None
MANUAL_LY: Optional[int] = None
MANUAL_RX_YAW: Optional[int] = None
MANUAL_RY_ALT: Optional[int] = None


# =========================
# Utility helpers
# =========================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_deadzone(x: float, deadzone: float) -> float:
    return 0.0 if abs(x) < deadzone else x


def pump() -> None:
    """Ensure pygame updates controller state."""
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


# =========================
# Crash logging / diagnostics
# =========================

_LOGGER: Optional[logging.Logger] = None
_LOG_PATH: Optional[Path] = None
_FAULT_FH: Optional[Any] = None  # file-like

# Lightweight breadcrumb context (keep values JSON-serializable and small).
RUNTIME_CTX: Dict[str, Any] = {
    "phase": "startup",
    "last_sim_call": None,
    "last_sim_args": None,
    "last_sim_kwargs": None,
    "reconnects": 0,
}


def get_default_log_path() -> Path:
    """
    User requested: ~/Desktop/logs/controller_crash.log
    Fallback: ./logs/controller_crash.log
    """
    home = Path.home()
    desktop = home / "Desktop"
    if desktop.exists():
        return desktop / "logs" / "controller_crash.log"
    return Path.cwd() / "logs" / "controller_crash.log"


def setup_logging() -> logging.Logger:
    """
    Initialize rotating file logger + console logger.
    Also enables faulthandler into the same log file when supported.

    On some Windows builds, faulthandler.register may not exist; we guard it.
    """
    global _LOGGER, _LOG_PATH, _FAULT_FH

    if _LOGGER is not None:
        return _LOGGER

    # 1) Compute log path first
    _LOG_PATH = get_default_log_path()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 2) Open a dedicated file handle for faulthandler (keep it open)
    if _FAULT_FH is None:
        _FAULT_FH = open(_LOG_PATH, "a", encoding="utf-8")

    logger = logging.getLogger("coppelia_xbox_control")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File (rotating)
    fh = RotatingFileHandler(
        _LOG_PATH,
        maxBytes=2_000_000,   # 2 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console (info+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Header block
    logger.info("=" * 72)
    logger.info("Starting coppelia_xbox_control.py")
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", platform.platform())
    try:
        logger.info("pygame: %s", pygame.version.ver)
        logger.info("faulthandler module: %r", getattr(faulthandler, "__file__", "<built-in?>"))
        logger.info("faulthandler has register: %s", hasattr(faulthandler, "register"))
    except Exception:
        pass
    logger.info("Log file: %s", str(_LOG_PATH))
    logger.info("=" * 72)

    # Enable faulthandler (best-effort)
    try:
        faulthandler.enable(file=_FAULT_FH, all_threads=True)

        if hasattr(faulthandler, "register"):
            # Windows: SIGTERM is not always meaningful; keep this guarded.
            if hasattr(signal, "SIGTERM"):
                try:
                    faulthandler.register(signal.SIGTERM, file=_FAULT_FH, all_threads=True)
                except Exception:
                    logger.warning("faulthandler.register(SIGTERM) not supported here", exc_info=True)

        logger.info("faulthandler enabled")
    except Exception:
        logger.warning("faulthandler not available / not fully supported on this environment", exc_info=True)

    # Unhandled exception hook
    def _excepthook(exc_type, exc, tb):
        logger.critical("UNHANDLED EXCEPTION. ctx=%s", RUNTIME_CTX)
        logger.critical("".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = _excepthook

    _LOGGER = logger
    return logger


def set_ctx(**kwargs: Any) -> None:
    """Update breadcrumb context (kept small)."""
    RUNTIME_CTX.update(kwargs)


def log_exception(where: str, exc: BaseException) -> None:
    logger = setup_logging()
    logger.exception("Exception in %s. ctx=%s", where, RUNTIME_CTX)


def sim_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Wrapper for sim.* calls so we know which call triggered a crash.
    Stores the last call in RUNTIME_CTX for post-mortem.
    """
    setup_logging()
    set_ctx(
        last_sim_call=getattr(fn, "__name__", str(fn)),
        last_sim_args=args,
        last_sim_kwargs=kwargs,
    )
    return fn(*args, **kwargs)


def close_logging() -> None:
    global _FAULT_FH
    try:
        if _FAULT_FH:
            _FAULT_FH.flush()
            _FAULT_FH.close()
    except Exception:
        pass


# =========================
# CoppeliaSim helpers
# =========================

def connect_sim() -> Any:
    client = RemoteAPIClient()
    return client.require("sim")


def sim_is_running(sim: Any) -> bool:
    try:
        return sim_call(sim.getSimulationState) == sim.simulation_advancing_running
    except Exception:
        return False


def require_one(sim: Any, paths: Iterable[str]) -> int:
    """
    Try multiple paths, return first handle found, else raise with helpful info.
    ZMQ Remote API typically throws if object doesn't exist.

    Complexity: O(P) for P candidate paths (tiny).
    """
    last_err: Optional[BaseException] = None
    for p in paths:
        try:
            return int(sim_call(sim.getObject, p))
        except Exception as e:
            last_err = e
    raise RuntimeError(f"None of these objects exist: {list(paths)}. Last error: {last_err}")


# Data-driven handle resolution: easy to adjust for different scenes/models. used so many "potential Paths"
# because i kept running into different naming conventions in different CoppeliaSim quadcopter models.
HANDLE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "target": ("/target", "/Quadcopter/base/target", "/Quadcopter/target"),
    "base": ("/Quadcopter/base", "/Quadcopter"),
    "light": ("/Quadcopter/light", "/light"),
    "alt_sensor": ("/Quadcopter/altimeter", "/Quadcopter/altSensor", "/altimeter"),
    "front_sensor": ("/Quadcopter/frontSensor", "/Quadcopter/front", "/frontSensor"),
}


def connect_all() -> Tuple[Any, Handles]:
    """Connect to sim + resolve required handles."""
    sim = connect_sim()

    resolved: Dict[str, int] = {}
    for name, paths in HANDLE_CANDIDATES.items():
        resolved[name] = require_one(sim, paths)

    handles = Handles(
        target=resolved["target"],
        base=resolved["base"],
        light=resolved["light"],
        alt_sensor=resolved["alt_sensor"],
        front_sensor=resolved["front_sensor"],
    )
    return sim, handles


# =========================
# Axis discovery / calibration
# =========================

def first_existing_axis(joy: pygame.joystick.Joystick, candidates: Iterable[int]) -> Optional[int]:
    n = joy.get_numaxes()
    for ax in candidates:
        if ax < n:
            return ax
    return None


def measure_resting_abs(
    joy: pygame.joystick.Joystick,
    axes: Iterable[int],
    seconds: float = 0.6
) -> Dict[int, float]:
    """
    Measure avg(abs(value)) per axis while user is not touching sticks.

    Complexity:
      O(S * A) where:
        - S = number of samples taken over 'seconds' (seconds / 0.01)
        - A = number of axes in 'axes'
    """
    end = time.perf_counter() + seconds
    axes = list(axes)
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


def capture_peak_delta(
    joy: pygame.joystick.Joystick,
    axes: Iterable[int],
    prompt: str,
    seconds: float = 0.9
) -> Dict[int, float]:
    """
    Ask user to flick in a direction; return the peak delta per axis.

    Complexity: O(S * A).
    """
    print(prompt)
    pump()
    axes = list(axes)
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


def calibrate_right_stick_axes(joy: pygame.joystick.Joystick) -> Tuple[int, int]:
    """
    Explicitly map:
      - yaw axis  = responds most when you flick RIGHT
      - alt axis  = responds most when you flick UP

    Purpose: prevent yaw being tied to up/down (or vice-versa) on different controllers.

    Complexity: O(S * A) sampling over a short duration; A is small (<= 4).
    """
    n = joy.get_numaxes()
    candidates = [ax for ax in (2, 3, 4, 5) if ax < n]
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
    return int(yaw_axis), int(alt_axis)


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

    return AxisMapping(lx=int(lx), ly=int(ly), rx_yaw=int(rx_yaw), ry_alt=int(ry_alt))


# =========================
# Controller IO
# =========================

def read_axes(joy: pygame.joystick.Joystick, mapping: AxisMapping, deadzone: float) -> Tuple[float, float, float, float]:
    """
    Read stick axes and apply deadzone.

    Note on sign conventions:
      - Many controllers report 'up' as negative on Y axes.
      - We handle sign later when forming commands so +forward/+climb feels intuitive.
    Complexity: O(1).
    """
    lx = apply_deadzone(joy.get_axis(mapping.lx), deadzone)
    ly = apply_deadzone(joy.get_axis(mapping.ly), deadzone)
    rx = apply_deadzone(joy.get_axis(mapping.rx_yaw), deadzone)
    ry = apply_deadzone(joy.get_axis(mapping.ry_alt), deadzone)
    return lx, ly, rx, ry


# =========================
# Motion math + safety filters
# =========================

def body_to_world_velocity(forward: float, strafe_right: float, yaw: float, xy_speed_mps: float) -> Tuple[float, float]:
    """
    Convert body-frame commands to world-frame velocities.

    Intuition (rotation matrix):
      [vx_w]   [ cos(y)  -sin(y) ] [vx_b]
      [vy_w] = [ sin(y)   cos(y) ] [vy_b]

    Complexity: O(1).
    """
    vx_body = forward * xy_speed_mps
    vy_body = -strafe_right * xy_speed_mps

    vx_world = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
    vy_world = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
    return vx_world, vy_world


def apply_front_obstacle_brake(
    forward_cmd: float,
    front_hit: bool,
    front_dist: float,
    slow_dist: float,
    stop_dist: float
) -> float:
    """
    Only affects forward motion (positive forward_cmd).

    Piecewise behavior:
      - If no hit: unchanged
      - If distance <= stop_dist: block forward motion
      - If stop_dist < distance < slow_dist: linearly scale down (0..1)

    Complexity: O(1).
    """
    if not front_hit:
        return forward_cmd

    if front_dist <= stop_dist and forward_cmd > 0:
        return 0.0

    if front_dist < slow_dist and forward_cmd > 0:
        scale = (front_dist - stop_dist) / max((slow_dist - stop_dist), K.SENSOR_POLL_JITTER_EPS)
        scale = clamp(scale, 0.0, 1.0)
        return forward_cmd * scale

    return forward_cmd


def apply_ground_margin(climb_cmd: float, alt_hit: bool, alt_dist: float, margin: float) -> float:
    """
    If too close to ground, block descending (negative climb).
    Complexity: O(1).
    """
    if alt_hit and alt_dist < margin and climb_cmd < 0:
        return 0.0
    return climb_cmd


# =========================
# Target + special actions
# =========================

def reset_target_above_drone(sim: Any, handles: Handles) -> float:
    """Teleport target above drone base and align yaw (reset waypoint)."""
    base_pos = sim_call(sim.getObjectPosition, handles.base, -1)
    safe_z = max(float(base_pos[2]) + K.RESET_ABOVE_Z_M, K.MIN_SAFE_Z_M)
    sim_call(sim.setObjectPosition, handles.target, -1, [float(base_pos[0]), float(base_pos[1]), safe_z])

    yaw = float(sim_call(sim.getObjectOrientation, handles.base, -1)[2])
    sim_call(sim.setObjectOrientation, handles.target, -1, [0.0, 0.0, yaw])
    return yaw


def start_flip(sim: Any, handles: Handles, flip: FlipState, duration: float = 0.9) -> None:
    """Start smooth 180° yaw rotation."""
    base_pos_raw = sim_call(sim.getObjectPosition, handles.base, -1)
    tgt_pos_raw = sim_call(sim.getObjectPosition, handles.target, -1)

    base_pos = (float(base_pos_raw[0]), float(base_pos_raw[1]), float(base_pos_raw[2]))
    tgt_pos = (float(tgt_pos_raw[0]), float(tgt_pos_raw[1]), float(tgt_pos_raw[2]))

    dx = tgt_pos[0] - base_pos[0]
    dy = tgt_pos[1] - base_pos[1]
    radius = math.hypot(dx, dy)
    angle0 = math.atan2(dy, dx) if radius > K.FLIP_RADIUS_EPS else 0.0

    yaw0 = float(sim_call(sim.getObjectOrientation, handles.target, -1)[2])
    yaw1 = normalize_angle(yaw0 + math.pi)

    flip.active = True
    flip.t = 0.0
    flip.duration = max(0.1, float(duration))
    flip.base_pos = base_pos
    flip.z = tgt_pos[2]
    flip.radius = radius
    flip.angle0 = angle0
    flip.yaw0 = yaw0
    flip.yaw1 = yaw1


def update_flip(sim: Any, handles: Handles, flip: FlipState, dt: float) -> Tuple[bool, Optional[float]]:
    """Advance flip animation. Returns (active, yaw_if_updated)."""
    if not flip.active:
        return False, None

    flip.t += dt
    T = flip.duration
    u = min(1.0, flip.t / T)
    u = smoothstep(u)

    base_pos = flip.base_pos
    z = flip.z

    if flip.radius > K.FLIP_RADIUS_EPS:
        ang = flip.angle0 + math.pi * u
        x = base_pos[0] + flip.radius * math.cos(ang)
        y = base_pos[1] + flip.radius * math.sin(ang)
        sim_call(sim.setObjectPosition, handles.target, -1, [x, y, z])

    yaw = normalize_angle(lerp(flip.yaw0, flip.yaw1, u))
    sim_call(sim.setObjectOrientation, handles.target, -1, [0.0, 0.0, yaw])

    if u >= 1.0:
        flip.active = False
        return False, yaw

    return True, yaw


def set_light_enabled(sim: Any, light_handle: int, enabled: bool) -> bool:
    """
    easy topggle light.
    Enable/disable a light object. Returns True on success.
    """
    state = 1 if enabled else 0

    try:
        sim_call(sim.setLightParameters, light_handle, state, None, None, None)
        return True
    except Exception:
        pass

    return False


# =========================
# Sensor helpers
# =========================

def read_proximity(sim: Any, sensor_handle: int) -> Tuple[bool, float]:
    """Read a proximity sensor; return (hit, distance_m)."""
    r, d, _, _, _ = sim_call(sim.readProximitySensor, sensor_handle)
    hit = (r > 0)
    return hit, float(d) if hit else float("inf")


# =========================
# SEARCH_ON signal helpers
# =========================

def toggle_search_on(sim: Any) -> bool:
    """
    Toggle the SEARCH_ON int signal (0/1).
    When SEARCH_ON=1, this controller becomes read-only and does NOT write /target.
    Returns the new state (True=on, False=off).
    """
    cur = sim_call(sim.getInt32Signal, "SEARCH_ON") or 0
    new = 0 if int(cur) == 1 else 1
    sim_call(sim.setInt32Signal, "SEARCH_ON", int(new))
    return bool(new)


def is_search_on(sim: Any) -> bool:
    """Return whether SEARCH_ON signal is currently set."""
    cur = sim_call(sim.getInt32Signal, "SEARCH_ON") or 0
    return int(cur) == 1


# =========================
# Main control loop helpers
# =========================

def init_pygame_controller() -> pygame.joystick.Joystick:
    """Initialize pygame and return the first detected joystick."""
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected by pygame.")

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print(f"Controller: {joy.get_name()}")
    print(f"Axes: {joy.get_numaxes()} | Buttons: {joy.get_numbuttons()}")
    return joy


def configure_sim_sensors(sim: Any, handles: Handles) -> None:
    """
    Disable explicit handling on sensors (best-effort).
    Do not add scripts to sensors in the sim scene.
    """
    try:
        sim_call(sim.setExplicitHandling, handles.alt_sensor, 0)
        sim_call(sim.setExplicitHandling, handles.front_sensor, 0)
    except Exception:
        pass


def maybe_poll_sensors(sim: Any, handles: Handles, state: LoopState, dt: float) -> None:
    """
    Poll sensors at a fixed rate (sensor_hz).
    Was crashing occasionally when polling every loop due to sim load.
    Complexity: O(1) per call; O(Hz) over time.
    """
    set_ctx(phase="read_sensors")

    sensor_period = 1.0 / CFG.sensor_hz
    state.sensors.accum_s += dt
    if state.sensors.accum_s < sensor_period:
        return

    state.sensors.accum_s = 0.0
    state.sensors.alt_hit, state.sensors.alt_dist = read_proximity(sim, handles.alt_sensor)
    state.sensors.front_hit, state.sensors.front_dist = read_proximity(sim, handles.front_sensor)


def update_target(sim: Any, handles: Handles, state: LoopState, axis_map: AxisMapping, joy: pygame.joystick.Joystick, dt: float) -> Tuple[float, float, float, float]:
    """
    Manual control update when not in flip and not in SEARCH_ON mode.
    Returns the raw axes (lx, ly, rx, ry) for optional debug.
    """
    set_ctx(phase="read_sticks")

    lx, ly, rx_yaw, ry_alt = read_axes(joy, axis_map, CFG.deadzone)

    # pygame Y axes: up is typically negative; invert so +forward/+climb are intuitive.
    forward_cmd = -ly
    strafe_cmd = lx
    climb_cmd = -ry_alt

    # Apply safety filters
    forward_cmd = apply_front_obstacle_brake(
        forward_cmd,
        state.sensors.front_hit, state.sensors.front_dist,
        CFG.slow_dist_m, CFG.stop_dist_m
    )
    climb_cmd = apply_ground_margin(
        climb_cmd,
        state.sensors.alt_hit, state.sensors.alt_dist,
        CFG.ground_margin_m
    )

    # Convert to world-space velocity
    vx, vy = body_to_world_velocity(forward_cmd, strafe_cmd, state.yaw, CFG.xy_speed_mps)

    set_ctx(phase="update_target")

    pos_raw = sim_call(sim.getObjectPosition, handles.target, -1)
    if pos_raw is None:
        raise RuntimeError("getObjectPosition returned None")
    pos = list(pos_raw)
    if len(pos) != 3:
        raise RuntimeError(f"getObjectPosition returned {pos!r}")

    pos[0] = float(pos[0]) + vx * dt
    pos[1] = float(pos[1]) + vy * dt
    pos[2] = clamp(float(pos[2]) + climb_cmd * CFG.z_speed_mps * dt, CFG.min_z, CFG.max_z)

    set_ctx(
        last_axes={"lx": lx, "ly": ly, "rx_yaw": rx_yaw, "ry_alt": ry_alt},
        last_cmd={"forward": forward_cmd, "strafe": strafe_cmd, "climb": climb_cmd},
        last_pos=pos,
    )

    sim_call(sim.setObjectPosition, handles.target, -1, pos)

    # Update yaw
    state.yaw = normalize_angle(state.yaw + (-rx_yaw) * CFG.yaw_rate_rps * dt)
    sim_call(sim.setObjectOrientation, handles.target, -1, [0.0, 0.0, state.yaw])

    return lx, ly, rx_yaw, ry_alt


def print_debug(state: LoopState, axes: Tuple[float, float, float, float]) -> None:
    """
    Print periodic debug info to console. Helps user see their inputs + sensor states.
    Complexity: O(1).
    """
    lx, ly, rx, ry = axes
    print(
        f"axes: lx={lx:+.2f} ly={ly:+.2f} yaw(rx)={rx:+.2f} alt(ry)={ry:+.2f} | "
        f"front: hit={int(state.sensors.front_hit)} d={state.sensors.front_dist if state.sensors.front_hit else -1:.2f} | "
        f"alt: hit={int(state.sensors.alt_hit)} d={state.sensors.alt_dist if state.sensors.alt_hit else -1:.2f}"
    )

# =========================
# Main loop
# =========================

def main() -> None:
    setup_logging()
    set_ctx(phase="init")

    joy = init_pygame_controller()
    axis_map = resolve_axis_mapping(joy)

    print(f"Final mapping -> LX:{axis_map.lx} LY:{axis_map.ly} YAW(RX):{axis_map.rx_yaw} ALT(RY):{axis_map.ry_alt}")
    print("Controls: Left stick = move | Right stick X = yaw | Right stick Y = altitude")
    print("Buttons: A = flip (smooth) | B = reset target | X = toggle light | Y = toggle SEARCH_ON | Ctrl+C to quit")

    sim, handles = connect_all()
    configure_sim_sensors(sim, handles)

    state = LoopState()
    state.yaw = float(sim_call(sim.getObjectOrientation, handles.base, -1)[2])
    state.last_t = time.perf_counter()
    state.last_debug_t = 0.0

    state.light_enabled = True
    set_light_enabled(sim, handles.light, state.light_enabled)

    """
    Main control loop.
    1) Read controller inputs
    2) Update target position/orientation unless in SEARCH_ON mode
    3) Handle special actions (flip, reset, light toggle)
    4) Poll sensors at a fixed rate
    5) Sleep to limit loop rate
    """
    while True:
        try:
            set_ctx(phase="loop_top", reconnects=state.reconnects)
            pump()

            # Edge-trigger buttons
            a_now = 1 if joy.get_button(BTN.a_flip) else 0
            b_now = 1 if joy.get_button(BTN.b_reset) else 0
            x_now = 1 if joy.get_button(BTN.x_light) else 0
            y_now = 1 if joy.get_button(BTN.y_search) else 0

            if not sim_is_running(sim):
                time.sleep(0.1)
                continue

            # Y toggles SEARCH_ON
            if y_now and not state.prev.y:
                new_state = toggle_search_on(sim)
                print(f"SEARCH_ON = {new_state}")
            state.prev.y = y_now

            # If in SEARCH_ON mode, skip target updates
            if is_search_on(sim):
                time.sleep(CFG.loop_sleep_s)
                continue

            # Timing 
            now = time.perf_counter()
            dt = clamp(now - state.last_t, 0.0, K.DT_MAX_S)
            state.last_t = now

            # X toggles light
            if x_now and not state.prev.x:
                state.light_enabled = not state.light_enabled
                ok = set_light_enabled(sim, handles.light, state.light_enabled)
                print(f"Light enabled = {state.light_enabled} (method_ok={ok})")
            state.prev.x = x_now

            # B resets target
            if b_now and not state.prev.b:
                state.yaw = reset_target_above_drone(sim, handles)
            state.prev.b = b_now

            # A triggers flip
            if a_now and not state.prev.a and not state.flip.active:
                start_flip(sim, handles, state.flip, duration=0.9)
            state.prev.a = a_now

            active, flip_yaw = update_flip(sim, handles, state.flip, dt)
            if active:
                if flip_yaw is not None:
                    state.yaw = float(flip_yaw)
                time.sleep(CFG.loop_sleep_s)
                continue

            maybe_poll_sensors(sim, handles, state, dt)

            axes = update_target(sim, handles, state, axis_map, joy, dt)

            if CFG.verbose_console and (now - state.last_debug_t > CFG.debug_period_s):
                state.last_debug_t = now
                print_debug(state, axes)

            time.sleep(CFG.loop_sleep_s)            
        except Exception as e:
            """
            Errors be gone! jk but serious - catch and log all exceptions to avoid silent crashes.
            These crashes were driving me nuts so hopefully this helps diagnose any future issues.
            Just remember to clear the log if you are actually using it for real debugging later.
            """
            log_exception("main_loop", e)
            time.sleep(K.LOOP_EXCEPT_SLEEP_S)

            # Attempt reconnect
            try:
                state.reconnects += 1
                set_ctx(phase="reconnect", reconnects=state.reconnects)

                sim, handles = connect_all()
                configure_sim_sensors(sim, handles)

                state.yaw = float(sim_call(sim.getObjectOrientation, handles.base, -1)[2])
            except Exception as e2:
                log_exception("reconnect", e2)
                time.sleep(K.RECONNECT_SLEEP_S)
            continue


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        setup_logging().info("Stopped by user (Ctrl+C).")
    except Exception as e:
        log_exception("top_level", e)
        raise
    finally:
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        close_logging()
