"""Seed-deterministic camera trajectories (dynamic-camera.md §3).

ONE parametrization covers every family: a camera is a point on a sphere around
the body-cloud centroid, and each degree of freedom carries an optional
time-varying term. A *static* camera is simply the case where every
time-varying coefficient is zero — there is no separate static code path
(dynamic-camera.md §1, and the framing that motivated it: "static motion camera
is just a dynamic one with zero velocity").

    az(t)   = az0 + omega_az·t                        # orbit
    el(t)   = el0 + el_amp·sin(2π·f_el·t + φ_el)      # arc
    d(t)    = d0·(1 + d_amp·sin(2π·f_d·t + φ_d))      # dolly
    roll(t) = roll_amp·sin(2π·f_roll·t + φ_roll)      # THE degeneracy breaker
    look(t) = look0 + Σ_k jitter_k(t)                 # handheld / gait
    pos(t)  = look(t) + d(t)·û(az(t), el(t)) + bob(t)

Why `roll` matters enough to be its own term: gravity tilt in camera frame is
`g_cam = R_cw @ (0,0,-1)`, and a plain look-at basis has ZERO roll by
construction (`up = cross(right, forward)`), which forces `g_cam.x ≡ 0` — the
label collapses from a 2-DoF direction to a 1-parameter arc readable off the
horizon. Sampling roll is what makes the tilt target non-degenerate; camera
*motion alone does not fix it* (dynamic-camera.md §1.1, §6).

Jitter/bob are sums of sinusoids rather than filtered noise on purpose: `pose(t)`
stays a pure function of `t` with no filter state, so frames can be evaluated in
any order, resumed, or re-derived from `params` alone — which is what keeps
twins (same seed, same path) bit-reproducible (dynamic-camera.md §4).

Pure numpy: no omni/isaac import, so this module is importable and testable
outside Isaac (unlike clip_capture, which pulls Replicator at import time).

`look_at_basis` here is the roll-capable superset of
`clip_capture.world_to_camera_frame`; that legacy function is exactly the
`roll_rad=0.0` case and stays untouched for backward compatibility. When the
dyncam collection path lands, clip_capture should delegate to this (one-line
change) rather than keep two copies of the basis math.
"""

import math
from dataclasses import dataclass

import numpy as np

WORLD_UP = (0.0, 0.0, 1.0)

# --- family tunables --------------------------------------------------------
# Bands are deliberately conservative: the acceptance gate (dynamic-camera.md
# §9.4) requires egomotion-induced image flow NOT to dominate object-induced
# flow, so angular rates stay small next to typical body motion.
ORBIT_RATE_DPS = (2.0, 12.0)          # deg/s of azimuth drift (sign sampled)
ARC_AMP_DEG = (3.0, 10.0)             # elevation oscillation amplitude
ARC_FREQ_HZ = (0.05, 0.25)
DOLLY_AMP_FRAC = (0.05, 0.20)         # fraction of the framing distance
DOLLY_FREQ_HZ = (0.05, 0.20)
ROLL_AMP_DEG = (5.0, 25.0)            # dynamic-camera.md §3
ROLL_FREQ_HZ = (0.10, 0.50)
HANDHELD_COMPONENTS = 3               # sinusoids per axis
HANDHELD_AMP = (0.004, 0.020)         # m, per component
HANDHELD_FREQ_HZ = (0.3, 1.6)
GAIT_BOB_AMP = (0.010, 0.030)         # m, vertical
GAIT_BOB_FREQ_HZ = (1.5, 2.5)         # human step cadence
GAIT_ROLL_AMP_DEG = (2.0, 6.0)        # step-synced roll (half the bob rate)

FAMILIES = ("static", "orbit", "arc", "dolly", "roll", "handheld", "gait")


def look_at_basis(camera_pos, look_at, roll_rad=0.0, world_up=WORLD_UP):
    """(R_camera_world, camera_pos) for a USD/OpenGL camera, with optional roll.

    USD convention: +X right, +Y up, -Z forward. `R_camera_world` rotates a
    *vector* from world into camera coords; a *point* goes
    `p_cam = R_camera_world @ (p_world - camera_pos)`.

    `roll_rad` rotates the (right, up) pair about the forward axis. At
    roll_rad=0 this reproduces clip_capture.world_to_camera_frame exactly.
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = look_at - camera_pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError(f"Degenerate camera: look_at == camera_pos ({camera_pos.tolist()})")
    forward /= n
    # Same degenerate-camera guard as the legacy basis (critique #12): if
    # forward ∥ world_up the cross product vanishes and the basis goes NaN.
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError(
            "Degenerate camera: forward is parallel to world_up "
            f"(camera_pos={camera_pos.tolist()}, look_at={look_at.tolist()})."
        )
    right /= right_norm
    up = np.cross(right, forward)

    if roll_rad:
        # Rotate the image plane about the viewing axis. This is what puts a
        # non-zero x-component into g_cam (see module docstring).
        c, s = math.cos(roll_rad), math.sin(roll_rad)
        right, up = c * right + s * up, -s * right + c * up

    R_world_camera = np.stack([right, up, -forward], axis=1)
    return R_world_camera.T, camera_pos


def usd_rotation_xyz_deg(R_camera_world):
    """R_camera_world -> USD `rotateXYZ` Euler angles (degrees) for the camera prim.

    Needed because `rep.modify.pose(look_at=...)` CANNOT express roll — a look-at
    always rebuilds the zero-roll basis, which is exactly the degeneracy we are
    removing. So the dynamic path sets an explicit `rotation=` instead.

    USD `rotateXYZ` applies X, then Y, then Z, i.e. R = Rz·Ry·Rx acting on column
    vectors, and the camera prim's local frame IS the camera frame (+X right,
    +Y up, -Z forward). The prim rotation is therefore camera->world =
    R_camera_world.T. Decomposition (with cb = cos(pitch) ≠ 0, guaranteed by the
    elevation clamp):

        pitch = asin(-M[2,0]);  roll_x = atan2(M[2,1], M[2,2]);  yaw = atan2(M[1,0], M[0,0])
    """
    M = np.asarray(R_camera_world, dtype=np.float64).T      # camera -> world
    sy = float(np.clip(-M[2, 0], -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(M[2, 2]) < 1e-9 and abs(M[2, 1]) < 1e-9:         # gimbal lock guard
        rx, rz = 0.0, math.atan2(-M[0, 1], M[1, 1])
    else:
        rx = math.atan2(M[2, 1], M[2, 2])
        rz = math.atan2(M[1, 0], M[0, 0])
    return tuple(math.degrees(a) for a in (rx, pitch, rz))


# Replicator does not build a camera as one prim. `rep.create.camera` creates
# `Camera_Xform` (the prim `rep.modify.pose` writes to) with a CHILD `Camera`
# prim carrying a fixed `rotateXYZ=(90, 0, 90)` — the adapter from the USD camera
# frame (+Y up, -Z forward) to the Z-up stage. USD composes child-then-parent, so
# in column convention the rendered camera->world matrix is
#
#     M_world = E_parent @ C          with C = Rz(90) @ Rx(90)
#
# Posing the parent with M_target therefore renders M_target @ C — a CYCLIC ROW
# PERMUTATION of the intended basis, which reads on screen as the camera moving
# sideways rather than forward. Measured against the live stage and reproduced
# exactly by this model (diagnose_camera_pose.py, 2026-07-28).
#
# The correction is a property of Replicator's prim layout, NOT of USD's Euler
# convention — `usd_rotation_xyz_deg` above is a correct decomposition and stays
# untouched. Only the value handed to `rep.modify.pose` needs pre-multiplying.
REPLICATOR_CAMERA_CORRECTION = np.array([[0.0, 0.0, 1.0],
                                         [1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]])


def replicator_rotation_xyz_deg(R_camera_world, correction=REPLICATOR_CAMERA_CORRECTION):
    """Euler angles for `rep.modify.pose(rotation=)` on a Replicator camera.

    Solves `E @ C = R_camera_world.T` for E, i.e. decomposes `C @ R_camera_world`.
    Validated two ways: the algebra round-trips offline (verify_camera_paths.py),
    and on a live stage it reproduces the parent rotation that the known-good
    `rep.create.camera(look_at=...)` path writes for the same pose.
    """
    return usd_rotation_xyz_deg(np.asarray(correction) @ np.asarray(R_camera_world))


def euler_xyz_deg_to_rotmat(rx, ry, rz):
    """Inverse of `usd_rotation_xyz_deg` (camera->world matrix). Test hook: the
    round trip must reproduce R_camera_world.T to machine precision, otherwise
    the camera prim would be posed differently from the logged extrinsics —
    a silent, total label corruption."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _sinusoids(rng, n, amp_range, freq_range):
    """n (amplitude, frequency, phase) triples — the deterministic jitter basis."""
    return [(float(rng.uniform(*amp_range)),
             float(rng.uniform(*freq_range)),
             float(rng.uniform(0.0, 2.0 * math.pi))) for _ in range(n)]


def _eval_sinusoids(comps, t):
    return sum(a * math.sin(2.0 * math.pi * f * t + p) for a, f, p in comps)


@dataclass
class CameraPath:
    """frame index -> (camera_pos_world, look_at_world, roll_rad).

    Fully described by its fields, so `to_dict()` round-trips into meta.json as
    provenance and the path can be re-derived without re-running the sampler.
    """
    kind: str
    az0: float                      # rad
    el0_deg: float
    d0: float                       # m, framing distance
    look0: tuple                    # world-frame look-at anchor
    dt: float                       # seconds per rendered frame (render_dt)
    el_clamp_deg: tuple = (12.0, 68.0)
    d_clamp: tuple = (0.8, 4.0)
    omega_az_dps: float = 0.0
    el_amp_deg: float = 0.0
    el_freq_hz: float = 0.0
    el_phase: float = 0.0
    d_amp_frac: float = 0.0
    d_freq_hz: float = 0.0
    d_phase: float = 0.0
    roll_amp_deg: float = 0.0
    roll_freq_hz: float = 0.0
    roll_phase: float = 0.0
    jitter: tuple = ()               # 3 tuples of sinusoid components (x, y, z)
    bob_amp: float = 0.0
    bob_freq_hz: float = 0.0
    # TIME WARP — the same shot traversed `speed_scale` times faster. This model
    # is a POSITIONAL parametrization (velocity is only the derivative of
    # `pose`), so unlike the free 6-DoF model the warp is exact and needs no
    # output gain: scaling the time argument IS x(s*t). Default 1.0 leaves every
    # existing path and every stored meta.json bit-identical.
    speed_scale: float = 1.0

    def pose(self, frame_idx):
        """(camera_pos_world, look_at_world, roll_rad) at a rendered frame."""
        t = float(frame_idx) * self.dt * float(self.speed_scale)

        az = self.az0 + math.radians(self.omega_az_dps) * t
        el_deg = self.el0_deg + self.el_amp_deg * math.sin(
            2.0 * math.pi * self.el_freq_hz * t + self.el_phase)
        el_deg = float(np.clip(el_deg, *self.el_clamp_deg))
        el = math.radians(el_deg)

        d = self.d0 * (1.0 + self.d_amp_frac * math.sin(
            2.0 * math.pi * self.d_freq_hz * t + self.d_phase))
        d = float(np.clip(d, *self.d_clamp))

        roll = math.radians(self.roll_amp_deg) * math.sin(
            2.0 * math.pi * self.roll_freq_hz * t + self.roll_phase)

        look = np.asarray(self.look0, dtype=np.float64).copy()
        if self.jitter:
            look += np.array([_eval_sinusoids(c, t) for c in self.jitter])

        offset = d * np.array([math.cos(el) * math.cos(az),
                               math.cos(el) * math.sin(az),
                               math.sin(el)])
        pos = look + offset
        if self.bob_amp:
            # Vertical bob rides on the CAMERA, not the look-at, so it produces
            # the pitch oscillation a walking observer sees.
            pos = pos + np.array([0.0, 0.0, self.bob_amp * math.sin(
                2.0 * math.pi * self.bob_freq_hz * t)])
        return pos, look, roll

    def extrinsics(self, frame_idx):
        """(R_camera_world, camera_pos_world) at a rendered frame."""
        pos, look, roll = self.pose(frame_idx)
        return look_at_basis(pos, look, roll_rad=roll)

    def gravity_camera(self, frame_idx):
        """Unit gravity direction in camera frame — the tilt label (T',3 stacked)."""
        R_cw, _ = self.extrinsics(frame_idx)
        return R_cw @ np.array([0.0, 0.0, -1.0])

    def is_static(self):
        # speed_scale == 0 freezes every time-dependent term, so the path is
        # static in fact — report it, or the rig pays the along-path gate and
        # per-frame re-posing for a camera that never moves.
        return self.kind == "static" or float(self.speed_scale) == 0.0 or not (
            self.omega_az_dps or self.el_amp_deg or self.d_amp_frac
            or self.roll_amp_deg or self.jitter or self.bob_amp)

    def to_dict(self):
        d = {k: (list(v) if isinstance(v, (tuple, np.ndarray)) and k != "jitter" else v)
             for k, v in self.__dict__.items()}
        d["jitter"] = [[list(c) for c in axis] for axis in self.jitter]
        d["look0"] = [float(x) for x in self.look0]
        return d


def sample_camera_path(rng, kind, az_center, el_center_deg, distance, look_at,
                       render_dt, el_clamp_deg=(12.0, 68.0), d_clamp=(0.8, 4.0),
                       roll_always=False, speed_scale=1.0):
    """Sample one seed-deterministic path of the given family.

    `az_center` / `el_center_deg` / `distance` / `look_at` come from the EXISTING
    rig solve (clip_capture.sample_camera_pose + framing_distance) — this module
    only adds the time dimension, so framing behaviour is unchanged at t=0.

    RNG DISCIPLINE (dynamic-camera.md §4): every draw here must come from a
    stream that no physics-affecting sampler shares, so switching camera family
    leaves the rollout bit-identical. That is what makes egomotion twins (same
    seed, different path) valid.

    `roll_always` forces a roll component onto any family — use it to guarantee
    tilt-label coverage (§9.3) even in otherwise-static batches.
    """
    if kind not in FAMILIES:
        raise ValueError(f"unknown camera path family {kind!r}; expected {FAMILIES}")

    p = dict(kind=kind, az0=float(az_center), el0_deg=float(el_center_deg),
             d0=float(distance), look0=tuple(float(x) for x in look_at),
             dt=float(render_dt), el_clamp_deg=tuple(el_clamp_deg),
             d_clamp=tuple(d_clamp), speed_scale=float(speed_scale))

    def _roll():
        p["roll_amp_deg"] = float(rng.uniform(*ROLL_AMP_DEG)) * rng.choice([-1.0, 1.0])
        p["roll_freq_hz"] = float(rng.uniform(*ROLL_FREQ_HZ))
        p["roll_phase"] = float(rng.uniform(0.0, 2.0 * math.pi))

    if kind == "orbit":
        p["omega_az_dps"] = float(rng.uniform(*ORBIT_RATE_DPS)) * rng.choice([-1.0, 1.0])
    elif kind == "arc":
        p["el_amp_deg"] = float(rng.uniform(*ARC_AMP_DEG))
        p["el_freq_hz"] = float(rng.uniform(*ARC_FREQ_HZ))
        p["el_phase"] = float(rng.uniform(0.0, 2.0 * math.pi))
    elif kind == "dolly":
        # NOTE: pure translation contributes ZERO gravity-tilt variation —
        # tilt depends only on camera ORIENTATION. Dolly buys parallax, not
        # tilt coverage (dynamic-camera.md §10).
        p["d_amp_frac"] = float(rng.uniform(*DOLLY_AMP_FRAC))
        p["d_freq_hz"] = float(rng.uniform(*DOLLY_FREQ_HZ))
        p["d_phase"] = float(rng.uniform(0.0, 2.0 * math.pi))
    elif kind == "roll":
        _roll()
    elif kind in ("handheld", "gait"):
        p["jitter"] = tuple(
            tuple(_sinusoids(rng, HANDHELD_COMPONENTS, HANDHELD_AMP, HANDHELD_FREQ_HZ))
            for _ in range(3))
        p["omega_az_dps"] = float(rng.uniform(0.0, ORBIT_RATE_DPS[0])) * rng.choice([-1.0, 1.0])
        _roll()
        if kind == "gait":
            p["bob_amp"] = float(rng.uniform(*GAIT_BOB_AMP))
            p["bob_freq_hz"] = float(rng.uniform(*GAIT_BOB_FREQ_HZ))
            # Step-synced roll at half the bob rate: one roll cycle per stride
            # (two steps) — the signature that separates walking from shake.
            p["roll_amp_deg"] = float(rng.uniform(*GAIT_ROLL_AMP_DEG)) * rng.choice([-1.0, 1.0])
            p["roll_freq_hz"] = p["bob_freq_hz"] / 2.0

    if roll_always and not p.get("roll_amp_deg"):
        _roll()
    return CameraPath(**p)


def egomotion_labels(path, frame_idx, dt=None):
    """(v_cam, omega_cam) — camera self-motion in the INSTANTANEOUS camera frame.

    Same type/frame convention as the per-body `lin_vel_camera` /
    `ang_vel_camera` records, so the camera is logged as "just another body"
    and existing target builders apply unchanged.

    Absolute camera pose is deliberately NOT a label: world origin (3 DoF) and
    yaw about gravity (1 DoF) are gauge — unobservable from pixels. Only
    relative motion is well-posed. Window-anchored relative pose (the target
    form that matches the window-aligned lesson from GATE-B) is derived at
    target-build time from the logged per-frame extrinsics, not here.
    """
    dt = float(dt if dt is not None else path.dt)
    R0, p0 = path.extrinsics(frame_idx)
    R1, p1 = path.extrinsics(frame_idx + 1)

    v_cam = R0 @ ((p1 - p0) / dt)                 # translation in camera-t axes
    R_rel = R1 @ R0.T                             # rotation over the step
    # log map of R_rel -> rotation vector; omega = theta * axis / dt.
    cos_th = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_th)
    if theta < 1e-9:
        omega = np.zeros(3)
    else:
        axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                         R_rel[0, 2] - R_rel[2, 0],
                         R_rel[1, 0] - R_rel[0, 1]]) / (2.0 * math.sin(theta))
        omega = axis * (theta / dt)
    return v_cam, omega
