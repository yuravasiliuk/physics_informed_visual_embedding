"""Free 6-DoF camera motion — the egocentric camera model (dynamic-camera.md §3).

`camera_paths.CameraPath` is **orbit-constrained**: every pose is
(azimuth, elevation, distance) around a look-at point, so the camera is always
aimed at the subject. That parametrization *cannot express* pan, tilt, or true
lateral travel — aim is anchored by construction. This module adds the
egocentric model, i.e. the standard cinematography basis:

    surge  forward / backward along the view axis   (dolly)
    sway   lateral                                  (truck)
    heave  vertical                                 (pedestal)
    pan    rotate about the camera's up axis        (steering)
    tilt   rotate about the camera's right axis
    roll   rotate about the view axis               (the degeneracy breaker)

The old families are special cases: `orbit` = truck + pan locked on the subject,
`dolly` = pure surge.

**Why the poses are PRECOMPUTED, not closed-form.** "Drive forward while
steering" is an ODE — rotations do not commute, so body-frame velocities must be
integrated. The integration runs ONCE at construction from a seeded rng and is
cached, so paths stay bit-reproducible (the twin machinery's precondition,
dynamic-camera.md §4); we only give up O(1) random access, which nothing needs
because frames are always evaluated in order.

**Interface parity:** `pose`/`extrinsics`/`gravity_camera`/`is_static`/`to_dict`
match `CameraPath`, so `collect_clip` and `egomotion_labels` consume either class
unchanged (duck-typed on `.extrinsics`).

CONVENTIONS (USD camera): +X right, +Y up, **−Z forward**. Body-frame linear
velocity is returned in camera axes, so *forward is negative z* — use
`surge_sway_heave()` rather than writing the sign by hand.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from camera_paths import look_at_basis   # shared basis; no omni deps

WORLD_UP = np.array([0.0, 0.0, 1.0])


def surge_sway_heave(surge=0.0, sway=0.0, heave=0.0):
    """(forward, right, up) m/s -> camera-axis velocity vector.

    Exists so no family has to remember that forward is −Z."""
    return np.array([float(sway), float(heave), -float(surge)])


def pan_tilt_roll(pan=0.0, tilt=0.0, roll=0.0):
    """(yaw, pitch, roll) rad/s about the camera's own axes -> ω in camera axes.

    pan  = about +Y (up)      — steering left/right
    tilt = about +X (right)   — looking up/down
    roll = about −Z (forward) — the g_cam.x degeneracy breaker
    """
    return np.array([float(tilt), float(pan), -float(roll)])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _expm_so3(omega, dt):
    """Exact exponential map (Rodrigues) for a constant body-rate over dt."""
    theta = float(np.linalg.norm(omega)) * dt
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(omega, dtype=np.float64) / np.linalg.norm(omega)
    K = _skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _reorthonormalize(R):
    """Project back onto SO(3). 64 chained rotations drift only ~1e-15, but the
    drift is unbounded in clip length and a non-rotation R would silently
    corrupt every camera-frame label — so pay the SVD."""
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:                    # never expected; fail loud-ish
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _track_toward(R_world_camera, position, target, tau):
    """Rotate the camera fractionally toward `target` WITHOUT touching roll.

    The correction is the minimal rotation taking the current forward axis to
    the desired one; its axis is perpendicular to both forwards, so rotation
    *about the view axis* — the roll that breaks the g_cam.x ≡ 0 degeneracy — is
    untouched by construction. (Re-aiming with a look-at basis instead would
    silently reset roll to zero and undo the whole point of the roll term.)

    tau = 0 free flight · tau = 1 subject locked to frame centre.
    """
    if tau <= 0.0:
        return R_world_camera
    f_cur = -R_world_camera[:, 2]
    d = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return R_world_camera
    f_des = d / n
    axis = np.cross(f_cur, f_des)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(np.dot(f_cur, f_des), -1.0, 1.0))
    if s < 1e-12:
        return R_world_camera                   # already aimed (or exactly opposite)
    axis = axis / s
    angle = math.atan2(s, c) * float(np.clip(tau, 0.0, 1.0))
    K = _skew(axis)
    R_corr = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return _reorthonormalize(R_corr @ R_world_camera)


# ---------------------------------------------------------------------------
# MOTION MODEL — the per-frame body-frame velocity command.
# ---------------------------------------------------------------------------

@dataclass
class MotionModel:
    """Body-frame velocity command as a function of frame index.

    `velocity(t)` returns `(v_cam, w_cam)`:
        v_cam : (3,) m/s in CAMERA axes  — build it with surge_sway_heave()
        w_cam : (3,) rad/s in CAMERA axes — build it with pan_tilt_roll()

    `terms` is a list of additive components, each a dict the family author
    fills in; keeping them declarative (not closures) is what lets `to_dict()`
    round-trip a path into meta.json and back.

    Supported term shapes:
        {"kind": "const", "v": (surge, sway, heave), "w": (pan, tilt, roll)}
        {"kind": "sine",  "v": (...), "w": (...), "freq_hz": f, "phase": p}
        {"kind": "ramp",  "v": (...), "w": (...), "t0": a, "t1": b}   # linear 0->1 over [a,b] frames
    """
    kind: str
    dt: float
    terms: tuple = ()

    def velocity(self, t):
        v = np.zeros(3)
        w = np.zeros(3)
        tt = float(t) * self.dt
        for term in self.terms:
            k = term.get("kind", "const")
            if k == "const":
                g = 1.0
            elif k == "sine":
                g = math.sin(2.0 * math.pi * float(term.get("freq_hz", 0.0)) * tt
                             + float(term.get("phase", 0.0)))
            elif k == "ramp":
                t0, t1 = float(term.get("t0", 0.0)), float(term.get("t1", 1.0))
                g = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
            else:
                raise ValueError(f"unknown motion term kind {k!r}")
            sv, ss, sh = term.get("v", (0.0, 0.0, 0.0))
            pn, tl, rl = term.get("w", (0.0, 0.0, 0.0))
            v = v + g * surge_sway_heave(sv, ss, sh)
            w = w + g * pan_tilt_roll(pn, tl, rl)
        return v, w

    def is_zero(self):
        return not self.terms

    def to_dict(self):
        return {"kind": self.kind, "dt": self.dt,
                "terms": [dict(t) for t in self.terms]}


class FreeCameraPath:
    """6-DoF camera path: integrate a MotionModel, cache the poses.

    Duck-compatible with `camera_paths.CameraPath` — same
    pose/extrinsics/gravity_camera/is_static/to_dict surface, so `collect_clip`,
    `set_camera_pose` and `egomotion_labels` all work unchanged.
    """

    def __init__(self, motion, position0, R_camera_world0, n_frames,
                 track_target=None, track_tau=0.0):
        self.motion = motion
        self.dt = float(motion.dt)
        self.n_frames = int(n_frames)
        self.track_target = (None if track_target is None
                             else np.asarray(track_target, dtype=np.float64))
        self.track_tau = float(track_tau)
        self.kind = motion.kind

        p = np.asarray(position0, dtype=np.float64).copy()
        R_wc = np.asarray(R_camera_world0, dtype=np.float64).T   # camera -> world
        pos = np.empty((self.n_frames, 3))
        rot = np.empty((self.n_frames, 3, 3))                    # camera -> world
        for t in range(self.n_frames):
            if self.track_target is not None and self.track_tau > 0.0:
                R_wc = _track_toward(R_wc, p, self.track_target, self.track_tau)
            pos[t], rot[t] = p, R_wc
            v_cam, w_cam = motion.velocity(t)
            p = p + R_wc @ v_cam * self.dt       # body velocity -> world
            R_wc = _reorthonormalize(R_wc @ _expm_so3(w_cam, self.dt))  # body rate
        self._pos, self._rot = pos, rot

    # --- CameraPath-compatible surface -----------------------------------
    def _idx(self, t):
        return int(np.clip(int(t), 0, self.n_frames - 1))

    def pose(self, frame_idx):
        """(camera_pos_world, look_at_world, roll_rad).

        `look_at` is a POINT ON THE VIEW AXIS, provided only for API parity —
        it does NOT determine orientation for this class (a look-at cannot carry
        roll or a free aim). Anything posing the prim must use `extrinsics()`.
        """
        i = self._idx(frame_idx)
        p, R_wc = self._pos[i], self._rot[i]
        return p, p + (-R_wc[:, 2]), 0.0

    def extrinsics(self, frame_idx):
        i = self._idx(frame_idx)
        return self._rot[i].T, self._pos[i]

    def gravity_camera(self, frame_idx):
        R_cw, _ = self.extrinsics(frame_idx)
        return R_cw @ np.array([0.0, 0.0, -1.0])

    def is_static(self):
        return self.motion.is_zero()

    def to_dict(self):
        return {"kind": self.kind, "model": "free6dof", "dt": self.dt,
                "n_frames": self.n_frames, "track_tau": self.track_tau,
                "track_target": (None if self.track_target is None
                                 else [float(x) for x in self.track_target]),
                "position0": [float(x) for x in self._pos[0]],
                "R_camera_world0": self._rot[0].T.tolist(),
                "motion": self.motion.to_dict()}


# ---------------------------------------------------------------------------
# >>> FAMILY REGISTRY — YOUR SLOT <<<
# ---------------------------------------------------------------------------
# `sample_motion` is the ONLY thing a new camera behaviour needs. Everything
# else (integration, extrinsics, egomotion, USD posing, gating, logging) is
# already wired and gated by verify_camera_motion.py.
#
# Contract:
#   in  : rng (np.random.Generator — the ONLY randomness source, so paths stay
#         seed-deterministic), kind (str), dt (seconds/frame), n_frames
#   out : MotionModel whose terms describe body-frame velocity over time
#
# Units: surge/sway/heave in m/s, pan/tilt/roll in rad/s. Keep speeds modest —
# the acceptance gate (dynamic-camera.md §9.4) requires egomotion-induced image
# flow NOT to dominate object-induced flow.
#
# ★ Reminder that outranks realism: ONLY ROLL breaks the g_cam.x ≡ 0
#   degeneracy. Pure surge/sway/heave are translations, and pan is yaw about
#   (approximately) the gravity axis — measured, both leave gravity tilt
#   invariant. A family with no roll term contributes NO tilt coverage.

MOTION_FAMILIES = (
    "still",        # implemented below (the zero case / regression anchor)
    "dolly_in",     # implemented below as the worked example
    "dolly_out",
    "truck",
    "pedestal",
    "steer",
    "handheld",
    "gait",
    "orbit_free",
    "survey",
)


def _still(rng, dt, n_frames):
    return MotionModel(kind="still", dt=dt, terms=())


def _dolly_in(rng, dt, n_frames):
    """WORKED EXAMPLE — push in on the subject with a slight roll.

    Read this as the template for the rest: draw every random quantity from
    `rng`, return declarative terms, add a roll component if the family is meant
    to carry the gravity-tilt label.
    """
    speed = float(rng.uniform(0.10, 0.45))          # m/s forward
    roll_rate = float(rng.uniform(0.02, 0.10)) * float(rng.choice([-1.0, 1.0]))
    return MotionModel(kind="dolly_in", dt=dt, terms=(
        # ease in over the first ~15% of the clip so frame 0 isn't a jerk
        {"kind": "ramp", "v": (speed, 0.0, 0.0), "w": (0.0, 0.0, 0.0),
         "t0": 0.0, "t1": max(1.0, 0.15 * n_frames)},
        {"kind": "sine", "v": (0.0, 0.0, 0.0), "w": (0.0, 0.0, roll_rate),
         "freq_hz": float(rng.uniform(0.08, 0.25)),
         "phase": float(rng.uniform(0.0, 2.0 * math.pi))},
    ))


def sample_motion(rng, kind, dt, n_frames):
    """Sample one family's MotionModel.  <<< ADD FAMILIES HERE >>>

    Each branch should return a MotionModel; see `_dolly_in` for the template.
    Unimplemented families raise NotImplementedError rather than silently
    falling back to `still` — a silent fallback would produce a corpus that
    looks dynamic and is not.
    """
    if kind == "still":
        return _still(rng, dt, n_frames)
    if kind == "dolly_in":
        return _dolly_in(rng, dt, n_frames)

    # --- TODO(you) -------------------------------------------------------
    # dolly_out : pull back; mirror of dolly_in with negative surge
    # truck     : lateral travel; consider a slight counter-pan so the subject
    #             drifts across frame instead of leaving it
    # pedestal  : vertical travel; note this changes ELEVATION, which moves
    #             g_cam along the degenerate arc — pair it with roll
    # steer     : constant surge + pan rate = a circular arc through the scene
    #             (the "driving" shot); pan sign and rate from rng
    # handheld  : several sine terms per axis at 0.3-1.6 Hz, small amplitudes,
    #             plus roll — the "fuzzy" motion
    # gait      : handheld + 1.5-2.5 Hz heave bob + roll at half the bob rate
    #             (one roll cycle per stride); keep it subtle, that is what
    #             real walking looks like
    # orbit_free: sway + pan tuned so the camera circles the subject WITHOUT a
    #             look-at constraint (compare against track_tau to see drift)
    # survey    : slow pan across a static-ish scene; tests whether the encoder
    #             separates egomotion from object motion
    raise NotImplementedError(
        f"camera motion family {kind!r} not implemented yet — add it in "
        "camera_motion.sample_motion (see _dolly_in for the template)")


def sample_free_path(rng, kind, position0, look_at0, n_frames, dt,
                     roll_rad0=0.0, track_target=None, track_tau=0.0):
    """Sample a 6-DoF path starting from a gated rig pose.

    `position0`/`look_at0` come from the existing rig solve, so frame 0 is
    exactly the pose the static visibility gate approved; the motion model then
    takes over. `track_tau` blends re-aiming at `track_target` (default: the
    look-at point) — free 6-DoF loses the subject fast, and a partial track is
    also what a real operator does.
    """
    R_cw0, _ = look_at_basis(position0, look_at0, roll_rad=roll_rad0)
    motion = sample_motion(rng, kind, dt, n_frames)
    return FreeCameraPath(
        motion, position0, R_cw0, n_frames,
        track_target=(look_at0 if track_target is None else track_target),
        track_tau=track_tau,
    )
