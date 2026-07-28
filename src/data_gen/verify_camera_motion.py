"""Closed-form checks for camera_motion.py — runs WITHOUT Isaac.

Companion to verify_camera_paths.py, for the free 6-DoF model. Run this after
adding a family to `camera_motion.sample_motion`: `--family <name>` re-runs the
generic invariants against YOUR family, so a behaviour that silently loses the
subject, drifts off SO(3), or commands motion it doesn't actually execute fails
here instead of after a GPU batch.

    python3 src/data_gen/verify_camera_motion.py                 # built-ins
    python3 src/data_gen/verify_camera_motion.py --family steer  # yours
    python3 src/data_gen/verify_camera_motion.py --all           # every family
"""

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from camera_motion import (  # noqa: E402
    MOTION_FAMILIES, FreeCameraPath, MotionModel, sample_free_path,
    sample_motion, pan_tilt_roll, surge_sway_heave,
)
from camera_paths import look_at_basis, usd_rotation_xyz_deg, euler_xyz_deg_to_rotmat  # noqa: E402

DT = 1.0 / 30.0
T = 64
POS0 = np.array([2.4, 0.0, 1.7])
LOOK0 = np.array([0.0, 0.0, 0.1])
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def generic_invariants(path, label, expect_motion=True):
    """Invariants EVERY family must satisfy, whatever it does."""
    # 1. Every cached orientation is a genuine rotation. Chained products drift
    #    off SO(3), and a non-rotation R corrupts every camera-frame label.
    ok = True
    for t in range(T):
        R, _ = path.extrinsics(t)
        ok &= np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    check(f"[{label}] extrinsics stay in SO(3)", ok)

    # 2. ★ COMMAND ROUND TRIP — the integrated poses must reproduce the body
    #    velocity that was commanded. This is the check that catches a family
    #    whose motion is not what it claims: differentiate the cached poses and
    #    compare against MotionModel.velocity().
    vs, ws, ok = [], [], True
    for t in range(T - 1):
        R0, p0 = path.extrinsics(t)
        R1, p1 = path.extrinsics(t + 1)
        v_cmd, w_cmd = path.motion.velocity(t)
        v_meas = R0 @ ((p1 - p0) / DT)                  # world delta -> camera axes
        R_rel = R0 @ R1.T                               # body-frame increment
        cth = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        th = math.acos(cth)
        w_meas = np.zeros(3) if th < 1e-9 else (np.array([
            R_rel[2, 1] - R_rel[1, 2], R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1]]) / (2 * math.sin(th)) * (th / DT))
        vs.append(np.linalg.norm(v_meas - v_cmd))
        ws.append(np.linalg.norm(w_meas - w_cmd))
    if path.track_tau == 0.0:      # tracking legitimately adds rotation
        ok = max(vs) < 1e-6 and max(ws) < 1e-6
        check(f"[{label}] integrated poses match the commanded velocity",
              ok, f"max dv={max(vs):.2e} dw={max(ws):.2e}")
    else:
        check(f"[{label}] translation matches command under tracking",
              max(vs) < 1e-6, f"max dv={max(vs):.2e}")

    # 3. USD Euler round trip — the prim must be posed exactly as the labels say.
    ok = True
    for t in range(0, T, 7):
        R, _ = path.extrinsics(t)
        rx, ry, rz = usd_rotation_xyz_deg(R)
        ok &= np.allclose(euler_xyz_deg_to_rotmat(rx, ry, rz), R.T, atol=1e-9)
    check(f"[{label}] USD rotateXYZ round-trips", ok)

    # 4. Gravity label stays unit and equivariant with the camera's own motion.
    g = np.stack([path.gravity_camera(t) for t in range(T)])
    eq = all(np.allclose((path.extrinsics(t + 1)[0] @ path.extrinsics(t)[0].T)
                         @ g[t], g[t + 1], atol=1e-9) for t in range(T - 1))
    check(f"[{label}] g_cam unit-norm + temporally equivariant",
          np.allclose(np.linalg.norm(g, axis=1), 1.0) and eq)

    # 5. Frame 0 is exactly the gated rig pose (visibility gating stays valid).
    check(f"[{label}] frame 0 == the gated rig pose",
          np.allclose(path.extrinsics(0)[1], POS0, atol=1e-12))

    if expect_motion:
        moved = np.linalg.norm(path.extrinsics(T - 1)[1] - POS0) > 1e-6
        rotated = not np.allclose(path.extrinsics(T - 1)[0], path.extrinsics(0)[0], atol=1e-9)
        check(f"[{label}] actually moves (not a silent `still`)", moved or rotated)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, help="verify one family by name")
    ap.add_argument("--all", action="store_true", help="verify every family")
    args = ap.parse_args()

    # --- the zero case must reproduce the static rig EXACTLY ---------------
    p_still = sample_free_path(np.random.default_rng(0), "still", POS0, LOOK0, T, DT)
    R_legacy, _ = look_at_basis(POS0, LOOK0)
    check("still: frame-0 basis == look_at_basis", np.allclose(p_still.extrinsics(0)[0], R_legacy))
    check("still: pose constant over the clip",
          all(np.allclose(p_still.extrinsics(t)[1], POS0) for t in range(T))
          and all(np.allclose(p_still.extrinsics(t)[0], R_legacy) for t in range(T)))
    check("still: is_static() is True", p_still.is_static())
    g_still = np.stack([p_still.gravity_camera(t) for t in range(T)])
    check("still: g_cam.x == 0 (the degeneracy, as expected with no roll)",
          np.allclose(g_still[:, 0], 0.0, atol=1e-12))
    generic_invariants(p_still, "still", expect_motion=False)

    # --- the worked example ------------------------------------------------
    p_dolly = sample_free_path(np.random.default_rng(1), "dolly_in", POS0, LOOK0, T, DT)
    g = generic_invariants(p_dolly, "dolly_in")
    check("dolly_in: roll term breaks g_cam.x == 0",
          np.abs(g[:, 0]).max() > 1e-3, f"|g.x|max={np.abs(g[:,0]).max():.4f}")
    d0 = np.linalg.norm(p_dolly.extrinsics(0)[1] - LOOK0)
    d1 = np.linalg.norm(p_dolly.extrinsics(T - 1)[1] - LOOK0)
    check("dolly_in: actually moves TOWARD the subject", d1 < d0,
          f"{d0:.2f} m -> {d1:.2f} m")

    # --- tracking blend ----------------------------------------------------
    #  A pure sway with tau=0 loses the subject; tau=1 keeps it centred; and
    #  crucially tracking must NOT reset roll (that would undo the degeneracy fix).
    m = MotionModel(kind="_t", dt=DT, terms=(
        {"kind": "const", "v": (0.0, 0.6, 0.0), "w": (0.0, 0.0, 0.15)},))
    R0, _ = look_at_basis(POS0, LOOK0)

    def off_axis(path):
        R, p = path.extrinsics(T - 1)
        f = -R.T[:, 2]
        d = LOOK0 - p
        return math.degrees(math.acos(float(np.clip(
            np.dot(f, d / np.linalg.norm(d)), -1, 1))))

    free = FreeCameraPath(m, POS0, R0, T, track_target=LOOK0, track_tau=0.0)
    lock = FreeCameraPath(m, POS0, R0, T, track_target=LOOK0, track_tau=1.0)
    check("tracking: tau=0 drifts off the subject", off_axis(free) > 10.0,
          f"{off_axis(free):.1f} deg off-axis")
    check("tracking: tau=1 keeps the subject centred", off_axis(lock) < 1.0,
          f"{off_axis(lock):.2f} deg off-axis")
    g_lock = np.stack([lock.gravity_camera(t) for t in range(T)])
    check("★ tracking preserves ROLL (does not reset the degeneracy fix)",
          np.abs(g_lock[:, 0]).max() > 1e-3,
          f"|g.x|max={np.abs(g_lock[:,0]).max():.4f}")
    generic_invariants(lock, "tracked", expect_motion=True)

    # --- determinism -------------------------------------------------------
    a = sample_free_path(np.random.default_rng(7), "dolly_in", POS0, LOOK0, T, DT)
    b = sample_free_path(np.random.default_rng(7), "dolly_in", POS0, LOOK0, T, DT)
    check("same seed -> identical path",
          all(np.allclose(a.extrinsics(t)[1], b.extrinsics(t)[1])
              and np.allclose(a.extrinsics(t)[0], b.extrinsics(t)[0]) for t in range(T)))

    # --- user families -----------------------------------------------------
    todo = ([args.family] if args.family
            else (list(MOTION_FAMILIES) if args.all else []))
    for kind in todo:
        if kind in ("still", "dolly_in") and not args.family:
            continue                     # already verified in full above
        try:
            p = sample_free_path(np.random.default_rng(3), kind, POS0, LOOK0, T, DT,
                                 track_tau=0.5)
        except NotImplementedError:
            print(f"SKIP  [{kind}] not implemented yet")
            continue
        gk = generic_invariants(p, kind, expect_motion=(kind != "still"))
        span = np.degrees(np.arccos(np.clip(gk @ gk[0], -1, 1))).max()
        carries = np.abs(gk[:, 0]).max() > 1e-3
        note = ("(carries tilt)" if carries
                else "(zero case — no tilt expected)" if kind == "still"
                else "(NO tilt coverage — add a roll term)")
        print(f"      [{kind}] tilt: |g.x|max={np.abs(gk[:,0]).max():.4f} "
              f"span={span:.2f} deg   {note}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all camera-motion invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
