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
    MOTION_FAMILIES, FreeCameraPath, MotionModel, resolve_family, sample_any_path,
    sample_free_path, sample_motion, pan_tilt_roll, surge_sway_heave,
)
from camera_paths import (  # noqa: E402
    egomotion_labels, euler_xyz_deg_to_rotmat, look_at_basis, sample_camera_path,
    usd_rotation_xyz_deg,
)

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

    # --- last-frame egomotion ----------------------------------------------
    #  FreeCameraPath caches a FIXED number of poses, so an index clamp at
    #  n_frames-1 would make the final frame difference against itself and log
    #  v = w = 0 — a corrupted label on 1/T of every dynamic clip, and one that
    #  looks like plausible data rather than a crash. The extra integration step
    #  in __init__ is what this asserts.
    v_last, w_last = egomotion_labels(p_dolly, T - 1)
    v_mid, _ = egomotion_labels(p_dolly, T // 2)
    check("★ last-frame egomotion is real, not a clamped zero",
          np.linalg.norm(v_last) > 0.5 * np.linalg.norm(v_mid),
          f"|v[T-1]|={np.linalg.norm(v_last):.4f} vs |v[T/2]|={np.linalg.norm(v_mid):.4f}")

    # --- speed_scale: ClipConfig.camera_speed_scale as a TIME WARP -----------
    #  Defined as "the same shot traversed s times faster", i.e. x(s*t) — NOT a
    #  gain on the velocity. The distinction is testable and these checks are
    #  what make it more than a claim in a docstring: a pure output gain would
    #  pass the travel-doubling check below and FAIL the warp identity, because
    #  it stretches oscillation amplitudes instead of replaying them faster.
    for name in ("free:dolly_in", "subject:orbit", "subject:handheld"):
        base = sample_any_path(np.random.default_rng(5), name, POS0, LOOK0, T, DT,
                               roll_always=True)
        same = sample_any_path(np.random.default_rng(5), name, POS0, LOOK0, T, DT,
                               roll_always=True, speed_scale=1.0)
        d = max(np.abs(np.asarray(base.extrinsics(t)[1])
                       - np.asarray(same.extrinsics(t)[1])).max() for t in range(T))
        check(f"speed_scale=1.0 is bit-identical to the default [{name}]", d == 0.0,
              f"max|dpos|={d:.1e}")

    #  Warp identity. EXACT for the subject-anchored model (closed form in t);
    #  first-order in dt for the free model, which integrates. The motion object
    #  is shared deliberately: `sample_motion` takes n_frames and `_dolly_in`
    #  scales its ramp by it, so re-sampling at a different n_frames would
    #  compare two different shots and the residual would not converge.
    m_warp = sample_motion(np.random.default_rng(5), "dolly_in", DT, T)
    import dataclasses as _dc
    errs = []
    for dt_scale in (1, 2, 4):
        dt_i, k_i = DT / dt_scale, T * dt_scale
        p1 = FreeCameraPath(_dc.replace(m_warp, dt=dt_i, speed_scale=1.0), POS0, R0, 2 * k_i)
        p2 = FreeCameraPath(_dc.replace(m_warp, dt=dt_i, speed_scale=2.0), POS0, R0, k_i)
        errs.append(max(np.abs(p2.extrinsics(t)[1] - p1.extrinsics(2 * t)[1]).max()
                        for t in range(k_i)))
    check("★ free: pose(t)@s=2 == pose(2t)@s=1, first-order in dt",
          errs[0] < 1e-2 and errs[1] < 0.6 * errs[0] and errs[2] < 0.6 * errs[1],
          f"err {errs[0]:.2e} -> {errs[1]:.2e} -> {errs[2]:.2e} as dt halves")

    for name in ("subject:orbit", "subject:handheld", "subject:arc", "subject:roll"):
        p1 = sample_any_path(np.random.default_rng(5), name, POS0, LOOK0, 2 * T, DT,
                             roll_always=True, speed_scale=1.0)
        p2 = sample_any_path(np.random.default_rng(5), name, POS0, LOOK0, 2 * T, DT,
                             roll_always=True, speed_scale=2.0)
        e = max(np.abs(p2.extrinsics(t)[1] - p1.extrinsics(2 * t)[1]).max() for t in range(T))
        check(f"★ subject: warp is EXACT (closed form) [{name}]", e < 1e-12,
              f"max|dpos|={e:.1e}")

    #  ...and it must actually double the egomotion label, or the knob is inert.
    slow = sample_any_path(np.random.default_rng(5), "free:dolly_in", POS0, LOOK0, T, DT)
    fast = sample_any_path(np.random.default_rng(5), "free:dolly_in", POS0, LOOK0, T, DT,
                           speed_scale=2.0)
    r = (np.mean([np.linalg.norm(egomotion_labels(fast, t)[0]) for t in range(T - 1)])
         / max(np.mean([np.linalg.norm(egomotion_labels(slow, t)[0]) for t in range(T - 1)]), 1e-12))
    check("speed_scale=2 doubles the logged |v_cam|", 1.9 < r < 2.2, f"ratio={r:.3f}")

    #  s=0 must report STATIC, not "dynamic but stationary" — a dynamic rig pays
    #  the along-path gate and per-frame prim re-posing for nothing.
    z_free = sample_any_path(np.random.default_rng(5), "free:dolly_in", POS0, LOOK0, T, DT,
                             speed_scale=0.0)
    z_sub = sample_any_path(np.random.default_rng(5), "subject:orbit", POS0, LOOK0, T, DT,
                            speed_scale=0.0)
    check("speed_scale=0 reports is_static() in both models",
          z_free.is_static() and z_sub.is_static())
    check("speed_scale=0 really does freeze the pose",
          all(np.allclose(z_free.extrinsics(t)[1], POS0) for t in range(T))
          and all(np.allclose(z_sub.extrinsics(t)[1], z_sub.extrinsics(0)[1]) for t in range(T)))

    #  Provenance: a warped path must round-trip, or meta.json no longer
    #  describes the camera the pixels came from (the property gated on the
    #  rig_check corpus).
    check("speed_scale survives to_dict() in both models",
          fast.to_dict()["motion"]["speed_scale"] == 2.0
          and sample_any_path(np.random.default_rng(5), "subject:orbit", POS0, LOOK0,
                              T, DT, speed_scale=2.0).to_dict()["speed_scale"] == 2.0)

    # --- cross-layer dispatch ----------------------------------------------
    #  One family namespace over two camera models. The failure this guards is
    #  silent: `gait` means different motion in each model, so a bare name that
    #  quietly picked one would produce a corpus that is not the one requested.
    check("resolve: bare unique name -> right layer",
          resolve_family("dolly") == ("subject", "dolly")
          and resolve_family("dolly_in") == ("free", "dolly_in"))
    check("resolve: prefixed names both work",
          resolve_family("subject:gait") == ("subject", "gait")
          and resolve_family("free:gait") == ("free", "gait"))
    for bad, why in (("gait", "ambiguous bare name"),
                     ("handheld", "ambiguous bare name"),
                     ("nope", "unknown family"),
                     ("free:nope", "unknown kind in a known layer"),
                     ("weird:gait", "unknown layer")):
        try:
            resolve_family(bad)
            check(f"resolve: {bad!r} rejected ({why})", False)
        except ValueError:
            check(f"resolve: {bad!r} rejected ({why})", True)

    #  The az/el/d derivation moved into sample_any_path; if it round-trips the
    #  rig pose wrong, every subject-anchored camera is posed somewhere the
    #  visibility gate never approved.
    rng_a = np.random.default_rng(11)
    sub = sample_any_path(rng_a, "subject:orbit", POS0, LOOK0, T, DT)
    check("dispatch: subject layer reproduces the rig pose at t=0",
          np.allclose(sub.extrinsics(0)[1], POS0, atol=1e-9)
          and np.allclose(sub.extrinsics(0)[0], R_legacy, atol=1e-9),
          f"|dp|={np.linalg.norm(sub.extrinsics(0)[1] - POS0):.2e}")
    check("dispatch: layers return their own path types",
          type(sub).__name__ == "CameraPath"
          and isinstance(sample_any_path(np.random.default_rng(11), "free:dolly_in",
                                         POS0, LOOK0, T, DT), FreeCameraPath))
    check("dispatch: sample_any_path == sample_camera_path for the same seed",
          np.allclose(sub.to_dict()["az0"],
                      sample_camera_path(
                          np.random.default_rng(11), "orbit",
                          math.atan2(POS0[1] - LOOK0[1], POS0[0] - LOOK0[0]),
                          math.degrees(math.asin((POS0[2] - LOOK0[2])
                                                 / np.linalg.norm(POS0 - LOOK0))),
                          float(np.linalg.norm(POS0 - LOOK0)), LOOK0, DT).to_dict()["az0"]))

    # --- roll_always on the free layer -------------------------------------
    #  Different mechanism from the subject layer (constant frame-0 offset, not
    #  an oscillating amplitude) and it must be a NO-OP on a family that already
    #  commands roll — otherwise it would fight the family whose roll dynamics
    #  are the point, and burn an rng draw that breaks twin reproducibility.
    p_roll = sample_free_path(np.random.default_rng(5), "still", POS0, LOOK0, T, DT,
                              roll_always=True)
    g_roll = np.stack([p_roll.gravity_camera(t) for t in range(T)])
    check("roll_always: injects tilt into a roll-free family",
          np.abs(g_roll[:, 0]).max() > 1e-2,
          f"|g.x|={np.abs(g_roll[:,0]).max():.4f} (constant over the clip)")
    on = sample_free_path(np.random.default_rng(9), "dolly_in", POS0, LOOK0, T, DT,
                          roll_always=True)
    off = sample_free_path(np.random.default_rng(9), "dolly_in", POS0, LOOK0, T, DT)
    check("roll_always: no-op (bit-identical) when the family already rolls",
          all(np.allclose(on.extrinsics(t)[0], off.extrinsics(t)[0], atol=0.0)
              and np.allclose(on.extrinsics(t)[1], off.extrinsics(t)[1], atol=0.0)
              for t in range(T)))

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
