"""Closed-form checks for camera_paths.py — runs WITHOUT Isaac.

Unlike the rest of src/data_gen/, camera_paths is pure numpy, so its invariants
can be gated here in CI-style instead of only on a rendered batch. These are the
data-gen-side preconditions for the dynamic-camera acceptance gate
(dynamic-camera.md §7, §9); the pixel-side checks still need a real run.

    python3 src/data_gen/verify_camera_paths.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from camera_paths import (  # noqa: E402
    CameraPath, egomotion_labels, euler_xyz_deg_to_rotmat, look_at_basis,
    sample_camera_path, usd_rotation_xyz_deg,
)

G_WORLD = np.array([0.0, 0.0, -1.0])
DT = 1.0 / 30.0
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def legacy_basis(camera_pos, look_at, world_up=(0.0, 0.0, 1.0)):
    """Verbatim clip_capture.world_to_camera_frame (the zero-roll legacy path)."""
    camera_pos = np.asarray(camera_pos, float)
    forward = np.asarray(look_at, float) - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(world_up, float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1).T


def main():
    rng = np.random.default_rng(0)
    look, d, az, el = (0.0, 0.0, 0.1), 3.0, 0.7, 35.0

    # 1. BACKWARD COMPATIBILITY — roll=0 must reproduce the legacy basis bit-wise.
    p_static = sample_camera_path(rng, "static", az, el, d, look, DT)
    pos0, look0, roll0 = p_static.pose(0)
    R_new, _ = p_static.extrinsics(0)
    check("static basis == legacy world_to_camera_frame",
          np.allclose(R_new, legacy_basis(pos0, look0), atol=1e-12)
          and roll0 == 0.0)

    # 2. THE DEGENERACY — a zero-roll camera forces g_cam.x == 0 exactly.
    gs = np.stack([p_static.gravity_camera(t) for t in range(64)])
    check("static path: g_cam.x identically 0 (the degeneracy)",
          np.allclose(gs[:, 0], 0.0, atol=1e-12)
          and np.allclose(gs, gs[0], atol=1e-12),
          f"g_cam={gs[0].round(4).tolist()}")

    # 3. ROLL BREAKS IT — this is the whole point of the roll term.
    p_roll = sample_camera_path(np.random.default_rng(1), "roll", az, el, d, look, DT)
    gr = np.stack([p_roll.gravity_camera(t) for t in range(64)])
    check("roll path: g_cam.x has real spread",
          np.abs(gr[:, 0]).max() > 0.02 and gr[:, 0].std() > 0.01,
          f"|x|max={np.abs(gr[:,0]).max():.4f}  std={gr[:,0].std():.4f}")

    # 4. UNIT NORM — the label is a direction; the loss is cosine.
    check("g_cam is unit-norm on every family", all(
        np.allclose(np.linalg.norm(
            np.stack([sample_camera_path(np.random.default_rng(s), k, az, el, d, look, DT)
                      .gravity_camera(t) for t in range(32)]), axis=1), 1.0)
        for s, k in enumerate(("orbit", "arc", "dolly", "roll", "handheld", "gait"))))

    # 5. TEMPORAL EQUIVARIANCE — g_cam(t+1) == R_rel(t) @ g_cam(t) exactly.
    #    This identity is the free cross-frame supervision (dynamic-camera.md §7).
    p_gait = sample_camera_path(np.random.default_rng(2), "gait", az, el, d, look, DT)
    ok = True
    for t in range(48):
        R0, _ = p_gait.extrinsics(t)
        R1, _ = p_gait.extrinsics(t + 1)
        ok &= np.allclose((R1 @ R0.T) @ p_gait.gravity_camera(t),
                          p_gait.gravity_camera(t + 1), atol=1e-12)
    check("temporal equivariance g(t+1) == R_rel(t) @ g(t)", ok)

    # 6. DOLLY CONTRIBUTES NO TILT — gravity is a uniform field, so pure
    #    translation cannot change g_cam. Guards trajectory-mix design.
    p_dolly = sample_camera_path(np.random.default_rng(3), "dolly", az, el, d, look, DT)
    gd = np.stack([p_dolly.gravity_camera(t) for t in range(64)])
    check("dolly (pure translation) gives ZERO tilt variation",
          np.allclose(gd, gd[0], atol=1e-12))

    # 6b. ORBIT CONTRIBUTES NO TILT EITHER — azimuth drift is rotation about the
    #     world-Z (= gravity) axis, and gravity is invariant under rotation about
    #     itself. The 4 gauge DoF (3 translation + yaw) are EXACTLY the motions
    #     that leave g_cam fixed, so orbit+dolly buy parallax, never tilt.
    p_orbit = sample_camera_path(np.random.default_rng(5), "orbit", az, el, d, look, DT)
    go = np.stack([p_orbit.gravity_camera(t) for t in range(64)])
    check("orbit (pure yaw about gravity) gives ZERO tilt variation",
          np.allclose(go, go[0], atol=1e-12))

    # 6c. ARC MOVES ALONG THE DEGENERATE ARC, it does not break it. Elevation
    #     change is pure pitch: g_cam.x stays 0, so the label remains the
    #     1-parameter family a single-frame horizon detector can read. ONLY roll
    #     breaks x ≡ 0 — that is why roll is mandatory in the dynamic mix, not
    #     one family among many (dynamic-camera.md §3).
    p_arc2 = sample_camera_path(np.random.default_rng(6), "arc", az, el, d, look, DT)
    ga = np.stack([p_arc2.gravity_camera(t) for t in range(64)])
    span = np.degrees(np.arccos(np.clip(ga @ ga[0], -1.0, 1.0))).max()
    check("arc varies tilt but keeps x == 0 (moves ALONG the degeneracy)",
          np.allclose(ga[:, 0], 0.0, atol=1e-12) and span > 0.5,
          f"tilt span {span:.2f} deg, |x|max={np.abs(ga[:,0]).max():.1e}")

    # 7. DETERMINISM — same seed, same path (twin machinery depends on it).
    a = sample_camera_path(np.random.default_rng(7), "gait", az, el, d, look, DT)
    b = sample_camera_path(np.random.default_rng(7), "gait", az, el, d, look, DT)
    check("same seed -> identical path", all(
        np.allclose(a.pose(t)[0], b.pose(t)[0]) and a.pose(t)[2] == b.pose(t)[2]
        for t in range(32)))

    # 8. CLAMPS hold for the whole clip, not just at t=0.
    p_arc = sample_camera_path(np.random.default_rng(4), "arc", az, 66.0, d, look, DT,
                               el_clamp_deg=(12.0, 68.0), d_clamp=(0.8, 4.0))
    zs = [p_arc.pose(t)[0][2] for t in range(256)]
    els = [math.degrees(math.asin(np.clip(
        (p_arc.pose(t)[0] - np.asarray(look))[2]
        / np.linalg.norm(p_arc.pose(t)[0] - np.asarray(look)), -1, 1))) for t in range(256)]
    check("elevation clamp holds across the clip",
          max(els) <= 68.0 + 1e-6 and min(els) >= 12.0 - 1e-6,
          f"el range [{min(els):.1f}, {max(els):.1f}] deg")
    check("camera stays above the floor", min(zs) > 0.0, f"min z={min(zs):.3f} m")

    # 9. EGOMOTION — zero for a static camera, non-zero and finite otherwise.
    v0, w0 = egomotion_labels(p_static, 0)
    check("static camera: egomotion == 0",
          np.allclose(v0, 0, atol=1e-12) and np.allclose(w0, 0, atol=1e-12))
    v1, w1 = egomotion_labels(p_gait, 10)
    check("gait camera: egomotion non-zero and finite",
          np.isfinite(v1).all() and np.isfinite(w1).all()
          and np.linalg.norm(v1) > 1e-6 and np.linalg.norm(w1) > 1e-6,
          f"|v|={np.linalg.norm(v1):.4f} m/s  |w|={np.linalg.norm(w1):.4f} rad/s")

    # 10. is_static() must agree with the actual pose sequence.
    check("is_static() agrees with the pose sequence",
          p_static.is_static() and not p_gait.is_static() and not p_roll.is_static())

    # 11. Provenance round-trips: to_dict() -> CameraPath -> same poses.
    d2 = p_gait.to_dict()
    d2["jitter"] = tuple(tuple(tuple(c) for c in axis) for axis in d2["jitter"])
    d2["look0"] = tuple(d2["look0"])
    rebuilt = CameraPath(**d2)
    check("to_dict() round-trips to identical poses",
          all(np.allclose(rebuilt.pose(t)[0], p_gait.pose(t)[0]) for t in range(32)))

    # 11b. USD EULER ROUND TRIP — the camera prim must be posed EXACTLY as the
    #      logged extrinsics say, or every camera-frame label is silently wrong.
    #      Checked across families (roll present) and along the whole clip.
    ok = True
    for kind in ("static", "roll", "handheld", "gait", "arc"):
        pth = sample_camera_path(np.random.default_rng(11), kind, az, el, d, look, DT)
        for t in range(0, 64, 7):
            R_cw, _ = pth.extrinsics(t)
            rx, ry, rz = usd_rotation_xyz_deg(R_cw)
            ok &= np.allclose(euler_xyz_deg_to_rotmat(rx, ry, rz), R_cw.T, atol=1e-9)
    check("USD rotateXYZ round-trips to the logged extrinsics", ok)

    # 11c. A look-at-only camera CANNOT express roll — the reason the dynamic
    #      path must set an explicit rotation (usd_rotation_xyz_deg docstring).
    p_r = sample_camera_path(np.random.default_rng(12), "roll", az, el, d, look, DT)
    pos_t, look_t, roll_t = p_r.pose(7)
    R_with, _ = look_at_basis(pos_t, look_t, roll_rad=roll_t)
    R_without, _ = look_at_basis(pos_t, look_t, roll_rad=0.0)
    check("look_at alone loses roll (so rotation= is mandatory)",
          abs(roll_t) > 1e-3 and not np.allclose(R_with, R_without, atol=1e-6),
          f"roll={np.degrees(roll_t):+.2f} deg")

    # 12. Degenerate-camera guard still fires (near-top-down NaN, critique #12).
    try:
        look_at_basis((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
        check("degenerate camera raises", False, "no exception")
    except ValueError:
        check("degenerate camera raises", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all camera-path invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
