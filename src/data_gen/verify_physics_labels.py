"""Closed-form correctness checks for the risky physics-label code paths.

W4 of `plans/data-gen/physics-diversification.md` (critique #4 / #11). The original
`verify_collect_clip.py` only exercises a single CoM=0 sphere in free flight,
so it never touches the label code that W2 (authored offset CoM) and W3
(spin / collision) now lean on. This sibling adds four closed-form cases, all
shot from the PRODUCTION camera (2.2,2.2,1.8 look-at 0,0,0.1) so verify tests
what we actually ship rather than a bespoke broadside rig:

All checks are deliberately warmup-independent (no absolute frame->time closed
form) — an earlier draft tripped on a one-render-step physics warmup; the checks
below use only constant/relative/body-local quantities.

  A. spin_cuboid    — uniform cube spun about world +Z in free fall. ang_vel is
                      torque-free constant; orientation is a pure +Z rotation.
                      Validates R_camera_world @ ang_vel_world (sign/scale) and
                      quat_wxyz_mul order (a wrong order leaks camera rotation
                      into the reconstructed body quat -> non-zero x,y); the
                      per-frame angular increment must equal omega*render_dt.

  B. (removed 2026-07-28 with the CoM target — see the note where it lived.)

  C. two_body       — two bodies at well-separated start points; asserts the
                      per-body state records stay matched to the right body
                      (no body-index <-> label swap).

  D. composite      — composite glued body through the PRODUCTION path
                      (body_factory.spawn_composite_body, W5): a cuboid split
                      into density slabs. Asserts PhysX's integrated mass / CoM
                      equal the analytic ΣρV / ΣρV·t form the label assumes —
                      the per-shape-density composition is the one uncertain
                      runtime behaviour. Same xy-constant / gravity trajectory
                      sanity the retired offset_com case carried.

  E. production_cam  — satisfied implicitly: every case above runs at the
                      production camera pose.

  F. moving_camera  — ★ the DYNAMIC-camera gate (dynamic-camera.md §7). Every
                      camera-frame label now depends on the frame-`t` camera, so
                      a single off-by-one silently rewrites the whole corpus and
                      still looks physically plausible. Asserts, per frame:
                        * exact reprojection — world CoM reconstructed from
                          `com_camera(t)` through `R_cw(t)` equals the
                          camera-agnostic `get_world_pose` (collect_raw_pose),
                          so this is independent of what the bodies did;
                        * the same for orientation via quat composition;
                        * a NEGATIVE CONTROL: redoing the reprojection with the
                          frame-shifted extrinsics must be orders of magnitude
                          worse, proving the check can actually see an
                          off-by-one rather than passing vacuously;
                        * on the logged camera records, `g_cam` unit-norm and
                          the equivariance identity g(t+1) == R_rel(t)·g(t), and
                          egomotion matching finite differences of the logged
                          extrinsics.

Out of scope (no clean closed form): general tumbling (Euler precession),
PhysX CoM on real meshes, and the composite *inertia* tensor (PhysX-computed,
no closed form — only mass + CoM are verified).

    ${ISAAC_SIM}/python.sh src/data_gen/verify_physics_labels.py \
        --result ~/data/verify_physics_labels
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

# `os` is imported out of alphabetical order on purpose: the carb override below
# needs it and must run before the omni/isaac imports.
import os

import carb
# Local Isaac asset root — same override collect_clip applies at import, kept
# here because this module boots its own SimulationApp before importing it.
# `~` is expanded rather than hardcoded so the path is not machine-specific.
ISAAC_ASSETS_DIR = os.path.expanduser("~/isaac_assets")
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", f"file://{ISAAC_ASSETS_DIR}",
)

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, GroundPlane
from omni.isaac.core.utils.stage import create_new_stage

import body_factory
import lighting
from collect_clip import simulate_clip
from seg_labels import apply_class_label
from camera_motion import sample_any_path
from camera_rig import CameraRig
from clip_capture import (
    build_metadata,
    quat_wxyz_mul,
    quat_wxyz_to_rotmat,
    read_local_com,
    rotmat_to_quat_wxyz,
    save_clip,
    world_to_camera_frame,
)
from config import ClipConfig

GRAVITY = 9.81
DEFAULT_OUT = os.path.expanduser("~/data/verify_physics_labels")

# Tolerances. All checks are now warmup-independent (no absolute-time closed
# form), so these are tight. COM_READBACK is the scale-fix regression: measured
# CoM vs authored offset must agree to sub-mm.
ANGVEL_TOL = 5e-3
COM_READBACK_TOL_M = 1e-3


# --- helpers ----------------------------------------------------------------

def _airborne_mask(config, bottom_height_m, margin_frames=1.5):
    """Boolean per-frame mask: True while the body is in free flight.

    A body dropped from rest whose lowest point starts `bottom_height_m` above
    the ground impacts at t = sqrt(2 h / g). We keep only frames safely before
    that, so the contact-free closed-form trajectory holds; `margin_frames`
    trims a couple of frames before nominal impact to clear contact onset.
    Every case keeps a ground plane (data-gen scenes always land on something),
    so the closed-form check is scoped to this airborne window rather than the
    whole clip.
    """
    t = _frame_times(config)
    t_land = np.sqrt(2.0 * bottom_height_m / GRAVITY)
    return t < (t_land - margin_frames * config.render_dt)


def _records_for(state_records, body_name):
    """state records for one body, in frame order."""
    return [r for r in state_records if r["body"] == body_name]


def _world_from_cam_point(p_cam, R_cw, cam_pos):
    return R_cw.T @ np.asarray(p_cam, dtype=np.float64) + cam_pos


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_angle_between(qa, qb):
    """Geodesic angle (rad) between two wxyz quaternions (sign-agnostic).

    Normalizes first, then uses the atan2 half-angle form instead of
    `2*arccos(|dot|)`. Both changes matter, and only the tight `moving_camera`
    tolerance exposed why: Isaac's `get_world_pose` returns quaternions
    normalized only to float32 (‖q‖ = 1 − ~2e-7), and arccos AMPLIFIES that
    near zero — arccos(1−ε) ≈ √(2ε) turns a 2e-7 norm deficit into ~9e-4 rad of
    phantom angle. `case_spin_cuboid` never noticed because its tolerance is
    0.02 rad. atan2 is well-conditioned at small angles, so the residual becomes
    the true float64 round-trip error (~1e-15).
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    qa = qa / max(float(np.linalg.norm(qa)), 1e-300)
    qb = qb / max(float(np.linalg.norm(qb)), 1e-300)
    if float(np.dot(qa, qb)) < 0.0:      # q and -q are the same rotation
        qb = -qb
    # dot = cos(θ/2)  =>  ‖qa−qb‖ = 2|sin(θ/4)|, ‖qa+qb‖ = 2|cos(θ/4)|,
    # so the atan2 is θ/4 and the factor is FOUR, not two.
    return 4.0 * float(np.arctan2(float(np.linalg.norm(qa - qb)),
                                  float(np.linalg.norm(qa + qb))))


def _frame_times(config):
    """Elapsed sim time at each logged frame. Velocity is applied before the
    first step, so frame i has advanced (i+1) render steps."""
    return (np.arange(config.frame_count) + 1) * config.render_dt


# --- case A: spinning uniform cuboid ----------------------------------------

def case_spin_cuboid(world, config, R_cw, cam_pos, q_cw, seed):
    omega = 4.0  # rad/s about world +Z
    side = 0.3
    start = np.array([0.0, 0.0, 1.0])            # high enough for a clean window
    cuboid = world.scene.add(DynamicCuboid(
        prim_path="/World/spin_cuboid", name="spin_cuboid",
        position=start, scale=[side, side, side],
        color=np.array([0.2, 0.5, 0.9], dtype=np.float32),
    ))
    spin_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/spin_cuboid")
    apply_class_label(spin_prim, "spin_cuboid")
    body_factory.zero_damping(spin_prim)   # match production: no PhysX default damping
    init_v = [(np.zeros(3), np.array([0.0, 0.0, omega]))]
    rollout = simulate_clip(
        world, [cuboid], init_v, _rig(config), config,
    )
    frames, recs = rollout.frames[0], rollout.state_records[0]

    # Torque-free (constant omega about +Z) only while airborne; a cube spinning
    # about +Z keeps its bottom face flat, so the lowest point stays at
    # start_z - side/2 until impact. Both checks below are warmup-independent
    # (they never reference absolute t), so the frame->time mapping is irrelevant.
    air = _airborne_mask(config, bottom_height_m=start[2] - side / 2.0)
    air_idx = [i for i in range(config.frame_count) if air[i]]
    body_recs = _records_for(recs, "spin_cuboid")

    # (1) ang_vel: constant in world, logged in camera frame.
    expected_angvel_cam = R_cw @ np.array([0.0, 0.0, omega])
    angvel_err = max(
        float(np.linalg.norm(np.asarray(body_recs[i]["ang_vel_camera"]) - expected_angvel_cam))
        for i in air_idx
    )

    # (2) orientation: reconstruct quat_body_world = conj(q_cw) (x) quat_cam.
    # A pure +Z spin has zero x,y vector components — if quat_wxyz_mul's order or
    # sign were wrong, the camera rotation would leak in and break that. Then the
    # per-frame angular increment must equal omega*render_dt (rotation integrates
    # exactly for constant omega), validating magnitude/sense without absolute t.
    qbodies = [
        quat_wxyz_mul(_quat_conj(q_cw), np.asarray(body_recs[i]["quat_camera_wxyz"]))
        for i in air_idx
    ]
    purez_err = float(max(np.hypot(q[1], q[2]) for q in qbodies))
    incs = [_quat_angle_between(a, b) for a, b in zip(qbodies[:-1], qbodies[1:])]
    inc_err = float(max(abs(d - omega * config.render_dt) for d in incs)) if incs else 0.0

    passed = (len(air_idx) >= 4 and angvel_err < ANGVEL_TOL
              and purez_err < 0.02 and inc_err < 0.02)
    return frames, recs, {
        "case": "spin_cuboid",
        "omega_rad_s": omega,
        "airborne_frames": len(air_idx),
        "angvel_camera_max_err": angvel_err,
        "orientation_purez_err": purez_err,
        "orientation_increment_err_rad": inc_err,
        "passed": bool(passed),
    }


# --- case B: REMOVED 2026-07-28 ---------------------------------------------
# `case_offset_com` verified that a primitive's PhysX CoM equalled the offset
# authored by `body_factory.attach_center_of_mass` (the W2/W4 scale-fix
# regression). That authoring was removed with the CoM target, so the case had
# nothing left to assert — a uniform primitive's CoM is now the prim origin by
# construction. The two guarantees it *also* carried are not lost: the
# get_world_pose-double-count guard and the world<-camera reconstruction check
# both live on in `case_composite` (whose CoM offset is real and closed-form)
# and, per frame under a MOVING camera, in `case_moving_camera` below.


# --- case D: composite glued body (W5) --------------------------------------

def case_composite(world, config, R_cw, cam_pos, q_cw, seed):
    """Composite built through the PRODUCTION path (body_factory.
    spawn_composite_body, W5): a uniform-colour cuboid internally split into
    density slabs. PhysX integrates the slab densities into the body mass/CoM;
    the analytic label is mass=ΣρV, com=ΣρV·t/ΣρV. This case proves PhysX agrees
    with that formula — the one genuinely uncertain runtime behaviour W5 leans
    on (per-shape density composition into a single rigid body).

    All checks are body-local / frame-free or relative, so warmup-independent:
      * get_mass() == Σ slab masses (the analytic composite mass), and
      * read_local_com() == ΣρV·t/ΣρV (the analytic offset, recomputed
        independently here from the logged per-slab parts), and
      * the reconstructed world CoM has constant xy under spin (no get_world_pose
        double-count) and a z second difference of -g·dt² (constant gravity).
    """
    rng = np.random.default_rng(seed)
    omega = 3.0
    side = 0.4
    start = np.array([0.0, 0.0, 1.2])
    # Force a clean 2-slab split for an interpretable, single-axis CoM offset.
    case_config = dataclasses.replace(config, composite_n_parts_range=(2, 2))
    body, spec = body_factory.spawn_composite_body(
        world, rng, case_config, 0, (side, side, side), start,
    )

    # Independent analytic recompute from the logged parts (don't just trust
    # spec["com_local_analytic"] — recompute mass=ΣρV, com=ΣρV·t/ΣρV here).
    parts = spec["parts"]
    mass_analytic = float(sum(p["mass"] for p in parts))
    com_analytic = np.sum(
        [p["mass"] * np.asarray(p["offset_local"]) for p in parts], axis=0,
    ) / mass_analytic

    init_v = [(np.zeros(3), np.array([0.0, 0.0, omega]))]
    rollout = simulate_clip(
        world, [body], init_v, _rig(config), config,
    )
    frames, recs = rollout.frames[0], rollout.state_records[0]

    # (1) THE label checks: PhysX-integrated mass / CoM == the analytic ΣρV form.
    mass_measured = float(body.get_mass())
    com_measured = read_local_com(body)
    mass_rel_err = abs(mass_measured - mass_analytic) / mass_analytic
    com_label_err = float(np.linalg.norm(com_measured - com_analytic))

    # (2) Trajectory reconstruction sanity (as in offset_com): xy of the world
    # CoM is constant under spin, z tracks constant gravity.
    air = _airborne_mask(config, bottom_height_m=start[2] - side / 2.0)
    idx = np.where(air)[0]
    com_world = np.stack([
        _world_from_cam_point(r["com_camera"], R_cw, cam_pos)
        for r in _records_for(recs, body.name)
    ])
    xy_std = float(np.linalg.norm(com_world[idx, :2].std(axis=0)))
    dt = config.render_dt
    zz = com_world[idx, 2]
    if len(zz) >= 3:
        second_diff = zz[2:] - 2.0 * zz[1:-1] + zz[:-2]
        accel_err = float(np.max(np.abs(second_diff - (-GRAVITY * dt * dt))))
    else:
        accel_err = float("inf")

    passed = (len(idx) >= 4
              and mass_rel_err < 1e-3
              and com_label_err < COM_READBACK_TOL_M
              and xy_std < 5e-3
              and accel_err < 0.05 * GRAVITY * dt * dt)
    return frames, recs, {
        "case": "composite",
        "airborne_frames": int(len(idx)),
        "n_parts": spec["n_parts"],
        "split_axis": spec["split_axis"],
        "densities": [p["density"] for p in parts],
        "mass_analytic": mass_analytic,
        "mass_measured": mass_measured,
        "mass_rel_err": mass_rel_err,
        "com_local_analytic": com_analytic.tolist(),
        "com_local_measured": com_measured.tolist(),
        "com_offset_norm_m": float(np.linalg.norm(com_analytic)),
        "com_label_err_m": com_label_err,
        "traj_xy_std_m": xy_std,
        "traj_accel_err": accel_err,
        "passed": bool(passed),
        "note": ("mass_rel_err / com_label_err prove PhysX integrates the "
                 "per-slab densities into the body mass/CoM the way the analytic "
                 "ΣρV label assumes. com_offset_norm_m should be clearly non-zero "
                 "(invisible density asymmetry); if ~0 the two slabs drew similar "
                 "densities — bump --seed for a stronger check."),
    }


# --- case F: moving camera — per-frame extrinsics ---------------------------

def case_moving_camera(world, config, R_cw, cam_pos, q_cw, seed):
    """★ Per-frame extrinsics gate (dynamic-camera.md §7).

    Isolation, deliberately: the check compares the logged camera-frame labels
    against `get_world_pose` — the camera-AGNOSTIC world pose — so it is exact
    whether the bodies are moving or still, and any residual is a camera/logging
    fault with no physics component to argue about. That is what test-ladder
    step 2 ("static bodies + moving camera") is really asking, without needing
    the bodies to actually hold still.

    The path is `handheld` with `roll_always`: translation AND rotation AND roll
    together. A pure-translation path would leave `R_cw` constant and the whole
    per-frame question untested.
    """
    rng = np.random.default_rng(seed)
    path = sample_any_path(
        rng, "subject:handheld", config.camera_position, config.camera_look_at,
        config.frame_count, config.render_dt,
        el_clamp_deg=config.camera_elevation_clamp_deg,
        d_clamp=config.camera_radius_clamp, roll_always=True,
    )
    bodies = []
    for i, p in enumerate((np.array([-0.25, 0.0, 0.75]), np.array([0.3, 0.1, 0.95]))):
        b, _ = body_factory.spawn_body(world, rng, config, i, "cuboid", (0.12, 0.1, 0.16), p)
        bodies.append(b)
    init_v = [(np.zeros(3), np.array([0.0, 0.0, 2.5])) for _ in bodies]

    # collect_raw_pose gives the camera-agnostic ground truth; it takes priority
    # over the cam_track return, so the camera records are rebuilt below through
    # the SAME `log_camera_frame` production uses.
    rollout = simulate_clip(
        world, bodies, init_v, CameraRig.from_paths([path], config), config,
        collect_raw_pose=True,
    )
    frames, recs = rollout.frames[0], rollout.state_records[0]
    raw_pose, raw_quat = rollout.raw_pose_world, rollout.raw_quat_world
    local_coms = [read_local_com(b) for b in bodies]

    # (1) EXACT reprojection, per frame, per body.
    pos_err = quat_err = 0.0
    shifted_err = 0.0          # negative control, see below
    T = config.frame_count
    for b_i, body in enumerate(bodies):
        brecs = _records_for(recs, body.name)
        for t in range(T):
            R_t, p_t = path.extrinsics(t)
            R_body = quat_wxyz_to_rotmat(np.asarray(raw_quat[b_i][t]))
            com_w_true = np.asarray(raw_pose[b_i][t]) + R_body @ local_coms[b_i]
            com_w_got = _world_from_cam_point(brecs[t]["com_camera"], R_t, p_t)
            pos_err = max(pos_err, float(np.linalg.norm(com_w_got - com_w_true)))

            q_got = quat_wxyz_mul(_quat_conj(rotmat_to_quat_wxyz(R_t)),
                                  np.asarray(brecs[t]["quat_camera_wxyz"]))
            quat_err = max(quat_err, _quat_angle_between(
                q_got, np.asarray(raw_quat[b_i][t])))

            # NEGATIVE CONTROL: the same reprojection with the NEXT frame's
            # extrinsics. If this is not far worse, the camera barely moved
            # between frames and (1) is passing vacuously — the failure mode
            # that would let a genuine off-by-one through unnoticed.
            if t + 1 < T:
                R_s, p_s = path.extrinsics(t + 1)
                shifted_err = max(shifted_err, float(np.linalg.norm(
                    _world_from_cam_point(brecs[t]["com_camera"], R_s, p_s) - com_w_true)))

    # (2) The camera records THE PRODUCTION LOOP WROTE — not a re-derivation.
    # Before RolloutResult these were unreachable here (collect_raw_pose took
    # priority over the cam_track return and silently dropped them), so this
    # check had to rebuild them by calling log_camera_frame itself — which
    # tested the logger but not the wiring that feeds it. Now it tests both.
    cam_track = rollout.camera_records[0]
    assert len(cam_track) == T, f"camera_records has {len(cam_track)} of {T} frames"
    g = np.array([r["g_camera"] for r in cam_track])
    g_norm_err = float(np.abs(np.linalg.norm(g, axis=1) - 1.0).max())
    equiv_err = 0.0
    ego_err = 0.0
    for t in range(T - 1):
        R0, p0 = path.extrinsics(t)
        R1, p1 = path.extrinsics(t + 1)
        equiv_err = max(equiv_err, float(np.abs((R1 @ R0.T) @ g[t] - g[t + 1]).max()))
        v_fd = R0 @ ((p1 - p0) / config.render_dt)
        ego_err = max(ego_err, float(np.abs(
            np.asarray(cam_track[t]["lin_vel_camera"]) - v_fd).max()))
    tilt_span = float(np.abs(g[:, 0]).max())

    passed = (pos_err < 1e-6 and quat_err < 1e-6
              and shifted_err > 100.0 * max(pos_err, 1e-9)
              and g_norm_err < 1e-9 and equiv_err < 1e-9 and ego_err < 1e-9
              and tilt_span > 1e-3)
    return frames, recs, {
        "case": "moving_camera",
        "camera_family": path.kind,
        "frames": T,
        "reproj_pos_max_err_m": pos_err,
        "reproj_quat_max_err_rad": quat_err,
        "frame_shifted_control_err_m": shifted_err,
        "g_camera_norm_err": g_norm_err,
        "g_camera_equivariance_err": equiv_err,
        "egomotion_vs_finite_diff_err": ego_err,
        "tilt_abs_gx_max": tilt_span,
        "passed": bool(passed),
        "note": ("reproj_* must be ~0: the camera-frame labels are compared "
                 "against get_world_pose, which no camera touches. "
                 "frame_shifted_control_err_m is the SAME reprojection done with "
                 "frame t+1's extrinsics and must be far larger — otherwise the "
                 "camera hardly moved and the test proves nothing. "
                 "tilt_abs_gx_max > 0 confirms roll is present, without which "
                 "g_cam.x is identically 0 and the tilt label is degenerate."),
    }, cam_track


# --- case C: two-body index alignment ---------------------------------------

def case_two_body(world, config, R_cw, cam_pos, q_cw, seed):
    rng = np.random.default_rng(seed)
    starts = [np.array([-0.6, 0.0, 0.5]), np.array([0.6, 0.0, 0.5])]
    bodies = []
    for i, p in enumerate(starts):
        b, _ = body_factory.spawn_body(
            world, rng, config, i, "sphere", (0.08,), p,
        )
        bodies.append(b)
    init_v = [(np.zeros(3), np.zeros(3)) for _ in bodies]
    rollout = simulate_clip(
        world, bodies, init_v, _rig(config), config,
    )
    frames, recs = rollout.frames[0], rollout.state_records[0]

    # First logged frame: each body's com_world must sit near its own spawn xy,
    # not the other's. A swap shows up as the wrong start point.
    max_xy_err = 0.0
    for b_i, body in enumerate(bodies):
        first = _records_for(recs, body.name)[0]
        com_w = _world_from_cam_point(first["com_camera"], R_cw, cam_pos)
        max_xy_err = max(max_xy_err, float(np.linalg.norm(com_w[:2] - starts[b_i][:2])))
    passed = max_xy_err < 0.05
    return frames, recs, {
        "case": "two_body",
        "names": [b.name for b in bodies],
        "max_start_xy_err_m": max_xy_err,
        "passed": bool(passed),
    }


# --- small helpers ----------------------------------------------------------

def _rig(config):
    """The production static camera, as a one-camera rig (no gating)."""
    return CameraRig.fixed(config.camera_position, config.camera_look_at, config)


# --- driver -----------------------------------------------------------------

CASES = {
    "spin_cuboid": case_spin_cuboid,
    "moving_camera": case_moving_camera,
    "two_body": case_two_body,
    "composite": case_composite,
}


def run_case(name, base_out, config, seed):
    World.clear_instance()
    create_new_stage()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    # Ground plane + lighting, matching production scenes — bodies must land on
    # something. Each case spawns high enough that its closed-form check runs
    # over the airborne window before impact (see _airborne_mask); post-impact
    # frames are still captured/saved but excluded from the checks.
    lighting.install_fallback_distant_light()
    GroundPlane(
        prim_path="/World/ground_plane", z_position=0.0,
        color=np.array([0.4, 0.4, 0.4], dtype=np.float32),
    )

    R_cw, cam_pos = world_to_camera_frame(config.camera_position, config.camera_look_at)
    q_cw = rotmat_to_quat_wxyz(R_cw)

    out = CASES[name](world, config, R_cw, cam_pos, q_cw, seed)
    # A case may return a 4th element: the per-frame camera records for a moving
    # camera. Length-checked so the three static cases stay untouched.
    frames, recs, result = out[0], out[1], out[2]
    camera_records = out[3] if len(out) > 3 else None

    out_dir = str(Path(base_out) / name)
    meta = build_metadata(config, [], {"static_assets": [], "scenario_specs": {}},
                          f"verify_{name}", R_cw, seed)
    meta["verify"] = result
    save_clip(out_dir, frames, recs, meta, save_resolution=config.save_resolution,
              camera_records=camera_records)
    with open(os.path.join(out_dir, "verify.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", type=str, default=None, choices=sorted(CASES),
                   help="Run a single case (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    base_out = Path(os.path.abspath(args.result))
    base_out.mkdir(parents=True, exist_ok=True)

    # Free-fall cases leave the FOV quickly; a short clip keeps state logging
    # cheap (physics state is logged regardless of visibility).
    config = dataclasses.replace(ClipConfig(), frame_count=24)

    names = [args.case] if args.case else list(CASES)
    results = []
    for i, name in enumerate(names):
        r = run_case(name, base_out, config, seed=args.seed + i)
        results.append(r)
        print(f"[{name}] passed={r['passed']}  {json.dumps({k: v for k, v in r.items() if k not in ('case', 'passed', 'note')})}")

    n_pass = sum(int(r["passed"]) for r in results)
    print(f"summary: {n_pass}/{len(results)} cases passed")
    simulation_app.close()


if __name__ == "__main__":
    main()
