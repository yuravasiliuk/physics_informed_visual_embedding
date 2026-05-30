"""Ballistic-trajectory correctness check for collect_clip.py.

Setup (per magnitude):
    1. A single sphere sits at rest on the procedural ground plane (z = R).
    2. For 1.0 s of clip time the sphere settles (velocity stays ~zero).
    3. At t = 1.0 s an instantaneous impulse is applied so the sphere's
       resulting velocity points 45 deg above +X with magnitude v0. This is
       equivalent to a force F integrated over one physics step such that
       J = F * physics_dt = m * v0.
    4. The clip continues capturing for ~1.5 s of free flight.

The recorded `state.json` (camera-frame CoM) is inverted back to world frame
via R_camera_world from meta.json, then compared against the closed-form
projectile trajectory `x0 + v0*t + 0.5*g*t^2`. Comparison is masked to the
airborne phase (frames between launch and ground impact at z = R).

Three impulse magnitudes are exercised so residuals can be checked for
proportional scaling. Reuses collect_clip's annotator + frame-logging code
path so any bug in pose/velocity logging, camera-frame conversion, or save
layout surfaces here without touching the production module.

    ${ISAAC_SIM}/python.sh src/data_gen/verify_collect_clip.py \
        --result ~/data/verify_collect_clip
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import carb
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", "file:///home/yura/isaac_assets",
)

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

import omni.replicator.core as rep
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere, GroundPlane, VisualCuboid
from omni.isaac.core.utils.stage import create_new_stage

import lighting
from clip_capture import (
    ClipConfig,
    build_metadata,
    capture,
    log_state_frame,
    make_annotators,
    read_local_com,
    rotmat_to_quat_wxyz,
    save_clip,
    world_to_camera_frame,
)
from seg_labels import apply_class_label


SPHERE_RADIUS = 0.1
# DynamicSphere defaults to 0.02 kg when neither mass nor density is passed
# (isaacsim/core/api/objects/sphere.py). At that mass, even J=1 N*s yields
# v=50 m/s and the ball clears the FOV in the first frame. Pin to 1 kg so
# impulse-in-N*s equals speed-in-m/s, and impulses below stay legible.
SPHERE_MASS_KG = 1.0
SPHERE_NAME = "probe_sphere"
SPHERE_PRIM = f"/World/{SPHERE_NAME}"
SPHERE_INIT_POS = np.array([0.0, 0.0, SPHERE_RADIUS])
LAUNCH_TIME_S = 1.0
LAUNCH_ANGLE_DEG = 45.0
LAUNCH_DIRECTION_HORIZ = np.array([1.0, 0.0, 0.0])
GRAVITY = 9.81
IMPULSE_MAGNITUDES = (1.0, 2.0, 3.0)  # N*s; speed = J / mass (~kg^-1)
DEFAULT_OUT = os.path.expanduser("~/data/verify_collect_clip")

# Small colored tiles dropped along +X so the broadside camera has visible
# scale references and the ball's progress against the ground is readable.
GROUND_MARKER_X_POSITIONS = (0.0, 0.5, 1.0, 1.5, 2.0)
GROUND_MARKER_COLORS = (
    (0.9, 0.9, 0.1),  # yellow at origin (launch)
    (0.1, 0.7, 0.9),  # cyan
    (0.9, 0.5, 0.1),  # orange
    (0.1, 0.9, 0.3),  # green
    (0.7, 0.1, 0.9),  # purple
)


def _add_ground_markers():
    h = 0.005
    side = 0.08
    for i, (x, color) in enumerate(zip(GROUND_MARKER_X_POSITIONS, GROUND_MARKER_COLORS)):
        VisualCuboid(
            prim_path=f"/World/marker_{i:02d}",
            name=f"marker_{i:02d}",
            position=np.array([x, 0.0, h * 0.5]),
            scale=np.array([side, side, h]),
            color=np.array(color, dtype=np.float32),
        )


def build_sphere_scene(world):
    """Place a ground plane + a single sphere resting on it at the origin.

    Installs the same fallback distant light collect_clip uses when no HDRI
    or environment is loaded — without it the scene renders all-black.
    """
    lighting.install_fallback_distant_light()
    GroundPlane(
        prim_path="/World/ground_plane", z_position=0.0,
        color=np.array([0.4, 0.4, 0.4], dtype=np.float32),
    )
    _add_ground_markers()
    sphere = world.scene.add(DynamicSphere(
        prim_path=SPHERE_PRIM,
        name=SPHERE_NAME,
        position=SPHERE_INIT_POS,
        radius=SPHERE_RADIUS,
        mass=SPHERE_MASS_KG,
        color=np.array([0.85, 0.2, 0.2], dtype=np.float32),
    ))
    stage = omni.usd.get_context().get_stage()
    apply_class_label(stage.GetPrimAtPath(SPHERE_PRIM), SPHERE_NAME)
    body_specs = [{
        "id": 1,
        "name": SPHERE_NAME,
        "shape": "sphere",
        "asset_url": None,
        "dims": [SPHERE_RADIUS],
        "init_position_world": SPHERE_INIT_POS.tolist(),
    }]
    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "scenario": "ballistic_verify",
            "launch_time_s": LAUNCH_TIME_S,
            "launch_angle_deg": LAUNCH_ANGLE_DEG,
            "launch_direction_world": LAUNCH_DIRECTION_HORIZ.tolist(),
        },
        "ground": {"color_rgb": [0.4, 0.4, 0.4]},
    }
    return [sphere], body_specs, scene_specs


def launch_velocity_world(speed):
    angle = np.deg2rad(LAUNCH_ANGLE_DEG)
    return speed * (np.cos(angle) * LAUNCH_DIRECTION_HORIZ
                    + np.sin(angle) * np.array([0.0, 0.0, 1.0]))


def simulate_with_launch(world, bodies, launch_frame, v_launch_world, annots,
                         config, R_camera_world, camera_pos_world, q_camera_world):
    """Step the world; on `launch_frame` set the sphere's linear velocity.

    Mirrors collect_clip.simulate_clip's per-frame loop so log_state_frame
    output is bit-compatible with production state.json.
    """
    world.reset()
    local_coms = [read_local_com(body) for body in bodies]

    frames = []
    state_records = []
    for frame_idx in range(config.frame_count):
        if frame_idx == launch_frame:
            bodies[0].set_linear_velocity(np.asarray(v_launch_world))
        world.step(render=True)
        rep.orchestrator.step()
        frames.append(capture(annots))
        log_state_frame(
            bodies, local_coms, frame_idx, state_records,
            R_camera_world=R_camera_world,
            camera_pos_world=camera_pos_world,
            q_camera_world=q_camera_world,
        )
    return frames, state_records


def analytical_world_positions(launch_frame, v_launch_world, config):
    """Closed-form CoM positions per captured frame, world frame.

    Frame indexing: on `launch_frame`, the velocity is set BEFORE world.step,
    so the logged pose at frame == launch_frame has already advanced by one
    render_dt. Hence t_since_launch = (i - launch_frame + 1) * render_dt for
    i >= launch_frame.
    """
    poses = np.zeros((config.frame_count, 3))
    g_vec = np.array([0.0, 0.0, -GRAVITY])
    for i in range(config.frame_count):
        if i < launch_frame:
            poses[i] = SPHERE_INIT_POS
        else:
            t = (i - launch_frame + 1) * config.render_dt
            poses[i] = SPHERE_INIT_POS + v_launch_world * t + 0.5 * g_vec * t * t
    return poses


def world_pos_from_camera(state_records, R_camera_world, camera_pos_world):
    R_inv = R_camera_world.T
    out = np.zeros((len(state_records), 3))
    for i, rec in enumerate(state_records):
        com_cam = np.asarray(rec["com_camera"], dtype=np.float64)
        out[i] = R_inv @ com_cam + camera_pos_world
    return out


def compare(state_records, expected_world, launch_frame, v_launch_world, config,
            R_camera_world, camera_pos_world):
    actual_world = world_pos_from_camera(state_records, R_camera_world, camera_pos_world)
    residual = actual_world - expected_world
    err = np.linalg.norm(residual, axis=1)

    # Airborne window: from launch up to predicted ground impact (z back to R).
    vz0 = float(v_launch_world[2])
    t_land = 2.0 * vz0 / GRAVITY if vz0 > 0 else 0.0
    last_airborne = launch_frame + int(np.floor(t_land / config.render_dt)) - 1
    last_airborne = min(last_airborne, config.frame_count - 1)
    air_slice = slice(launch_frame, last_airborne + 1)

    return {
        "max_err_m": float(err.max()),
        "median_err_m": float(np.median(err)),
        "airborne_max_err_m": float(err[air_slice].max()) if last_airborne >= launch_frame else None,
        "airborne_median_err_m": float(np.median(err[air_slice])) if last_airborne >= launch_frame else None,
        "predicted_t_land_s": t_land,
        "predicted_landing_frame": last_airborne,
        "per_frame_err_m": err.tolist(),
        "actual_world_xyz": actual_world.tolist(),
        "expected_world_xyz": expected_world.tolist(),
    }


def run_one(out_dir, impulse_magnitude, config, seed):
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    bodies, body_specs, scene_specs = build_sphere_scene(world)
    annots = make_annotators(
        config.camera_position, config.camera_look_at, config.resolution,
    )
    R_camera_world, camera_pos_world = world_to_camera_frame(
        config.camera_position, config.camera_look_at,
    )
    q_camera_world = rotmat_to_quat_wxyz(R_camera_world)

    # Mass is read post-reset (PhysX assigns it from density + radius), so
    # do a no-op reset first to materialize the rigid body view.
    world.reset()
    mass = float(bodies[0].get_mass())
    speed = impulse_magnitude / mass
    v_launch = launch_velocity_world(speed)

    launch_frame = int(round(LAUNCH_TIME_S / config.render_dt))
    frames, state_records = simulate_with_launch(
        world, bodies, launch_frame, v_launch, annots, config,
        R_camera_world, camera_pos_world, q_camera_world,
    )
    expected = analytical_world_positions(launch_frame, v_launch, config)
    comparison = compare(
        state_records, expected, launch_frame, v_launch, config,
        R_camera_world, camera_pos_world,
    )

    meta = build_metadata(
        config, body_specs, scene_specs, "ballistic_verify",
        R_camera_world, seed,
    )
    meta["verify"] = {
        "impulse_N_s": float(impulse_magnitude),
        "sphere_mass_kg": mass,
        "launch_frame": launch_frame,
        "launch_speed_m_s": float(speed),
        "launch_velocity_world": v_launch.tolist(),
        "max_err_m": comparison["max_err_m"],
        "median_err_m": comparison["median_err_m"],
        "airborne_max_err_m": comparison["airborne_max_err_m"],
        "airborne_median_err_m": comparison["airborne_median_err_m"],
        "predicted_t_land_s": comparison["predicted_t_land_s"],
        "predicted_landing_frame": comparison["predicted_landing_frame"],
    }
    save_clip(out_dir, frames, state_records, meta)
    with open(os.path.join(out_dir, "verify.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(
        f"[{out_dir}] J={impulse_magnitude:.2f} N*s  m={mass:.3f} kg  "
        f"v0={speed:.2f} m/s  airborne_max_err={comparison['airborne_max_err_m']*1000:.1f} mm  "
        f"airborne_median_err={comparison['airborne_median_err_m']*1000:.1f} mm"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    base_out = Path(os.path.abspath(args.result))
    base_out.mkdir(parents=True, exist_ok=True)

    # Broadside camera: looking along -Y at the +X flight path. The arc
    # then projects onto the image as a clean 2D parabola (image-X tracks
    # world-X, image-Y tracks world-Z), making FOV budgeting predictable
    # and making it visually obvious if the trajectory ever curves out of
    # the launch plane.
    config = dataclasses.replace(
        ClipConfig(),
        frame_count=60,                          # 2.0 s at 30 fps render
        camera_position=(1.0, -3.0, 1.2),
        camera_look_at=(1.0, 0.0, 0.4),
    )

    for i, J in enumerate(IMPULSE_MAGNITUDES):
        if i > 0:
            World.clear_instance()
            create_new_stage()
        out_dir = str(base_out / f"impulse_{J:.2f}".replace(".", "p", 1))
        run_one(out_dir, J, config, seed=args.seed + i)

    simulation_app.close()


if __name__ == "__main__":
    main()
