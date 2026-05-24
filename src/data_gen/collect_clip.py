"""Reusable Isaac Sim clip-collection module.

Refactor of `collect_smoke.py` (steps 1-2 of `plans/probe-dataset.md` §11).
Every piece is a pure function with explicit parameters so a shard-level
driver can reuse it without copy-paste.

Step 2 adds the randomization knobs called out in §7 of the plan:
    - object count (3-8)
    - shape mix (cuboid / sphere / cylinder / capsule)
    - per-axis scale
    - per-body color (HSV-sampled)
    - per-body physics material (friction + restitution)
    - initial angular velocity range
    - rejection-sampled non-overlapping initial poses
    - per-body specs captured in meta.json

Step 3 is the camera-frame contract change (plan §13, §14):
    - lin_vel, ang_vel, body orientation are all stored in camera frame.
    - World-frame velocities are not persisted; world frame is recoverable
      from `R_camera_world` in meta.json if needed.

Step 5 adds the §9 sanity-check pass. `validate_clip` returns (ok, reason);
`collect_clip` embeds the result in `meta.json` and propagates it so the
shard driver (step 7) can re-seed on failure. Step 4 (compression encoders)
is deliberately deferred — current `save_clip` layout is fine for end-to-end
validation under Isaac Sim.

    ${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py --result out/clip_0000
"""
from isaacsim import SimulationApp

# SimulationApp must be constructed before any other omni / isaac import.
simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

# Local Isaac asset root — avoids re-fetching USDs from S3 every clip.
# Tree under this path mirrors s3://omniverse-content-production/Assets/Isaac/5.1/.
# Must be set after SimulationApp init: Kit loads ~/.local/share/ov/data/Kit/
# Isaac-Sim*/user.config.json on startup (which pins the setting to the S3 URL),
# and that file is read *after* the SimulationApp({}) dict — so an override has
# to happen at runtime.
import carb
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", "file:///home/yura/isaac_assets"
)

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane

import lighting
import scenarios


@dataclass(frozen=True)
class ClipConfig:
    """Per-clip generation parameters. Defaults follow §7 of plans/probe-dataset.md."""

    # Capture
    frame_count: int = 32                     # K — plan §7
    resolution: tuple = (640, 640)            # (width, height) — plan §7
    physics_dt: float = 1.0 / 60.0            # physics step — plan §7 (60 Hz)
    render_dt: float = 1.0 / 30.0             # render step — plan §7 (30 fps)
    camera_position: tuple = (1.2, 1.2, 0.8)  # world frame
    camera_look_at: tuple = (0.0, 0.0, 0.1)   # world frame

    # Scene randomization
    scenarios: tuple = (
        "drop", "roll", "throw",
        "drop_on_platform", "settle", "push",
        "cluttered",
    )                                                                         # uniform sampling per clip
    n_bodies_range: tuple = (3, 8)                                            # inclusive
    shape_choices: tuple = ("cuboid", "sphere", "cylinder", "capsule")
    scale_range: tuple = (0.05, 0.15)                                         # m; per-axis (cuboid) or radius/height
    color_hue_range: tuple = (0.0, 1.0)
    color_sat_range: tuple = (0.5, 1.0)
    color_val_range: tuple = (0.4, 0.9)
    static_friction_range: tuple = (0.3, 0.8)
    dynamic_friction_range: tuple = (0.3, 0.8)
    restitution_range: tuple = (0.0, 0.4)
    init_xy_range: tuple = (-0.3, 0.3)
    init_height_range: tuple = (0.4, 1.2)
    init_ang_vel_range: tuple = (-2.0, 2.0)                                   # rad/s, uniform per axis
    pose_rejection_max_attempts: int = 100

    # PBR visual materials — Tier 3 (plan §5 / step 5.10).
    roughness_range: tuple = (0.2, 0.9)
    metallic_prob: float = 0.15
    metallic_values: tuple = (0.8, 1.0)

    # Ground colour — Tier 3. Greys with a slight tint (per RGB channel,
    # uniform in this range). Texture-based ground materials and "room"
    # enclosures (plan §5) deferred — they need external PBR assets.
    ground_color_range: tuple = (0.25, 0.55)

    # Lighting (HDRI). Empty `hdri_paths` = no-op, Isaac default lighting applies.
    # Populate with file paths or nucleus URIs to enable HDRI randomization.
    hdri_paths: tuple = ()
    hdri_rotation_range_deg: tuple = (0.0, 360.0)
    hdri_intensity: float = 1000.0

DEFAULT_DATA_PATH = os.path.expanduser('~/data/sim_physics')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=Path(DEFAULT_DATA_PATH))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scenario", type=str, default=None,
        help="Force a single scenario (overrides ClipConfig.scenarios). Used by "
             "the smoke-test loop to exercise one archetype at a time.",
    )
    return p.parse_args()


def build_scene(world: World, rng: np.random.Generator, config: ClipConfig):
    """Sample a scenario, build it, return everything the rest of the pipeline needs.

    Returns:
        bodies              : list of dynamic body handles
        body_specs          : list of dicts for meta.json["bodies"]
        scene_specs         : dict for meta.json["scene"] (static_assets, scenario_specs, ground)
        initial_velocities  : list of (lin_vel_world, ang_vel_world) numpy
                              arrays — applied by simulate_clip AFTER world.reset()
        scenario_name       : str — sampled archetype (e.g., "roll")
    """
    # Procedural ground plane via PhysicsSchemaTools — avoids the asset-server
    # lookup that `world.scene.add_default_ground_plane()` triggers (which
    # requires both /Isaac and /NVIDIA under the asset root and fails on our
    # local /Isaac-only sync). For HDRI-lit scenes we don't need the textured
    # grid environment anyway.
    ground_color = np.array(
        rng.uniform(*config.ground_color_range, size=3), dtype=np.float32,
    )
    GroundPlane(
        prim_path="/World/ground_plane", z_position=0.0, color=ground_color,
    )
    scenario_name = rng.choice(config.scenarios).item()
    bodies, body_specs, scene_specs, initial_velocities = scenarios.build(
        scenario_name, world, rng, config,
    )
    scene_specs["ground"] = {"color_rgb": [float(c) for c in ground_color]}
    return bodies, body_specs, scene_specs, initial_velocities, scenario_name


def make_annotators(camera_position, look_at, resolution):
    """Attach RGB, depth, normals, semantic-seg annotators to a render product.

    Segmentation uses `semantic_segmentation` (not `instance_segmentation_fast`)
    so prims sharing a `class` label collapse to one ID — step 5.8 of
    `plans/scene-diversification.md`. Each primitive/asset body carries its
    own class (its body name) and all scenery shares `class=scenery`, so the
    annotator yields exactly the per-body-plus-scenery grouping the probe
    needs. `colorize=False` keeps the output as raw class IDs.
    """
    camera = rep.create.camera(position=tuple(camera_position), look_at=tuple(look_at))
    rp = rep.create.render_product(camera, tuple(resolution))
    annots = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
        "normals": rep.AnnotatorRegistry.get_annotator("normals"),
        "seg": rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False},
        ),
    }
    for a in annots.values():
        a.attach(rp)
    return annots


def capture(annots):
    """Pull one frame from every annotator and return it as plain arrays."""
    rgb = annots["rgb"].get_data()           # (H, W, 4) uint8
    depth = annots["depth"].get_data()       # (H, W) float32
    normals = annots["normals"].get_data()   # (H, W, 4) float32 in [-1, 1]
    seg = annots["seg"].get_data()           # {"data": uint32 (H,W), "info": {...}} or ndarray

    seg_array = seg["data"] if isinstance(seg, dict) else seg
    seg_info = seg["info"] if isinstance(seg, dict) else None

    return {
        "rgb": rgb[..., :3].astype(np.uint8),
        "depth": depth.astype(np.float16),
        "normals": normals[..., :3].astype(np.float16),
        "seg": seg_array.astype(np.uint32),
        "seg_info": seg_info,
    }


def world_to_camera_frame(camera_pos, look_at, world_up=(0.0, 0.0, 1.0)):
    """Return (R_camera_world, camera_pos_world) for a USD/OpenGL camera.

    USD camera convention: +X right, +Y up, -Z forward (camera looks down -Z).
    R_camera_world rotates a *vector* expressed in world coords into camera coords.
    A *point* transforms as p_cam = R_camera_world @ (p_world - camera_pos_world).
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = look_at - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    R_world_camera = np.stack([right, up, -forward], axis=1)
    R_camera_world = R_world_camera.T
    return R_camera_world, camera_pos


def com_world_to_camera(com_world, R_camera_world, camera_pos_world):
    """Convert a CoM 3D point from world frame to camera frame."""
    return R_camera_world @ (np.asarray(com_world, dtype=np.float64) - camera_pos_world)


def quat_wxyz_mul(q1, q2):
    """Hamilton product of two wxyz quaternions. Returns (4,) float64."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a wxyz quaternion. Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_wxyz_to_rotmat(q):
    """Convert a wxyz quaternion to a 3x3 rotation matrix (inverse of rotmat_to_quat_wxyz)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def read_local_com(body):
    """Body-local CoM offset (3,) via SingleRigidPrim.get_com().

    Reads PhysX's rigid-body view rather than the USD MassAPI attribute:
    PhysX computes a CoM from the collision mesh for assets that ship
    without an authored `centerOfMass`, and the view exposes that computed
    value even though it never round-trips back to USD. A MassAPI-direct
    read would return zeros in that case and mis-report CoM as the prim
    origin — the exact bug step 5.7 is fixing.

    Must be called after `world.reset()` (when the physics view is created).
    Returns zeros for primitives, whose CoM coincides with the prim origin.
    """
    com_pos, _ = body.get_com()
    return np.asarray(com_pos, dtype=np.float64).reshape(3)


def log_state_frame(bodies, local_coms, frame_idx, records, R_camera_world, camera_pos_world, q_camera_world):
    """Append one record per body for the current simulation frame.

    `local_coms[i]` is the body-local CoM offset (read once at scene-build time
    via read_local_com). World CoM = prim_origin + R(quat_body_world) @ com_local
    — correct for asymmetric assets, and reduces to the prim origin for
    uniform-density primitives (where com_local is zero).

    All vectors and orientations are stored in **camera frame**. World frame is
    recoverable from `R_camera_world` in meta.json if downstream code needs it.
    """
    for body, com_local in zip(bodies, local_coms):
        pos_world, quat_body_world = body.get_world_pose()
        pos_world = np.asarray(pos_world, dtype=np.float64)
        quat_body_world = np.asarray(quat_body_world, dtype=np.float64)
        R_body_world = quat_wxyz_to_rotmat(quat_body_world)
        com_world = pos_world + R_body_world @ com_local
        lin_vel_world = np.asarray(body.get_linear_velocity())
        ang_vel_world = np.asarray(body.get_angular_velocity())

        records.append({
            "frame": frame_idx,
            "body": body.name,
            "com_camera": com_world_to_camera(com_world, R_camera_world, camera_pos_world).tolist(),
            "lin_vel_camera": (R_camera_world @ lin_vel_world).tolist(),
            "ang_vel_camera": (R_camera_world @ ang_vel_world).tolist(),
            "quat_camera_wxyz": quat_wxyz_mul(q_camera_world, quat_body_world).tolist(),
            "mass": float(body.get_mass()),
        })


def simulate_clip(world, bodies, initial_velocities, annots, config, R_camera_world, camera_pos_world, q_camera_world):
    """Run the K-frame loop, returning captured frames and per-body state records.

    `initial_velocities` is a list of (lin_vel_world, ang_vel_world) numpy
    arrays aligned with `bodies`. Velocities are applied AFTER `world.reset()`
    because Isaac discards velocities set before reset.
    """
    world.reset()
    for body, (lin_vel, ang_vel) in zip(bodies, initial_velocities):
        body.set_linear_velocity(np.asarray(lin_vel))
        body.set_angular_velocity(np.asarray(ang_vel))

    # MassAPI is authored at scene-build time and constant through the clip,
    # so cache the local CoM offset once instead of re-reading USD per frame.
    local_coms = [read_local_com(body) for body in bodies]

    frames = []
    state_records = []
    for frame_idx in range(config.frame_count):
        world.step(render=True)
        rep.orchestrator.step()  # flush annotator outputs for this frame
        frames.append(capture(annots))
        log_state_frame(
            bodies, local_coms, frame_idx, state_records,
            R_camera_world=R_camera_world,
            camera_pos_world=camera_pos_world,
            q_camera_world=q_camera_world,
        )
    return frames, state_records


def validate_clip(frames, state_records, config: ClipConfig):
    """Run the §9 sanity checks. Returns (ok: bool, reason: str | None).

    Pure function — no side effects, no logging. Caller (CLI or shard driver)
    decides what to do with the result.
    """
    K = config.frame_count

    if len(frames) != K:
        return False, f"frame_count_mismatch: got {len(frames)}, expected {K}"

    for rec in state_records:
        for key in ("com_camera", "lin_vel_camera", "ang_vel_camera", "quat_camera_wxyz"):
            arr = np.asarray(rec[key], dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                return False, f"nonfinite_state: frame={rec['frame']} body={rec['body']} key={key}"

    for f_idx, f in enumerate(frames):
        if int(np.asarray(f["seg"]).max()) == 0:
            return False, f"empty_seg: frame={f_idx} has no foreground"

    by_body: dict = {}
    for rec in state_records:
        by_body.setdefault(rec["body"], []).append(rec)
    all_stationary = True
    for recs in by_body.values():
        peak_v = max(float(np.linalg.norm(r["lin_vel_camera"])) for r in recs)
        peak_w = max(float(np.linalg.norm(r["ang_vel_camera"])) for r in recs)
        if peak_v > 0.01 or peak_w > 0.1:
            all_stationary = False
            break
    if all_stationary:
        return False, "all_bodies_stationary"

    rgb_std = float(np.stack([f["rgb"] for f in frames]).std())
    if rgb_std < 1.0:
        return False, f"rgb_low_variance: std={rgb_std:.3f}"

    return True, None


def build_metadata(config: ClipConfig, body_specs, scene_specs, scenario_name, R_camera_world, seed: int):
    return {
        "schema_version": "rigid-probe/v0-draft",
        "frame_count": config.frame_count,
        "resolution": list(config.resolution),
        "render_fps": 1.0 / config.render_dt,
        "physics_hz": 1.0 / config.physics_dt,
        "seed": seed,
        "scenario": scenario_name,
        "scene": scene_specs,
        "bodies": body_specs,
        "camera_position_world": list(config.camera_position),
        "camera_look_at_world": list(config.camera_look_at),
        "R_camera_world": R_camera_world.tolist(),
    }


def save_clip(out_dir: str, frames, state_records, meta):
    """Write the smoke-style output layout: mp4 + .npy stacks + JSON sidecars."""
    os.makedirs(out_dir, exist_ok=True)

    rgb_stack = np.stack([f["rgb"] for f in frames])           # (T, H, W, 3) uint8
    depth_stack = np.stack([f["depth"] for f in frames])       # (T, H, W) float16
    normals_stack = np.stack([f["normals"] for f in frames])   # (T, H, W, 3) float16
    seg_stack = np.stack([f["seg"] for f in frames])           # (T, H, W) uint32

    height, width = rgb_stack.shape[1:3]
    video_file = os.path.join(out_dir, "rgb.mp4")
    writer = cv2.VideoWriter(
        video_file, cv2.VideoWriter_fourcc(*"mp4v"),
        float(meta["render_fps"]), (width, height),
    )
    for frame in rgb_stack:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    np.save(os.path.join(out_dir, "depth.npy"), depth_stack)
    np.save(os.path.join(out_dir, "normals.npy"), normals_stack)
    np.save(os.path.join(out_dir, "seg.npy"), seg_stack)

    seg_info = frames[0].get("seg_info")
    if seg_info is not None:
        with open(os.path.join(out_dir, "seg_id_to_class.json"), "w") as f:
            json.dump(seg_info, f, indent=2, default=str)

    with open(os.path.join(out_dir, "state.json"), "w") as f:
        json.dump(state_records, f, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def collect_clip(out_dir: str, config: ClipConfig, seed: int):
    """End-to-end single-clip collection. Reusable from a shard-level driver."""
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    hdri_spec = lighting.setup_random_hdri(rng, config)
    bodies, body_specs, scene_specs, initial_velocities, scenario_name = build_scene(
        world, rng, config,
    )
    if hdri_spec is not None:
        scene_specs["hdri"] = hdri_spec
    annots = make_annotators(config.camera_position, config.camera_look_at, config.resolution)

    R_camera_world, camera_pos_world = world_to_camera_frame(
        config.camera_position, config.camera_look_at,
    )
    q_camera_world = rotmat_to_quat_wxyz(R_camera_world)

    frames, state_records = simulate_clip(
        world, bodies, initial_velocities, annots, config,
        R_camera_world, camera_pos_world, q_camera_world,
    )
    ok, reason = validate_clip(frames, state_records, config)

    meta = build_metadata(config, body_specs, scene_specs, scenario_name, R_camera_world, seed)
    meta["validation"] = {"ok": ok, "reason": reason}
    save_clip(out_dir, frames, state_records, meta)
    return ok, reason


def main():
    args = parse_args()
    config = ClipConfig()
    if args.scenario is not None:
        if args.scenario not in scenarios.BUILDERS:
            raise SystemExit(
                f"--scenario {args.scenario!r} not in {sorted(scenarios.BUILDERS)}"
            )
        config = dataclasses.replace(config, scenarios=(args.scenario,))
    ok, reason = collect_clip(os.path.abspath(args.result), config, args.seed)
    print(f"validation: ok={ok} reason={reason}")
    simulation_app.close()


if __name__ == "__main__":
    main()
