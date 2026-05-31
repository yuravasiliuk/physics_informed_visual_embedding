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
# SimulationApp must be constructed before any other omni / isaac import.
# Gated behind `__main__` so importing this module from another script (e.g.
# verify_collect_clip.py, or a future shard-level driver) does not spawn a
# second SimulationApp — Kit segfaults if two are alive in one process. The
# caller owns SimApp construction; this branch only fires for the CLI use.
if __name__ == "__main__":
    from isaacsim import SimulationApp
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
import os
from pathlib import Path

import numpy as np

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.stage import create_new_stage

import body_factory
import environments
import lighting
import scenarios
from clip_capture import (
    ClipConfig,
    build_metadata,
    capture,
    log_state_frame,
    make_rig_annotators,
    read_local_com,
    rotmat_to_quat_wxyz,
    sample_camera_rig,
    save_clip,
    world_to_camera_frame,
)


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
    p.add_argument(
        "--environment", type=str, default=None,
        choices=environments.library.names + [environments.NONE_NAME],
        help="Force a specific environment backdrop. Default (omitted) samples "
             "uniformly over discovered + 'none'.",
    )
    p.add_argument(
        "--n_clips", type=int, default=1,
        help="Generate N clips in one SimulationApp invocation (amortizes the "
             "~30-60s Isaac boot). N>1 writes each clip to --result/clip_NNNN/; "
             "N==1 writes directly to --result (legacy layout). Per-clip seed = "
             "--seed + i.",
    )
    return p.parse_args()


def build_scene(world: World, rng: np.random.Generator, config: ClipConfig, env_loaded: bool):
    """Sample a scenario, build it, return everything the rest of the pipeline needs.

    `env_loaded=True` means an environment USD was referenced upstream and
    brings its own authored floor — we skip the procedural `GroundPlane` to
    avoid Z-fighting at z=0. Otherwise (env_loaded=False) we author the
    procedural ground as before.

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
    ground_spec = None
    if not env_loaded:
        if config.ground_grid_enabled:
            # Grid-line texture (scene-diversification.md §10.4) — a translation/
            # parallax cue the flat colour lacks. UV tiling randomized per clip.
            grid_mat = body_factory.make_ground_grid_material(
                "/World/Looks/ground_grid", rng, config,
            )
            GroundPlane(
                prim_path="/World/ground_plane", z_position=0.0, visual_material=grid_mat,
            )
            ground_spec = {"grid": True}
        else:
            ground_color = np.array(
                rng.uniform(*config.ground_color_range, size=3), dtype=np.float32,
            )
            GroundPlane(
                prim_path="/World/ground_plane", z_position=0.0, color=ground_color,
            )
            ground_spec = {"color_rgb": [float(c) for c in ground_color]}
    scenario_name = rng.choice(config.scenarios).item()
    bodies, body_specs, scene_specs, initial_velocities = scenarios.build(
        scenario_name, world, rng, config,
    )
    if ground_spec is not None:
        scene_specs["ground"] = ground_spec
    return bodies, body_specs, scene_specs, initial_velocities, scenario_name


def simulate_clip(world, bodies, initial_velocities, cam_annots, config, cam_extrinsics, collect_raw_pose=False):
    """Run the K-frame loop once, capturing from N cameras simultaneously.

    Multi-view (scene-diversification.md §10): the physics steps ONCE per frame;
    each frame is then captured from every camera and the per-body state is
    logged once per camera (in that camera's frame). The N cameras observe
    identical dynamics — the basis for the independent-clip fan-out in
    `collect_clip`.

    `cam_annots` is a list (length C) of annotator dicts, one per camera (from
    `make_rig_annotators`). `cam_extrinsics` is the parallel list of
    `(R_camera_world, camera_pos_world, q_camera_world)` tuples. A single camera
    is just C==1 — that's how verify_physics_labels.py drives this loop.

    `initial_velocities` is a list of (lin_vel_world, ang_vel_world) arrays
    aligned with `bodies`, applied AFTER `world.reset()` (Isaac discards
    velocities set before reset).

    Returns `(cam_frames, cam_records)` — each a list of length C, where
    `cam_frames[c]` is that camera's frame list and `cam_records[c]` its state
    records. `collect_raw_pose=True` (verify-only) appends camera-agnostic
    `raw_pose_world, raw_quat_world` (per-body world `get_world_pose()`), used to
    probe origin-vs-CoM independently of the reconstruction (W4).
    """
    world.reset()
    for body, (lin_vel, ang_vel) in zip(bodies, initial_velocities):
        body.set_linear_velocity(np.asarray(lin_vel))
        body.set_angular_velocity(np.asarray(ang_vel))

    # MassAPI is authored at scene-build time and constant through the clip,
    # so cache the local CoM offset once instead of re-reading USD per frame.
    local_coms = [read_local_com(body) for body in bodies]

    C = len(cam_annots)
    cam_frames = [[] for _ in range(C)]
    cam_records = [[] for _ in range(C)]
    raw_pose_world = [[] for _ in bodies] if collect_raw_pose else None
    raw_quat_world = [[] for _ in bodies] if collect_raw_pose else None
    for frame_idx in range(config.frame_count):
        world.step(render=True)
        rep.orchestrator.step()  # flush all render products for this frame
        for c in range(C):
            cam_frames[c].append(capture(cam_annots[c]))
            R_camera_world, camera_pos_world, q_camera_world = cam_extrinsics[c]
            log_state_frame(
                bodies, local_coms, frame_idx, cam_records[c],
                R_camera_world=R_camera_world,
                camera_pos_world=camera_pos_world,
                q_camera_world=q_camera_world,
            )
        if collect_raw_pose:
            for b_i, body in enumerate(bodies):
                p, q = body.get_world_pose()
                raw_pose_world[b_i].append(np.asarray(p, dtype=np.float64))
                raw_quat_world[b_i].append(np.asarray(q, dtype=np.float64))

    if collect_raw_pose:
        return cam_frames, cam_records, raw_pose_world, raw_quat_world
    return cam_frames, cam_records


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


def collect_clip(out_dirs, config: ClipConfig, seed: int, env_override: str = None):
    """End-to-end collection of ONE physics rollout, fanned out to N clips.

    `out_dirs` is a list of `config.n_cameras` output directories — one per rig
    camera. The physics rollout runs once; each camera's view is written to its
    own directory as an INDEPENDENT clip (single-camera contract unchanged,
    scene-diversification.md §10). Returns a list of `(ok, reason)` aligned with
    `out_dirs`. `env_override` pins the environment choice.
    """
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    # Env first: a loaded backdrop ships its own floor + lights, which
    # suppresses both the procedural ground and the HDRI/fallback distant
    # light to avoid Z-fighting and double-bright scenes (plan §7 q3).
    env_spec = environments.setup_random_environment(rng, force_name=env_override)
    if env_spec is None:
        hdri_spec = lighting.setup_random_hdri(rng, config)
    else:
        # Env brings its own lights — but they reference HDRIs under
        # /NVIDIA/Assets/Skies/ that may not be in the local mirror. Install
        # a low-intensity distant light as a defensive floor so missing
        # textures don't render a black scene. Harmless overlay if the env's
        # authored lighting works.
        hdri_spec = None
        lighting.install_fallback_distant_light(
            intensity=lighting.FALLBACK_LIGHT_INTENSITY_WITH_ENV,
        )
    bodies, body_specs, scene_specs, initial_velocities, scenario_name = build_scene(
        world, rng, config, env_loaded=env_spec is not None,
    )
    if env_spec is not None:
        scene_specs["environment"] = env_spec
    if hdri_spec is not None:
        scene_specs["hdri"] = hdri_spec
    # Camera rig: N jittered viewpoints on a shell around the scene (§10.2).
    # Each becomes its own clip; the physics rollout is shared.
    rig = sample_camera_rig(rng, config)
    cam_annots = make_rig_annotators(rig, config.resolution)
    cam_extrinsics = []
    for cam_pos, look_at in rig:
        R_cw, cam_pos_w = world_to_camera_frame(cam_pos, look_at)
        cam_extrinsics.append((R_cw, cam_pos_w, rotmat_to_quat_wxyz(R_cw)))

    cam_frames, cam_records = simulate_clip(
        world, bodies, initial_velocities, cam_annots, config, cam_extrinsics,
    )

    # Post-reset MEASURED mass properties — the training-truth labels, camera-
    # agnostic. PhysX only materializes mass/CoM at world.reset() (done inside
    # simulate_clip) and both are constant thereafter, so read them back now. The
    # measured CoM must equal the authored offset (W2 + scale fix); logging the
    # measured value makes the label self-consistent with the physics.
    for spec, body in zip(body_specs, bodies):
        spec["com_local_measured"] = read_local_com(body).tolist()
        spec["mass_measured"] = float(body.get_mass())

    # Fan out: one independent clip per camera. body_specs is shared except the
    # camera-frame initial-velocity projection (critique #2), which is recomputed
    # per camera; the *_world fields are identical across siblings.
    results = []
    C = len(rig)
    for c in range(C):
        R_cw, cam_pos_w, _ = cam_extrinsics[c]
        specs_c = [dict(s) for s in body_specs]
        for spec, (lin_vel_world, ang_vel_world) in zip(specs_c, initial_velocities):
            lin_vel_world = np.asarray(lin_vel_world, dtype=np.float64)
            ang_vel_world = np.asarray(ang_vel_world, dtype=np.float64)
            spec["init_lin_vel_camera"] = (R_cw @ lin_vel_world).tolist()
            spec["init_ang_vel_camera"] = (R_cw @ ang_vel_world).tolist()
            spec["init_lin_vel_world"] = lin_vel_world.tolist()
            spec["init_ang_vel_world"] = ang_vel_world.tolist()

        ok, reason = validate_clip(cam_frames[c], cam_records[c], config)
        meta = build_metadata(
            config, specs_c, scene_specs, scenario_name, R_cw, seed,
            camera_position_world=[float(x) for x in rig[c][0]],
            camera_look_at_world=[float(x) for x in rig[c][1]],
            multiview={"rollout_seed": seed, "camera_index": c, "n_cameras": C},
        )
        meta["validation"] = {"ok": ok, "reason": reason}
        save_clip(out_dirs[c], cam_frames[c], cam_records[c], meta,
                  save_resolution=config.save_resolution)
        results.append((ok, reason))
    return results


def main():
    args = parse_args()
    config = ClipConfig()
    if args.scenario is not None:
        if args.scenario not in scenarios.BUILDERS:
            raise SystemExit(
                f"--scenario {args.scenario!r} not in {sorted(scenarios.BUILDERS)}"
            )
        config = dataclasses.replace(config, scenarios=(args.scenario,))

    base_out = os.path.abspath(args.result)
    C = config.n_cameras
    n_ok = 0
    n_total = 0
    global_idx = 0    # running clip index across all (rollout, camera) pairs
    for i in range(args.n_clips):
        # Isolate between rollouts: World is a singleton with cached physics
        # views, and the USD stage retains every prim from the prior rollout.
        # Clearing both gives the next collect_clip() a clean slate without
        # re-booting SimulationApp.
        if i > 0:
            World.clear_instance()
            create_new_stage()

        # Each physics rollout fans out into C independent sibling clips
        # (one per rig camera), written as flat clip_NNNNN dirs.
        out_dirs = [os.path.join(base_out, f"clip_{global_idx + c:05d}") for c in range(C)]
        seed = args.seed + i
        results = collect_clip(out_dirs, config, seed, env_override=args.environment)
        for c, (ok, reason) in enumerate(results):
            n_total += 1
            n_ok += int(ok)
            print(f"[rollout {i + 1}/{args.n_clips} cam {c}] "
                  f"-> {os.path.basename(out_dirs[c])} seed={seed} ok={ok} reason={reason}")
        global_idx += C

    print(f"summary: {n_ok}/{n_total} clips validated "
          f"({args.n_clips} rollouts x {C} cameras)")
    simulation_app.close()


if __name__ == "__main__":
    main()
