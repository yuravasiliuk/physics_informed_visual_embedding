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

import environments
import lighting
import scenarios
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
    if env_loaded:
        ground_color = None
    else:
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
    if ground_color is not None:
        scene_specs["ground"] = {"color_rgb": [float(c) for c in ground_color]}
    return bodies, body_specs, scene_specs, initial_velocities, scenario_name


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


def collect_clip(out_dir: str, config: ClipConfig, seed: int, env_override: str = None):
    """End-to-end single-clip collection. Reusable from a shard-level driver.

    `env_override` pins the environment choice (see environments.setup_random_environment).
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
    annots = make_annotators(config.camera_position, config.camera_look_at, config.resolution)

    R_camera_world, camera_pos_world = world_to_camera_frame(
        config.camera_position, config.camera_look_at,
    )
    q_camera_world = rotmat_to_quat_wxyz(R_camera_world)

    # Persist the realized initial-velocity draw per body (critique #2). The
    # scenario builders sample these; they are otherwise applied in
    # simulate_clip and discarded, leaving the clip unreconstructable from
    # meta.json (only the sampling *range* is recorded) and frame-0 state
    # already corrupted by one gravity step + any contact impulse.
    #
    # Primary fields are in CAMERA frame, matching state.json's per-frame
    # *_camera convention (R_camera_world @ v_world, the same rotation
    # log_state_frame applies — velocities are free vectors, no translation).
    # The raw world-frame draw is kept in parallel *_world fields: it's exactly
    # what set_{linear,angular}_velocity received, so a replay can reapply it
    # verbatim. initial_velocities is aligned with body_specs by spawn order.
    for spec, (lin_vel_world, ang_vel_world) in zip(body_specs, initial_velocities):
        lin_vel_world = np.asarray(lin_vel_world, dtype=np.float64)
        ang_vel_world = np.asarray(ang_vel_world, dtype=np.float64)
        spec["init_lin_vel_camera"] = (R_camera_world @ lin_vel_world).tolist()
        spec["init_ang_vel_camera"] = (R_camera_world @ ang_vel_world).tolist()
        spec["init_lin_vel_world"] = lin_vel_world.tolist()
        spec["init_ang_vel_world"] = ang_vel_world.tolist()

    frames, state_records = simulate_clip(
        world, bodies, initial_velocities, annots, config,
        R_camera_world, camera_pos_world, q_camera_world,
    )
    ok, reason = validate_clip(frames, state_records, config)

    meta = build_metadata(config, body_specs, scene_specs, scenario_name, R_camera_world, seed)
    meta["validation"] = {"ok": ok, "reason": reason}
    save_clip(out_dir, frames, state_records, meta, save_resolution=config.save_resolution)
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

    base_out = os.path.abspath(args.result)
    n_ok = 0
    for i in range(args.n_clips):
        # Isolate between clips: World is a singleton with cached physics
        # views, and the USD stage retains every prim from the prior clip.
        # Clearing both gives the next collect_clip() a clean slate without
        # re-booting SimulationApp.
        if i > 0:
            World.clear_instance()
            create_new_stage()

        if args.n_clips == 1:
            out_dir = base_out
        else:
            out_dir = os.path.join(base_out, f"clip_{i:04d}")

        seed = args.seed + i
        ok, reason = collect_clip(
            out_dir, config, seed, env_override=args.environment,
        )
        n_ok += int(ok)
        print(f"[clip {i + 1}/{args.n_clips}] seed={seed} ok={ok} reason={reason}")

    if args.n_clips > 1:
        print(f"summary: {n_ok}/{args.n_clips} clips validated")
    simulation_app.close()


if __name__ == "__main__":
    main()
