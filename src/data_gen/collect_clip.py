"""Reusable Isaac Sim clip-collection module.

Refactor of `collect_smoke.py` (steps 1-2 of `plans/data-gen/probe-dataset.md` §11).
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

# `os` is imported OUT OF ALPHABETICAL ORDER on purpose: the carb override below
# needs it, and that override has to run before the omni/isaac imports. Don't let
# a formatter move it back down — it would NameError at import time.
import os

ISAAC_ASSETS_DIR = os.path.expanduser("~/isaac_assets")
import carb
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", f"file://{ISAAC_ASSETS_DIR}"
)

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane
from omni.isaac.core.utils.stage import create_new_stage

import body_factory
import environments
import lighting
import cli
import scenarios
from camera_rig import CameraRig                             # owns poses, paths, gating, per-frame logging
from contact_log import to_camera_frame                      # pure helper
from clip_capture import (
    body_seg_coverage,
    build_metadata,
    capture,
    log_state_frame,
    read_local_com,
    save_clip,
)
from config import ClipConfig, DEFAULT_DATA_PATH


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
    # Every ClipConfig override lives in cli.py, shared with collect_corpus so
    # the two drivers cannot drift. No `defaults=`: an omitted flag leaves the
    # ClipConfig value alone, keeping ClipConfig the single source of truth.
    cli.add_config_args(p)
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
    # Scale bodies up inside a loaded environment so they read at the env's
    # scale instead of being a ~1% speck in a warehouse (Stage 3,
    # camera-framing-fix.md). Procedural-ground scenes keep the base range.
    if env_loaded:
        config = dataclasses.replace(config, scale_range=config.scale_range_env)
    scenario_name = rng.choice(config.scenarios).item()
    bodies, body_specs, scene_specs, initial_velocities = scenarios.build(
        scenario_name, world, rng, config,
    )
    if ground_spec is not None:
        scene_specs["ground"] = ground_spec
    return bodies, body_specs, scene_specs, initial_velocities, scenario_name


def _teleport_bodies(bodies, snapshot):
    """Pose bodies at a pre-rolled (pos, quat) snapshot and zero their velocity.

    Used to hold the bodies still at the probe pose: the probe's render step can
    integrate physics by a frame, so re-posing before every probe keeps all
    candidate cameras seeing the same configuration.
    """
    for body, (pos, quat) in zip(bodies, snapshot):
        body.set_world_pose(np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))


def preroll_trajectory(world, bodies, initial_velocities, body_specs, config,
                       gate_policy="max_spread"):
    """Deterministic physics-only pre-roll (no rendering) to learn where the
    bodies go, so the rig can frame + gate on the action (camera-framing-fix.md
    §5, Stage 1).

    Steps `config.frame_count` physics frames exactly as `simulate_clip` will,
    recording each body's world pose. Returns `(centroid, cloud_radius,
    snapshot)`, computed under `gate_policy`:

    **"max_spread"** (default, unchanged): frame EVERYTHING for the WHOLE clip —
    swept-AABB centroid, `cloud_radius` = farthest body surface point over all
    frames, snapshot at `spread.argmax()` (worst case for occlusion). Right when
    the payload is spread over the clip, as in `drop`/`settle`/`topple`.

    **"event"**: frame the MOMENT, and let debris leave afterwards. `f_star` is
    the frame of MINIMUM spread — closest approach, i.e. the impact — and both
    centroid and radius are measured AT that frame instead of swept.

    Why the policy exists (2026-07-28): `collide`'s payload is the momentum
    exchange, which happens early and near the origin, but "max_spread" framed
    the instant the bodies were furthest apart — the single worst choice for it.
    With `framing_distance` clamped at 4 m the largest framable cloud is 0.93 m
    while collide spread past 1.6 m, so 5/6 rollouts were dropped for "no
    viewable camera". That is a policy mismatch, not bad luck, and it will bite
    any future scenario whose payload is a brief event.

    The bodies are left wherever the pre-roll ends; the caller must re-pose or
    rely on the `world.reset()` at the top of `simulate_clip` to rewind.
    """
    world.reset()
    for body, (lin_vel, ang_vel) in zip(bodies, initial_velocities):
        body.set_linear_velocity(np.asarray(lin_vel))
        body.set_angular_velocity(np.asarray(ang_vel))

    bound = np.array([
        float(np.linalg.norm(body_factory.half_extents(s["shape"], s["dims"])))
        for s in body_specs
    ])
    poses, quats = [], []
    for _ in range(config.frame_count):
        world.step(render=False)
        frame_pos, frame_quat = [], []
        for body in bodies:
            p, q = body.get_world_pose()
            frame_pos.append(np.asarray(p, dtype=np.float64))
            frame_quat.append(np.asarray(q, dtype=np.float64))
        poses.append(np.stack(frame_pos))
        quats.append(np.stack(frame_quat))

    P = np.stack(poses)   # (F, B, 3)
    Q = np.stack(quats)   # (F, B, 4)
    spread = np.linalg.norm(P.max(axis=1) - P.min(axis=1), axis=1)  # (F,)

    if gate_policy == "event":
        # Closest approach = the impact. Measure the cloud AT that frame, not
        # swept: a swept radius would still push the camera out to the clamp and
        # reproduce the very problem this policy exists to fix.
        f_star = int(spread.argmin())
        pts = P[f_star]
        centroid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        cloud_radius = float((np.linalg.norm(pts - centroid, axis=1) + bound).max())
    elif gate_policy == "max_spread":
        pts = P.reshape(-1, 3)
        centroid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        dist = np.linalg.norm(P - centroid, axis=2) + bound[None, :]   # (F, B)
        cloud_radius = float(dist.max())
        f_star = int(spread.argmax())
    else:
        raise ValueError(f"unknown gate_policy {gate_policy!r}; "
                         "expected 'max_spread' or 'event'")

    snapshot = list(zip(P[f_star], Q[f_star]))
    return centroid, cloud_radius, snapshot


# RELOCATED 2026-07-28 -> camera_rig.CameraRig: `gate_camera_rig`,
# `build_camera_paths` and `gate_camera_paths` lived here. They were three
# stages of one job (propose -> gate -> path -> gate) split across a module that
# also owns scene building and the capture loop, and their outputs had to be
# threaded to `simulate_clip` as two parallel lists it then duck-typed. The rig
# owns them now; RNG draw order is preserved exactly (poses for every camera,
# THEN paths for every camera), so the corpus for a given seed is unchanged.


@dataclasses.dataclass
class RolloutResult:
    """Everything one rollout produced, in ONE shape regardless of flags.

    `simulate_clip` used to return 2, 3 or 4 values depending on
    `collect_raw_pose` and whether the rig was dynamic, so every caller had to
    know which arm it was on and unpack accordingly. Worse, the arms were
    mutually exclusive: `collect_raw_pose` took priority and silently discarded
    the camera records, which is why `verify_physics_labels.case_moving_camera`
    had to rebuild them by hand. Optional products are now optional FIELDS.

    Every list is per-camera (index c), length = number of surviving cameras.
    `camera_records[c]` is [] for a static rig — the zero case, not a None.
    """
    frames: list
    state_records: list
    camera_records: list
    raw_pose_world: list = None
    raw_quat_world: list = None

    def __iter__(self):
        # Back-compat for `frames, records = simulate_clip(...)`: the two
        # always-present products unpack exactly as before.
        return iter((self.frames, self.state_records))


def simulate_clip(world, bodies, initial_velocities, rig, config,
                  collect_raw_pose=False, contact_logger=None):
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

    C = len(rig)
    cam_frames = [[] for _ in range(C)]
    cam_records = [[] for _ in range(C)]
    raw_pose_world = [[] for _ in bodies] if collect_raw_pose else None
    raw_quat_world = [[] for _ in bodies] if collect_raw_pose else None
    # Dynamic-camera support (dynamic-camera.md §6): `cam_extrinsics[c]` may be
    # either the legacy constant (R, pos, q) tuple OR a CameraPath. Duck-typed
    # on `.extrinsics` so the static path is untouched — additive by design,
    # the same pattern the (now retired) collect_pairs driver used.
    # Progress every 16 frames. A dynamic rig re-poses C camera prims through
    # Replicator per frame on top of the render, so this loop is materially
    # slower than the static one and used to run silently for minutes.
    t_loop = t_mark = time.time()
    for frame_idx in range(config.frame_count):
        if frame_idx and frame_idx % 16 == 0:
            # stdout: Kit relabels stderr as [Error] (see camera_rig._progress).
            # Marginal s/frame, not cumulative — cumulative hides a per-frame
            # cost that GROWS with frame index, which is exactly what a
            # Replicator graph accumulating one node per pose write looks like.
            print(f"  [clip] frame {frame_idx}/{config.frame_count} "
                  f"x{C} cam | {time.time() - t_loop:.0f}s total | "
                  f"{(time.time() - t_mark) / 16:.2f}s/frame last 16",
                  file=sys.stdout, flush=True)
            t_mark = time.time()
        # Re-pose every moving camera BEFORE stepping, so frame t is rendered
        # from pose t. No-op for a static rig, so there is ONE loop body here
        # rather than a static branch and a dynamic branch.
        rig.pose_for_frame(frame_idx)

        # Tag contact events with the rendered frame they fall in. Several
        # physics substeps happen inside one rendered frame, and the events
        # carry `substep` so nothing is lost to the render rate (contact_log.py).
        if contact_logger is not None:
            contact_logger.begin_frame(frame_idx)

        world.step(render=True)
        rep.orchestrator.step()  # flush all render products for this frame
        for c in range(C):
            cam_frames[c].append(capture(rig.annotators(c)))
            # Frame-t extrinsics — the SAME values used to pose the prim above,
            # so pixels and labels cannot describe different cameras. Constant
            # for a static rig, which is why no branch is needed.
            R_camera_world, camera_pos_world, q_camera_world = rig.extrinsics(c, frame_idx)
            rig.log_frame(c, frame_idx, R_camera_world, camera_pos_world)
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

    return RolloutResult(
        frames=cam_frames,
        state_records=cam_records,
        camera_records=rig.all_camera_records(),
        raw_pose_world=raw_pose_world,
        raw_quat_world=raw_quat_world,
    )


def make_contact_logger(bodies, config):
    """ContactLogger if `config.log_contacts`, else None (import is deferred so
    contact_log's omni deps never load for a static/no-contact run)."""
    if not getattr(config, "log_contacts", False):
        return None
    from contact_log import ContactLogger
    return ContactLogger(bodies, config,
                         threshold=float(getattr(config, "contact_threshold", 0.0)))


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

    # Body visibility: the old `seg.max()==0` (any foreground) check is fooled by
    # loaded environments — floor/walls are foreground, so a clip showing zero
    # dynamic bodies still passed. Require the bodies to cover at least
    # `min_body_coverage` of the frame at their peak (camera-framing-fix.md
    # Stage 4). This is the write-time net behind the render-probe gating.
    body_cov_peak = max(
        (body_seg_coverage(f["seg"], f.get("seg_info")) for f in frames),
        default=0.0,
    )
    if body_cov_peak < config.min_body_coverage:
        return False, (
            f"low_body_coverage: peak={body_cov_peak:.4f} < {config.min_body_coverage}"
        )

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


def collect_clip(base_out, start_idx, config: ClipConfig, seed: int, env_override: str = None):
    """End-to-end collection of ONE physics rollout, fanned out to ≤N clips.

    The physics rollout runs once and is rendered from the cameras that survive
    visibility gating (camera-framing-fix.md): each is written to its own
    `base_out/clip_{start_idx+c:05d}` directory as an INDEPENDENT clip
    (single-camera contract unchanged, scene-diversification.md §10). A rollout
    may yield FEWER than `config.n_cameras` clips — or zero, if nothing is
    viewable. Returns a list of `(out_dir, ok, reason)` for the clips written.
    `env_override` pins the environment choice.
    """
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    world.get_physics_context().enable_ccd(True)
    # Env first: a loaded backdrop ships its own floor + lights, which
    # suppresses both the procedural ground and the HDRI/fallback distant
    # light to avoid Z-fighting and double-bright scenes (plan §7 q3).
    # When `config.use_isaac_envs` is False we hard-pin "none" so every clip
    # runs the procedural ground + sky path. (The textureless-floor asset that
    # once motivated disabling envs wholesale, office.usd, is now excluded at
    # the library level instead — see environments.CANDIDATE_ENVIRONMENTS.) An
    # explicit `--environment` CLI override still wins.
    effective_env_name = env_override
    if effective_env_name is None and not config.use_isaac_envs:
        effective_env_name = environments.NONE_NAME
    env_spec = environments.setup_random_environment(rng, force_name=effective_env_name)
    if env_spec is None:
        sky_spec = lighting.setup_random_sky(rng, config)
    else:
        # Env brings its own lights — but they reference HDRIs under
        # /NVIDIA/Assets/Skies/ that may not be in the local mirror. Install
        # a low-intensity distant light as a defensive floor so missing
        # textures don't render a black scene. Harmless overlay if the env's
        # authored lighting works.
        sky_spec = None
        lighting.install_fallback_distant_light(
            intensity=lighting.FALLBACK_LIGHT_INTENSITY_WITH_ENV,
        )
    bodies, body_specs, scene_specs, initial_velocities, scenario_name = build_scene(
        world, rng, config, env_loaded=env_spec is not None,
    )
    if env_spec is not None:
        scene_specs["environment"] = env_spec
    if sky_spec is not None:
        scene_specs["sky"] = sky_spec
    # Pre-roll the deterministic physics (no render) to learn where the bodies
    # go, then frame + gate the rig on that cloud rather than the floor
    # (camera-framing-fix.md §5, Stages 1–2). The body-cloud anchoring fixes the
    # "looking at the floor / tiny speck" framing; the render-probe gate drops
    # viewpoints occluded by env geometry or pointed at a wall/void.
    centroid, cloud_radius, snapshot = preroll_trajectory(
        world, bodies, initial_velocities, body_specs, config,
        gate_policy=scenarios.GATE_FRAME_POLICY.get(scenario_name, "max_spread"),
    )
    # Camera paths (dynamic-camera.md §3-§4). RNG DISCIPLINE: these draws happen
    # AFTER every physics-affecting sampler (scene build + pre-roll are done
    # above), so switching camera family leaves the rollout bit-identical — the
    # precondition for egomotion twins. `static` consumes no draws at all and
    # reproduces the legacy behaviour exactly.
    rig = CameraRig.solve(rng, config, centroid, cloud_radius,
                          teleport=lambda: _teleport_bodies(bodies, snapshot))
    if rig.is_empty():
        # Nothing viewable from any proposed viewpoint, or every viewpoint drifts
        # off the action somewhere in the clip — emit no clips rather than save
        # garbage (quality > count).
        return []

    contact_logger = make_contact_logger(bodies, config)
    rollout = simulate_clip(world, bodies, initial_velocities, rig, config,
                            contact_logger=contact_logger)
    cam_frames, cam_records = rollout.frames, rollout.state_records
    contact_events = []
    if contact_logger is not None:
        contact_events = contact_logger.drain()
        contact_logger.close()
        print(f"  contacts: {len(contact_events)} events", flush=True)
    # Post-reset MEASURED mass properties — the training-truth labels, camera-
    # agnostic. PhysX only materializes mass/CoM at world.reset() (done inside
    # simulate_clip) and both are constant thereafter, so read them back now. The
    # measured CoM must equal the authored offset (W2 + scale fix); logging the
    # measured value makes the label self-consistent with the physics.
    for spec, body in zip(body_specs, bodies):
        spec["com_local_measured"] = read_local_com(body).tolist()
        spec["mass_measured"] = float(body.get_mass())

    # Fan out: one independent clip per surviving camera. body_specs is shared
    # except the camera-frame initial-velocity projection (critique #2), which is
    # recomputed per camera; the *_world fields are identical across siblings.
    results = []
    C = len(rig)
    for c in range(C):
        out_dir = os.path.join(base_out, f"clip_{start_idx + c:05d}")
        R_cw, cam_pos_w, _ = rig.extrinsics(c, 0)
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
            camera_position_world=[float(x) for x in rig.poses[c][0]],
            camera_look_at_world=[float(x) for x in rig.poses[c][1]],
            multiview={"rollout_seed": seed, "camera_index": c, "n_cameras": C},
            camera_path=rig.path(c),
        )
        meta["validation"] = {"ok": ok, "reason": reason}
        # Contacts are pairwise SCENE facts, so they are logged once in world
        # frame and converted per clip using that camera's extrinsics — the same
        # camera-frame invariant the body records follow. For a moving camera the
        # conversion uses the event's own frame-t extrinsics.
        # `rig.camera_records(c)` is [] for a static rig (the zero case), so
        # emptiness — not None — marks "no per-frame extrinsics".
        contacts_c = None
        if contact_events:
            if rig.is_dynamic():
                contacts_c = []
                for ev in contact_events:
                    R_t, p_t, _ = rig.extrinsics(c, ev["frame"])
                    contacts_c.extend(to_camera_frame([ev], R_t, p_t))
            else:
                contacts_c = to_camera_frame(contact_events, R_cw, cam_pos_w)
        save_clip(out_dir, cam_frames[c], cam_records[c], meta,
                  camera_records=(rig.camera_records(c) or None),
                  contact_records=contacts_c,
                  save_resolution=config.save_resolution,
                  depth_format=config.depth_format)
        results.append((out_dir, ok, reason))
    return results


def main():
    args = parse_args()
    cli.validate_camera_args(args)
    config = cli.config_from_args(args)
    if args.scenario is not None:
        if args.scenario not in scenarios.BUILDERS:
            raise SystemExit(
                f"--scenario {args.scenario!r} not in {sorted(scenarios.BUILDERS)}"
            )
        config = dataclasses.replace(config, scenarios=(args.scenario,))

    base_out = os.path.abspath(args.result)
    n_ok = 0
    n_total = 0
    global_idx = 0    # running clip index across all surviving (rollout, camera) pairs
    for i in range(args.n_clips):
        # Isolate between rollouts: World is a singleton with cached physics
        # views, and the USD stage retains every prim from the prior rollout.
        # Clearing both gives the next collect_clip() a clean slate without
        # re-booting SimulationApp.
        if i > 0:
            World.clear_instance()
            create_new_stage()

        # Each physics rollout fans out into the cameras that survive visibility
        # gating (≤ n_cameras), written as flat clip_NNNNN dirs.
        seed = args.seed + i
        results = collect_clip(base_out, global_idx, config, seed, env_override=args.environment)
        for c, (out_dir, ok, reason) in enumerate(results):
            n_total += 1
            n_ok += int(ok)
            print(f"[rollout {i + 1}/{args.n_clips} cam {c}] "
                  f"-> {os.path.basename(out_dir)} seed={seed} ok={ok} reason={reason}")
        if not results:
            print(f"[rollout {i + 1}/{args.n_clips}] seed={seed} -> 0 clips (no viewable camera)")
        global_idx += len(results)

    print(f"summary: {n_ok}/{n_total} clips validated "
          f"from {args.n_clips} rollouts (≤{config.n_cameras} cameras each)")
    simulation_app.close()


if __name__ == "__main__":
    main()
