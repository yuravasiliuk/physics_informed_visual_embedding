"""Paired-corpus driver for controlled-intervention divergence (gate-b-probe.md §13).

Each rollout is built ONCE (scene, pre-roll, gated camera rig) and then
simulated once per variant from the identical initial state:

    base    — CoM offsets exactly as sampled (scale 1.0)
    com<α>  — every body's authored CoM re-authored to α·sampled, one twin
              per --com_scales value (α=0 → centroid; the classic zeroing)

Twins share the clip-index stem and differ by dir suffix, carry a
meta["pair"] provenance block, and share frame-0 pixels exactly: the
re-authoring consumes no RNG draws and PhysX re-materializes mass properties
at the world.reset() inside simulate_clip. R8 holds by construction — the CoM
shift has zero appearance signature, so twin feature divergence can only come
from motion.

Per twin a ground-truth kinematic-divergence diagnostic is printed and logged
(max CoM-path separation vs the base rollout, fork frame). That is the tuning
signal for --com_scales: pick α where twins fork visibly (divergence ≫ 0)
but don't saturate into different scenarios (separation ≲ body scale,
late fork) — see gate-b-probe.md §13 on why saturated divergence is
uninformative. Tune from this log BEFORE spending GPU on probe extraction.

Additive by design (CLAUDE.md §Principles): functionality is imported from
collect_clip / clip_capture / body_factory; only orchestration glue lives
here. The env-setup and per-camera save blocks mirror collect_clip.collect_clip
(not importable without restructuring it) — if that fan-out changes, mirror
the change here manually.

Usage (user-run; Isaac required, do not run from the agent):
    ${ISAAC_SIM}/python.sh src/data_gen/collect_pairs.py --result ~/data/pairs \\
        --n_rollouts 20 --com_scales 0.0 0.5 --scenario topple
"""
# SimulationApp must be constructed before any omni / isaac import (CLAUDE.md);
# gated behind __main__ exactly like collect_clip so importing this module
# never spawns a second Kit instance.
if __name__ == "__main__":
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

# Importing collect_clip performs all omni imports and applies its carb
# asset-root override (module level, after its own __main__ gate) — safe here
# because SimulationApp is already alive.
import collect_clip
import body_factory
import environments
import lighting
import scenarios
# Side-effect import: registers the pair-tuned builders (weighted_roll —
# periodic CoM wobble while rolling; collide_static — single contact against
# a static wall, no second dynamic body) into scenarios.BUILDERS. Both honor
# --n_objects exactly via scenarios._sample_n_bodies, unlike topple/collide,
# which use their own counts. collect_clip never imports this module, so the
# main corpus pipeline is unaffected.
import pair_scenarios  # noqa: F401
from clip_capture import (
    build_metadata,
    make_rig_annotators,
    read_local_com,
    rotmat_to_quat_wxyz,
    save_clip,
    world_to_camera_frame,
)
from config import ClipConfig

from omni.isaac.core import World
from omni.isaac.core.utils.stage import create_new_stage, get_current_stage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n_rollouts", type=int, default=1,
        help="Physics rollouts; each yields (1 + len(--com_scales)) × ≤n_cameras clips.",
    )
    p.add_argument(
        "--com_scales", type=float, nargs="+", default=[0.0],
        help="One twin per value α: authored CoM = α · sampled offset. α=0 is "
             "the centroid intervention (max dose); sweep e.g. 0.0 0.33 0.66 "
             "to tune dose vs scenario divergence from the printed diagnostic.",
    )
    p.add_argument(
        "--scenario", type=str, default=None,
        help="Force a single scenario. The pair-tuned builders weighted_roll "
             "(sustained periodic CoM signature) and collide_static (single "
             "clean wall impact) are registered too — prefer them over "
             "topple/collide for pair corpora (R6; free-fall twins barely fork).",
    )
    p.add_argument(
        "--n_objects", type=int, default=None,
        help="Force a deterministic body count per clip (e.g. 1, 2, 3). "
             "Pins config.n_bodies_range=(n, n) so scenarios._sample_n_bodies "
             "returns exactly n. Default: keep config's sampled range. Fewer "
             "bodies = cleaner per-body attribution of twin divergence.",
    )
    p.add_argument(
        "--environment", type=str, default=None,
        choices=environments.library.names + [environments.NONE_NAME],
        help="Force a specific environment backdrop (same semantics as collect_clip).",
    )
    return p.parse_args()


def author_scaled_com(bodies, body_specs, base_offsets, scale):
    """Re-author every body's CoM to scale·sampled_offset (absolute, so twins
    can be generated in any order). Primitive bodies only — main() forces
    composite_frac=0 because a composite's CoM lives in its per-part densities
    (W5) and can't be re-authored without changing mass/inertia structure.
    Consumes no RNG draws; takes effect at the next world.reset().
    """
    stage = get_current_stage()
    for body, spec, offset in zip(bodies, body_specs, base_offsets):
        assert spec["kind"] == "primitive", "pair corpora are primitives-only"
        prim = stage.GetPrimAtPath(body.prim_path)
        body_factory.attach_center_of_mass(
            prim, scale * offset, body_factory.prim_local_scale(prim),
        )


def twin_divergence(records_base, records_twin, body_specs, fork_frac=0.1):
    """Ground-truth kinematic divergence between a twin and its base rollout.

    Compares per-(frame, body) CoM positions from the state records (camera
    frame; distances are camera-invariant since both variants share the rig).
    Returns a JSON-ready dict:
        max_divergence_m    worst CoM-path separation over all frames/bodies
        rel_divergence      same, relative to the median body bbox diagonal —
                            ≳1 means twins reached different-scenario territory
        fork_frame          first frame any body separates by fork_frac·its
                            own diagonal (-1 = never forked: dose invisible)
        final_divergence_m  mean separation at the last frame
    """
    base = {(r["frame"], r["body"]): np.asarray(r["com_camera"]) for r in records_base}
    diag = {
        s["name"]: 2.0 * float(np.linalg.norm(
            body_factory.half_extents(s["shape"], s["dims"])))
        for s in body_specs
    }
    max_div, fork_frame = 0.0, -1
    last_frame = max(r["frame"] for r in records_twin)
    final = []
    for r in sorted(records_twin, key=lambda r: r["frame"]):
        key = (r["frame"], r["body"])
        if key not in base:
            continue
        d = float(np.linalg.norm(np.asarray(r["com_camera"]) - base[key]))
        max_div = max(max_div, d)
        if fork_frame < 0 and d > fork_frac * diag[r["body"]]:
            fork_frame = int(r["frame"])
        if r["frame"] == last_frame:
            final.append(d)
    med_diag = float(np.median(list(diag.values())))
    return {
        "max_divergence_m": max_div,
        "rel_divergence": max_div / med_diag if med_diag > 0 else float("nan"),
        "fork_frame": fork_frame,
        "final_divergence_m": float(np.mean(final)) if final else 0.0,
    }


def collect_pair_rollout(base_out, start_idx, config: ClipConfig, seed: int,
                         com_scales, env_override=None):
    """One rollout → base + twins, all sharing scene/rig/frame-0 state.

    Returns (results, stats_by_label); results entries are
    (out_dir, ok, reason). Empty results if no camera survives gating
    (quality > count, as in collect_clip).
    """
    rng = np.random.default_rng(seed)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    world.get_physics_context().enable_ccd(True)
    # Env-or-procedural-backdrop glue — mirrors collect_clip (plan §7 q3 / the
    # fallback-light rationale documented there).
    effective_env_name = env_override
    if effective_env_name is None and not config.use_isaac_envs:
        effective_env_name = environments.NONE_NAME
    env_spec = environments.setup_random_environment(rng, force_name=effective_env_name)
    if env_spec is None:
        sky_spec = lighting.setup_random_sky(rng, config)
    else:
        sky_spec = None
        lighting.install_fallback_distant_light(
            intensity=lighting.FALLBACK_LIGHT_INTENSITY_WITH_ENV,
        )
    bodies, body_specs, scene_specs, initial_velocities, scenario_name = \
        collect_clip.build_scene(world, rng, config, env_loaded=env_spec is not None)
    if env_spec is not None:
        scene_specs["environment"] = env_spec
    if sky_spec is not None:
        scene_specs["sky"] = sky_spec

    # Pre-roll + body-cloud-anchored gated rig (camera-framing-fix.md §5). The
    # rig is gated on the BASE trajectory and reused by every twin — required:
    # a per-variant rig would change pixels at frame 0 and break the
    # "identical visual starting point" contract of the pair experiment.
    centroid, cloud_radius, snapshot = collect_clip.preroll_trajectory(
        world, bodies, initial_velocities, body_specs, config,
    )
    rig = collect_clip.gate_camera_rig(bodies, rng, config, centroid, cloud_radius, snapshot)
    if not rig:
        return [], {}
    cam_annots = make_rig_annotators(rig, config.resolution)
    cam_extrinsics = []
    for cam_pos, look_at in rig:
        R_cw, cam_pos_w = world_to_camera_frame(cam_pos, look_at)
        cam_extrinsics.append((R_cw, cam_pos_w, rotmat_to_quat_wxyz(R_cw)))

    # Sampled offsets, captured before any re-authoring (absolute per-twin
    # authoring uses these, so variant order can't compound).
    base_offsets = [
        np.asarray(s["com_local_authored"], dtype=np.float64) for s in body_specs
    ]

    variants = [("base", 1.0)] + [(f"com{a:g}", float(a)) for a in com_scales]
    results, stats_by_label, base_records = [], {}, None
    C = len(rig)
    for label, scale in variants:
        if label != "base":
            author_scaled_com(bodies, body_specs, base_offsets, scale)
        cam_frames, cam_records = collect_clip.simulate_clip(
            world, bodies, initial_velocities, cam_annots, config, cam_extrinsics,
        )

        # Post-reset measured labels, per variant (the re-authoring changes
        # them); authored field updated to what was actually authored.
        specs_v = [dict(s) for s in body_specs]
        for spec, body, offset in zip(specs_v, bodies, base_offsets):
            spec["com_local_measured"] = read_local_com(body).tolist()
            spec["mass_measured"] = float(body.get_mass())
            if label != "base":
                spec["com_local_authored"] = (scale * offset).tolist()

        if label == "base":
            base_records = cam_records[0]
        else:
            stats_by_label[label] = twin_divergence(base_records, cam_records[0], specs_v)

        # Per-camera fan-out — mirrors collect_clip.collect_clip (critique #2
        # camera-frame init-velocity projection included). Twins share the
        # camera index; only the suffix differs, so the probe matches twins on
        # (rollout_seed, camera_index, body).
        for c in range(C):
            out_dir = os.path.join(base_out, f"clip_{start_idx + c:05d}_{label}")
            R_cw, cam_pos_w, _ = cam_extrinsics[c]
            specs_c = [dict(s) for s in specs_v]
            for spec, (lin_vel_world, ang_vel_world) in zip(specs_c, initial_velocities):
                lin_vel_world = np.asarray(lin_vel_world, dtype=np.float64)
                ang_vel_world = np.asarray(ang_vel_world, dtype=np.float64)
                spec["init_lin_vel_camera"] = (R_cw @ lin_vel_world).tolist()
                spec["init_ang_vel_camera"] = (R_cw @ ang_vel_world).tolist()
                spec["init_lin_vel_world"] = lin_vel_world.tolist()
                spec["init_ang_vel_world"] = ang_vel_world.tolist()

            ok, reason = collect_clip.validate_clip(cam_frames[c], cam_records[c], config)
            meta = build_metadata(
                config, specs_c, scene_specs, scenario_name, R_cw, seed,
                camera_position_world=[float(x) for x in rig[c][0]],
                camera_look_at_world=[float(x) for x in rig[c][1]],
                multiview={"rollout_seed": seed, "camera_index": c, "n_cameras": C},
            )
            meta["validation"] = {"ok": ok, "reason": reason}
            meta["pair"] = {
                "pair_seed": seed,
                "variant": label,
                "intervention": "com",
                "com_scale": scale,
            }
            if label != "base":
                meta["pair"]["kinematic_divergence"] = stats_by_label[label]
            save_clip(out_dir, cam_frames[c], cam_records[c], meta,
                      save_resolution=config.save_resolution)
            results.append((out_dir, ok, reason))
    return results, stats_by_label


def main():
    args = parse_args()
    config = ClipConfig()
    if args.scenario is not None:
        if args.scenario not in scenarios.BUILDERS:
            raise SystemExit(
                f"--scenario {args.scenario!r} not in {sorted(scenarios.BUILDERS)}"
            )
        config = dataclasses.replace(config, scenarios=(args.scenario,))
    # Primitives only: author_scaled_com can't re-author a composite (W5).
    config = dataclasses.replace(config, composite_frac=0.0)
    if args.n_objects is not None:
        if args.n_objects < 1:
            raise SystemExit("--n_objects must be >= 1")
        # Collapse the sampled range to a point so every clip has exactly n
        # bodies (scenarios._sample_n_bodies draws from this inclusive range).
        config = dataclasses.replace(config, n_bodies_range=(args.n_objects, args.n_objects))

    base_out = os.path.abspath(args.result)
    n_ok = n_total = global_idx = 0
    for i in range(args.n_rollouts):
        if i > 0:  # same World/stage isolation as collect_clip.main
            World.clear_instance()
            create_new_stage()
        seed = args.seed + i
        results, stats = collect_pair_rollout(
            base_out, global_idx, config, seed, args.com_scales,
            env_override=args.environment,
        )
        for out_dir, ok, reason in results:
            n_total += 1
            n_ok += int(ok)
            print(f"[rollout {i + 1}/{args.n_rollouts}] -> "
                  f"{os.path.basename(out_dir)} seed={seed} ok={ok} reason={reason}")
        for label, s in stats.items():
            print(f"[rollout {i + 1}/{args.n_rollouts}] {label}: "
                  f"max div {s['max_divergence_m']:.3f} m "
                  f"({s['rel_divergence']:.2f}x body diag), "
                  f"fork@frame {s['fork_frame']}, "
                  f"final {s['final_divergence_m']:.3f} m")
        if not results:
            print(f"[rollout {i + 1}/{args.n_rollouts}] seed={seed} -> 0 clips "
                  f"(no viewable camera)")
        global_idx += len(results) // (1 + len(args.com_scales)) if results else 0

    print(f"summary: {n_ok}/{n_total} clips validated from {args.n_rollouts} rollouts "
          f"x {1 + len(args.com_scales)} variants")
    simulation_app.close()


if __name__ == "__main__":
    main()
