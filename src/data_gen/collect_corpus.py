"""Single-boot corpus generator — all scenarios in ONE Isaac session.

Fuses scripts/gen_probe_corpus.sh into one process. The shell script re-booted
Kit once PER scenario — 9× the ~30–60 s, machine-freezing SimulationApp load.
This boots SimulationApp ONCE and loops scenarios × rollouts in the same session,
reusing collect_clip's proven per-rollout isolation (`World.clear_instance()` +
`create_new_stage()` between rollouts — exactly what collect_clip.main() does for
`--n_clips`). Only the *orchestration* moves here; generation is unchanged.

Layout matches the .sh: one subdir per scenario, `result/<scenario>/clip_NNNNN/`,
with disjoint per-scenario seed bands (scenario_index × --seed_stride) so rollout
groups stay globally unique.

MUST run under the Isaac conda env (env_isaaclab). The very first line imports
`isaacsim`, so a wrong env fails instantly here — before any GPU/Kit spend:
    conda activate env_isaaclab
    python src/data_gen/collect_corpus.py --result ~/data/probe_corpus --n_clips 30

Progress prints with flush=True (no block-buffering — the log shows live, unlike
the redirected .sh logs). Watch clips land with:
    watch -n 30 'find ~/data/probe_corpus -name meta.json | wc -l'
"""

# SimulationApp MUST be constructed before ANY omni/isaac import (Kit segfaults
# otherwise) — mirrors collect_clip.py's __main__ boot block. Importing
# collect_clip below re-applies the carb asset-root and pulls omni, so it has to
# come AFTER this line.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import argparse
import dataclasses
import json
import os
import sys
import time

from omni.isaac.core import World
from omni.isaac.core.utils.stage import create_new_stage

from collect_clip import collect_clip          # sets carb asset-root + imports omni at load (SimApp exists now)
from config import ClipConfig, DEFAULT_DATA_PATH
import scenarios                                # already imported by collect_clip; here for BUILDERS validation

DEFAULT_SCENARIOS = ("drop", "roll", "throw", "drop_on_platform", "settle",
                     "push", "cluttered", "topple", "collide")


def _resume_state(out_s, fresh_seed_base):
    """Scan an existing scenario dir so new clips APPEND, never overwrite.

    Returns (start_idx, seed_base):
      * start_idx = max existing `clip_NNNNN` index + 1  (unique folder, never clobbers)
      * seed_base = max `seed` in existing meta.json + 1  (new rollouts are DISTINCT
        physics — reusing a seed would just regenerate an identical clip, not a new
        group). Continues within the scenario's seed band.
    Empty/missing dir → (0, fresh_seed_base). NB: a rollout gated to 0 clips leaves
    no meta, so its seed isn't counted — worst case a few gated seeds get re-attempted
    (harmless: no duplicate data, just a little wasted render).
    """
    if not os.path.isdir(out_s):
        return 0, fresh_seed_base
    max_idx, max_seed = -1, -1
    for name in os.listdir(out_s):
        if not name.startswith("clip_"):
            continue
        try:
            max_idx = max(max_idx, int(name[5:]))
        except ValueError:
            continue
        try:
            with open(os.path.join(out_s, name, "meta.json")) as f:
                s = json.load(f).get("seed")
            if s is not None:
                max_seed = max(max_seed, int(s))
        except (OSError, ValueError, KeyError):
            pass
    return max_idx + 1, (max_seed + 1 if max_seed >= 0 else fresh_seed_base)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", default=os.path.join(os.path.dirname(DEFAULT_DATA_PATH), "probe_corpus"),
                   help="base output dir; each scenario writes to <result>/<scenario>/")
    p.add_argument("--n_clips", type=int, default=30,
                   help="rollouts (R) per scenario")
    p.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS),
                   help="scenario names to generate (default: all 9)")
    p.add_argument("--n_cameras", type=int, default=1,
                   help="cameras per rollout (1 = groups-not-views, the probe default)")
    p.add_argument("--seed_stride", type=int, default=1000,
                   help="seed band width per scenario (seed = scenario_index*stride + rollout)")
    p.add_argument("--com_offset_abs", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="absolute CoM offset range (m); omit for legacy fraction-of-extent")
    p.add_argument("--depth_format", default="npz", choices=["npz", "ffv1", "hevc12"])
    p.add_argument("--environment", default=None,
                   help="force an environment backdrop; default samples per clip")
    # --- dynamic camera (dynamic-camera.md) ---------------------------------
    p.add_argument("--camera_families", nargs="+", default=["static"],
                   help="camera path families to sample per camera "
                        "(static orbit arc dolly roll handheld gait). 'static' "
                        "is the zero-coefficient case = legacy behaviour.")
    p.add_argument("--camera_weights", nargs="+", type=float, default=None,
                   help="sampling weights parallel to --camera_families "
                        "(default uniform)")
    p.add_argument("--camera_roll_always", action="store_true",
                   help="force a roll component on every path. REQUIRED for any "
                        "batch that must carry the gravity-tilt label: only roll "
                        "breaks g_cam.x==0 — orbit is pure yaw about the gravity "
                        "axis and dolly is pure translation, and both leave tilt "
                        "EXACTLY invariant (verify_camera_paths.py).")
    p.add_argument("--log_contacts", action="store_true",
                   help="log PhysX contact events (solver-exact impulses at "
                        "physics rate + the event timestamps that event-anchored "
                        "targets are built on). REQUIRED for restitution and for "
                        "mass-ratio-at-collisions — see contact_log.py.")
    p.add_argument("--camera_path_gate_samples", type=int, default=4,
                   help="poses probed along each path for visibility gating; "
                        "1 disables the along-path gate")
    return p.parse_args()


def main():
    args = parse_args()
    bad = [s for s in args.scenarios if s not in scenarios.BUILDERS]
    if bad:
        sys.exit(f"unknown scenarios {bad}; have {sorted(scenarios.BUILDERS)}")

    from camera_paths import FAMILIES as _CAM_FAMILIES
    bad = [f for f in args.camera_families if f not in _CAM_FAMILIES]
    if bad:
        sys.exit(f"unknown camera families {bad}; expected {list(_CAM_FAMILIES)}")
    if args.camera_weights is not None and len(args.camera_weights) != len(args.camera_families):
        sys.exit("--camera_weights must be parallel to --camera_families")
    base_cfg = dataclasses.replace(ClipConfig(), n_cameras=args.n_cameras,
                                   depth_format=args.depth_format,
                                   camera_path_families=tuple(args.camera_families),
                                   camera_path_weights=(tuple(args.camera_weights)
                                                        if args.camera_weights else None),
                                   camera_roll_always=args.camera_roll_always,
                                   camera_path_gate_samples=args.camera_path_gate_samples,
                                   log_contacts=args.log_contacts)
    if args.com_offset_abs is not None:
        base_cfg = dataclasses.replace(base_cfg, com_offset_abs_range=tuple(args.com_offset_abs))

    base_out = os.path.abspath(os.path.expanduser(args.result))
    print(f"env={os.environ.get('CONDA_DEFAULT_ENV', '?')} | python={sys.executable}", flush=True)
    print(f"corpus: {len(args.scenarios)} scenarios × {args.n_clips} rollouts "
          f"→ {base_out}", flush=True)
    print(f"scenarios: {' '.join(args.scenarios)}", flush=True)

    counts = {}
    first = True                 # skip stage-reset before the very first rollout
    t_start = time.time()
    for si, scen in enumerate(args.scenarios):
        cfg = dataclasses.replace(base_cfg, scenarios=(scen,))
        out_s = os.path.join(base_out, scen)
        # Resume-safe: never overwrite existing clips; append with unique folder
        # indices and continue the seed band so new rollouts are distinct physics.
        start_idx, seed_base = _resume_state(out_s, si * args.seed_stride)
        written = start_idx
        t0 = time.time()
        n_ok = 0
        n_fail = 0
        resume = f"  (appending after {start_idx} existing)" if start_idx else ""
        print(f"\n--- {scen}: +{args.n_clips} rollouts → clip_{start_idx:05d}.., "
              f"seeds from {seed_base}{resume} ---", flush=True)
        for i in range(args.n_clips):
            # Per-rollout isolation — identical to collect_clip.main(): the World
            # is a singleton with cached physics views and the stage retains prior
            # prims, so clear both between rollouts (never before the first).
            if not first:
                World.clear_instance()
                create_new_stage()
            first = False
            seed = seed_base + i
            # One bad rollout must not kill the whole corpus. Seed-dependent scene
            # builds can fail (e.g. "could not place N bodies without overlap" in a
            # tight scenario); render/physics can raise too. Log, skip, continue —
            # the next iteration's clear_instance()+create_new_stage() resets the
            # stage. (Exception, not BaseException: Ctrl-C still stops the run.)
            try:
                results = collect_clip(out_s, written, cfg, seed, env_override=args.environment)
            except Exception as e:
                n_fail += 1
                print(f"[{scen} {i + 1}/{args.n_clips}] seed={seed} -> FAILED "
                      f"({type(e).__name__}: {e}) — skipping", flush=True)
                continue
            ok = sum(int(o) for _, o, _ in results)
            written += len(results)
            n_ok += ok
            note = "" if results else "  (no viewable camera)"
            print(f"[{scen} {i + 1}/{args.n_clips}] seed={seed} "
                  f"-> {len(results)} clip(s), {ok} ok{note}", flush=True)
        added = written - start_idx
        counts[scen] = (added, written)     # (new this run, total on disk)
        fail_note = f", {n_fail} errored" if n_fail else ""
        print(f"--- {scen}: +{added} new (total {written}, {n_ok} validated{fail_note}) in "
              f"{time.time() - t0:.0f}s ---", flush=True)

    # --- summary (parity with the .sh) ---
    print("\n=== SUMMARY ===", flush=True)
    total = 0
    for scen in args.scenarios:
        added, tot = counts.get(scen, (0, 0))
        total += tot
        flag = "   <-- FEW NEW (<70% of R gated?)" if added < args.n_clips * 0.7 else ""
        print(f"  {scen:18s} +{added} new, {tot} total{flag}", flush=True)
    print(f"TOTAL: {total} clips on disk / {len(args.scenarios)} scenarios in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    print("Next (amass2, GPU): extract + coverage_audit + density_audit + "
          "replicate_gate_b (probe-corpus-spec.md §6)", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
