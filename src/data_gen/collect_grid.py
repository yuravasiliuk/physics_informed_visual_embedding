"""Grid corpus driver — run a (room × scenario × view) plan in ONE Isaac session.

`collect_corpus.py` loops scenarios with a single camera policy. This drives
`ClipsGenerator`, so the two axes you actually want to control — which ROOM and
which EGOMOTION — are explicit, labelled, and reported per cell instead of
sampled and averaged away.

    ${ISAAC_SIM}/python.sh src/data_gen/collect_grid.py \
        --result ~/data/cg_smoke --n_rollouts 2 --frame_count 32 \
        --rooms warehouse simple_room --scenarios drop \
        --views forward=free:dolly_in+roll fixed=static

ORDER OF OPERATIONS IS THE POINT. Everything checkable offline runs BEFORE
`SimulationApp` is constructed: arg parsing, view-spec grammar, family dispatch,
duplicate-name detection, and the printed plan. A typo, a stub family, or a
mis-sized grid costs a second — not a ~60 s Kit boot plus a rollout. That is
only possible because `clips_generator`, `cli` and `camera_motion` import no
omni; this driver would be a lie in a module that did.

Outputs, all under `--result`:
    <room>/<scenario>/<view>/clip_NNNNN/    the clips
    manifest.json                           the PLAN (every cell, pre-run)
    records.json                            what was REALIZED (per attempt)
Requested vs realized are kept as separate files on purpose: their difference is
the gating yield, which is the number that tells you a cell is un-framable.

MUST run under the Isaac conda env (env_isaaclab).
"""

import argparse
import dataclasses
import json
import os
import sys

# Pure imports only — no omni anywhere in this chain, which is what lets the
# whole pre-flight below run before Kit exists. Do not add an omni-importing
# module here; move it under `main()`'s post-boot section instead.
from clips_generator import ClipsGenerator, parse_view
from config import DEFAULT_DATA_PATH
import cli


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", default=os.path.join(
        os.path.dirname(DEFAULT_DATA_PATH), "grid_corpus"),
        help="base output dir; clips land in <result>/<room>/<scenario>/<view>/")
    p.add_argument("--rooms", nargs="+", default=[""],
                   help="environment names (environments.CANDIDATE_ENVIRONMENTS). "
                        "Omit for the current behaviour: sample one per clip.")
    p.add_argument("--scenarios", nargs="+", default=["drop"],
                   help="scenario names (scenarios.BUILDERS)")
    p.add_argument("--views", nargs="+", default=["static"],
                   help="view specs, '[label=]family[@speed][+roll]'. e.g. "
                        "forward=free:dolly_in+roll  steering=subject:orbit@2+roll")
    p.add_argument("--n_rollouts", type=int, default=2,
                   help="physics rollouts per (room, scenario) — each is shot by EVERY view")
    p.add_argument("--seed_stride", type=int, default=1000,
                   help="seed band width per scenario, so rollout groups stay globally unique")
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan, write manifest.json, and exit WITHOUT booting Isaac")
    p.add_argument("--on_error", choices=["skip", "raise"], default="skip",
                   help="skip: one bad seed can't kill a long corpus. raise: while developing")
    # Shared ClipConfig overrides. --camera_* knobs here act as the DEFAULT for
    # every view; a view spec's @speed / +roll override them per view.
    cli.add_config_args(p, defaults={"n_cameras": 1})
    return p.parse_args()


def build_generator(args):
    """args -> ClipsGenerator. Pure: no omni, so it is testable and pre-flightable."""
    base = cli.config_from_args(args)
    views = [parse_view(s, n_cameras=base.n_cameras,
                        track_tau=base.camera_track_tau) for s in args.views]
    # A view spec without @speed/+roll inherits the global camera flags, so
    # `--camera_speed_scale 2 --views a b` means "both views at 2x" and
    # `--views a@3` still wins locally. Least-surprise: local beats global.
    views = [v if v.speed_scale != 1.0 else
             dataclasses.replace(v, speed_scale=base.camera_speed_scale)
             for v in views]
    views = [v if v.roll_always else
             dataclasses.replace(v, roll_always=base.camera_roll_always)
             for v in views]
    return ClipsGenerator(
        result_dir=args.result, rooms=args.rooms, scenario_names=args.scenarios,
        views=views, n_rollouts=args.n_rollouts, config=base,
        seed_stride=args.seed_stride)


def main():
    args = parse_args()

    # ---- PRE-FLIGHT: everything that does not need Kit -------------------
    cli.validate_camera_args(args)
    gen = build_generator(args)
    try:
        gen.validate_pure()          # families, duplicate view names, cardinality
    except ValueError as e:
        sys.exit(f"invalid plan: {e}")
    print(gen.summary(), flush=True)
    manifest = gen.write_manifest()
    print(f"plan written: {manifest}", flush=True)
    if args.dry_run:
        print("--dry_run: stopping before Isaac boot", flush=True)
        return

    # ---- BOOT: nothing above this line may import omni -------------------
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    # Importing collect_clip here (not in run()) mirrors collect_corpus: its
    # module body installs the carb local-asset-root override, and that has to
    # be in place before the first environment reference is resolved.
    import collect_clip
    _ = collect_clip            # imported for its side effect, not its symbols
    print(f"env={os.environ.get('CONDA_DEFAULT_ENV', '?')} | python={sys.executable}",
          flush=True)

    records = gen.run(on_error=args.on_error)

    # ---- REPORT ----------------------------------------------------------
    out = os.path.join(gen.result_dir, "records.json")
    with open(out, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=1)
    ok = sum(1 for r in records if r.ok)
    print("\n=== YIELD (per cell — a global average hides an un-framable cell) ===")
    print(ClipsGenerator.yield_table(records), flush=True)
    print("\n=== COST (is room-major ordering paying for itself?) ===")
    print(ClipsGenerator.cost_table(records), flush=True)
    print(f"\n{ok}/{len(records)} clips validated over {len(gen.plan())} cells "
          f"-> {gen.result_dir}\nrecords: {out}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
