"""Offline checks for clips_generator.py — runs WITHOUT Isaac.

Two tiers, and the split is the point:

  * **pure tier** — `plan`, `config_for`, `validate`, `yield_table`,
    `cost_table`, `write_manifest`. These are the reason the class exists: a
    corpus you can inspect and cost before booting Kit. If they need Isaac to
    test, they were not pure and the whole design claim is false.
  * **orchestration tier** — `run()`'s bookkeeping (clip indices, record
    building, error containment, the room-reload flag), exercised against
    STUBBED omni + collect_clip. This tests wiring only, never generation, and
    says nothing about whether a clip is correct — that is what the Isaac
    ladder in data-gen-testing.md is for.

    python3 src/data_gen/verify_clips_generator.py
"""

import dataclasses
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from clips_generator import (  # noqa: E402
    ClipRecord, ClipsGenerator, ViewSpec, parse_view,
)
from config import ClipConfig  # noqa: E402

FAILURES = []

# The ONLY ClipConfig fields a view is allowed to touch. Anything else differing
# between two views of one rollout would mean the camera perturbed the scene,
# which breaks the twin guarantee this class is built on.
VIEW_FIELDS = {"scenarios", "camera_path_families", "camera_path_weights",
               "camera_speed_scale", "camera_roll_always", "camera_track_tau",
               "n_cameras"}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_gen(result_dir="/tmp/cg", n_rollouts=3, **kw):
    return ClipsGenerator(
        result_dir=result_dir,
        rooms=kw.pop("rooms", ["warehouse", "simple_room"]),
        scenario_names=kw.pop("scenario_names", ["drop", "collide"]),
        views=kw.pop("views", [
            ViewSpec("free:dolly_in", roll_always=True, label="forward"),
            ViewSpec("free:dolly_in", speed_scale=2.0, roll_always=True, label="forward_fast"),
            ViewSpec("subject:orbit", roll_always=True, label="steering"),
        ]),
        n_rollouts=n_rollouts, **kw)


def diff_fields(a, b):
    return {f.name for f in dataclasses.fields(a)
            if getattr(a, f.name) != getattr(b, f.name)}


# --- pure tier --------------------------------------------------------------

def test_plan():
    g = make_gen()
    cells = g.plan()
    check("plan: cardinality = rooms x scenarios x rollouts x views",
          len(cells) == 2 * 2 * 3 * 3, f"{len(cells)} cells")

    # Room-major: the room axis must never revisit a value it has left, or the
    # ordering claim (and any cache benefit it bets on) is not what we think.
    seen, order_ok = [], True
    for c in cells:
        if not seen or seen[-1] != c.scene.room:
            order_ok &= c.scene.room not in seen
            seen.append(c.scene.room)
    check("plan: room-major (each room is one contiguous block)", order_ok,
          f"room blocks: {seen}")

    # View innermost — consecutive cells share a rollout until the views run out.
    idx = [(c.scene.room, c.scene.scenario, c.scene.rollout_seed) for c in cells]
    check("plan: view is the INNERMOST axis (twins land adjacent)",
          all(idx[i] == idx[i + 1] for i in range(0, len(idx) - 1, 3)))

    check("plan: deterministic", [dataclasses.astuple(c) for c in g.plan()]
          == [dataclasses.astuple(c) for c in make_gen().plan()])

    # Seed bands: scenario i owns [i*stride, i*stride + n_rollouts).
    bands = {}
    for c in cells:
        bands.setdefault(c.scene.scenario, set()).add(c.scene.rollout_seed)
    check("plan: per-scenario seed bands are disjoint",
          not set.intersection(*bands.values()), f"{ {k: sorted(v) for k, v in bands.items()} }")

    # A room is a DIFFERENT shoot of the SAME physics — same seeds, different dir.
    by_room = {}
    for c in cells:
        by_room.setdefault(c.scene.room, set()).add((c.scene.scenario, c.scene.rollout_seed))
    check("plan: rooms reuse the same rollout seeds (same physics, new backdrop)",
          len(set(map(frozenset, by_room.values()))) == 1)

    dirs = {c.out_dir for c in cells}
    check("plan: out_dir is unique per (room, scenario, view) cell",
          len(dirs) == 2 * 2 * 3, f"{len(dirs)} dirs")
    check("plan: out_dir encodes the factor structure",
          cells[0].out_dir.endswith(os.path.join("warehouse", "drop", "forward")),
          cells[0].out_dir)


def test_defaults():
    g = ClipsGenerator("/tmp/cg", rooms=[], scenario_names=["drop"], views=[])
    cells = g.plan()
    check("defaults: no rooms -> one <sampled> cell, no views -> one static view",
          len(cells) == 1 and cells[0].scene.room == ""
          and cells[0].view.family == "static")
    check("defaults: sampled room writes under 'sampled/'",
          cells[0].out_dir.endswith(os.path.join("sampled", "drop", "static")))


def test_config_for():
    g = make_gen()
    cells = g.plan()
    base = ClipConfig()
    cfg = g.config_for(cells[1])          # forward_fast
    check("config_for: view fields are applied",
          cfg.camera_path_families == ("free:dolly_in",) and cfg.camera_speed_scale == 2.0
          and cfg.camera_roll_always is True and cfg.n_cameras == 1
          and cfg.scenarios == ("drop",))

    # ★ THE TWIN GUARANTEE, at config level: two views of one rollout may differ
    # ONLY in camera fields. A physics-affecting field leaking in would make the
    # "same scene, different angle" claim false and the pair useless as a twin.
    a, b = g.config_for(cells[0]), g.config_for(cells[2])
    d = diff_fields(a, b)
    check("★ config_for: two views of one rollout differ ONLY in camera fields",
          d <= VIEW_FIELDS and "scenarios" not in d, f"differs in {sorted(d)}")

    d_base = diff_fields(base, cfg)
    check("config_for: nothing outside the view surface is touched",
          d_base <= VIEW_FIELDS, f"differs from ClipConfig in {sorted(d_base)}")

    # A custom base must survive — ClipConfig stays the single default source.
    g2 = make_gen(config=dataclasses.replace(ClipConfig(), frame_count=32,
                                             min_body_coverage=0.05))
    c2 = g2.config_for(g2.plan()[0])
    check("config_for: non-view knobs fall through from the base config",
          c2.frame_count == 32 and c2.min_body_coverage == 0.05)

    # weights must be cleared: a stale parallel weight vector against a
    # single-family tuple would raise inside the sampler, mid-Isaac.
    g3 = make_gen(config=dataclasses.replace(
        ClipConfig(), camera_path_families=("a", "b"), camera_path_weights=(0.5, 0.5)))
    check("config_for: stale camera_path_weights are cleared",
          g3.config_for(g3.plan()[0]).camera_path_weights is None)


def test_validate():
    # validate_pure ONLY here: full validate() imports environments/scenarios,
    # which import omni — and importing omni without a live SimulationApp is the
    # documented way to segfault Kit, so it is not something a no-Isaac verifier
    # may reach for. That split is the reason validate_pure exists.
    check("validate_pure: accepts implemented families",
          make_gen().validate_pure() is None)
    for bad, why in (("free:steer", "implemented-but-stub"),
                     ("free:nope", "unknown kind"),
                     ("gait", "ambiguous bare name")):
        g = make_gen(views=[ViewSpec(bad)])
        try:
            g.validate_pure()
            check(f"validate_pure: rejects {bad!r} ({why})", False)
        except ValueError:
            check(f"validate_pure: rejects {bad!r} ({why})", True)
    check("validate_pure: imports no omni (module list is clean)",
          not any(m == "omni" or m.startswith("omni.") for m in sys.modules))


def test_view_naming():
    check("ViewSpec: label wins over the derived name",
          ViewSpec("free:dolly_in", label="forward").name == "forward")
    n = ViewSpec("free:dolly_in", speed_scale=2.0, roll_always=True).name
    check("ViewSpec: derived name is filesystem-safe and carries the knobs",
          ":" not in n and "s2" in n and "roll" in n, n)
    check("ViewSpec: distinct speeds -> distinct cell names",
          ViewSpec("subject:orbit", speed_scale=1.0).name
          != ViewSpec("subject:orbit", speed_scale=2.0).name)


def test_parse_view():
    """The CLI grammar. A mis-parse here is silent: you get a corpus shot with
    the wrong camera and nothing complains, so every field is asserted."""
    cases = {
        "static": ("static", 1.0, False, ""),
        "free:dolly_in": ("free:dolly_in", 1.0, False, ""),
        "free:dolly_in+roll": ("free:dolly_in", 1.0, True, ""),
        "free:dolly_in@2": ("free:dolly_in", 2.0, False, ""),
        "free:dolly_in@2.5+roll": ("free:dolly_in", 2.5, True, ""),
        "forward=free:dolly_in+roll": ("free:dolly_in", 1.0, True, "forward"),
        "fast=subject:orbit@3+roll": ("subject:orbit", 3.0, True, "fast"),
    }
    ok = True
    for spec, (fam, sp, roll, label) in cases.items():
        v = parse_view(spec)
        got = (v.family, v.speed_scale, v.roll_always, v.label)
        if got != (fam, sp, roll, label):
            ok = False
            print(f"      {spec!r} -> {got}, expected {(fam, sp, roll, label)}")
    check("parse_view: [label=]family[@speed][+roll] round-trips", ok,
          f"{len(cases)} specs")

    # The family separator ':' must survive '@' and '+' handling — 'free:gait'
    # losing its prefix would silently resolve to the WRONG camera model.
    check("parse_view: prefixed family keeps its ':' (model is not lost)",
          parse_view("x=free:gait@2+roll").family == "free:gait")
    check("parse_view: label becomes the cell name",
          parse_view("forward=free:dolly_in").name == "forward")
    check("parse_view: rig knobs come from the globals, not the spec",
          parse_view("static", n_cameras=7, track_tau=0.3).n_cameras == 7
          and parse_view("static", track_tau=0.3).track_tau == 0.3)
    try:
        parse_view("free:dolly_in@fast")
        check("parse_view: rejects a non-numeric speed", False)
    except ValueError:
        check("parse_view: rejects a non-numeric speed", True)

    # Two views that differ only in a knob must still get distinct directories.
    a, b = parse_view("free:dolly_in"), parse_view("free:dolly_in@2")
    check("parse_view: unlabelled views differing in speed stay separable",
          a.name != b.name, f"{a.name} vs {b.name}")


def test_duplicate_view_names():
    g = make_gen(views=[ViewSpec("free:dolly_in", label="fwd"),
                        ViewSpec("subject:orbit", label="fwd")])
    try:
        g.validate_pure()
        check("validate_pure: rejects duplicate view names (shared out_dir)", False)
    except ValueError as e:
        check("validate_pure: rejects duplicate view names (shared out_dir)",
              "duplicate view names" in str(e))
    g2 = make_gen(n_rollouts=0)
    try:
        g2.validate_pure()
        check("validate_pure: rejects n_rollouts < 1", False)
    except ValueError:
        check("validate_pure: rejects n_rollouts < 1", True)


def test_reporting():
    R = ClipRecord
    recs = ([R("warehouse", "drop", 0, "fwd", "/d", True, None) for _ in range(9)]
            + [R("warehouse", "drop", 1, "fwd", None, False, "no viewable camera")]
            + [R("warehouse", "collide", 0, "fwd", None, False, "no viewable camera")
               for _ in range(5)]
            + [R("warehouse", "collide", 1, "fwd", "/d", True, None)])
    t = ClipsGenerator.yield_table(recs)
    check("yield_table: per-cell yield, not a global average",
          "16.7%" in t and "90.0%" in t, t.splitlines()[-1].strip())

    # cost_table must count each CELL once even though a cell can emit many
    # clips — otherwise a 7-camera cell would outvote six 1-camera cells.
    multi = [R("warehouse", "drop", 0, "fwd", f"/d{i}", True, None, 90.0, True)
             for i in range(7)] + \
            [R("warehouse", "drop", 1, "fwd", "/d", True, None, 50.0, False)]
    ct = ClipsGenerator.cost_table(multi)
    check("cost_table: one timing per CELL, not per clip",
          " 1 " in ct.replace("    ", " ") and "90.0" in ct and "50.0" in ct)


def test_manifest():
    with tempfile.TemporaryDirectory() as d:
        g = make_gen(result_dir=d)
        p = g.write_manifest()
        rows = json.load(open(p))
        check("write_manifest: one row per planned cell",
              len(rows) == len(g.plan()))
        check("write_manifest: rows carry the full ViewSpec (corpus is re-derivable)",
              set(rows[0]["view_spec"]) == {f.name for f in dataclasses.fields(ViewSpec)})


def test_driver_preflight():
    """collect_grid's pre-flight must work with Kit closed, or its whole
    order-of-operations claim is false. Importing the driver at all is half the
    test: a single omni import at its module level would fail right here."""
    import collect_grid

    class A:                       # stand-in for argparse.Namespace
        pass

    def args(**kw):
        a = A()
        a.result, a.rooms, a.scenarios = "/tmp/cg", ["warehouse"], ["drop"]
        a.views, a.n_rollouts, a.seed_stride = ["static"], 2, 1000
        for f, _field, _kw, _h in cli_config_args():
            setattr(a, f.lstrip("-"), None)
        a.n_cameras = 1
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    g = collect_grid.build_generator(args(views=["fwd=free:dolly_in+roll",
                                                 "fixed=static"]))
    check("driver: build_generator is pure (no Isaac) and builds the grid",
          len(g.plan()) == 1 * 1 * 2 * 2, f"{len(g.plan())} cells")
    check("driver: view specs reach the ClipConfig",
          g.config_for(g.plan()[0]).camera_roll_always is True)

    # ★ local-beats-global: a global --camera_speed_scale is the DEFAULT for
    #   views that don't say, and a view's @speed overrides it. Getting this
    #   backwards would silently run a whole sweep at one speed.
    g2 = collect_grid.build_generator(
        args(views=["slow=free:dolly_in", "fast=free:dolly_in@3"],
             camera_speed_scale=2.0))
    speeds = {v.label: v.speed_scale for v in g2.views}
    check("★ driver: global speed is the default, a view's @speed overrides it",
          speeds == {"slow": 2.0, "fast": 3.0}, f"{speeds}")

    g3 = collect_grid.build_generator(args(views=["free:dolly_in"],
                                           camera_roll_always=True))
    check("driver: global --camera_roll_always reaches a bare view",
          g3.views[0].roll_always is True)


def cli_config_args():
    import cli
    return cli._CONFIG_ARGS


# --- orchestration tier (stubbed omni) --------------------------------------

def _install_stubs(collect_impl):
    """Stand in for the three symbols `run()` imports, so bookkeeping is testable.

    Deliberately narrow: this proves indices, records and error containment are
    wired right. It cannot and must not say anything about generated data — the
    stub never renders. (The lesson from `verify_forward_motion`: a test seam
    that quietly stops matching the code is worse than no seam, so `run()`
    imports these by name and a rename breaks HERE, loudly.)
    """
    core = types.ModuleType("omni.isaac.core")
    core.World = type("World", (), {"clear_instance": staticmethod(lambda: None)})
    stage = types.ModuleType("omni.isaac.core.utils.stage")
    stage.create_new_stage = lambda: None
    cc = types.ModuleType("collect_clip")
    cc.collect_clip = collect_impl
    # `run()` calls the FULL validate(), which reads the scenario and room
    # registries. Stub those too: the real modules import omni.client, and this
    # tier is testing bookkeeping, not name validation (which the pure tier
    # already covers). Stubbing them here is also what keeps the earlier
    # "validate_pure imports no omni" check meaningful — it runs before this.
    sc = types.ModuleType("scenarios")
    sc.BUILDERS = {"drop": None, "collide": None}
    env = types.ModuleType("environments")
    env.CANDIDATE_ENVIRONMENTS = [("warehouse", "u"), ("simple_room", "u")]
    env.NONE_NAME = "none"
    for name, mod in (("omni", types.ModuleType("omni")),
                      ("omni.isaac", types.ModuleType("omni.isaac")),
                      ("omni.isaac.core", core),
                      ("omni.isaac.core.utils", types.ModuleType("omni.isaac.core.utils")),
                      ("omni.isaac.core.utils.stage", stage),
                      ("collect_clip", cc), ("scenarios", sc),
                      ("environments", env)):
        sys.modules[name] = mod


def test_run_bookkeeping():
    calls = []

    def fake_collect(out_dir, start_idx, config, seed, env_override=None):
        calls.append((out_dir, start_idx, seed, env_override,
                      config.camera_path_families[0]))
        if config.scenarios[0] == "collide":
            return []                                   # gate refusal, every rollout
        if seed == 1:
            # A DROP seed, deliberately: `collide` already returns [] above, so
            # raising on a collide seed would be unreachable and the check would
            # pass vacuously. (It did, on the first run of this file.)
            raise RuntimeError("could not place bodies without overlap")
        return [(os.path.join(out_dir, f"clip_{start_idx:05d}"), True, None)]

    _install_stubs(fake_collect)
    g = make_gen(n_rollouts=2, rooms=["warehouse", "simple_room"],
                 scenario_names=["drop", "collide"],
                 views=[ViewSpec("free:dolly_in", label="fwd"),
                        ViewSpec("subject:orbit", label="orb")])
    recs = g.run(verbose=False)

    check("run: one record per attempted cell", len(recs) == len(g.plan()),
          f"{len(recs)} records / {len(g.plan())} cells")
    check("run: env_override is the pinned room (None when sampled)",
          {c[3] for c in calls} == {"warehouse", "simple_room"})
    # Expected counts are DERIVED from the plan, not hardcoded: hardcoding them
    # made two of these checks assert the wrong number and fail on correct code.
    plan = g.plan()
    n_collide = sum(1 for c in plan if c.scene.scenario == "collide")
    n_raise = sum(1 for c in plan
                  if c.scene.scenario == "drop" and c.scene.rollout_seed == 1)
    check("run: gate refusal is recorded, not dropped",
          sum(1 for r in recs if r.reason == "no viewable camera") == n_collide,
          f"{n_collide} expected")
    check("run: a raising cell is contained and labelled",
          sum(1 for r in recs if r.reason and "RuntimeError" in r.reason) == n_raise,
          f"{n_raise} expected")
    check("run: on_error='raise' propagates instead", _raises(g))

    # ★ The invariant that actually matters: no two WRITTEN clips may share a
    #   path. Repeated start_idx is legitimate — a rollout that writes nothing
    #   must not advance the counter, or the corpus grows index holes. (My first
    #   version asserted "indices never repeat" and failed on exactly that
    #   legitimate case: `collide` returns [] twice, so both calls start at 0.)
    written = [r.clip_dir for r in recs if r.clip_dir]
    check("★ run: no two written clips share a directory",
          len(written) == len(set(written)), f"{len(written)} written")
    per_dir = {}
    for out_dir, start_idx, *_ in calls:
        per_dir.setdefault(out_dir, []).append(start_idx)
    check("run: per-dir clip index is non-decreasing and starts at 0",
          all(v == sorted(v) and v[0] == 0 for v in per_dir.values()))
    check("run: a cell that writes nothing does NOT advance its index",
          all(v == [0] * len(v) for d, v in per_dir.items() if "collide" in d))

    check("run: one family per cell reaches collect_clip",
          {c[4] for c in calls} == {"free:dolly_in", "subject:orbit"})
    check("run: room_reload marks exactly one cell per room block",
          sum(1 for r in recs if r.room_reload) == len({r.room for r in recs}),
          f"{sum(1 for r in recs if r.room_reload)} reloads / "
          f"{len({r.room for r in recs})} rooms")
    check("run: records carry timings", all(r.seconds >= 0.0 for r in recs))


def _raises(g):
    def boom(*a, **k):
        raise RuntimeError("nope")
    _install_stubs(boom)
    try:
        g.run(on_error="raise", verbose=False)
        return False
    except RuntimeError:
        return True


def main():
    for t in (test_plan, test_defaults, test_config_for, test_validate,
              test_view_naming, test_parse_view, test_duplicate_view_names,
              test_reporting, test_manifest, test_driver_preflight,
              test_run_bookkeeping):
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all clips-generator invariants hold")


if __name__ == "__main__":
    main()
