"""ClipsGenerator — a corpus is a PLAN over (room × scenario × view).

`collect_corpus.py` treats a corpus as "one ClipConfig, looped over scenarios".
That makes the two axes you actually want to control invisible: which ROOM a
clip was shot in, and which EGOMOTION it was shot with. Both currently arrive by
sampling — `camera_path_families` is drawn per camera, the environment per clip
— so you cannot ask for "40 clips of forward motion in a warehouse" or read the
per-cell yield afterwards. This module adds that axis structure WITHOUT touching
the generation path: it builds ClipConfigs and calls `collect_clip` exactly as
the existing drivers do.

THREE IDEAS, in the order they matter:

1. **The plan is a value, not a side effect.** `plan()` is pure — no omni, no
   Isaac, no GPU — and returns the full list of cells. Print it, diff it, count
   it, cost it BEFORE booting Kit. Generation is then "execute a manifest".

2. **Requested ≠ realized.** Visibility gating rejects cells; a rollout can emit
   zero clips. `run()` returns one `ClipRecord` per attempt carrying the cell
   that produced it, so `yield_table()` can say *which cell* is dropping (the
   `collide` × dynamic-camera failure was invisible in a global "20/21").

3. **Seed decomposition is what makes multi-view honest.** Views of one rollout
   are independent data only if the camera cannot perturb physics. Two
   structural facts make that true here and both are load-bearing:
     * camera draws happen strictly AFTER every physics-affecting draw
       (dynamic-camera.md §4), so changing `family` leaves the rollout
       bit-identical — that is the egomotion-twin precondition; and
     * `environments.setup_random_environment(rng, force_name=...)` returns
       WITHOUT consuming an rng draw, so pinning `room` also leaves physics
       bit-identical.
   Together: one `rollout_seed` = one physics trajectory, reproducible across
   every room and every view. Same scene, different camera = a controlled pair.

EXECUTION ORDER IS AN OPTIMIZATION, NOT PART OF THE CORPUS. Cells are emitted
room-major (room → scenario → rollout → view) because a room change reloads USD
assets and a view change costs nothing extra. Reordering changes cost and
nothing else — the set of cells, and therefore the corpus, is identical.

    gen = ClipsGenerator(
        result_dir="~/data/corpus",
        rooms=["warehouse", "simple_room"],
        scenario_names=["drop", "collide"],
        views=[ViewSpec("free:dolly_in", roll_always=True),
               ViewSpec("subject:orbit", speed_scale=2.0, roll_always=True)],
        n_rollouts=5,
    )
    print(gen.summary())          # offline — costs nothing
    records = gen.run()           # needs a live SimulationApp
    print(ClipsGenerator.yield_table(records))
"""

import dataclasses
import json
import os
import time

from config import ClipConfig                       # pure: no omni at import


# --- the axes ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SceneSpec:
    """One physics rollout. Changing anything here costs a stage rebuild."""
    room: str                  # environments name; "" = sample per clip, "none" = bare
    scenario: str              # key in scenarios.BUILDERS
    rollout_seed: int          # THE physics identity — see module docstring §3


@dataclasses.dataclass(frozen=True)
class ViewSpec:
    """How a rollout is observed. Free to vary: physics is untouched.

    `family` is a `camera_motion` name and MUST be prefixed when ambiguous
    ('subject:gait' vs 'free:gait'). One family per view by design — a view is a
    named experimental condition ("forward", "steering"), so sampling a family
    per camera would make the condition unlabelled, which is the thing this
    class exists to fix.
    """
    family: str = "static"
    speed_scale: float = 1.0
    roll_always: bool = False
    track_tau: float = 0.0
    n_cameras: int = 1
    label: str = ""            # optional human name ("forward"); defaults to derived

    @property
    def name(self):
        """Filesystem-safe cell name. `label` wins so a corpus can speak your
        vocabulary ("forward") rather than the implementation's."""
        if self.label:
            return self.label
        n = self.family.replace(":", "-")
        if self.speed_scale != 1.0:
            n += f"_s{self.speed_scale:g}"
        if self.roll_always:
            n += "_roll"
        return n


def parse_view(spec, n_cameras=1, track_tau=0.0):
    """`[label=]family[@speed][+roll]` -> ViewSpec. The CLI surface for a view.

        forward=free:dolly_in+roll        labelled, rolled, speed 1
        free:dolly_in@2+roll              derived label, twice as fast
        static                            the legacy zero-coefficient case

    A compact grammar rather than a JSON file because a view is exactly four
    scalars and a name; a second config format would be more surface than the
    thing it configures. `n_cameras`/`track_tau` stay global (cli.py) — they are
    rig properties, not what distinguishes one condition from another.

    Note `@` and `+` are parsed BEFORE the family is read, so `free:dolly_in`
    keeps its colon: the family separator is `:` and never collides with these.
    """
    s = str(spec).strip()
    label, _, rest = s.rpartition("=")          # rpartition: no '=' -> label ""
    roll = False
    while rest.endswith("+roll"):
        roll, rest = True, rest[:-len("+roll")]
    family, _, speed = rest.partition("@")
    try:
        speed_scale = float(speed) if speed else 1.0
    except ValueError:
        raise ValueError(f"bad speed {speed!r} in view spec {spec!r} "
                         f"(expected [label=]family[@speed][+roll])")
    return ViewSpec(family=family, speed_scale=speed_scale, roll_always=roll,
                    track_tau=float(track_tau), n_cameras=int(n_cameras),
                    label=label)


@dataclasses.dataclass(frozen=True)
class Cell:
    """One `collect_clip` call: a scene observed one way, into one directory."""
    scene: SceneSpec
    view: ViewSpec
    out_dir: str

    @property
    def key(self):
        """Grouping key for yield reporting — the cell WITHOUT the rollout."""
        return (self.scene.room, self.scene.scenario, self.view.name)


@dataclasses.dataclass
class ClipRecord:
    """One realized (or refused) clip. The corpus's actual shape, not the plan's.

    `ok=False` with `clip_dir=None` means the rollout produced no viewable
    camera at all — a gate refusal, not a validation failure. The two are
    different diagnoses and are kept distinguishable on purpose.
    """
    room: str
    scenario: str
    rollout_seed: int
    view: str
    clip_dir: str = None
    ok: bool = False
    reason: str = None
    seconds: float = 0.0       # wall time for the whole cell
    room_reload: bool = False  # was this the FIRST cell in its room?

    def to_dict(self):
        return dataclasses.asdict(self)


# --- the generator ----------------------------------------------------------

class ClipsGenerator:
    """Plan a (room × scenario × view × rollout) corpus, then execute it.

    Owns NO generation logic: `config_for()` produces a `ClipConfig` and `run()`
    hands it to `collect_clip`, exactly as `collect_corpus` does. Everything
    physics- or camera-side stays where it already lives.
    """

    def __init__(self, result_dir, rooms, scenario_names, views,
                 n_rollouts=1, config=None, seed_stride=1000):
        self.result_dir = os.path.abspath(os.path.expanduser(result_dir))
        # "" = let collect_clip sample the environment, preserving current
        # behaviour as the zero-configuration case rather than a special mode.
        self.rooms = list(rooms) if rooms else [""]
        self.scenario_names = list(scenario_names)
        self.views = list(views) if views else [ViewSpec()]
        self.n_rollouts = int(n_rollouts)
        self.base_config = config if config is not None else ClipConfig()
        # Disjoint seed band per scenario, same convention as collect_corpus, so
        # rollout groups stay globally unique across a corpus.
        self.seed_stride = int(seed_stride)

    # --- pure: safe to call with no Isaac, no GPU, no stage ------------------

    def plan(self):
        """Every cell, room-major. Pure — call it before spending anything.

        Room is the OUTER loop because it is the only axis that reloads USD
        assets; view is innermost because it is nearly free, which also lands
        the twins (same rollout, different camera) adjacent in the manifest.
        """
        cells = []
        for room in self.rooms:
            for si, scenario in enumerate(self.scenario_names):
                for r in range(self.n_rollouts):
                    seed = si * self.seed_stride + r
                    scene = SceneSpec(room=room, scenario=scenario, rollout_seed=seed)
                    for view in self.views:
                        cells.append(Cell(
                            scene=scene, view=view,
                            out_dir=os.path.join(self.result_dir, room or "sampled",
                                                 scenario, view.name)))
        return cells

    def config_for(self, cell):
        """The `ClipConfig` for one cell. Pure, so a cell's config is testable.

        Only the view's fields are overridden — every other knob falls through
        to `base_config`, keeping ClipConfig the single source of default truth
        (the same rule cli.py follows).
        """
        return dataclasses.replace(
            self.base_config,
            scenarios=(cell.scene.scenario,),
            camera_path_families=(cell.view.family,),
            camera_path_weights=None,
            camera_speed_scale=float(cell.view.speed_scale),
            camera_roll_always=bool(cell.view.roll_always),
            camera_track_tau=float(cell.view.track_tau),
            n_cameras=int(cell.view.n_cameras),
        )

    def validate_pure(self):
        """Everything checkable with Kit closed. Raises ValueError. Call FIRST.

        Camera families go through the REAL dispatch
        (`check_family_implemented` — 8 of 15 families are still stubs, and a
        stub used to survive validation and then raise 12 s into a run, stage
        already built).

        Deliberately does NOT touch `scenarios`/`environments`: both import omni
        at module level, and importing omni before `SimulationApp` exists is the
        documented way to segfault Kit. A `try: import` would be gambling a
        crash against a diagnostic — so room and scenario names wait for
        `validate()`, which runs after boot.
        """
        from camera_motion import check_family_implemented    # pure
        for v in self.views:
            check_family_implemented(v.family)
        if self.n_rollouts < 1:
            raise ValueError(f"n_rollouts must be >= 1, got {self.n_rollouts}")
        dupes = [n for n in {v.name for v in self.views}
                 if sum(1 for v in self.views if v.name == n) > 1]
        if dupes:
            # Two views sharing a name share an out_dir, so their clips would
            # interleave in one directory and stop being separable conditions.
            raise ValueError(f"duplicate view names {dupes} — give them distinct "
                             f"labels, they would share an output directory")

    def validate(self):
        """`validate_pure()` plus the checks that need omni. Requires a live Kit."""
        self.validate_pure()
        import environments as _env
        import scenarios as _sc
        bad = [s for s in self.scenario_names if s not in _sc.BUILDERS]
        if bad:
            raise ValueError(f"unknown scenarios {bad}; have {sorted(_sc.BUILDERS)}")
        known = {n for n, _ in _env.CANDIDATE_ENVIRONMENTS} | {_env.NONE_NAME, ""}
        bad = [r for r in self.rooms if r not in known]
        if bad:
            raise ValueError(f"unknown rooms {bad}; have {sorted(known - {''})}")

    def summary(self):
        """One-screen description of what `run()` would do, and what it costs."""
        cells = self.plan()
        rollouts = len({(c.scene.room, c.scene.scenario, c.scene.rollout_seed)
                        for c in cells})
        return (f"{len(cells)} cells = {len(self.rooms)} room(s) × "
                f"{len(self.scenario_names)} scenario(s) × {self.n_rollouts} rollout(s) "
                f"× {len(self.views)} view(s)\n"
                f"  {rollouts} distinct rollouts, ≤{cells[0].view.n_cameras if cells else 0} "
                f"camera(s) each, {self.base_config.frame_count} frames\n"
                f"  rooms:     {', '.join(r or '<sampled>' for r in self.rooms)}\n"
                f"  scenarios: {', '.join(self.scenario_names)}\n"
                f"  views:     {', '.join(v.name for v in self.views)}\n"
                f"  -> {self.result_dir}")

    def write_manifest(self, path=None):
        """Persist the plan next to the data. The corpus's definition, on disk."""
        path = path or os.path.join(self.result_dir, "manifest.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([{"room": c.scene.room, "scenario": c.scene.scenario,
                        "rollout_seed": c.scene.rollout_seed,
                        "view": c.view.name, "view_spec": dataclasses.asdict(c.view),
                        "out_dir": c.out_dir} for c in self.plan()], f, indent=1)
        return path

    # --- impure: needs a live SimulationApp ---------------------------------

    def run(self, on_error="skip", verbose=True):
        """Execute the plan. Requires `SimulationApp` to already exist.

        Per-cell isolation is `World.clear_instance()` + `create_new_stage()` —
        the same reset `collect_clip.main` and `collect_corpus` use between
        rollouts, not a new one. `on_error="skip"` keeps one bad seed from
        killing a long corpus (scene builds can fail on overlap); "raise"
        surfaces it, which is what you want while developing a scenario.
        """
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import create_new_stage
        from collect_clip import collect_clip

        self.validate()
        cells = self.plan()
        records, counters, first, prev_room = [], {}, True, object()
        for i, cell in enumerate(cells):
            if not first:
                # NOTE — this DISCARDS the stage, room included, and the next
                # collect_clip re-references the room USD from scratch
                # (collect_clip.py: setup_random_environment runs per call).
                # Room-major ordering therefore does NOT skip a reload; it can
                # only win via caches BELOW composition (omni.client's asset
                # cache, a live Sdf.Layer). `cost_table` measures whether that
                # win is real on this machine instead of assuming it.
                World.clear_instance()
                create_new_stage()
            first = False
            reload_room = cell.scene.room != prev_room
            prev_room = cell.scene.room
            idx = counters.get(cell.out_dir, 0)
            if verbose:
                print(f"[cell {i + 1}/{len(cells)}] {cell.scene.room or '<sampled>'} | "
                      f"{cell.scene.scenario} | {cell.view.name} | "
                      f"seed={cell.scene.rollout_seed}", flush=True)
            common = dict(room=cell.scene.room, scenario=cell.scene.scenario,
                          rollout_seed=cell.scene.rollout_seed, view=cell.view.name,
                          room_reload=reload_room)
            t0 = time.time()
            try:
                results = collect_clip(cell.out_dir, idx, self.config_for(cell),
                                       cell.scene.rollout_seed,
                                       env_override=(cell.scene.room or None))
            except Exception as e:
                if on_error == "raise":
                    raise
                records.append(ClipRecord(**common, seconds=time.time() - t0,
                                          reason=f"{type(e).__name__}: {e}"))
                continue
            dt = time.time() - t0
            counters[cell.out_dir] = idx + len(results)
            if not results:
                # A gate refusal, not a failure: no camera framed the action.
                records.append(ClipRecord(**common, seconds=dt,
                                          reason="no viewable camera"))
                continue
            for out_dir, ok, reason in results:
                records.append(ClipRecord(**common, clip_dir=out_dir, seconds=dt,
                                          ok=bool(ok), reason=reason))
        return records

    # --- reading the result -------------------------------------------------

    @staticmethod
    def yield_table(records):
        """Per-cell yield. The point of the whole class: a global pass rate hides
        a structurally un-framable cell, which is exactly how `collide` lost 5/6
        rollouts while the total still looked survivable."""
        cells = {}
        for r in records:
            k = (r.room or "<sampled>", r.scenario, r.view)
            got, ok = cells.get(k, (0, 0))
            cells[k] = (got + 1, ok + int(r.ok))
        w = max((len(f"{a} | {b} | {c}") for a, b, c in cells), default=20)
        lines = [f"{'room | scenario | view':<{w}}  {'ok':>5} / {'att':<5} yield"]
        for (room, scen, view), (got, ok) in sorted(cells.items()):
            lines.append(f"{room + ' | ' + scen + ' | ' + view:<{w}}  "
                         f"{ok:>5} / {got:<5} {100.0 * ok / max(got, 1):5.1f}%")
        return "\n".join(lines)

    @staticmethod
    def cost_table(records):
        """Does room-major ordering actually pay? Measure, don't assume.

        Every cell resets the stage and re-references its room, so the only
        possible saving from grouping same-room cells is cache hits below USD
        composition. If `first in room` and `same room as previous` cost the
        same, the ordering buys nothing and the plan is free to be ordered for
        balance (e.g. interleaving rooms so a crash leaves a usable corpus)
        instead. One `collect_clip` call per cell, deduplicated by clip.
        """
        seen, groups = set(), {True: [], False: []}
        for r in records:
            k = (r.room, r.scenario, r.rollout_seed, r.view)
            if k in seen:
                continue                      # one timing per cell, not per clip
            seen.add(k)
            groups[bool(r.room_reload)].append(r.seconds)
        lines = [f"{'cell':<22} {'n':>4} {'mean s':>9} {'median s':>9}"]
        for flag, label in ((True, "first in room"), (False, "same room as prev")):
            v = sorted(groups[flag])
            if not v:
                continue
            lines.append(f"{label:<22} {len(v):>4} {sum(v) / len(v):9.1f} "
                         f"{v[len(v) // 2]:9.1f}")
        return "\n".join(lines)
