# CLAUDE.md — operating manual for this repo

### Principles
1. Backward compatibility. The exisiting codebase should remain untouched unless user explicitly specified structural changes in functionality. Otherwise create new file. <br />
2. Avoid code duplication. If there is possibility to import function prefer it over naive copy-paste. <br />
3. When conducting training experiments always save train logs, tensorboard logs, TRAIN_RESULTS_AND_CONCLUSIONS.md and HOW_TO_TRAIN_AND_VERIFY.md

Agent-facing working notes. **Not** an architecture doc — for *what* the project is
and *why*, read [README.md](README.md) (the map) and the source-of-truth docs in
[plans/](plans/). This file is only the *how to work here* layer.

## Orientation
- Research codebase for a **physics-informed visual encoder** (sim→real bridge). Only
  the Isaac Sim **data-generation** code exists so far ([src/data_gen/](src/data_gen/));
  encoder/training/probes are planned, not written.
- Docs are the source of truth, grouped under `plans/`: [theory/](plans/theory/)
  (modeling/paper), [data-gen/](plans/data-gen/) (collection pipeline),
  [critique/](plans/critique/) (reviews). Three hubs stay at `plans/` root, on
  different axes: [BRAINSTORM.md](plans/BRAINSTORM.md) (**why** — architecture +
  risks R1–R11), [IMPLEMENTATION.md](plans/IMPLEMENTATION.md) (**what's built** —
  status / what's next / hard gates), and [EXPERIMENTS.md](plans/EXPERIMENTS.md)
  (**what we're learning** — the hypothesis frontier; check it first for the
  current active line and what's parked/killed).

## Where the data-gen code lives (2026-07-28)
`src/data_gen/` is not flat any more — three modules own one concern each and the
rest of the pipeline consumes them:
- [camera_rig.py](src/data_gen/camera_rig.py) — **all** camera logic: propose,
  gate, path, along-path gate, annotators, per-frame pose, per-frame log.
  `extrinsics()` is world-frame and **gauge** (transform tier, not a target);
  `egomotion()`/`gravity_camera()` are the label tier. Touch camera behaviour
  here, not in `collect_clip`.
- [cli.py](src/data_gen/cli.py) — every `ClipConfig`-override flag, declared once
  and shared by both drivers. Adding a knob is a one-line table entry; adding it
  to a driver's argparse instead re-creates the drift this replaced.
- [camera_paths.py](src/data_gen/camera_paths.py) /
  [camera_motion.py](src/data_gen/camera_motion.py) — pure numpy, no omni,
  61 offline checks. Two camera *models*, one family namespace (`subject:` /
  `free:`); the rig consumes both.
- [clips_generator.py](src/data_gen/clips_generator.py) +
  [collect_grid.py](src/data_gen/collect_grid.py) — corpus as a **plan** over
  (room × scenario × view), not a config. `plan()`/`config_for()`/
  `validate_pure()` import **no omni** and that is load-bearing: the driver
  validates and prints the whole grid *before* `SimulationApp` is constructed
  (`--dry_run` stops there). Keep it that way — one omni import at module level
  in this chain silently costs a Kit boot per typo.

## Which driver
`collect_clip` = one scenario, N rollouts. `collect_corpus` = all scenarios, one
camera policy, flat per-scenario dirs. `collect_grid` = explicit room/egomotion
axes with per-cell yield + cost reporting. All three share `cli.py`, and all
three call the same `collect_clip()` — only the *orchestration* differs.

## Source-of-truth rule
When you change a load-bearing decision (anything under README §Best tactics or a
`plans/*.md` spec), **update the relevant `plans/*.md` in the same commit.** The plans
are canonical; README and this file are pointers. Keep IMPLEMENTATION.md's status
markers current as work lands.

## Isaac Sim workflow (read before touching `src/data_gen/`)
- The data-gen code **cannot run without Isaac Sim** and is not unit-testable here. The
  loop is: **write → compile-check with `ast.parse` → the *user* runs it** under
  `${ISAAC_SIM}/python.sh`. Do **not** run Isaac yourself or modify any Isaac install.
- `SimulationApp({...})` must be constructed **before any `omni`/`isaac` import**
  (Kit segfaults otherwise); see the gated `__main__` block in
  [collect_clip.py](src/data_gen/collect_clip.py).
- Validation procedure for a run: [data-gen/data-gen-testing.md](plans/data-gen/data-gen-testing.md).
- `--n_clips N` = N physics **rollouts**; each fans out to ≤ `n_cameras` independent
  clip dirs — so it is *not* a clip count.

## Verification discipline (do not skip)
- [verify_physics_labels.py](src/data_gen/verify_physics_labels.py) is the **label
  trust-gate** (closed-form checks at the production camera). Physics labels must pass
  it **before any GPU/scale spend** (GATE A). GATE A (dynamics learnable) and GATE B
  (frozen-encoder probe → backbone) gate downstream work — see IMPLEMENTATION.md.

## Data-gen invariants — don't break silently
- All state labels are in **camera frame** (world stays recoverable via `R_camera_world`).
- **No appearance cue for latent physics (R8):** CoM/mass/density must be inferable from
  geometry + dynamics only, never colour/texture. One uniform material per body.
- Clips are seed-deterministic; **non-retrofittable knobs** (density, CoM offset, save
  resolution — they change dynamics/pixels) must be settled *before* any scale batch.

## Conventions
- Match the surrounding code: `src/data_gen/` uses **dense comments that cite the plan
  section** driving each choice — keep that density and the citations.
- Commits go **directly to `main`** in this repo (no feature branch). End commit messages
  with the `Co-Authored-By: Claude Opus 4.8` trailer.
- **Retire, don't delete.** Code whose question is settled moves to
  [src/legacy/](src/legacy/) with a row in its README (why + what superseded it) —
  never `rm`. Invariant: **nothing in `legacy/` may be imported by live code**;
  if it is, it belongs elsewhere. Whole modules move; dead code *inside* a live
  module gets a `DEPRECATED:` marker and goes on the next substantial edit.
  Zero importers ≠ obsolete — check before moving.
- **Don't spawn subagents** unless explicitly asked.
- When moving/renaming docs, rewrite all relative links (README + hubs + cross-refs +
  `plans/...` strings in `.py` docstrings) and re-audit — links carry section anchors.

## Handy commands
```bash
# Compile-check all data-gen modules (no Isaac needed)
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('src/data_gen/*.py')]"

# Run / verify under Isaac — hand these to the user (do not run yourself):
${ISAAC_SIM}/python.sh src/data_gen/verify_physics_labels.py
${ISAAC_SIM}/python.sh src/data_gen/collect_clip.py --result ~/data/out --n_clips 8
```
