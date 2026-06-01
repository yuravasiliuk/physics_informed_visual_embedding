# CLAUDE.md — operating manual for this repo

Agent-facing working notes. **Not** an architecture doc — for *what* the project is
and *why*, read [README.md](README.md) (the map) and the source-of-truth docs in
[plans/](plans/). This file is only the *how to work here* layer.

## Orientation
- Research codebase for a **physics-informed visual encoder** (sim→real bridge). Only
  the Isaac Sim **data-generation** code exists so far ([src/data_gen/](src/data_gen/));
  encoder/training/probes are planned, not written.
- Docs are the source of truth, grouped under `plans/`: [theory/](plans/theory/)
  (modeling/paper), [data-gen/](plans/data-gen/) (collection pipeline),
  [critique/](plans/critique/) (reviews). Two hubs stay at `plans/` root:
  [BRAINSTORM.md](plans/BRAINSTORM.md) (architecture + risks R1–R11) and
  [IMPLEMENTATION.md](plans/IMPLEMENTATION.md) (status / what's next / hard gates).

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
