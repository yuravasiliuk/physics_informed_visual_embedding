# Data-Gen Critique — Round 1 (post-restructure, pre-supervisor)

Brutal review of the data-gen plan stack and the surrounding mental-map docs
*as the student left them on 2026-06-01*. The student asked me to read it before
the supervisor does and call out things a strict reviewer would flag.

The training side is in progress and **explicitly out of scope** for this
critique. So is the codebase itself (the student wants to fix it himself).
Comments below stay at the level of the *plans* and the *documentation* — what
the docs commit to, what they leave dangling, and what a supervisor will
hammer on.

What I read for this round:

- [README.md](../../README.md), [CLAUDE.md](../../CLAUDE.md)
- [plans/BRAINSTORM.md](../BRAINSTORM.md), [plans/IMPLEMENTATION.md](../IMPLEMENTATION.md)
- [plans/data-gen/probe-dataset.md](../data-gen/probe-dataset.md)
- [plans/data-gen/synthetic-data-collection.md](../data-gen/synthetic-data-collection.md)
- [plans/data-gen/scene-diversification.md](../data-gen/scene-diversification.md)
- [plans/data-gen/isaac-asset-diversification.md](../data-gen/isaac-asset-diversification.md)
- [plans/data-gen/physics-diversification.md](../data-gen/physics-diversification.md)
- [plans/data-gen/camera-framing-fix.md](../data-gen/camera-framing-fix.md)
- [plans/data-gen/data-gen-testing.md](../data-gen/data-gen-testing.md)
- [plans/data-gen/w5_composite_glued_body.md](../data-gen/w5_composite_glued_body.md)
- spot-checked `src/data_gen/{config,collect_clip,clip_capture,scenarios,body_factory}.py`
  against the plans' "DONE" claims (to verify the docs match reality — not to
  critique the code itself)

The headline up front: **the documentation is in much better shape than three
weeks ago, the verify discipline is genuinely good, and the W1–W5 physics
fidelity pass closes the most damning gaps from critique 0.** A supervisor will
nevertheless find ~10 substantive cracks, of which 3–4 are blocking and the
rest are "address before the thesis chapter."

---

## Critical — a supervisor will lead with these

### C1. The whole stack still hasn't been run end-to-end under Isaac

[IMPLEMENTATION.md P0.4](../IMPLEMENTATION.md) is the only unchecked box on the
P0 hardening list, and it's the one that matters: **no clip has been
successfully written through the production path against the current code.**
Every "✅ DONE" item above it is verified either off-Isaac (compile checks) or
through `verify_physics_labels.py` (driving `simulate_clip` directly, not
`collect_clip`).

This is the single biggest unaddressed risk. Five claims now sit untested in
sequence:

- the camera-framing-fix preroll → rig-gate → fan-out pipeline (camera-framing-fix.md §5)
- the W5 composite-body path on the production driver (the smoke run in
  physics-diversification.md §W5 was a partial check, not a full corpus run
  with all scenarios + envs + camera rig)
- the `topple` energy-barrier fix (W3a is implemented but [IMPLEMENTATION.md L142](../IMPLEMENTATION.md)
  still has it `⬜ TO IMPLEMENT`)
- the multi-view fan-out (scene-diversification.md §10) emitting independent
  clip dirs with consistent `multiview` provenance
- the env-backdrop loader (isaac-asset-diversification.md Tier 2) actually
  rendering with `office.usd` excluded

If even one of these chains fails on first run, several "DONE" items have to
re-open. The supervisor will note that the doc stack reads as if the pipeline
works; until P0.4 passes, nothing in P0 is actually validated together.

**Fix for the supervisor read:** demote a few ✅s to 🟡 ("verified in isolation,
not on the integrated path") until P0.4 closes, and put a one-line caveat at
the top of [IMPLEMENTATION.md](../IMPLEMENTATION.md) above "Where we are right
now" stating this explicitly. The honesty will read better than the appearance
of completeness.

---

### C2. Phase-0.5 GATE A is mis-described (and that protects nothing)

[README §Baseline Step 3 / Phase 0.5](../../README.md) says GATE A trains a
"small clip-level encoder + per-object dynamics heads" on
**Kubric MOVi-E / CLEVRER**, with pass thresholds ω R² > 0.5, v R² > 0.5,
CoM L2 < ½ median bbox diagonal. [IMPLEMENTATION.md §Hard gates](../IMPLEMENTATION.md)
repeats this. The synthetic-data-collection.md §Implementation order §3 also
repeats it.

Three problems a supervisor will spot in 30 seconds:

1. **"Perfect RGB"** is overclaimed. MOVi-E is rendered RGB with all the same
   compression / shading / texture issues as anything else. The claim "is
   dynamics learnable from *perfect data*" should be "is dynamics learnable
   from a dataset with *clean labels*" — those are different things. The gate's
   intent (rule out the per-object-pooling formulation being broken) is sound;
   the framing isn't.

2. **CoM L2 < ½ median bbox diagonal** is so loose that "predict the geometric
   centroid" passes by construction (R7 in [BRAINSTORM.md](../BRAINSTORM.md)
   says exactly this, but the gate doesn't apply the fix). The threshold should
   be "lift over the centroid baseline > X." Otherwise the gate measures
   "encoder finds the object," not "encoder encodes mass distribution."

3. **The dataset is wrong for the question.** MOVi-E does not have authored
   offset CoM — every body has CoM at the centroid by construction (Kubric
   primitives are uniform-density convex shapes). So a CoM-offset gate on
   MOVi-E asks a question the dataset cannot answer; the gate would
   "pass" trivially (offsets are zero everywhere → predict-zero scores
   perfectly). ω/v are fine on MOVi-E; CoM-offset is not.

A supervisor will read this and conclude either the gate is not well-thought-
through or the student is conflating "dynamics" with "all four physics heads."
Both readings are bad.

**Fix:** in the gate description, scope CoM-offset out of MOVi-E (it cannot
be tested there) and either (a) use ω + v as the GATE A signal, or (b) run the
CoM-offset half of GATE A on the project's own corpus (W5 composites give the
needed signal). State this trade-off explicitly.

---

### C3. The "no domain cue" claim (R8) is asserted but never measured

[BRAINSTORM.md R8](../BRAINSTORM.md) commits to "internal mass distribution
the only difference" — meaning a probe cannot cheat CoM/mass from appearance.
[physics-diversification.md §W5](../data-gen/physics-diversification.md)
extends this to composites: the density asymmetry is **invisible** because
slabs share one material.

The W5 doc does report one R8 check: composite vs primitive-cuboid saturation
0.40 vs 0.38, with a small ~0.04 gap under brightness control. That's a single
summary statistic on one corpus, on saturation alone. A supervisor will ask:

- **Is mass linearly recoverable from RGB?** If you train a linear regressor
  on `mean(rgb)` and `std(rgb)` per body, does it explain mass variance? If R²
  > say 0.1, the R8 claim is shaky — the encoder doesn't need to find the
  density slab; just the appearance correlate. This needs to be a regression
  baseline before any encoder probe, not after.
- **Is the CoM-offset *direction* recoverable from silhouette asymmetry?**
  Authored-offset primitives keep the geometric centroid at the prim origin,
  but the segmentation mask + ground-plane shadow can still cue the offset
  through (e.g.) which face is in contact in `topple`. A
  silhouette-only regression baseline should be reported.
- **Is the composite-vs-primitive distinction itself a free label?** If the
  composite share is ~33% and an encoder can tell them apart from pixels
  alone, the "two halves of the corpus" idea collapses into "two visually
  distinct populations."

These are 1 GPU-hour each and the supervisor *will* ask for them. None of the
plans currently commit to running them.

**Fix:** add a short subsection in
[physics-diversification.md](../data-gen/physics-diversification.md) called
"R8 cheat-detector baselines," listing the three regressions above as
**required before** GATE B is interpreted, with explicit pass thresholds. This
is exactly the kind of pre-registration a supervisor will reward.

---

### C4. `composite_frac=0.3` is unjustified and creates a hidden category label

The composite path is invisible at single-body level (R8 enforced), but at
**clip level** there's a deterministic generator side-channel: how many
composites appeared in this clip. The encoder can in principle learn

> P(this clip is a "composite-heavy" clip) → P(corpus subset) → mass-distribution prior

…without ever recovering CoM from dynamics. This is a subtler R8 leak: not at
the body level but at the *batch / clip* level.

The fix is principled but the doc doesn't address it: composites should be
sampled **per body independently** (which they are) AND the corpus should be
**rebalanced** so the composite count distribution is approximately the same
across categories (across density bins, scenario types, etc.). Neither is
documented.

A 0.3 fraction is also picked by feel. The doc never says why it isn't 0.5,
0.2, or "all bodies are composites with `n_parts ∈ {1, 2, 3}`" (where
`n_parts=1` is the W2 primitive and you sweep n_parts uniformly). The
last option is cleaner — composites and W2-primitives are then the same code
path at different `n_parts`, and the corpus has no "is this body a composite"
binary label at all.

**Fix:** in [w5_composite_glued_body.md](../data-gen/w5_composite_glued_body.md)
or [physics-diversification.md](../data-gen/physics-diversification.md),
either (a) defend `composite_frac=0.3` with a sensitivity analysis, or (b)
collapse the binary to a continuous `n_parts ∈ {1, 2, 3}` knob and retire the
flag. (b) is the cleaner long-term answer.

---

## High — blocking before the thesis chapter, not before the supervisor read

### H5. The mental-map docs disagree on the architecture

This one matters a lot for the supervisor read because it shows up
*immediately*:

- [README §Best tactics → Training data](../../README.md) commits to the B/F
  config split with both running the domain regularizer and a frozen-first
  backbone. It explicitly flags the two-stream as the escape hatch.
- [BRAINSTORM.md §0](../BRAINSTORM.md) says (paraphrased): "⚠ open tension —
  the README's Best tactics commit to the two-stream as decided. §2 of this
  doc **supersedes that for the main line.**"

So the canonical hub document **declares itself the source of truth** while
also flagging that the source-of-truth doc (the README) hasn't been updated.
A supervisor will see this and either: (a) lose ~20 minutes reading the wrong
half, or (b) ask whether the project has settled on its architecture or not.

The data-gen plans inherit the same problem in miniature:
[probe-dataset.md](../data-gen/probe-dataset.md) §13 ("Settled decisions")
records v0 contracts that
[scene-diversification.md](../data-gen/scene-diversification.md) §6 *explicitly
asks to overwrite* ("regenerate v0, or call this v0.1?"), recommends Option A
(promote into v0), but the README and probe-dataset.md still describe v0 as
"primitives only, fixed camera, fixed lighting."

**Fix:** reconcile the README's §Best tactics with BRAINSTORM §2 in the same
commit, and either bump probe-dataset.md's v0 to v0.1 in place (with the
diversification baked in) or split it cleanly into "v0 (historical)" and "v0.1
(current target)". The current state — README says one thing, the active hub
supersedes part of it — is the kind of organizational debt supervisors flag in
the first 10 minutes.

---

### H6. The data-gen-testing.md "four layers" miss one layer the docs assume exists

[data-gen-testing.md](../data-gen/data-gen-testing.md) has four layers (static,
closed-form labels, smoke gen, visibility/quality), all good.

What's missing: **dataset-level statistics** (the "sanity dashboard" in
[synthetic-data-collection.md §Implementation order](../data-gen/synthetic-data-collection.md)
Step 2 and [IMPLEMENTATION.md P1](../IMPLEMENTATION.md) "Sanity dashboard"). A
clip can pass all four current layers and still produce a corpus that's
*globally* unusable:

- Mass distribution heavily bimodal because `density_range` × volume yields a
  multimodal mass distribution at small N.
- Per-body `quat_camera_wxyz` distribution concentrated on the
  upright-aligned axis-aligned quaternion (`scenarios` mostly spawn bodies
  axis-aligned; only `topple` deliberately tilts them). The encoder can then
  learn "predict identity quaternion" and pass any orientation probe.
- Velocity distributions wildly heavy-tailed in `throw`/`push` because the
  upper end of THROW_SPEED_RANGE = 12 m/s vs. lower end 0.5 m/s spans 1.5
  decades.
- Per-scenario clip count skewed by visibility-gate drops: `cluttered` clips
  fail the gate more often (occlusion by static props) so the corpus
  under-represents the only scenario with realistic occluders.

None of these are catchable in Layer 1–4. The supervisor will read the testing
doc and ask "where is the corpus statistic check?" and the honest answer is
"in IMPLEMENTATION.md P1, but unwritten."

**Fix:** add a "Layer 5 — corpus statistics" to data-gen-testing.md that
references the sanity-dashboard plan and lists the per-channel checks (mass
histogram, velocity distribution, per-scenario clip yield, quaternion sphere
coverage, body-coverage CDF). Even a one-paragraph stub raises the supervisor's
confidence dramatically.

---

### H7. The "rollouts vs clips" confusion is now load-bearing and under-documented

After the multi-view fan-out, `--n_clips N` does NOT produce N clips. It
produces N *rollouts*, each of which produces 0 to `n_cameras` clips after
visibility gating. [CLAUDE.md "Isaac Sim workflow"](../../CLAUDE.md) does call
this out, and [data-gen-testing.md](../data-gen/data-gen-testing.md) notes it.
But several other places don't:

- [README §Baseline / Step 1](../../README.md) talks about "10k-50k clips total"
  with no mention of which.
- [synthetic-data-collection.md](../data-gen/synthetic-data-collection.md)
  was written before multi-view and uses "clips" throughout to mean what is
  now "rollouts."
- [probe-dataset.md §6 Per-clip size budget](../data-gen/probe-dataset.md)
  computes corpus size by clip — multi-view fan-out now multiplies this by up
  to `n_cameras`. A 50k-rollout corpus at 7 cameras = 350k clips ≈ 1 TB at
  ~2.9 MB/clip. The disk budget needs revising.

A supervisor reading the doc cold will see different numbers in different
places. They'll either be confused or assume the student is.

**Fix:** pick a vocabulary (I'd suggest "rollouts" and "views," with "clip" =
one view of one rollout), grep all the data-gen plans + README for the term,
and rewrite each occurrence. Update the disk budgets in
[probe-dataset.md §6](../data-gen/probe-dataset.md) for the realistic
post-fan-out size.

---

### H8. The render resolution decision is overdue and now blocks both threads

`save_resolution=(1024, 1024)` (config default) is **the** biggest disk lever
left, and [IMPLEMENTATION.md L114-117](../IMPLEMENTATION.md) flags it as 🟡
"engage before the P1 batch." It also blocks the V-JEPA-2 checkpoint choice
([IMPLEMENTATION.md L172-174](../IMPLEMENTATION.md), 256² vs 384²) — which in
turn blocks GATE B.

This has been in flight long enough that a supervisor will ask why a single
dataclass field is gating two upstream decisions. The honest answer is "no one
benchmarked V-JEPA 2 throughput at 256 vs 384 yet" — that's a 1-hour task
that's been deferred. Either commit to a value now with a one-paragraph
rationale (V-JEPA 2 ViT-L is published at 384²; 256² is a downsampling that
loses the small-body pixel signal which the W5 R8 check just barely cleared)
or run the benchmark.

**Fix:** make the V-JEPA 2 resolution decision in
[probe/gate-b-probe.md](../probe/gate-b-probe.md) (which I haven't reviewed —
this critique assumes it exists per the README link), and update
`save_resolution` + state it as a frozen decision in
[probe-dataset.md §13 Settled decisions](../data-gen/probe-dataset.md).

---

### H9. The pre-roll determinism claim is structural but unproven

[camera-framing-fix.md §7](../data-gen/camera-framing-fix.md) Risks:
*"Pre-roll determinism: relies on `world.reset()` + re-applied velocities
reproducing the rendered run exactly. … verify the pre-roll and real
trajectories match to confirm no render-vs-no-render divergence."*

This is currently a hand-wave with a TODO. The whole multi-view fan-out is
predicated on the no-render preroll producing the *same* dynamics as the
rendered run that follows. If `world.step(render=True)` vs.
`world.step(render=False)` diverge (different substep counts under load,
rendering forcing a flush of any deferred physics), the rig is framed for
trajectory A but captures trajectory B — the bodies leave the framed cloud.

This is the kind of risk a strict supervisor will ask about because it's
exactly where Isaac/PhysX integration has burned other groups before. The
plan handles it correctly (call it out, plan a verification), but the
verification hasn't run.

**Fix:** add a Layer 1.5 verify case
("trajectory_determinism") that runs `preroll_trajectory` and `simulate_clip`
on the same seed/scene/velocities, compares per-frame body poses, asserts
agreement to ε = 1e-4 m (or wider — pick a number with a margin for
floating-point reorderings). Document this in
[data-gen-testing.md](../data-gen/data-gen-testing.md). Run it before P0.4.

---

### H10. `verify_physics_labels.py` doesn't cover the rig-gate path

The verify script (per [physics-diversification.md §W4](../data-gen/physics-diversification.md))
runs four cases at the **production camera** = the *single legacy camera*
`(2.2, 2.2, 1.8) → (0, 0, 0.1)`. After camera-framing-fix.md the production
path is no longer "single fixed camera"; it's the body-cloud-anchored rig with
visibility gating.

So:

- `world_to_camera_frame` may now be invoked at near-vertical elevations
  (camera_elevation_clamp_deg up to 68°), where the math is closer to the
  degenerate-camera assert. The verify cases never exercise that elevation.
- The per-camera `validate_clip` change (`min_body_coverage`) is not
  cross-checked against any label-correctness invariant.
- Multi-view fan-out: there's no verify that the same physics state, captured
  from two cameras simultaneously, produces *consistent* per-body world-CoM
  reconstructions (i.e., that the camera-frame projections invert to the same
  world point). This is a free closed-form check and an obvious supervisor ask.

**Fix:** add to `verify_physics_labels.py` (or sibling) two cases:

- **`rig_high_elevation`** — re-run `offset_com` at elevation 65° (just below
  the clamp). Validates the look-at and projection math at the rig's worst
  case.
- **`multiview_consistency`** — same scene rendered from 2+ cameras; the
  recovered world-CoM from each camera's per-frame label must agree to
  sub-mm across all frames.

Both are direct extensions of existing verify code and would be cheap.

---

## Medium — substantive but not blocking

### M11. The "free fall" / "everything is gravity" critique from scene-diversification.md §1 isn't reflected in the corpus *budget*

[scene-diversification.md §1](../data-gen/scene-diversification.md) correctly
diagnoses that early clips were free-fall-dominated. The fix adds 8 scenarios
(was 1) — but `config.scenarios` lists them with **uniform weighting**, so
4/9 ≈ 44% of clips are still in {drop, throw, drop_on_platform, push} where
gravity dominates the linear-velocity signal. After visibility-gate drops
(probably correlated with scenario type), the *effective* distribution is even
less clear.

This isn't fatal — the encoder sees enough diversity — but the supervisor will
ask "what fraction of your training signal is free-fall?" and the doc doesn't
have a number. The OBSERVABILITY_HINT mapping is the right primitive
(`free_flight` / `contact` / `collision`), but it's not used for re-weighting
in `config.scenarios`.

**Fix:** in [physics-diversification.md](../data-gen/physics-diversification.md)
or [scene-diversification.md](../data-gen/scene-diversification.md), add an
explicit scenario-weight default that targets a known split — e.g. 30%
free-flight / 60% contact / 10% collision — and report the realized split
after the P0.4 smoke run.

---

### M12. `density_range = (200, 8000)` is a 40× range — likely too wide

The default decouples mass from volume, which is correct for the W1 motivation.
But 200 kg/m³ → 8000 kg/m³ is styrofoam → steel, and in the typical small-body
volume range (0.05–0.15 m per axis), this gives mass ratios up to 40× between
two bodies of similar visible size.

Two consequences:

1. **PhysX numerical stability** with high-mass-ratio contacts is poor. Two
   bodies of mass ratio 40 colliding can produce visible interpenetration or
   energy injection at the default solver iterations. The `collide` scenario
   is the worst case.
2. **The mass-prediction probe** sees mass in log-space spanning ~1.5 orders
   of magnitude. The natural loss target is `log(mass)` not `mass`; the docs
   don't say.

A supervisor familiar with PhysX will ask about (1); the second-best ask is
about (2).

**Fix:** narrow the default to ~5× range (e.g. 600–3000 kg/m³, plastic→stone)
unless there's a reason for the wider span, and document the loss target for
the mass head as `log(mass)` in [training-data.md](../theory/training-data.md)
(not in this critique's scope but worth a one-line note in the data-gen plan).

---

### M13. The "scenario tag in scene_specs" should be a top-level meta field

[scenarios.py](../../src/data_gen/scenarios.py) injects `observability_hint`
and `scene_scale` under `scenario_specs`, which itself nests under
`scene_specs`. So in meta.json it lands at
`scene.scenario_specs.observability_hint`.

For a dataloader that wants to stratify by hint, this is now `meta["scene"]
["scenario_specs"]["observability_hint"]` — three levels of nesting for a
scalar. It'll get pulled up by hand into the dataloader anyway. Better to
mirror `scenario` (a top-level field) with `observability_hint` also at the
top level.

Cosmetic, but supervisors notice schema design.

---

### M14. The W3a "topple" lean-past-θ_crit fix is described but not in the run

[IMPLEMENTATION.md L141-143](../IMPLEMENTATION.md) still marks W3a as `⬜ TO
IMPLEMENT`, but [scenarios.py:622-683](../../src/data_gen/scenarios.py) is the
full lean-past-θ_crit implementation, and
[physics-diversification.md W3a](../data-gen/physics-diversification.md) is
phrased as "TO IMPLEMENT" too. The doc disagrees with the code.

A supervisor will check this and conclude either the implementation tracker is
stale (lowering trust in the rest of it) or W3a was implemented in haste and
not validated. Either reading is bad.

**Fix:** verify W3a end-to-end (a 6-clip smoke run with `--scenario topple`),
confirm toppling is now visible, and update IMPLEMENTATION.md +
physics-diversification.md to ✅. This is a 10-minute fix and removes a real
read-time confusion.

---

### M15. The Replicator-RNG-not-seeded issue from critique 0 is still unresolved

Critique 0 noted that `np.random.default_rng(seed)` controls scene-level
randomization but Replicator's ray-tracing / denoiser RNG is not seeded
through any public API, so clips are not bit-exact reproducible across runs.

None of the current plans address this. A supervisor who cares about
reproducibility (most do) will ask "is your data reproducible from
`{seed, ClipConfig}`?" The honest answer is "scene state yes, pixel-exact RGB
no." Worth documenting.

**Fix:** add a one-liner to
[data-gen-testing.md](../data-gen/data-gen-testing.md) "Known limitations":
"clips are reproducible at the scene-state / physics-label level; RGB is
ray-traced and Replicator's RNG is not externally seeded, so RGB differs at
the noise floor between runs of the same seed."

---

### M16. The data-gen plan doesn't commit to a corpus *manifest*

Going from per-clip `meta.json` to a corpus-level manifest (Parquet or
similar) is mentioned in
[synthetic-data-collection.md](../data-gen/synthetic-data-collection.md) §
Output format and storage as `manifest table (Parquet)` with concrete columns,
and in [probe-dataset.md §11](../data-gen/probe-dataset.md) as Step 6
(WebDataset shard writer). After the restructure, no plan has an owner for
this work; [IMPLEMENTATION.md P1](../IMPLEMENTATION.md) lists "WebDataset
packaging utilities" as a good-first-issue without further detail.

This is a real cliff at scale: with 10k+ clips spread across N rollouts × M
cameras across multiple scenarios + environments, **filtering** the corpus
(by scenario, by env, by `validation.ok`, by `composite_frac`, by
`observability_hint`) requires opening every meta.json. The dataloader will
slow to a crawl.

**Fix:** commit to a manifest schema (one row per clip, columns: clip_id,
rollout_seed, camera_index, scenario, env_name, observability_hint, n_bodies,
composite_count, validation_ok, validation_reason, …) in
[probe-dataset.md](../data-gen/probe-dataset.md). One-time cost; pays back
every dataloader query.

---

## Notes — minor, easy fixes

### N17. `density_range` and `composite_frac` aren't in [probe-dataset.md §7 "Generation parameters"](../data-gen/probe-dataset.md)

The "Generation parameters — v0" table is the canonical reference for what
distinguishes a v0 corpus, and it's frozen-in-time at the W1/W2/W5 pre-state.
A supervisor reading top-down will see "object color randomized, friction
randomized, restitution randomized" and miss that density/CoM-offset/composite
are *the* new contributions.

### N18. `--n_clips` CLI help in [collect_clip.py](../../src/data_gen/collect_clip.py) says "generate N clips" but
the post-fan-out semantics is N rollouts. Same vocabulary issue as H7,
surfaces in the CLI.

### N19. The composite GT — `mass = Σ ρ_k V_k` — assumes slabs don't overlap. [body_factory._partition_cuboid](../../src/data_gen/body_factory.py) does enforce non-overlap by tiling exactly. The doc says this; worth a one-line cross-reference in the verify case description so a reader can see the proof of correctness in one read.

### N20. [data-gen-testing.md §1](../data-gen/data-gen-testing.md) prescribes "Run these whenever you touch label or capture code" but doesn't quantify "whenever" — no CI hook, no pre-commit. For a thesis project this is fine; for a supervisor who's seen too many "we forgot to re-run verify" disasters, even a `make verify` target would buy goodwill.

### N21. The README links `summaries/scope.md` etc. but the summary index hasn't grown to cover the SCOPE→DPT→intermediate-features lineage that BRAINSTORM.md §2.2 now leans on. A supervisor following BRAINSTORM's reasoning will go looking for the DPT paper summary and not find one.

---

## Summary table

| #   | Severity | Issue                                                              | One-line fix                                                                              |
|-----|----------|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| C1  | Critical | Production pipeline untested end-to-end (P0.4)                     | Demote a few ✅→🟡; caveat at IMPLEMENTATION.md top                                       |
| C2  | Critical | GATE A description is wrong about MOVi-E and the CoM threshold     | Scope CoM out of MOVi-E; report lift-over-centroid                                        |
| C3  | Critical | R8 ("no appearance cue") asserted but not measured                 | Add three regression baselines: RGB-mean→mass, silhouette→CoM, pixels→composite-flag      |
| C4  | Critical | `composite_frac=0.3` unjustified; creates a clip-level cheat label | Defend the value or collapse to continuous `n_parts ∈ {1, 2, 3}`                          |
| H5  | High     | README and BRAINSTORM disagree on architecture (declared "open")   | Reconcile §Best tactics with BRAINSTORM §2 in one commit                                  |
| H6  | High     | data-gen-testing.md has no corpus-statistics layer                 | Add "Layer 5 — corpus statistics" stub                                                    |
| H7  | High     | "Rollouts vs clips" vocabulary is inconsistent across docs         | Pick terms, grep-rewrite, update disk budgets                                             |
| H8  | High     | `save_resolution` decision overdue; blocks GATE B                  | Make the V-JEPA 2 256² vs 384² call now, freeze save_resolution                           |
| H9  | High     | Pre-roll determinism unproven                                      | Add `trajectory_determinism` verify case before P0.4                                      |
| H10 | High     | Verify doesn't cover the rig-gate / multi-view path                | Add `rig_high_elevation` + `multiview_consistency` cases                                  |
| M11 | Medium   | Scenario weights uniform → 44% still free-fall-dominated           | Reweight to target a known free-flight/contact/collision split                            |
| M12 | Medium   | `density_range` 40× → PhysX stability + log-mass loss target       | Narrow to ~5×; document log-mass loss target                                              |
| M13 | Medium   | `observability_hint` buried under `scene_specs.scenario_specs`     | Lift to top-level meta.json field                                                         |
| M14 | Medium   | IMPLEMENTATION.md and physics-diversification.md disagree on W3a   | 10-minute smoke run + flip ✅                                                             |
| M15 | Medium   | Replicator RNG not seeded — RGB not bit-reproducible               | One-line in "Known limitations"                                                           |
| M16 | Medium   | No corpus manifest committed                                       | Commit a Parquet schema in probe-dataset.md                                               |
| N17 | Note     | probe-dataset.md §7 missing density/CoM-offset/composite           | Add table rows                                                                            |
| N18 | Note     | `--n_clips` CLI help misleading (now = rollouts)                   | Update help string                                                                        |
| N19 | Note     | Composite GT non-overlap proof not cross-referenced                | One-line in verify case description                                                       |
| N20 | Note     | No `make verify` target                                            | Add a Makefile / shell alias                                                              |
| N21 | Note     | DPT / intermediate-features summaries missing                      | Add to summaries/                                                                         |

---

## What to fix BEFORE the supervisor read (the 90-minute pass)

If the supervisor read is imminent and there's time for one focused fix pass,
prioritise:

1. **H5 (1 hr)** — reconcile README §Best tactics with BRAINSTORM §2. The
   supervisor will hit this in minute 5; fixing it is pure documentation.
2. **C1 (15 min)** — add the caveat at IMPLEMENTATION.md top + demote two or
   three ✅s to 🟡. Lower-trust signals than visible inconsistency.
3. **C2 (15 min)** — fix the GATE A description (CoM scoping + threshold).
4. **H7 (30 min)** — vocabulary cleanup across the data-gen plans.
5. **M14 (10 min)** — fix the W3a status divergence.

The remaining items (C3, C4, H6, H8, H9, H10, M11–M16, N17–N21) survive a
first-pass review fine if they're acknowledged on a "known follow-ups" list
visible in IMPLEMENTATION.md.

---

## What this critique does NOT cover (by request)

- Codebase issues (the student is fixing himself).
- Training-side plans (P2+ in IMPLEMENTATION.md, `probe/gate-b-probe.md`,
  `theory/training-data.md`).
- The theory paper (`theory/theoretical-paper-plan.md`).
- The domain-marginalization regularizer (`theory/domain-marginalization-regularizer.md`).

Those are for a separate round.
