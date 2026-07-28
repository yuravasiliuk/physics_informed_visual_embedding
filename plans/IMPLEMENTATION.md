# Implementation Status & Plan

*Operational tracker for the engineering build. Living document — update the
status markers as work lands. This is the **"where are we / what's next"** (what's
**built**) view; the **why** lives in the design docs linked from each item, the
milestone narrative is in [README §Baseline to implement](../README.md#baseline-to-implement-this-milestone),
and the **hypothesis frontier** (what we're **learning** — active/parked/killed
experiments) is the sibling hub [EXPERIMENTS.md](EXPERIMENTS.md).*

**Last updated: 2026-07-05.**

## Status legend

| Marker | Meaning |
|---|---|
| ✅ | done & verified |
| 🟡 | partial / in flight (note says what's missing) |
| ⬜ | not started |
| ⛔ | gated — blocked until a named gate passes |

## Milestone goal

A clean, runnable **B-config training loop end-to-end on a ~100-clip Phase-1a
corpus** (frozen encoder + dense depth/normals/seg heads + domain regularizer).
F config, sim-only physics heads, translated pairs, and ablations layer on top.

## Where we are right now

**GATE B iteration-1 read is LOCKED (2026-07-05)** on the 755-group v2 corpus
(E1-accepted; R8 holds at r=−0.01, 40× density spread):

- **Kinematics: YES.** v control **passes** with window-aligned targets
  (clip-mean v R² 0.60 linear / 0.70 MLP @ layers 28–32); ω 0.38 / 0.47 @ L30.
  The v1 "+0.05 plateau" is fully explained: group starvation (17→755 groups)
  + target↔pooling-window misalignment (+0.29). **Frozen-first survives; the
  escape hatch is NOT justified; the Rung-1 tap depth is layers ~28–32.**
- **CoM: the encoder is EXONERATED, not passed.** The null was uninformative —
  the ★oracle (R9 reference, `results/probe/com_oracle*`) proved CoM was not
  decodable even from GT trajectories on that corpus (R6 violated in practice;
  register collapse ruled out — body↔frame cosine *falls* with depth).
- **R6 is now a measurable gate**: `com_oracle` PILOT mode = no-GPU scenario
  acceptance test. First **R6-certified scenario: `eccentric_roll`** (pooled
  pilots, n=96: ridge +0.0199±0.0058, MLP +0.0161±0.0087 — passes the
  pre-registered bar in both hypothesis classes). `balance` mechanism validated
  at the boundary (offset beats geometry) but underpowered at pilot n.

**b0 fast path DONE (2026-07-06)** — first *interpretable* encoder CoM read
(46 eccroll rollouts, `results/probe/ecc46/`): control replicates (v +0.56 @
L28) → **linear CoM null** at layers 28–32 vs oracle ceiling +0.02, plus an
ungated MLP hint (~+0.010 everywhere, control failed at n=96). Directional by
design — cannot separate "encoder misses CoM" from "n too small".

Next action (now decisive, not optional): W6 mid-batch (balance 200 @
`--com_offset_frac 0.45`, eccroll 150 @ 128 frames) → same oracle bar → fold
certified scenarios into the probe corpus → **re-read the GATE-B CoM row
powered.** Live tracker: [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Hard gates (do not skip)

- **GATE A — feasibility (P0.5a).** No Isaac Sim *scale* compute is committed until a
  small model learns dynamics from *perfect* RGB (Kubric/CLEVRER): ω R² > 0.5,
  v R² > 0.5, CoM L2 < ½ median bbox diagonal. If perfect data can't be learned, the
  full pipeline won't either. ~1 GPU-day.
- **GATE B — architecture (P0.5b, Rung 0).** The layer-wise frozen probe decides the
  backbone: signal present in the frozen encoder → frozen-first main line (and its
  peak depth sets which blocks the head taps); signal absent at *every* depth → the
  raw-pixel escape hatch (Rung 2) is justified — **but only after the
  register-collapse diagnostic (P0.5(b) below) rules out a fixable
  V-JEPA-2-specific artifact, not genuine absence of signal**. See
  [BRAINSTORM.md §2](BRAINSTORM.md#2-architecture) and
  [probe/vjepa-embedding-structure.md §5](probe/vjepa-embedding-structure.md).

---

## P0 — Single-clip data-gen pipeline (Isaac Sim)

*Spec: [synthetic-data-collection.md](data-gen/synthetic-data-collection.md),
[probe-dataset.md](data-gen/probe-dataset.md). Code: [../src/data_gen/](../src/data_gen/).*

- ✅ `collect_clip.py` refactored into reusable `collect_clip(out_dir, ClipConfig, seed)`
  with `build_scene / simulate_clip / validate_clip` stages.
- ✅ Supporting modules in place: `asset_library`, `asset_loader`, `body_factory`,
  `scenarios`, `lighting`, `environments`, `static_geometry`, `clip_capture`, `seg_labels`.
- ✅ `verify_collect_clip.py` written (CoM=0 sphere smoke test).
- ✅ **Verify coverage hardened (R11 / #4 / #11).** `verify_physics_labels.py`
  (W4) adds spin-cuboid (ω sign/scale + quat composition), authored-offset-CoM,
  and two-body index-swap cases, all at the production camera; 3/3 pass. Caught
  and fixed two real label bugs (unscaled `centerOfMass`, default angular
  damping). See [physics-diversification.md](data-gen/physics-diversification.md) W4.
- ✅ **P0.4 — Run `collect_clip.py` end-to-end** — field-proven at scale
  (2026-07: 763-clip 9-scenario v2 probe corpus + W6 pilots, all through the
  E1 acceptance audit green; see the corpus runbook). Contract reference:
  `save_clip` writes
  **quantized, compressed `.npz`** — not raw `.npy`:
  `rgb.mp4` (lossy mp4v), `depth.npz` (uint16, per-clip range carried in
  `depth_min`/`depth_max` keys — dataloader must dequantize), `normals.npz`
  (uint8, fixed [-1,1] map), `seg.npz` (uint8, or uint16 if >255 classes),
  `state.json`, `meta.json`, and optional `seg_id_to_class.json` (id→class map).
  This lands the **precision/format** disk wins from
  [data_gen_critique0_analysis.md](critique/data_gen_critique0_analysis.md) (#6 seg downcast,
  normals→uint8, depth→uint16, `.npz` compression). The **§14 resolution-decoupling**
  knob (`save_resolution`) exists but defaults to full render res (1024²) — *not yet
  engaged*, so the biggest disk lever is still on the table (see hardening below).
  *First action for a new contributor.*

**P0 hardening — critique 0** (*tracked in
[data_gen_critique0_analysis.md](critique/data_gen_critique0_analysis.md). Density and
resolution change dynamics/pixels and **cannot be retrofitted** — decide these
before any scale generation; the rest are rewrite-and-rerun.*):

- ✅ **#2 initial velocities persisted** — `body_specs` carries
  `init_lin_vel_world/_camera` + `init_ang_vel_world/_camera`; clips are now
  reconstructable and "predict launch condition" targets are possible.
- ✅ **#4 / #11 extend verify** per [BRAINSTORM §6](BRAINSTORM.md#6-what-is-and-isnt-verifiable-critique-4-scope):
  done as `verify_physics_labels.py` (W4) — uniform cuboid spin (ω closed form +
  quat composition), authored-offset CoM body, production camera pose, two-body
  index-swap; 3/3 pass. *Was the highest value-per-hour item; the GPU-spend gate
  (GATE A) is now satisfied on labels.* (Out of scope, as planned: general
  tumbling and PhysX CoM on real meshes — no clean ground truth.)
- ✅ **#7 density randomization (W1).** `ClipConfig.density_range` sampled per
  body and applied via `MassAPI.CreateDensityAttr` in `body_factory.spawn_body`;
  mass is now decoupled from visible volume. `density` + analytic `mass` +
  post-reset `mass_measured` logged in `body_specs`. Also fixes the
  non-retrofittable "mass ⟂ size" (Direction A) prerequisite.
- ✅ **W2 authored offset CoM / W3 observability scenarios** (physics-diversification.md).
  Per-body scale-corrected offset CoM authored in `spawn_body` (CoM-offset label
  now has real variance; verified to ~2 nm); new `topple` + `collide` scenarios
  make CoM/mass-ratio observable.
- ✅ **W5 composite glued bodies** (physics-diversification.md W5). Completes the
  W1–W5 physics-fidelity sub-plan. `body_factory.spawn_composite_body` builds a
  uniform-colour cuboid internally split into 2–3 density slabs
  (`config.composite_frac=0.3`); PhysX integrates the emergent mass/CoM, labels
  are closed-form `ΣρV` / `ΣρV·t/ΣρV`, R8-clean. Wired into `build_random_bodies`
  + `topple`/`collide` via `spawn_dynamic_body`. **`composite` verify case PASSED**
  (mass err 8e-9, CoM err 5 nm). Smoke run confirmed labels on the production path
  and caught a material bug (parent-Xform OmniPBR binding didn't reach the child
  Cubes → grey composites), **fixed** by binding materials per child; re-run
  confirms coloured composites indistinguishable from primitive cuboids
  (saturation 0.40 vs 0.38). **Mesh-asset bodies (W5b) deferred** — no closed-form
  GT for a mesh's CoM.
- 🟡 **§14 engage resolution decoupling** — the `save_resolution` knob exists but
  defaults to 1024²; dropping labels to ~256² is the ÷16 disk lever (and resolves
  the #1 lossy-RGB debate). Set before the P1 batch. Caveat: target res depends on
  the chosen V-JEPA 2 checkpoint (256² vs 384²).
- ⬜ **#3 retry + don't persist failed clips** — `main()` writes `validation.ok=False`
  dirs and steps `seed + i`; wrap in a bounded re-seed loop (large offset → independent
  draws) and route failures to `failed/`. *Required before the P1 run.*
- ⬜ **#5 seg_info merge across frames** — currently taken from `frames[0]` only; a
  body occluded at frame 0 can drop from the id→class map. 3-line last-write-wins merge.
- ⬜ **#12 degenerate-camera assert** — `world_to_camera_frame` NaNs when
  `forward ∥ world_up`; add a cheap `|cross| > eps` guard before a top-down scenario
  trips it silently.
- ✅ **#8 / #9 mass → `body_specs`** — `body_specs` now carries `mass` (analytic),
  `mass_measured` (post-reset `get_mass`), and `density` (W1), so mass no longer
  lives only per-frame in `state.json`. *Remaining by-design / documented, low
  priority: #1 RGB stays lossy mp4v pending the resolution decision; #10 frame 0
  is post-step, not t=0.*

**★ CoM ABANDONED (2026-07-27).** Not deferred — dropped. The oracle measured it
unobservable from perfect GT dynamics; the single certified channel (eccroll,
+0.020) doesn't repay its scenario budget. See EXPERIMENTS.md Graveyard.
- Dropped: `com_offset` / `com_local` authoring, `com_norm` targets,
  `--com_offset_frac` / `--com_offset_abs`, CoM-scale twins in `collect_pairs`,
  and the W6 mid-batch (balance ~200 / eccroll ~150) that was queued.
- **Kept:** the oracle *method* (target-agnostic, no-GPU scenario acceptance from
  `state.json`) — now repointed at materials; and **`com_camera`**, the body
  POSITION in camera frame, which is the kinematic channel velocities derive from
  and has nothing to do with the abandoned inertial offset.
- Replacement targets, observable by construction: **restitution** (a bounce is
  the readout), **friction** (slide/roll), **contact events** (`impulse ≈ Δ(m·v)`,
  closed-form verifiable).

**Corpus rebuild + dynamic camera (2026-07-27).** The probe corpus is being
**regenerated from scratch**, so non-retrofittable knobs are all in play. Spec:
[data-gen/dynamic-camera.md](data-gen/dynamic-camera.md).
- ✅ **Camera trajectory layer** — [camera_paths.py](../src/data_gen/camera_paths.py)
  (one parametrization; `static` = zero-coefficient case; roll term added because a
  look-at basis forces `g_cam.x ≡ 0`) + [verify_camera_paths.py](../src/data_gen/verify_camera_paths.py)
  — **14/14 pass, runs WITHOUT Isaac** (first unit-testable data-gen module).
  Proven: roll=0 reproduces the legacy basis bit-wise; dolly gives exactly zero
  tilt variation; temporal equivariance to 1e-12; seed-determinism.
- ✅ **Collection wiring** — per-frame extrinsics + egomotion (`v_cam`,
  `omega_cam` in the instantaneous camera frame; absolute pose is gauge and NOT
  a label) → `camera.json` + `schema_version: rigid-probe/v1-dyncam`; camera prim
  re-posed per step via an explicit `rotation=` (a `look_at` cannot carry roll);
  path-aware gating (`gate_camera_paths`, 4 poses along the path);
  `collect_corpus` flags. **Not yet run under Isaac.**
- ✅ **Free 6-DoF camera model** — [camera_motion.py](../src/data_gen/camera_motion.py)
  (surge/sway/heave + pan/tilt/roll, integrated once at construction; tracking
  blend `τ` that re-aims WITHOUT resetting roll) +
  [verify_camera_motion.py](../src/data_gen/verify_camera_motion.py).
  The orbit model could not express pan/tilt/truck at all. **8 motion families
  are stubs — user is writing them**; `build_camera_paths` does not yet dispatch
  to `sample_free_path`.
- ✅ **Contact logging** — [contact_log.py](../src/data_gen/contact_log.py)
  (`--log_contacts`, opt-in) + the trust gate
  [verify_contacts.py](../src/data_gen/verify_contacts.py) (`J_net ≈ m·Δv −
  m·g·Δt`, post-hoc, no Isaac). **Why it is load-bearing, not an extra:** the
  material oracle proved restitution is destroyed by render-rate sampling
  (median rebound 0.178 m/s vs a `g·Δt = 0.327 m/s` floor), and that
  event-anchored windows lift it from R² −0.018 → **+0.217** — which requires
  the event TIMESTAMPS this produces. Impulses are solver-exact at physics rate,
  so they bypass the floor entirely, and they are the only exact read of
  mass/density ratio at collisions. ⚠ The PhysX subscription is the one part not
  checkable off-Isaac — if a first run logs zero events, look there first.
- ⛔ **BLOCKING pre-flight (no Isaac, no GPU): material observability oracle** —
  restitution/friction decodability from GT trajectories on the *existing* corpus.
  `restitution_range` is dynamics-changing and must go **per-scenario** (bounces are
  the observable in drop/throw/collide; high restitution would wreck `balance` /
  `eccentric_roll`). Skipping this repeats the CoM mistake at corpus scale.

**Scene-diversity v0.2 — found on inspection 2026-05-30** (*next implementation
batch; one Isaac smoke cycle. Specs:
[scene-diversification.md §10](data-gen/scene-diversification.md) +
[physics-diversification.md W3a](data-gen/physics-diversification.md). Change dynamics
(topple) and pixels/viewpoint (camera, ground) → settle before the P1 batch.*):

- ✅ **W3a `topple` energy-barrier fix** — lean-past-θ_crit + small kick
  implemented (`scenarios._topple_lean_pose`) and **field-verified on the v2
  corpus** (bodies rotate 70–160°, falling done by frame ~20–25/64). Caveat →
  W6: the past-critical lean *overdetermines* the outcome, so topple motion is
  guaranteed but CoM-blind (com_oracle, 2026-07-05).
- ⬜ **W6 CoM-observability-first scenarios** — `balance` (lean band *around*
  geometric θ_crit, ~zero kick → the invisible CoM decides tip vs rock-back)
  + `eccentric_roll` (offset-CoM cylinders/spheres, rolling start → wobble +
  CoM-low rest pose). Implemented + compile-checked in `scenarios.py`/`config.py`;
  **pending Isaac pilot → `com_oracle` PILOT-mode acceptance (no-GPU) → scale.**
  Spec: [physics-diversification.md §W6](data-gen/physics-diversification.md).
- ⬜ **Multi-view capture (axis G)** — render each physics rollout from
  `n_cameras=7` jittered viewpoints and fan out into **N independent clips**
  (own folder, single-camera contract unchanged; siblings linked by a
  `multiview` provenance block). Physics steps once → big throughput win + free
  multi-view-consistent samples. Per-camera `validate_clip`. Target box 24 GB
  headless; pair with the §14 `save_resolution` drop.
- ⬜ **#12 degenerate-camera guard** — folded into the camera-rig work: the higher
  rig elevations make the `forward ∥ world_up` NaN reachable, so the `|cross|>eps`
  guard in `world_to_camera_frame` lands with axis G (was listed standalone above).
- ⬜ **Grid-line ground (axis D)** — replace the flat-color floor with a locally
  generated grid-line texture (`OmniPBR(texture_path=…)` on `GroundPlane`),
  randomized per clip, for a translation/parallax cue. Suppressed under an
  environment USD that ships its own floor.

## P0.5 — Gating probes  ⛔ GATE A + GATE B

*GATE A on Kubric MOVi-E / CLEVRER (shared GT ω/v/CoM, 4-frame RGB). GATE B
implementation sub-plan: [probe/gate-b-probe.md](probe/gate-b-probe.md). Spec:
[README §Baseline Step 3](../README.md#baseline-to-implement-this-milestone).*

- ⬜ **(a) Feasibility probe** — small clip encoder + per-object dynamics heads.
  *Is dynamics learnable from perfect RGB at all?* → **GATE A**.
- 🟡 **(b) Layer-wise frozen probe (Rung 0/1)** — **iteration-1 read LOCKED on
  the 755-group v2 corpus** (see "Where we are"): arm 2a run (`gate_b_v2` +
  `fit_matrix` target variants — v control PASSES window-aligned at 0.60 @
  L32); arm 2b (MLP + multi-layer concat, `arm_2b.py`) run — v 0.70, ω 0.47;
  CoM row untestable on that corpus (oracle: R6 violated) — reopens once W6
  scenarios land. **Tier-1 targets are now window-aligned clip-mean**
  (`build_targets_v2.py`; probe-walkthrough §2.1 update note). Still open:
  DINO/SigLIP comparison arms; CoM re-read on R6-certified slices.
  *Is the signal in the frozen encoder, at which depth?* → **GATE B**. **Decided
  with user:** frozen encoder only (no unfreeze), eval on a held-out split of our
  own sim corpus, `v` as the harness control; IntPhys2 is a later
  external-transfer iteration. Full design + kill criteria in
  [probe/gate-b-probe.md](probe/gate-b-probe.md).
  - ✅ **register-collapse diagnostic — RUN, collapse RULED OUT** (2026-07-05,
    `results/probe/register_collapse`): body↔frame cosine *falls* with depth
    (0.96 → 0.56) — objects are most distinct at depth; the CoM null was not
    the V-JEPA-2 register artifact. Original rationale kept below:
    [probe/vjepa-embedding-structure.md §5](probe/vjepa-embedding-structure.md):
    the V-JEPA 2.1 paper shows vanilla V-JEPA 2's context tokens collapse toward
    register-token behavior under the exact linear-per-token protocol Arm 2a
    uses (22.2 mIoU / 0.682 RMSE on dense targets, fixed to 47.9 / 0.307 by a
    different training recipe). CoM-offset is the Tier-1 target most exposed to
    the same failure — it needs fine static geometry, not just apparent motion,
    unlike `v`/`ω`. Cheap, label-free check: per layer, cosine-similarity an
    object's masked-pooled feature against the whole-frame-pooled feature;
    near-identical ⇒ collapse signature at that depth. Run this before reading a
    CoM null result as "physics absent."
  - ⬜ **add V-JEPA 2.1 as a second encoder arm** — *deprioritized 2026-07-05:
    its motivating hypothesis (register collapse) was ruled out; keep only as a
    generic comparison arm.* Original rationale:
    [summaries/v-jepa2-1.md](../summaries/v-jepa2-1.md): same JEPA family,
    drop-in checkpoint swap in `encoders.py`, fixes exactly the dense/local
    weakness the diagnostic above targets, and ships a 300M distilled ViT-L
    within ~1pt of the 2B teacher (fits the workstation budget better than
    committing to ViT-g/1B). If CoM-lift recovers under 2.1 but not vanilla
    V-JEPA 2, that confirms confound, not absence.
- ⬜ **Decide before running:** V-JEPA 2 **vs 2.1** checkpoint (and resolution
  variant) → `save_resolution` lock (§14); reference upper-bound model (R9);
  state-readout formulation (chosen). See
  [probe/gate-b-probe.md §12](probe/gate-b-probe.md#12-open-decisions-settle-before-code)
  and [BRAINSTORM §7](BRAINSTORM.md#7-open-decisions).

## P1 — Synthetic data at scale

*Spec: [synthetic-data-collection.md §Implementation order](data-gen/synthetic-data-collection.md#implementation-order),
[scene-diversification.md](data-gen/scene-diversification.md), [isaac-asset-diversification.md](data-gen/isaac-asset-diversification.md).*

> **Non-retrofittable dynamics decisions resolved** (W1 density, W2 offset CoM,
> W3 observability scenarios — all change dynamics and are landed + W4-verified).
> One non-retrofittable knob remains: **§14 save resolution** (changes pixels) —
> settle before the Phase-1a pilot. Also pending first: the P0.4 end-to-end smoke
> run across scenarios (incl. new `topple`/`collide`) to validate the full
> pipeline, not just the W4 label core.

- ⬜ **Sanity dashboard** (depth histogram, normals unit-norm, seg coverage, state
  distributions). *Good first contribution.*
- ⬜ **WebDataset packaging utilities** (per-clip dirs → shards). *Good first contribution.*
- ⬜ **Phase 1a — 100-clip pilot** across the five free-object templates (drop,
  stack-collapse, slope-roll, domino-chain, cluttered-surface) with full DR on.
- ⬜ **Dataloader** reads WebDataset shards; camera-frame conversion of state done
  offline at load time (raw log stays world-frame).

## P2 — Encoder + B-config heads (frozen-first)  ⛔ shape set by GATE B

*Spec: [README §Baseline Step 2](../README.md#baseline-to-implement-this-milestone),
[BRAINSTORM §2](BRAINSTORM.md#2-architecture), [training-data.md §Architecture](theory/training-data.md#architecture--temporal-modeling).*

- ⬜ Frozen **V-JEPA 2 (or 2.1 — checkpoint settled by GATE B, see P0.5)** backbone
  wrapper with intermediate-block taps (which blocks ← GATE B).
- ⬜ **Multi-layer nonlinear head** over intermediate features (DPT-style decoder) for
  dense depth / normals / segmentation. **No raw-pixel conv backbone** (that's the
  escape hatch — see Parked).
- ⬜ Camera-frame labels, ignore-masked background. Clip-level K = 4 to start.
- *No dynamics heads in B (those are F, and are per-object).*

## P3 — Training loop

*Spec: [domain-marginalization-regularizer.md](theory/domain-marginalization-regularizer.md),
[README §Baseline Step 3](../README.md#baseline-to-implement-this-milestone).*

- ⬜ Mixed sim + real batches **from the start** (acceptable stand-in: held-out
  heavy-DR sim as "real" to exercise the code path).
- ⬜ **Domain-marginalization regularizer** (DANN + CORAL), `λ_dom` warm-up over first
  ~5% of steps; tune domain-classifier accuracy into **0.55–0.65**. Wired for B, not just F.
- ⬜ Per-head loss weights explicit in code, not implicit.
- ⬜ Linear-probe diagnostic at fixed intervals (freeze φ, train a linear head per task).

## P4 — B-vs-F experimental loop

*Spec: [training-data.md](theory/training-data.md), [observability-weighted-loss.md](theory/observability-weighted-loss.md),
[README §Baseline Step 4](../README.md#baseline-to-implement-this-milestone).*

- ⬜ Branch **F config**: add sim-only heads (CoM offset, ω, v, contact), **per-object**
  masked-pool over predicted seg → MLP. (Predict CoM *offset*, score lift over centroid — R7.)
- ⬜ **Observability-weighted / heteroscedastic-σ loss** for CoM/mass (weight by
  angular displacement + contact, not linear velocity — R6).
- ⬜ Contact-force loss γ-multiplier on contacting bodies (≈5) so it doesn't collapse to zero.
- ⬜ **B-vs-F eval** on Physion++ / IntPhys 2 / RoboCasa-or-LIBERO. *Closes the first
  experimental loop.*

## P5 — Sim→real policy transfer  *(the load-bearing eval)*

- ⬜ Train a small policy on `φ(X_sim)` for one controlled task (RoboCasa pick-place /
  LIBERO), deploy on **real** frames, measure success rate. Domain-classifier accuracy
  and head losses are only proxies; this is the real test of the project thesis.

---

## Parked / conditional

- **Rung 2 — raw-pixel two-stream + cross-attention** (the PDF design). Built **only if
  GATE B finds no signal at any depth** *and* Rung 1 confirms the ceiling is the
  information bottleneck, not head capacity, **and the register-collapse
  diagnostic has ruled out a fixable V-JEPA-2-specific artifact (try V-JEPA 2.1
  first — P0.5(b))**. Requires R1's load-bearing-stream ablation measured on the
  *real* side. [BRAINSTORM §2.3](BRAINSTORM.md#23-alternative--escape-hatch-the-raw-pixel-two-stream-the-pdf-design).
- **Cosmos-Transfer translated pairs** — generative photorealization as an augmentation
  layer, gated by a registration / label-consistency check; treated as a third domain
  (R4/R5/R10). After regularizer A+C are stable. [BRAINSTORM §5](BRAINSTORM.md#5-simreal-domain-gap-strategy).
- **Jacobian-column-space regularizer** (the version the transfer theorem is about) —
  after DANN+CORAL are stable. [domain-marginalization-regularizer.md](theory/domain-marginalization-regularizer.md).
- **Physics-latent-dynamics ladder** (Neural-ODE / equivalence-class probe) — separate
  diagnostic track, Path A vs B undecided. [physics-latent-ladder.md](theory/physics-latent-ladder.md).
- **World-lines representation** (allocentric, dimensionless object/ego world-lines) —
  Path-B geometric framing of the latent ladder. Design: [world-lines.md](theory/world-lines.md).
  **Phase E0 (sim→world-line encoding) is implemented + round-trip verified to ~1e-16**
  ([world-lines-encoding.md](theory/world-lines-encoding.md), `src/worldlines/`); model /
  training (E1+: object-query head, set-matching, contact residual) gated by GATE A like
  the rest of the pipeline.

## Open decisions blocking specific items

See [BRAINSTORM §7](BRAINSTORM.md#7-open-decisions). The ones that gate code: probe
formulation (P0.5), reference upper-bound (P0.5), temporal handling (P2),
DR-vs-Cosmos mix + registration gate (Parked/Cosmos).

---

## Known bugs

*Append-and-prune ledger (not a work queue): record on sight, fix opportunistically
or when a bug blocks the active line in [EXPERIMENTS.md](EXPERIMENTS.md). Severity is
relative to the experiment it blocks — a bug in parked code is non-urgent by design.*

| id | location | symptom | sev | blocks |
|---|---|---|---|---|
| B1 | `src/worldlines/io.py:31,66` | `_SPEC` loads only `com_local_authored`; composites carry `com_local_analytic` (authored=None) → wrong/None CoM for composite bodies (cf. `probe/dataset.py` `_COM_FIELD`) | med | world-lines (PARKED) |
| B2 | `src/worldlines/io.py:66` | `visible` hard-set to `np.ones(...)` — every frame marked visible regardless of seg presence | med | world-lines (PARKED) |
| B3 | `src/worldlines/gauge.py:40` | gauge origin = mean of body initial positions → **composition-dependent** frame (shifts with #/which bodies) | high | world-lines (PARKED) |
| B4 | `src/worldlines/gauge.py:54` | `to_gauge` passes `b.quat_w` **unchanged** — orientation not rotated into the gauge frame (positions/vels are) | high | world-lines (PARKED) |

All four are in the PARKED world-lines (Tier-2) branch, so non-urgent — fix as a
batch when that branch is promoted, before extending `src/worldlines/`.

## Tech debt / deprecations

*Same discipline: record now, pay down at next-touch — NOT a scheduled cleanup sprint.*

| id | where | issue | trigger to fix |
|---|---|---|---|
| D1 | `src/probe/{learning_curve,diagnose_probe}.py` | import **private** helpers across modules (`_pca_train_test_r2` from `diagnose_probe`, `group_kfold` from `replicate_gate_b`) — fragile chain | when next touching the probe analysis scripts → lift `group_kfold` / `_pca_train_test_r2` / `MIN_TOKEN_WEIGHT` into `src/probe/_common.py` |
| D2 | `out/` (51M) **and** `src/out/` (3.5M) | two output dirs — ambiguous which is canonical | **decided 2026-07-04: canonical is `results/`** (E1+ artifacts write there; gitignored). `out/`+`src/out/` hold legacy artifacts referenced by EXPERIMENTS.md — migrate/delete at next touch |
| D3 | repo root | heavy regenerable artifacts were untracked-but-loose (`*.mkv`, `*.npz`, `src/result.png`) | done: added to `.gitignore`; delete the loose copies (see deletion list). NB `papers/*.pdf` is a tracked convention — NOT ignored |
