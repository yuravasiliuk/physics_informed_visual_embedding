# Implementation Status & Plan

*Operational tracker for the engineering build. Living document — update the
status markers as work lands. This is the **"where are we / what's next"** view;
the **why** lives in the design docs linked from each item, and the milestone
narrative is in [README §Baseline to implement](../README.md#baseline-to-implement-this-milestone).*

**Last updated: 2026-05-30.**

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

**Phase 0 (single-clip data-gen pipeline) — code written, not yet validated end-to-end.**
Next action: run `collect_clip.py` under the Isaac Sim Python env and confirm all
outputs are sane (see P0.4) — the validation procedure is in
[data-gen-testing.md](data-gen-testing.md). Nothing downstream is started, and
**no model work is committed until the Phase-0.5 feasibility probe passes** (see
the gate below).

---

## Hard gates (do not skip)

- **GATE A — feasibility (P0.5a).** No Isaac Sim *scale* compute is committed until a
  small model learns dynamics from *perfect* RGB (Kubric/CLEVRER): ω R² > 0.5,
  v R² > 0.5, CoM L2 < ½ median bbox diagonal. If perfect data can't be learned, the
  full pipeline won't either. ~1 GPU-day.
- **GATE B — architecture (P0.5b, Rung 0).** The layer-wise frozen probe decides the
  backbone: signal present in the frozen encoder → frozen-first main line (and its
  peak depth sets which blocks the head taps); signal absent at *every* depth → the
  raw-pixel escape hatch (Rung 2) is justified. See
  [BRAINSTORM.md §2](BRAINSTORM.md#2-architecture).

---

## P0 — Single-clip data-gen pipeline (Isaac Sim)

*Spec: [synthetic-data-collection.md](synthetic-data-collection.md),
[probe-dataset.md](probe-dataset.md). Code: [../src/data_gen/](../src/data_gen/).*

- ✅ `collect_clip.py` refactored into reusable `collect_clip(out_dir, ClipConfig, seed)`
  with `build_scene / simulate_clip / validate_clip` stages.
- ✅ Supporting modules in place: `asset_library`, `asset_loader`, `body_factory`,
  `scenarios`, `lighting`, `environments`, `static_geometry`, `clip_capture`, `seg_labels`.
- ✅ `verify_collect_clip.py` written (CoM=0 sphere smoke test).
- ✅ **Verify coverage hardened (R11 / #4 / #11).** `verify_physics_labels.py`
  (W4) adds spin-cuboid (ω sign/scale + quat composition), authored-offset-CoM,
  and two-body index-swap cases, all at the production camera; 3/3 pass. Caught
  and fixed two real label bugs (unscaled `centerOfMass`, default angular
  damping). See [physics-diversification.md](physics-diversification.md) W4.
- ⬜ **P0.4 — Run `collect_clip.py` end-to-end** under the Isaac Sim Python env;
  confirm the storage-optimized output contract is sane. `save_clip` writes
  **quantized, compressed `.npz`** — not raw `.npy`:
  `rgb.mp4` (lossy mp4v), `depth.npz` (uint16, per-clip range carried in
  `depth_min`/`depth_max` keys — dataloader must dequantize), `normals.npz`
  (uint8, fixed [-1,1] map), `seg.npz` (uint8, or uint16 if >255 classes),
  `state.json`, `meta.json`, and optional `seg_id_to_class.json` (id→class map).
  This lands the **precision/format** disk wins from
  [data_gen_critique0_analysis.md](data_gen_critique0_analysis.md) (#6 seg downcast,
  normals→uint8, depth→uint16, `.npz` compression). The **§14 resolution-decoupling**
  knob (`save_resolution`) exists but defaults to full render res (1024²) — *not yet
  engaged*, so the biggest disk lever is still on the table (see hardening below).
  *First action for a new contributor.*

**P0 hardening — critique 0** (*tracked in
[data_gen_critique0_analysis.md](data_gen_critique0_analysis.md). Density and
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

**Scene-diversity v0.2 — found on inspection 2026-05-30** (*next implementation
batch; one Isaac smoke cycle. Specs:
[scene-diversification.md §10](scene-diversification.md) +
[physics-diversification.md W3a](physics-diversification.md). Change dynamics
(topple) and pixels/viewpoint (camera, ground) → settle before the P1 batch.*):

- ⬜ **W3a `topple` energy-barrier fix** — the tip kick is below the tipping
  barrier, so `topple` looks static. Fix: spawn leaning just past θ_crit + small
  kick (gravity topples it; offset CoM biases the landing face). Adds an
  `orientation` param to `make_body`/`spawn_body`. *Was a silent physics bug —
  the new scenario produced near-zero motion.*
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

*On Kubric MOVi-E / CLEVRER (shared GT ω/v/CoM, 4-frame RGB). Spec:
[README §Baseline Step 3](../README.md#baseline-to-implement-this-milestone).*

- ⬜ **(a) Feasibility probe** — small clip encoder + per-object dynamics heads.
  *Is dynamics learnable from perfect RGB at all?* → **GATE A**.
- ⬜ **(b) Layer-wise frozen probe (Rung 0)** — linear probe of frozen V-JEPA 2 at
  *each* intermediate block; compare DINO / SigLIP. *Is the signal in the frozen
  encoder, at which depth?* → **GATE B**.
- ⬜ **Decide before running:** reference upper-bound model (R9); state-readout vs.
  future-prediction probe formulation (read-out preferred). See
  [BRAINSTORM §7](BRAINSTORM.md#7-open-decisions).

## P1 — Synthetic data at scale

*Spec: [synthetic-data-collection.md §Implementation order](synthetic-data-collection.md#implementation-order),
[scene-diversification.md](scene-diversification.md), [isaac-asset-diversification.md](isaac-asset-diversification.md).*

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
[BRAINSTORM §2](BRAINSTORM.md#2-architecture), [training-data.md §Architecture](training-data.md#architecture--temporal-modeling).*

- ⬜ Frozen **V-JEPA 2** backbone wrapper with intermediate-block taps (which blocks ← GATE B).
- ⬜ **Multi-layer nonlinear head** over intermediate features (DPT-style decoder) for
  dense depth / normals / segmentation. **No raw-pixel conv backbone** (that's the
  escape hatch — see Parked).
- ⬜ Camera-frame labels, ignore-masked background. Clip-level K = 4 to start.
- *No dynamics heads in B (those are F, and are per-object).*

## P3 — Training loop

*Spec: [domain-marginalization-regularizer.md](domain-marginalization-regularizer.md),
[README §Baseline Step 3](../README.md#baseline-to-implement-this-milestone).*

- ⬜ Mixed sim + real batches **from the start** (acceptable stand-in: held-out
  heavy-DR sim as "real" to exercise the code path).
- ⬜ **Domain-marginalization regularizer** (DANN + CORAL), `λ_dom` warm-up over first
  ~5% of steps; tune domain-classifier accuracy into **0.55–0.65**. Wired for B, not just F.
- ⬜ Per-head loss weights explicit in code, not implicit.
- ⬜ Linear-probe diagnostic at fixed intervals (freeze φ, train a linear head per task).

## P4 — B-vs-F experimental loop

*Spec: [training-data.md](training-data.md), [observability-weighted-loss.md](observability-weighted-loss.md),
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
  information bottleneck, not head capacity. Requires R1's load-bearing-stream ablation
  measured on the *real* side. [BRAINSTORM §2.3](BRAINSTORM.md#23-alternative--escape-hatch-the-raw-pixel-two-stream-the-pdf-design).
- **Cosmos-Transfer translated pairs** — generative photorealization as an augmentation
  layer, gated by a registration / label-consistency check; treated as a third domain
  (R4/R5/R10). After regularizer A+C are stable. [BRAINSTORM §5](BRAINSTORM.md#5-simreal-domain-gap-strategy).
- **Jacobian-column-space regularizer** (the version the transfer theorem is about) —
  after DANN+CORAL are stable. [domain-marginalization-regularizer.md](domain-marginalization-regularizer.md).
- **Physics-latent-dynamics ladder** (Neural-ODE / equivalence-class probe) — separate
  diagnostic track, Path A vs B undecided. [physics-latent-ladder.md](physics-latent-ladder.md).

## Open decisions blocking specific items

See [BRAINSTORM §7](BRAINSTORM.md#7-open-decisions). The ones that gate code: probe
formulation (P0.5), reference upper-bound (P0.5), temporal handling (P2),
DR-vs-Cosmos mix + registration gate (Parked/Cosmos).
