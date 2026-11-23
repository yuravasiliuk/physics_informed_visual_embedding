# Physics-Informed Visual Embedding

> **Authorship:** this research was conducted by Yura Vasiliuk, an employee of GlobalLogic LLC.
>
> **Usage:** all code in this repository may be used for **research purposes only**.

A research codebase building a **domain-agnostic visual encoder** so that downstream
policies can be trained **entirely in simulation** and deployed on real-world inputs
without a domain shift. The encoder is the sim→real bridge; physics-grounded
supervision (simulator-derived heads on top of geometric heads) is the mechanism that
forces the embedding to capture transferable structure rather than appearance.

Two co-equal goals:
1. **Close the sim→real domain gap** at the embedding level, so a policy trained on
   `φ(X_sim)` works on `φ(X_real)`.
2. **Encode physics-relevant structure** (geometry, dynamics, contact) that
   appearance-only encoders demonstrably miss (see [IntPhys 2](summaries/intphys2.md)).

The original sketch is in [Physics informed embedding.pdf](Physics%20informed%20embedding.pdf):

> RGB → frozen semantic encoder (JEPA / DINO / SigLIP) + learned conv "geometry" encoder
> → cross-attention fusion → multiple supervision heads.
> **sim+real heads:** depth, normals, segmentation.
> **sim-only heads:** center of mass, angular velocity ω, linear velocity v, contact force.

**Current main line has moved off this sketch.** The raw-pixel learned conv branch
is the most sim→real-fragile component, so it is demoted to an *escape hatch*; the
default is now a **frozen encoder + a multi-layer nonlinear head over its features**
(trainable parameters never touch raw pixels). The full argument and the capacity
ladder that decides when the conv branch is *earned* are in
[BRAINSTORM.md §2](plans/BRAINSTORM.md#2-architecture). The supervision-head split
below is unchanged.

Two refinements to the sketch the plans commit to (orthogonal to the backbone choice):
- The encoder is **clip-level**, K = 4–16 frames. ω, v, and time-derivatives of CoM
  are not observable from a single frame, so single-frame inputs are off the table.
  See [training-data.md §Architecture — temporal modeling](plans/theory/training-data.md#architecture--temporal-modeling).
- Dynamics heads (CoM, ω, v, contact) are **per-object**, not per-pixel — masked-pool
  the encoder features over each body's predicted segmentation, then one MLP per head.
  Per-pixel dynamics is degenerate (constant within each segment) and trivially
  satisfiable by `seg → MLP(seg_id)`. Per-pixel stays only for depth, normals, seg.

This README is the **entry point** for new researchers, engineers, and contributors.
For depth on any topic, follow the links into [plans/](plans/) and [summaries/](summaries/).

---

## Repo layout

| Path | What's in it |
|---|---|
| [Physics informed embedding.pdf](Physics%20informed%20embedding.pdf) | The original architecture sketch. |
| [papers/](papers/) | PDFs of the literature we read while scoping the project. |
| [summaries/](summaries/) | Per-paper working notes + the cross-paper index in [summaries/README.md](summaries/README.md). |
| [plans/](plans/) | Design docs, grouped into [theory/](plans/theory/) (modeling/paper), [data-gen/](plans/data-gen/) (collection pipeline), [critique/](plans/critique/) (reviews); [BRAINSTORM.md](plans/BRAINSTORM.md) + [IMPLEMENTATION.md](plans/IMPLEMENTATION.md) + [EXPERIMENTS.md](plans/EXPERIMENTS.md) stay at the root as hubs. Two lineages. **Committed plan (read in order):** [theoretical-paper-plan.md](plans/theory/theoretical-paper-plan.md) → [training-data.md](plans/theory/training-data.md) → [domain-marginalization-regularizer.md](plans/theory/domain-marginalization-regularizer.md) → [synthetic-data-collection.md](plans/data-gen/synthetic-data-collection.md); dataset spec stack [probe-dataset.md](plans/data-gen/probe-dataset.md) → [scene-diversification.md](plans/data-gen/scene-diversification.md) → [isaac-asset-diversification.md](plans/data-gen/isaac-asset-diversification.md). **Active brainstorm hub:** [BRAINSTORM.md](plans/BRAINSTORM.md) (its §0 maps the rest of that cluster). **Build status / what's-done-vs-next:** [IMPLEMENTATION.md](plans/IMPLEMENTATION.md). **Hypothesis frontier (active/parked/killed experiments — check first to see the current line):** [EXPERIMENTS.md](plans/EXPERIMENTS.md). **Critiques:** [critque.md](plans/critique/critque.md) → [critique2.md](plans/critique/critique2.md), [data_gen_critique0.md](plans/critique/data_gen_critique0.md). |
| [src/data_gen/](src/data_gen/) | Isaac Sim data-collection code (the only code so far). Entry point: [collect_clip.py](src/data_gen/collect_clip.py) (refactor of the original smoke script into reusable pieces; see [plans/probe-dataset.md](plans/data-gen/probe-dataset.md)). How to validate a run: [plans/data-gen-testing.md](plans/data-gen/data-gen-testing.md). Encoder / training / experiment code is **planned, not yet written** — see §Baseline to implement. |
| [notebooks/](notebooks/) | Scratch / exploratory notebooks. |

---

## Analysis — what the literature tells us

The full notes are in [summaries/README.md](summaries/README.md); the load-bearing
take-aways for our design are:

- **Architecture.** [SCOPE](summaries/scope.md) — frozen DINOv2 cross-attended into a
  learned conv branch on RGB + normals beats concatenation-style fusion (e.g.
  DiffusionNOCS) on out-of-distribution objects (ADD-S 59→76 on YCB-V). Note this is
  precedent for the **escape-hatch** two-stream (Rung 2), not the frozen-first main
  line — and *if* the conv branch is entered, it says use **cross-attention, not
  concat**. It does not bear on whether the conv branch is needed at all; that is the
  layer-wise-probe question ([BRAINSTORM.md §2](plans/BRAINSTORM.md#2-architecture)).
- **Aux-task supervision is principled, not arbitrary.**
  [Why and How Auxiliary Tasks Improve JEPA](summaries/why-auxiliary-tasks-jepa.md)
  shows each aux head induces an equivalence relation on the latent space; the encoder
  is constrained to the meet of those relations. Picking the head set *is* picking the
  geometry of the representation.
- **Video alone is not enough.**
  The [intuitive-physics opinion paper](summaries/opinion-intuitive-physics.md) +
  [IntPhys 2](summaries/intphys2.md): SOTA video models including V-JEPA 2 sit at
  ~56–58 % on intuitive physics where humans are at ~96 %. Strong external argument
  for explicit physics supervision from a simulator.
- **Two paradigms compete with us, both worth understanding:**
  - [Cosmos-Transfer2.5](summaries/cosmos-predict2.5.md) — translate sim pixels to real-looking
    pixels, then train normally. Pixel-space alignment.
  - Ours — train one encoder on sim+real with shared and asymmetric heads, with a
    regularizer pulling the embedding to be domain-invariant. Latent-space alignment.
- **Closest prior art in spirit:** [RoboScape](summaries/roboscape.md) — dual-branch world
  model with auxiliary depth + keypoint-dynamics heads. Different end product (generative
  video model), but the multi-head physics-aware recipe is exactly the family we sit in.
  Their ablation found keypoint dynamics carried most of the gain over depth; we expect
  an analogous head-importance ordering on our side.
- **Closest negative result:** [PI-JEPA](summaries/pi-jepa.md) reports PDE-residual loss
  as **neutral** on top of operator splitting + VICReg. We argue we escape this because
  our heads regress *observables* (CoM, ω, v) rather than enforcing a residual *constraint*
  on an operator with a large null space — see [theoretical-paper-plan.md "Why we expect
  to escape PI-JEPA's null result"](plans/theory/theoretical-paper-plan.md#why-we-expect-to-escape-pi-jepas-null-result).
  If F ≈ B in practice, that argument is wrong and the project pivots.
- **Encoder candidates for the frozen branch:** [V-JEPA 2](summaries/v-jepa2.md) is the
  current default (zero-shot robot planning from RGB, scales to internet video). DINOv2
  is the proven choice from SCOPE for static spatial consistency. Pick one to start; the
  theory is encoder-agnostic.

---

## Best tactics — decisions we're committing to

Pulled from the design docs; cite the source whenever you change one of these.

> **Architecture note.** The main line is **frozen-first**: a frozen encoder + a
> multi-layer nonlinear head over its features, *no* raw-pixel conv backbone. The
> two-stream + cross-attention is an escape hatch, earned only if a layer-wise probe
> finds no recoverable physics signal in the frozen encoder at any depth — see the
> capacity ladder in [BRAINSTORM.md §2](plans/BRAINSTORM.md#2-architecture). The
> **B/F configs below are about head sets** (B = dense only; F = + physics), which is
> orthogonal to the backbone choice and unchanged.

### Theoretical scope ([plans/theoretical-paper-plan.md](plans/theory/theoretical-paper-plan.md))
- The first paper is **theory-only**. No experiments needed to ship it.
- The distinctive contributions, in priority order: (1) domain-marginalization regularizer
  with a transfer bound for sim-only heads, (2) first-order / counterfactual supervision
  via differentiable rendering, (3) multi-head equivalence composition, (4) bias–variance
  comparison vs Cosmos-Transfer-style pixel translation.
- Equivariance structure of physical heads (SE(3)_camera × stab(g)_world) is the
  minimum group to design for.

### Training data ([plans/training-data.md](plans/theory/training-data.md))
- **Real-data starting mix:** AgiBot-World-Beta (manipulation, mirrors RoboScape/Cosmos)
  + ScanNet++ (sensor-grade depth & normals).
- **Simulator:** Isaac Lab. Matches Cosmos and IntPhys 2 stacks; full label coverage.
- **Clip-level inputs**, K = 4–16 frames. V-JEPA 2 (temporal) for the frozen encoder.
  DINOv2 only as a single-frame geometry-only ablation.
- **Frozen-first head** (main line): multi-layer nonlinear head over V-JEPA 2's
  intermediate features; no raw-pixel conv backbone unless the layer-wise probe
  earns it. See [training-data.md §Architecture](plans/theory/training-data.md#architecture--temporal-modeling)
  and [BRAINSTORM.md §2](plans/BRAINSTORM.md#2-architecture).
- **All labels in camera frame.** Dense heads (depth, normals, seg) are per-pixel
  with an *ignore mask* for background. Dynamics heads (CoM, ω, v, contact) are
  **per-object** — masked-pool over predicted segmentation, then MLP.
- **CoM head:** skip bodies that are non-compact (CoM far from bbox center) or
  architectural (high mass / kinematic). Other heads are unaffected.
- **Contact-force loss** gets a γ-multiplier on contacting bodies (cf. RoboScape γ = 5),
  otherwise the head collapses to "predict zero everywhere."
- **Two configs trained on the same corpus** (a *head-set* distinction — orthogonal
  to the frozen-first backbone choice above), both running the domain-marginalization
  regularizer (it's the sim→real bridge, not an F-only safeguard):
  - **B (baseline):** depth + normals + segmentation only. Symmetric heads only.
  - **F (full):** B + CoM, ω, v, contact. Sim-only heads add the asymmetric
    supervision the regularizer is also designed to defend against.

### Domain-marginalization regularizer ([plans/domain-marginalization-regularizer.md](plans/theory/domain-marginalization-regularizer.md))
- **Primary role:** sim→real bridging mechanism. The whole project hinges on
  `φ(X_sim) ≈ φ(X_real)` in distribution so that sim-trained policies deploy on real.
  The regularizer is the forcing function that makes that hold.
- **Secondary role:** prevent the sim-only heads (F config) from carving out a
  sim-only feature subspace that real inputs never enter.
- **First-pass regularizer:** DANN-style domain-adversarial classifier on `φ(X)`
  + CORAL moment matching, with `λ_dom` warm-up over the first ~5 % of steps.
- **Tuning target:** drive the domain classifier accuracy to 0.55–0.65. Lower → over-aligning;
  higher → regularizer not biting.
- Once available, add **translated-pair consistency** via Cosmos-Transfer2.5.
- The **Jacobian-column-space constraint** (the version the theorem is about) is
  parked for a later pass — implement it after A+C are stable.

### Synthetic data collection ([plans/synthetic-data-collection.md](plans/data-gen/synthetic-data-collection.md))
- Workstation-scale pilot, not cluster scale: **10k free-object + 5k manipulation**
  clips for Phase 1.
- Realism budget goes to **diversity** (lights, materials, cameras), not per-frame
  fidelity. RTX Real-Time or PT-low (4 spp), not full path tracing.
- Five free-object templates: drop, stack-collapse, slope-roll, domino-chain,
  cluttered-surface. Roughly equal share.
- Heavy domain randomization is what makes the "no Cosmos-Transfer up front" choice
  defensible — the regularizer assumes the encoder *cannot* trivially classify
  sim vs real from pixel statistics.
- **Output format:** WebDataset shards; per-clip directories with stacked NPY arrays,
  one MP4 for RGB, one Parquet for per-body state.
- Camera-frame conversion of state happens **offline** at dataloader time. The raw
  log stays in world frame.

---

## Baseline to implement (this milestone)

The goal of the first engineering milestone is to produce a clean, runnable
**B configuration training loop end-to-end on a small Phase-1a corpus** (~100 clips).
Everything else — F config, regularizer, translated pairs, leave-one-out ablations —
is layered on top.

Concretely:

1. **Data collection — finish Phase 1 of [synthetic-data-collection.md §Implementation order](plans/data-gen/synthetic-data-collection.md#implementation-order).**
   - **Step 1 (in flight).** Phase-0 collection script: [src/data_gen/collect_clip.py](src/data_gen/collect_clip.py)
     renders one free-object drop with 3 primitives, captures RGB + depth + normals +
     instance-seg per frame, logs per-body state to JSON. Refactored from the
     original smoke script into reusable functions so a future shard driver can
     call `collect_clip(out_dir, ClipConfig(...), seed)` directly.
     Status: the script is in place but **has not been run end-to-end** under the
     Isaac Sim Python environment yet. First action for a new contributor: run it
     and verify all outputs (`rgb.mp4`, `depth.npy`, `normals.npy`, `seg.npy`,
     `state.json`, `meta.json`, optional `seg_id_to_prim.json`) look sane.
   - **Step 2.** Sanity dashboard (depth histogram, normals unit-norm check, seg
     coverage, state distributions).
   - **Step 3 (Phase 0.5, hard gate).** Two probes on Kubric MOVi-E or CLEVRER
     (shared GT ω / v / CoM, 4-frame RGB):
     - **(a) Feasibility probe** — small clip-level encoder + per-object dynamics
       heads. Answers *is dynamics learnable from perfect RGB at all?* Required to
       pass: ω R² > 0.5, v R² > 0.5, CoM L2 < ½ median bbox diagonal on held-out
       scenes.
     - **(b) Layer-wise frozen probe (Rung 0)** — linear probe of **frozen V-JEPA 2
       at each intermediate block** (compare DINO / SigLIP) for the same targets.
       Answers *is the signal in the frozen encoder, and at which depth?* This is the
       **architecture gate**: signal present → frozen-first main line, and its peak
       depth sets which blocks the Step-2 head taps; signal absent at every depth →
       the escape-hatch conv branch is justified.

     **No Isaac Sim compute is committed until (a) passes** — if a small model can't
     learn dynamics on perfect data, the full pipeline won't either. ~1 GPU-day.
   - **Step 4 (Phase 1a).** 100-clip pilot across the five free-object templates with
     full DR on. Validate the dataloader can read WebDataset shards.
2. **Encoder & heads — B config first, frozen-first backbone.**
   - Clip-level inputs (K = 4 initially, K = 8 or 16 at Phase 1c).
   - Frozen encoder: V-JEPA 2 (temporal). DINOv2 disqualified for clip-level use except
     as a single-frame ablation.
   - **Main-line head:** multi-layer nonlinear head over V-JEPA 2's *intermediate*
     features (fuse several blocks, attentive-pool / shallow). **No raw-pixel conv
     backbone** — that is the escape hatch (Rung 2), built only if the layer-wise
     probe in Step 3 finds no recoverable physics signal at any depth. See
     [BRAINSTORM.md §2](plans/BRAINSTORM.md#2-architecture).
   - Heads: dense per-pixel depth, normals, instance segmentation. No dynamics heads
     in B (those come with F, and are per-object).
   - All labels in camera frame, ignore-masked background, per [training-data.md "Label semantics"](plans/theory/training-data.md#label-semantics).
3. **Training loop.**
   - Mixed sim + real batches **from the start** — the regularizer can only align
     domains it sees together. Acceptable temporary substitute while real ingestion
     is being wired: a held-out chunk of sim with heavy DR as a stand-in for "real,"
     so the regularizer code path is exercised end-to-end.
   - Domain-marginalization regularizer (DANN + CORAL) wired in for B, not just F.
   - Per-head loss weighting documented in code, *not* implicit.
   - Linear-probe diagnostic at fixed intervals: freeze φ, train a linear head per task
     on a held-out split.
4. **B-vs-F gate.** Once B is converging on Phase-1c (10k clips), branch the F config:
   add the sim-only heads on top of the regularizer that B is already running. The
   B-vs-F comparison on Physion++ / IntPhys 2 / RoboCasa-or-LIBERO is the milestone
   that closes the first experimental loop.
5. **Sim→real policy transfer eval (the real test).** Train a small policy on
   `φ(X_sim)` for one controlled task (RoboCasa pick-place or LIBERO), deploy on
   real frames, measure success rate. This is the load-bearing eval for the project
   thesis — domain classifier accuracy and head losses are proxies.

A successful baseline is one where:
- Phase 0.5 has passed (per-object dynamics learnable on perfect data).
- B's depth + normals + seg losses converge to numbers competitive with a non-physics
  encoder of similar size trained on the same data.
- Domain classifier accuracy on `φ(X)` is in the 0.55–0.65 target band.
- The dataloader, eval harness, and metric-logging are reusable verbatim for F.
- Code is structured so that adding a head is a localized change, not a refactor.

---

## How to contribute

- **New researchers** — start with [summaries/README.md](summaries/README.md). Skim
  every entry; read [scope.md](summaries/scope.md), [roboscape.md](summaries/roboscape.md),
  [why-auxiliary-tasks-jepa.md](summaries/why-auxiliary-tasks-jepa.md), and
  [opinion-intuitive-physics.md](summaries/opinion-intuitive-physics.md) in full.
  Then [theoretical-paper-plan.md](plans/theory/theoretical-paper-plan.md). After that you
  should be able to pick up any open question listed in the plans.
- **Engineers** — start with [synthetic-data-collection.md](plans/data-gen/synthetic-data-collection.md)
  and [probe-dataset.md](plans/data-gen/probe-dataset.md), then run
  [src/data_gen/collect_clip.py](src/data_gen/collect_clip.py) to verify the
  Isaac Sim pipeline produces the expected outputs. Anything in
  [probe-dataset.md §11 Implementation order](plans/data-gen/probe-dataset.md#11-implementation-order)
  Steps 2+ is fair game.
- **Free contributors** — the cheapest contributions right now are: (a) reading-pile
  gaps in [theoretical-paper-plan.md §"Reading still owed"](plans/theory/theoretical-paper-plan.md#reading-still-owed)
  and writing a summary in the style of the others; (b) the sanity-dashboard tool
  (Step 2); (c) WebDataset packaging utilities for Step 3.

When you change a load-bearing decision (anything under §Best tactics), update the
relevant `plans/*.md` file in the same commit. The plans are the source of truth;
this README is a map.
