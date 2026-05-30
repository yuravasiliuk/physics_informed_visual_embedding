# Physics-Informed Embedding — Brainstorm Summary

*Date: 2026-05-30. Design notes, not a spec. Captures the discussion around the
architecture sketched in `Physics informed embedding.pdf`, the physics-label
targets, a probabilistic CoM formulation, and the sim→real strategy.*

The architecture in one line: **a frozen encoder (V-JEPA/DINO/SigLIP) + a
multi-layer nonlinear head over its features**, with prediction heads split by
supervision availability. RGB-only input, real or sim. Heads: *sim+real* (depth,
normals, segmentation) and *sim-only* (angular velocity, center of mass, contact
forces, distance-to-ground). The PDF's raw-pixel two-stream + cross-attention is
demoted to an escape hatch (§2.3); the main line keeps the trainable parameters
off raw pixels so sim→real stability holds by construction. See §2.

Risks are listed first deliberately — most of them are load-bearing and several
can silently invalidate results.

---

## 0. Document map

This is the **hub for the active architecture/risks thinking**. For the
whole-repo map (code, data pipeline, literature, the engineering milestone) start
at the [root README](../README.md) — this doc does not duplicate it.

**This brainstorm cluster:**
- **BRAINSTORM.md** (this file) — architecture, risks (R1–R11), CoM formulation,
  sim→real strategy, verifiability, open decisions.
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — operational status tracker: what's done,
  what's next, and the two hard gates. The *why* for each item lives here; that doc
  tracks the *state*.
- [observability-weighted-loss.md](observability-weighted-loss.md) — spun off from
  **R6 / §4**: weight physics loss by angular displacement + contact (not linear
  velocity); learned heteroscedastic σ.
- [physics-latent-ladder.md](physics-latent-ladder.md) — sibling "pragmatic
  ladder" brainstorm; pre-experiment staging of physics-latent-dynamics targets.

**Critiques that feed the risks here:**
- [data_gen_critique0.md](data_gen_critique0.md) → its
  [analysis](data_gen_critique0_analysis.md) — the data-gen review driving the
  recent `collect_clip.py` work; source of **R11** (verification gaps).
- [critque.md](critque.md) → [critique2.md](critique2.md) — earlier
  proposal critiques (the single-frame-input fatal-flaw thread).

**Relationship to the committed plan (⚠ open tension):** the
[root README §Best tactics](../README.md#best-tactics--decisions-were-committing-to)
and its Era-1 lineage ([theoretical-paper-plan.md](theoretical-paper-plan.md),
[training-data.md](training-data.md), [probe-dataset.md](probe-dataset.md))
**commit** to the two-stream design (frozen + learned conv branch, cross-attention,
B/F configs) as decided. §2 of this doc **supersedes that for the main line**: the
default is now a frozen encoder + multi-layer nonlinear head (no raw-pixel
backbone), and the two-stream is demoted to an escape hatch entered only if a
layer-wise probe finds no physics signal in the frozen encoder at any depth (R1
warns the conv branch can otherwise make the frozen stream vestigial). Not yet
reconciled into the README's §Best tactics; treat §2 as the newer position.

---

## 1. Risks (read these first)

### R1 — The trainable geometry branch can make the frozen stream vestigial
The conv geometry encoder is trained end-to-end on sim; it can learn to predict
everything from sim pixels while cross-attention learns to **ignore** the
semantic stream. If that happens the domain bridge is gone and sim→real transfer
collapses — but training/val loss on sim looks great, so the failure is silent.
**Mitigation:** an ablation that proves the frozen semantic stream is
load-bearing, measured on the *real* side (not sim), e.g. attention-mass on
semantic tokens, or real-domain dense-task degradation when the stream is
removed.

### R2 — Sim→real transfer of the sim-only heads is fundamentally unverifiable
CoM / ω / contact forces have ground truth **only** in sim. On real data there
is no label to check against. The entire value of the project rests on a bet
that can't be directly measured. **Mitigation:** lean on proxies — the *sim+real*
dense heads (depth/normals/seg) as a transfer thermometer, plus qualitative ω/CoM
sanity on real clips. Accept that the core claim is supported only indirectly.

### R3 — Real-side "sim+real" labels are pseudo-labels
Depth/normals/seg in the real domain come from sensors or pretrained monocular
models. Their quality is precisely what gates whether the *physics* heads
transfer. Weak real-side dense supervision → weak cross-domain alignment →
physics heads don't transfer, silently.

### R4 — Generative photorealization (Cosmos-Transfer) can corrupt labels
Pushing sim frames through a structure-conditioned diffusion model can drift
object boundaries, swim textures, and break frame-to-frame motion coherence.
That is **label noise on exactly the quantities we care about**:
- dense depth/normals/seg are pixel-exact → small boundary drift misregisters them;
- ω/contact assume apparent motion matches the sim trajectory → diffusion temporal
  inconsistency desyncs them.
Fundamental tension: **more photorealism (more generative freedom) ⇒ more label
drift.** Non-negotiable: a registration / label-consistency gate before trusting
labels on generated frames.

### R5 — Cosmos output is a *third domain*, and only closes the appearance gap
Generated video has its own diffusion signature — it is "Cosmos's idea of real,"
not real. Mix it; don't replace sim with it. And it repaints appearance only —
it does **not** fix the scene-composition / motion-distribution gap (3–8
primitives on a ground plane stays that, however photorealistic).

### R6 — Unlearnable-but-perfectly-labeled targets
CoM offset and mass have **near-zero static visual signature**; they are
observable only through *dynamics* (off-center tumbling) and especially
*contact* (tipping, bouncing, settling). If the dataset lacks dynamic/contact
richness, these become targets with exact labels and no recoverable signal — the
worst kind of confound. Target learnability must be designed into the *scenarios*,
not just the labels. Partial mitigation (training-noise half only) via
observability-weighted loss — see
[observability-weighted-loss.md](observability-weighted-loss.md): weight by
angular displacement + contact (not linear velocity), preferably as a learned
heteroscedastic σ.

### R7 — Absolute-CoM is a confounded metric
A head predicting absolute CoM scores high R² for free just by localizing the
object in frame, telling you nothing about mass distribution. **Fix:** predict
the *offset* `CoM − geometric_centroid` and report **lift over the centroid
baseline**, not raw CoM error.

### R8 — Coarse targets can be cheated from appearance
If off-center / non-uniform objects look visually different, "is it off-center?"
degenerates into appearance classification, not physics-from-dynamics. **Fix:**
visually-controlled objects — same outer shape/texture, internal mass
distribution the only difference — so the answer is reachable only through
observed motion.

### R9 — Predictive uncertainty conflates two causes
A wide predicted distribution can mean "the info isn't in the video"
(identifiability) *or* "the model couldn't extract it" (capacity). This
architecture *softens* the confound (the geometry branch is trainable, so a wide
output leans toward genuine unobservability) but doesn't remove it. A reference
upper-bound model is still the clean way to separate the two.

### R10 — Compute and disk
Cosmos on a 10k-clip *video* dataset is a heavy compute bill and more data
volume, on top of an already-tight disk budget (see the data-gen analysis: the
current per-clip layout doesn't fit 10k clips as-is).

### R11 — Verification gaps (data-gen critique #4)
The current verify script only exercises a single non-spinning sphere
(`com_local == 0`), so ω sign/scale, quaternion composition, and the
asymmetric-CoM path are untested. Note two things that are *not* verifiable and
should stay out of scope: general tumbling (no stable closed form;
intermediate-axis chaos) and PhysX-computed CoM on real assets (no independent
ground truth).

---

## 2. Architecture

**Main line: a frozen encoder + a multi-layer, nonlinear head over its
features.** No raw-pixel second backbone. The trainable parameters never see raw
pixels — only frozen, domain-robust features — so sim→real stability is high *by
construction*. The raw-pixel two-stream + cross-attention from the PDF is demoted
to an escape hatch (§2.3), entered only if the gating experiment (§2.2, Rung 0)
shows the physics signal is absent from the frozen encoder at every depth.

The reframe that matters: **V-JEPA is the thing being probed for physics.** The
question is *"does a frozen generalist encoder encode physics that a head can
recover — and transfer?"* not *"can a sim-trained conv net predict physics with
SSL features as a side input?"* The second framing silently swaps the scientific
question for an engineering one and reintroduces the sim-overfit branch.

### 2.1 Why frozen-features-first is the stable choice (and its honest cost)

The only reason to freeze the encoder is domain robustness. A head whose inputs
are frozen features inherits that robustness — almost nothing can overfit sim
appearance, because the overfittable low-level signal (texture, lighting) never
reaches a trainable parameter. A raw-pixel conv branch is the opposite: trained on
**sim physics labels only** (no real CoM/ω label exists), it *will* specialize to
sim, and any fusion is the single fragile thing pulling it back toward real. That
is R1 + R2, failing silently.

The information-vs-gap trade-off, stated plainly:
- **Raw pixels** — maximum information (nothing discarded), maximum domain gap.
- **Frozen features** — minimum domain gap, but information bottlenecked to *what
  the encoder chose to keep*.

So a head over frozen features **cannot recover geometry the encoder discarded** —
that information is gone. Its "cost" is therefore a feature: if it fails, it fails
*honestly* ("the physics isn't recoverable from the embedding"), which is exactly
the question we want answered. The bet is that the encoder **encodes but does not
linearly expose** the physics — recoverable nonlinearly, and especially from
*intermediate* layers (early/mid blocks retain spatial/geometric detail the final
embedding has abstracted away). That bet is testable and cheap (§2.2, Rung 0).

### 2.2 The capacity ladder (each rung adds capacity, subtracts stability)

- **Rung 0 (gating experiment) — layer-wise linear probe.** Frozen encoder +
  linear probe per target, **probed at each intermediate block**. Max stability,
  cheapest, and it answers two things at once: *is the physics signal present at
  all*, and *at which depth does it peak* (→ which layer the Rung-1 head should
  tap). Run across encoders (**V-JEPA vs DINO vs SigLIP**); V-JEPA is
  video/motion-trained, so it is the one plausibly carrying ω-relevant dynamics.
  IntPhys 2 (V-JEPA 2 ~56–58% on intuitive physics) is a yellow flag that the
  signal may be weak — which is precisely why this probe runs *before* anything is
  built. **No second-stream work is committed until this locates a signal.**
- **Rung 1 (main line) — multi-layer nonlinear head over frozen features.** Fuse
  several intermediate blocks (à la DPT / DINO dense probes) into an
  attentive-pool / shallow-conv / MLP head; DPT-style decoder for the dense heads.
  This is a strict **superset of Rung 0's reach** (multiple layers, nonlinear)
  while keeping Rung 0's domain robustness — it recovers nonlinearly-encoded and
  early-layer geometry the final embedding lost, *without ever touching raw
  pixels*. Most achievable accuracy-with-stability lives here. **This is where the
  "use intermediate V-JEPA features instead of pixels" idea lands** — correctly
  understood, it is not a second stream, it is the frozen-encoder probe done
  properly.
- **Rung 2 (escape hatch, §2.3) — raw-pixel trainable branch + cross-attention.**
- **Rung 3 — LoRA / partial unfreeze.** Last resort: directly spends the domain
  robustness the freeze was buying.

**Job split** the ladder respects: *Job A — dense decoding capacity*
(depth/normals/seg) is a head problem (Rung 1 DPT decoder), never a reason for a
backbone. *Job B — supplying information the encoder lacks* (the physics) is the
**only** thing that could justify Rung 2, and only after Rung 1 is shown to hit a
ceiling on a signal Rung 0 confirmed exists.

Caveat that survives every rung: **R2** — no architecture earns a real-domain
physics label. Frozen-first doesn't fix R2; it avoids *also* paying the R1
sim-overfit tax while that quantity is unmeasurable anyway.

### 2.3 Alternative / escape hatch: the raw-pixel two-stream (the PDF design)

Two parallel streams off the same RGB: a frozen semantic stream (JEPA/DINO/SigLIP)
and a **trainable convolutional geometry stream from raw pixels**, fused by
**cross-attention**, heads on the fused representation ("embedding over
embedding"). Here V-JEPA is reframed as a frozen domain prior and the conv branch
does the geometric heavy lifting from pixels.

This is the PDF's design and the highest-capacity option — and the most
transfer-fragile. It is the **only** path that can recover physics signal the
frozen encoder discarded (it sees full-information pixels), so it is the right
fallback *if and only if* Rung 0 shows the signal is **absent from the frozen
encoder at every depth**. Conditions on entering it:
- Rung 0 found no recoverable signal anywhere in the frozen features, **and** Rung
  1 confirmed the ceiling is the information bottleneck, not head capacity;
- **R1's load-bearing ablation is enforced and measured on the *real* side** (not
  sim) — prove the frozen stream is doing work, or the domain bridge is gone;
- the domain-gap work (Section 5) is aimed primarily at regularizing *this* branch.

Note: feeding V-JEPA features (not pixels) into this "second stream" does **not**
make it a safer Rung 2 — it collapses back into Rung 1 (a head over frozen
features), because both streams then derive from the same bottlenecked source.
Rung 2's only justification is full-information pixels; take that away and there
is no second stream.

### Open architecture questions
- **Temporal handling.** ω / contact are dynamic. At the main line (Rungs 0–1)
  this is "frozen *video* encoder (V-JEPA, native temporal) vs. per-frame image
  encoder + a temporal head over the per-frame features." Decide explicitly.
- **Cross-attention direction** (only relevant if the §2.3 escape hatch is
  entered). Geometry-as-query (semantics modulates geometry) vs. the reverse;
  geometry-as-query is the natural default, but be deliberate.

---

## 3. Prediction targets

- **Sim + real (dense):** depth, normals, segmentation. Labelable in both domains
  (sim GT; real via sensors/pseudo-labels). These are the *transfer bridge*.
- **Sim only (physics):** angular velocity, center of mass, contact forces,
  distance-to-ground. Exact in sim; no real GT.

Ground-truth note: **Isaac/PhysX always knows the exact CoM and inertia tensor**
— they are mandatory inputs to rigid-body dynamics. Non-uniform density is not
"uncomputable"; it has to be *created* deliberately (a multi-density compound
body, or authored `MassAPI`), and then PhysX combines/derives it exactly. So
ground-truth labels for the physics heads always exist and are exact; the open
question is purely *recoverability from video*.

---

## 4. Center-of-mass: the well-posed formulation

Distinctions that matter:
- **First moment (CoM location)** — *where* the mass center sits. Offset from the
  geometric centroid = density asymmetry. Observable mainly via **contact/tipping**.
- **Second moment (inertia / radius of gyration)** — *how spread out* the mass is.
  Governs precession/tumbling. Observable mainly via **free-flight tumbling**.
  This is what "entropy / concentration of mass" actually maps to — distinct from
  CoM offset.

Recommended formulation:
1. **Predict the offset, not absolute CoM** (`CoM − centroid`), and score **lift
   over the centroid baseline** (R7). Centroid is computable from the silhouette
   and needs no physics; the lift is the encoded physics.
2. **Match target granularity to available signal.** Exact 3D offset may be
   under-determined from 16 frames; detection (off-center yes/no), magnitude
   (‖offset‖), or ordinal bins may be the well-posed task — subject to R8.
3. **Probabilistic head (the "sharp vs. wide Gaussian" idea).** The probe outputs
   a *distribution* over CoM, trained with Gaussian NLL
   (`½(‖μ−CoM‖²/σ² + log σ²)`), so widths *emerge*: sharp for well-determined
   (uniform) objects, wide for under-determined (complex) ones. The predictive
   **entropy is a readout of CoM identifiability from the observed dynamics.**
   - Use **full covariance**, not isotropic — identifiability is anisotropic
     (e.g. well-constrained perpendicular to the spin axis, poorly along it).
   - Applied to the *offset* distribution: uniform objects collapse to a spike at
     0; complex objects spread; the entropy is a single scalar "deviation from
     trivial, as far as the model can tell."
   - Caveat R9: separate identifiability from capacity with a reference model.

Dataset implication: to make CoM/mass learnable at all, scenarios must make them
**dynamically visible** — non-uniform compound bodies (real, exact offset) plus
tumbling + contact-rich motion. Logging the label is necessary but not
sufficient (R6).

---

## 5. Sim→real domain-gap strategy

Goal: reduce the appearance gap so the trainable geometry branch doesn't overfit
sim, while preserving the exact physics labels.

Recommended **sequencing** (label-exact first, generative last):
1. **Domain randomization** (textures, materials, lighting, HDRI, camera) —
   preserves labels perfectly because we control the renderer. Proven sim2real
   workhorse. First line.
2. **Higher in-sim render fidelity** (RTX path tracing, better PBR/HDRI) — more
   realism, zero label risk.
3. **Cosmos-Transfer as an augmentation layer on top** — high ceiling, high risk.
   Structure-conditioned (depth/seg/edge) photorealization; it is the intended
   tool for sim2real, *not* naive. But subject to R4/R5/R10:
   - gate every generated clip with a **registration / label-consistency check**;
   - treat output as a **third domain** mixed in, never a replacement for sim;
   - its real job here is **regularizing the geometry branch**, since the frozen
     semantic stream is already fairly domain-robust.

---

## 6. What is and isn't verifiable (critique #4 scope)

- **Verifiable with airtight closed form:** ω sign/scale and quaternion
  composition via a **uniform cuboid, edge-axis (principal-axis) spin, 45°
  launch** — gravity produces zero torque about the CoM, so rotation decouples
  from the parabola; ω stays constant in world frame and orientation rotates at
  constant rate. The asymmetric-CoM path via an **authored-offset CoM body**
  (`pos + R_body_world @ offset` is closed-form). Plus: run at the *production*
  camera pose, and a two-body scene for index-swap.
- **Out of scope (no clean ground truth):** general tumbling (Euler-equation
  precession; intermediate-axis instability makes any numerical reference
  diverge) and PhysX-computed CoM on real meshes (consistency-checkable only).

These verify the *logger*, which is all #4 asks. They do not — and need not —
verify dynamics fidelity or learnability.

---

## 7. Open decisions

- **Capacity ladder (gating decision, §2).** Start frozen-and-shallow (Rung 0/1)
  and let measured ceilings — not the PDF — decide whether the trainable geometry
  backbone (Rung 2) is earned. Begins with the frozen-encoder comparison
  (V-JEPA vs DINO vs SigLIP), which is cheap and answers the core premise.
- Probe formulation: **state read-out** (regress current ω/CoM) vs. **future
  prediction** (forecast next frames). Read-out is better-posed for "does it
  encode physics" (exact, interpretable labels); forecasting re-does V-JEPA's
  native task and conflates encoding with dynamics modeling.
- The **reference upper-bound** model (R9) — its choice determines what claim the
  whole experiment can support. Decide before running.
- Scenario design for **dynamic observability** of CoM/mass (R6) — compound-body
  construction + contact-rich motion, with visual control (R8).
- Temporal design and cross-attention direction (Section 2).
- DR-vs-Cosmos mix ratio and the registration-gate definition (Section 5).
