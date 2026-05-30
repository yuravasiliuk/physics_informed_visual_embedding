# Physics-Informed Embedding — Brainstorm Summary

*Date: 2026-05-30. Design notes, not a spec. Captures the discussion around the
architecture sketched in `Physics informed embedding.pdf`, the physics-label
targets, a probabilistic CoM formulation, and the sim→real strategy.*

The architecture in one line: **a frozen semantic encoder (JEPA/DINO/SigLIP) +
a trainable convolutional geometry encoder, fused by cross-attention, with
prediction heads split by supervision availability.** RGB-only input, real or
sim. Heads: *sim+real* (depth, normals, segmentation) and *sim-only* (angular
velocity, center of mass, contact forces, distance-to-ground).

Risks are listed first deliberately — most of them are load-bearing and several
can silently invalidate results.

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
not just the labels.

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

Two parallel streams off the same RGB (image or video, real or sim):

1. **Frozen semantic stream** — JEPA / DINO / SigLIP, weights frozen → semantic
   patches. Generalist, domain-robust, never trained.
2. **Trainable geometry stream** — convolutional geometry encoder → geometric
   patches. Learned from pixels.

**Cross-attention** fuses them; prediction heads hang off the fused
representation ("embedding over embedding").

The reframe that matters: **V-JEPA is not the thing being probed for physics.**
It is a frozen semantic / domain prior. The geometry branch does the geometric
heavy lifting from pixels. So the question is not "does V-JEPA encode CoM" but
**"does a frozen generalist semantic stream help a trainable geometry encoder
predict — and transfer — physics it couldn't get from geometry alone?"**

Per-stream domain sensitivity (why the design is shaped this way):
- the frozen semantic encoder already generalizes sim→real for *semantics*;
- the trainable geometry conv **will** overfit sim appearance — so the domain-gap
  work (Section 5) exists primarily to regularize *this* branch.

Open architecture questions:
- **Temporal handling.** ω / contact are dynamic → either the geometry conv is
  3D/temporal, or it is per-frame and the cross-attention/heads integrate time.
  Decide explicitly.
- **Cross-attention direction.** Geometry-as-query (semantics modulates geometry)
  vs. the reverse. Geometry-as-query is the natural default for
  physics-from-geometry-with-semantic-priors, but be deliberate.

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
