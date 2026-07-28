# Supervisor Instructions — Physics Probes and World-Lines

**Audience:** junior research intern  
**Status:** mandatory research-direction correction before further world-lines model work  
**Scope:** the physics-informed embedding brainstorm, GATE B probe, sim→real claims, and
`src/worldlines/`

## 1. Supervisor verdict

The overall research direction is not rejected, but its claims must be narrowed.

The **frozen-encoder physics probe** is a well-posed and potentially useful research
project: measure which physical quantities are recoverable from each layer of a video
foundation model, under controlled scenarios and leakage-safe splits.

The stronger claims are not yet established:

- marginal sim/real feature alignment does not guarantee policy transfer;
- monocular video does not uniquely determine a canonical allocentric scene state;
- predicting world-line coefficients from simulator ground truth is not evidence that an
RGB model can infer them;
- the current `src/worldlines/` round-trip verifies trajectory compression, not the
world-lines representation thesis.

Treat world-lines as an **experimental Tier-2 target representation**, not as foundational
truth and not as the main project until the Tier-1 probe produces credible results.

## 2. Primary assignment

Your first responsibility is to complete the Tier-1 frozen-encoder probe described in
[probe-walkthrough.md](../probe/probe-walkthrough.md) and
[gate-b-probe.md](../probe/gate-b-probe.md).

The question is:

> Which physical quantities are recoverable from frozen video-model features, at which
> layers, with what probe capacity, and under which observability conditions?

This is the main deliverable. Do not begin an object-query world-lines model, contact
residual, Neural ODE, or sim→real policy experiment before this gate has a defensible
answer.

Required probe order:

1. Validate the harness using linear velocity `v`.
2. Measure angular velocity `ω`.
3. Measure CoM offset, never absolute CoM.
4. Report every target by encoder layer.
5. Report CoM separately for free-flight and contact-rich scenarios.
6. Compare linear and shallow nonlinear probes using identical cached features and splits.
7. Compare V-JEPA 2 with V-JEPA 2.1 before interpreting a local-geometry null result.

Split by physics rollout, never by rendered camera clip. Multiview siblings in both train
and test invalidate the result.

## 3. Clarify what is being trained

The current documents sometimes imply that auxiliary physics heads will make a frozen
encoder's embedding more physics-aware. This is impossible if the encoder is frozen and
the exported embedding is its unchanged output.

Before any B-vs-F experiment, write down the exact exported representation:

- If it is the raw frozen V-JEPA feature, physics heads are **probes only** and cannot
  reshape it.
- If it is the output of a shared trainable adapter or fusion module, all heads must train
  that shared module, and the adapted feature must be the representation evaluated
  downstream.
- If each head has an independent readout, F-vs-B cannot demonstrate representation
  improvement; it only compares readout tasks.

Do not use the phrase “physics-supervised embedding” until a shared trainable
representation is explicitly defined and exported.

## 4. Restrict the sim→real claim

The domain-marginalization plan in
[domain-marginalization-regularizer.md](../theory/domain-marginalization-regularizer.md)
is a hypothesis, not a guarantee.

DANN, CORAL, MMD, and domain-classifier accuracy constrain feature marginals. They do not
by themselves guarantee:

- class-conditional or state-conditional alignment;
- preservation of task-relevant distinctions;
- transfer of sim-only physical quantities;
- transfer of a policy trained on simulation embeddings.

Therefore:

- describe DANN/CORAL as **alignment baselines**, not as “the forcing function that makes
  transfer hold”;
- never treat 50–65% domain-classifier accuracy as evidence of successful transfer;
- require downstream task performance or paired/state-conditioned evaluation;
- include an unregularized F baseline to establish whether the supposed shortcut occurs;
- do not claim real-side transfer of CoM, mass, restitution, or contact without a
  measurable real-side test.

The load-bearing evaluation remains a downstream sim→real task. Until that exists, use
“sim-domain physics decodability” rather than “domain-agnostic physics embedding.”

## 5. Reframe world-lines

Use the following conservative description:

> A compact, gravity-aligned representation of per-object and camera trajectories, with
> explicit residual gauge and observability assumptions.

Do not currently describe it as:

- a uniquely recoverable canonical world state;
- camera-invariant without qualification;
- required for object permanence;
- metric from monocular video in general;
- fully observer-invariant;
- verified from RGB.

Object permanence does not require allocentric coordinates. Persistent egocentric slots,
scene graphs, and object-centric memory can also represent an occluded object. Allocentric
world-lines are one useful inductive bias, not the only possible ontology.

Monocular separation of ego-motion, object motion, depth, scale, yaw, and origin is
underidentified without anchors. State the required anchors in every experiment:

- a usable static background;
- calibrated or estimated camera intrinsics;
- gravity direction;
- a ground or horizontal reference;
- sufficient parallax or physical acceleration for scale;
- visibility and object-association supervision.

When an anchor is missing, predict an equivalence class, uncertainty distribution, or
up-to-scale state instead of forcing a single ground-truth answer.

## 6. Required `src/worldlines/` corrections

Do not call Phase E0 verified until the following are addressed.

### 6.1 Gauge

The design says the gauge must not depend on object composition, but
[`build_gauge`](../../src/worldlines/gauge.py) currently uses the mean frame-0 body
position as its origin. Adding or removing an object therefore changes every other
object's coefficients.

Replace this with an explicitly defined anchor:

- ground-plane origin plus horizontal reference for the intended representation; or
- first-camera frame as a clearly named debugging baseline.

Implement and test yaw pinning. Store whether scale and yaw are known rather than silently
pretending every clip has a complete gauge.

### 6.2 Orientation conventions

[`to_gauge`](../../src/worldlines/gauge.py) transforms positions and vector rates but
passes body quaternions through unchanged. Body orientation must be represented in the
same declared frame as body position.

The ego-line design uses camera-to-world `T_WC`, while
[`encode_clip`](../../src/worldlines/encode.py) currently fits quaternions derived from
world-to-camera `R_cw`. Choose one convention, document it once, and enforce it in the
loader, encoder, decoder, projection functions, and tests.

### 6.3 General SO(3) trajectories

[`fit_rotations`](../../src/worldlines/basis.py) sums frame-to-frame rotation vectors.
This is exact only for restricted commuting or constant-axis motion. General rotations do
not commute.

A review stress test using alternating-axis increments produced approximately `0.321 rad`
maximum reconstruction error even though the fitted tangent curve reported numerical-zero
error. Replace the representation with a mathematically consistent method, for example:

- fit per-frame relative rotations with geodesic loss;
- use cumulative quaternion integration;
- use a Lie-group spline;
- or limit the representation explicitly to constant-axis motion.

The reported fit error must measure reconstructed orientation geodesic error, not only
error in the summed tangent surrogate.

### 6.4 Visibility and permanence

[`load_clip`](../../src/worldlines/io.py) currently marks every body visible at every
frame. This means no current round-trip tests object permanence or occlusion.

Read visibility from segmentation presence, retain bodies through invisible intervals,
and add synthetic occlusion tests. Distinguish:

- object existence;
- image visibility;
- tracker confidence;
- no-object padding.

These are not the same variable.

### 6.5 Composite-body intrinsics

The loader reads `com_local_authored` but composite bodies use
`com_local_analytic` and later `com_local_measured`. The current encoding therefore emits a
zero CoM-offset intrinsic for composite bodies.

Define a precedence rule:

1. measured post-reset value;
2. analytic composite value;
3. authored primitive value;
4. missing/unknown.

Apply the same rule to mass and other intrinsic labels.

### 6.6 Round-trip verification

[`reproject_error`](../../src/worldlines/project.py) reconstructs object positions but
projects them with the ground-truth camera trajectory. It therefore does not validate the
encoded ego-line.

Add separate tests for:

- object trajectory reconstruction;
- body orientation reconstruction;
- ego position reconstruction;
- ego orientation reconstruction;
- reprojection using **decoded object and decoded ego trajectories**;
- gauge invariance under a global scene transform;
- composition invariance when an unrelated object is added or removed;
- non-polynomial contact and drag motion;
- occlusion intervals;
- arbitrary non-commuting rotation.

The existing `~1e-16` result on ballistic translation and constant-axis spin is only a
sanity check for an easy analytic case.

## 7. Observability discipline

Invariant physical quantities are not automatically observable from video.

For each target, document:

1. the visual/dynamic evidence that can reveal it;
2. the transformations under which it is invariant or covariant;
3. the residual ambiguity after observing the clip;
4. the scenarios that contain sufficient evidence;
5. the trivial baseline;
6. the uncertainty or abstention behavior when evidence is absent.

Examples:

- `v`: visually observable from tracked translation, but metric scale may be unknown.
- `ω`: observable only when shape, texture, or geometry reveals orientation change.
- CoM offset: generally needs contact, tipping, or coupled rotational dynamics.
- mass ratio: requires known interaction forces or collision structure; free fall gives no
  mass information.
- restitution: requires a sufficiently observed impact and pre/post-contact velocity.
- density: generally not identifiable independently from geometry and mass.

Do not supervise an exact scalar on an unobservable clip without an ignore mask,
equivalence-class target, or calibrated uncertainty objective.

## 8. Contact scope

Do not begin with exact contact-force regression. Replace it initially with measurable
surrogates:

- contact/no-contact state;
- contact-event timing;
- collision partner;
- change in velocity or angular velocity across the event;
- impulse per unit mass where scale permits.

Exact force depends on collision geometry, material parameters, temporal resolution, and
simulator solver details. It is not a clean RGB target for the first experiment.

For world-lines, start with piecewise trajectories or event-aligned residual evaluation.
A globally smooth low-degree polynomial is least accurate precisely around the event that
contains the useful physical evidence.

## 9. Required deliverables

Submit work in this order.

### Deliverable A — probe report

- per-layer `v`, `ω`, and CoM-offset results;
- train/test and scenario-stratified metrics;
- trivial baselines;
- V-JEPA 2 versus V-JEPA 2.1;
- linear versus shallow nonlinear head;
- register-collapse diagnostic;
- exact split manifest proving no multiview leakage.

### Deliverable B — representation contract

A short document defining:

- the exported embedding;
- which parameters are frozen or trainable;
- which losses update the exported representation;
- frames, units, gauge, and uncertainty semantics;
- what can and cannot be claimed from each experiment.

### Deliverable C — corrected E0 tests

Automated tests covering the items in §6.6. Report real reconstruction errors for
position, orientation, ego-motion, and decoded reprojection.

### Deliverable D — world-lines ablation

Only after A–C:

- camera-frame per-object trajectory baseline;
- allocentric trajectory with ground-truth camera pose;
- allocentric trajectory with predicted camera pose;
- per-frame state versus basis coefficients;
- smooth basis versus contact-aware representation;
- comparison against a classical tracker/SfM baseline.

The allocentric representation earns further work only if it improves robustness to
camera motion, occlusion, or downstream prediction over these simpler baselines.

## 10. Stop conditions

Stop or substantially reframe the world-lines track if:

- a classical tracker plus camera-pose estimator matches it on the intended data;
- decoded world-line error grows with camera motion as much as the camera-frame baseline;
- performance depends on simulator ground-truth camera pose unavailable at deployment;
- gauge estimates change materially when unrelated objects are added;
- CoM or mass predictions succeed from static appearance cues;
- Tier-1 frozen features contain no usable signal after controlling for encoder and probe
  confounds;
- the representation adds complexity without improving any downstream or diagnostic
  metric.

Stopping under these conditions is a correct research result, not a failure to complete
the original plan.

## 11. Reporting language

Use precise language:

- “recoverable from frozen features on held-out simulated rollouts”;
- “gravity-aligned up to residual yaw/scale ambiguity”;
- “trajectory coefficient encoding”;
- “alignment baseline”;
- “hypothesis for sim→real transfer”;
- “contact-rich subset.”

Avoid:

- “understands physics”;
- “camera-invariant by construction” without stating the gauge and pose source;
- “domain-agnostic” before downstream transfer;
- “verified world-lines” when only GT encode/decode was tested;
- “mass is recoverable” without identifying the interaction that makes it observable;
- “object permanence requires allocentric coordinates.”

## 12. Immediate next action

Do not extend `src/worldlines/` yet.

First produce the Tier-1 probe report and a one-page representation contract. After those
are reviewed, correct the E0 gauge/orientation/visibility tests and decide whether
world-lines remain justified as Tier 2.
