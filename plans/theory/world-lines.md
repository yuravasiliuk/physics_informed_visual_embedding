# World-Lines — an allocentric, differentiable representation of scene dynamics

**Status:** brainstorm → design draft, pre-experiment. Written to be picked apart and
to be precise enough that careful code can be written from it. Sections marked
**[Honest take]** are deliberately critical — argue with them.

**Lineage.** This is the geometric framing of the equivalence-class / latent-dynamics
idea in [physics-latent-ladder.md](physics-latent-ladder.md) (Path B). It does **not**
supersede the committed main line ([../BRAINSTORM.md §2](../BRAINSTORM.md#2-architecture):
frozen encoder + probe). It is a *target representation* that could sit on top of the
frozen encoder as the F-config head, or become a separate research track. Read
[../BRAINSTORM.md](../BRAINSTORM.md) (R6/R7/R8, the capacity ladder),
[theoretical-paper-plan.md §3/§5](theoretical-paper-plan.md) (equivariance, conservation),
and [../../summaries/intphys2.md](../../summaries/intphys2.md) first.

---

## 0. One-paragraph statement

Represent a clip not as per-frame, camera-centric labels, but as a small set of
**world-lines** — continuous per-object trajectories in an **allocentric, gravity-pinned
world frame** — plus the camera's own world-line (the **ego-line**). The camera is not a
privileged frame; it is just another world-line that *observes* the others. The
camera-frame (observed) motion is a *derived* quantity, never the representation. Each
world-line is encoded differentiably as coefficients of a fixed temporal basis, factored
into **observer-invariant intrinsics** (mass, CoM-offset magnitude, restitution) and
**covariant kinematics** (position/orientation trajectory). We supervise the *structured
state* (ego-line + object world-lines + intrinsics), **not** reconstructed pixels.

---

## 1. Why allocentric world-lines (the motivation, not the elegance)

Three independent arguments converge on the same representation. None is "it's pretty."

### 1.1 Permanence *requires* an allocentric frame
In a camera/egocentric frame, an occluded object has **no representation** — it has left
the view, so it has left the state. In a world frame, the object's world-line simply
*continues behind the occluder*; the camera merely stops sampling it. Object permanence
— the single biggest model-vs-human gap on
[IntPhys 2](../../summaries/intphys2.md) (humans ~96 %, predictive models ~52–58 %, near
chance) — is **only expressible** once the representation is allocentric. This is the
load-bearing reason; the rest is corollary.

### 1.2 Camera motion: from "covariant" to "removed"
IntPhys 2 shows moving cameras cost models several points
([../../summaries/intphys2.md](../../summaries/intphys2.md)). A camera-frame descriptor
can at best be made *covariant* (transforms correctly under camera change). Estimating the
ego-line explicitly and factoring it out makes object world-lines **camera-invariant by
construction** — `X_world(t)` does not depend on how the camera moved. Camera motion then
appears exactly **once**, in the shared ego-line, instead of contaminating every object.

### 1.3 It is the ontology simulator-supervision is supposed to inject
[../../summaries/opinion-intuitive-physics.md](../../summaries/opinion-intuitive-physics.md):
scaling video alone does not install "the world is made of persistent objects." The
world-line target *is* that ontology, made differentiable. Per-object world-trajectory
supervision is the strongest version of the project thesis: not "add physics labels to
V-JEPA," but "use the simulator to install the persistent-object inductive bias the
benchmark proves is missing."

**[Honest take]** This overlaps heavily with object-centric world models (SAVi/SIMONe/
OSRT-style), dynamic monocular SLAM, and 4D scene reconstruction. None of those three
properties above is novel in isolation. The only defensible contribution is the *specific
combination*: structured world-line state as the **supervision target for a frozen-encoder
probe aimed at sim→real physics transfer**, with the invariant⊕covariant factorization and
the physics intrinsics. If a reviewer can point at one paper doing exactly that, this is a
diagnostic track, not a headline.

---

## 2. Frames, gauge, and notation (get this exactly right or nothing converges)

### 2.1 Frames
- **World frame `W`** — allocentric, **gravity-aligned, ground-anchored** (gauge fixed in
  §3). This is where world-lines live.
- **Camera frame `C(t)`** — the observer at time `t`. Pose `T_WC(t) ∈ SE(3)` maps
  camera→world. The **ego-line** is the trajectory `{T_WC(t)}_{t=1..K}`.
- Object `i` world pose: position `p_i^W(t) ∈ ℝ³`, orientation `R_i^W(t) ∈ SO(3)`.

### 2.2 The one identity everything hangs on
The **observed** (camera-frame) object position is *derived*, not predicted directly:

```
p_i^C(t) = T_WC(t)^{-1} · p_i^W(t)  =  T_CW(t) · p_i^W(t)
```

and (optionally) projected to the image with intrinsics `K`:
`u_i(t) = π(K · p_i^C(t))`. The model predicts `p_i^W` and `T_WC`; `p_i^C` and `u_i`
fall out by composition. This is what makes camera motion factor cleanly.

### 2.4 Units: dimensionless, gravity-natural (decided)
World-lines are **not** stored in meters-seconds. Monocular vision is metric only up to
the **similarity group** (scale × rotation × translation), so the native representation is
similarity-invariant, with metric as an *optional overlay* recovered from a
recognized-object size prior. We fix the residual gauge with the one force that breaks
scale-invariance — **gravity** — but only as far as gravity is *observable* (see direction
vs. magnitude below). Time is always available (fps), normalized to clip duration
(`τ ∈ [0,1]`).
- Gravity breaks the similarity group's **scale** symmetry exactly as it breaks the
  **rotational** symmetry to the yaw subgroup (`stab(g)_world`,
  [theoretical-paper-plan.md §3](theoretical-paper-plan.md)). Same phenomenon, two axes.
- **Direction vs. magnitude — they decouple, and this is the load-bearing distinction.**
  Gravity *direction* (the up axis) is observable in nearly **any** scene (ground, normals,
  resting/stacking) and fixes the **rotational** gauge. Gravity *magnitude* (the length
  scale) needs observable **acceleration** — a fall, topple, bounce, or roll — and is
  genuinely **absent in quasi-static scenes** (a resting object is in static equilibrium;
  nothing accelerates, so it shows *which way* is down but **nothing about scale**).
- **The length gauge is therefore tiered:**
  1. gravitational-dynamics cue present → **gravity-natural unit** (`‖g‖ ≡ 1`), which *is*
     metric-anchored (Earth `g`); lengths in "how far a body falls per clip."
  2. else → a stable **scene-geometric unit** (median depth / ground-plane scale from the
     depth map — does not depend on object count), **scale-free**, flagged
     `scale_known=False`.
  3. optional upgrade → metric via a **recognized-object size** prior (the human cue).
- **Read inter-object relations *out* of the gauge; do not build the gauge *from* the
  objects** (largest object / mean spacing rescales when an object is added/removed —
  composition-dependent, poison for learning). Both `‖g‖` and median depth are
  object-count-independent → stable.
- **It costs nothing for our targets**, which are scale-free: `ω` (time metric),
  restitution, CoM-offset/size (R7), mass **ratios** (vision never gives absolute mass —
  only ratios via collision momentum exchange), trajectory classes (`J/m`, `c/m`). Absolute
  metric scale is unknown in a no-cue scene, but those quantities have no real GT anyway
  (R2). **The no-cue case is a real-footage concern, not a sim one:** our scenarios
  (drop / stack-collapse / slope-roll / topple / domino) contain gravitational acceleration
  by design (R6 observability), so tier 1 holds; only `cluttered-surface`/resting falls back
  to tier 2.

The encoder ([world-lines-encoding.md](world-lines-encoding.md)) keeps the metric
world-frame arrays as ground truth and applies an **invertible** gauge transform, so the
scheme is a tweakable, reversible step — not baked in.

### 2.3 Data already speaks this language
The data-gen pipeline **logs state in world frame** and keeps `R_camera_world`
recoverable (the "raw log stays world-frame; camera-frame conversion happens offline at
dataloader time" invariant — see the Data-gen invariants in [../../CLAUDE.md](../../CLAUDE.md)
and [../data-gen/synthetic-data-collection.md](../data-gen/synthetic-data-collection.md)). So
**GT object world-lines and the GT ego-line both already exist**. We are not inventing new
labels; we are choosing to supervise the world-frame decomposition the sim already records.

---

## 3. Pinning the world frame (the gauge problem — §2's "still wondering how")

The world origin/orientation is arbitrary; **only relative quantities are observable**, so
an unconstrained `W` makes the loss degenerate (a global `SE(3)`×scale can be added to all
world-lines and the ego-line with zero cost). This is the literal diffeomorphism-gauge
freedom of the GR analogy. It **must** be pinned.

**Humans pin it sub-consciously** (gravity from the vestibular system, ground/horizon from
vision) and never "think about it" — but a trainable setup has no vestibular prior, so the
gauge has to be supplied. The good news: the cues humans use are exactly the signals we
already have.

**Recommended gauge `G` — gravity + ground + horizontal reference:**
1. **Up axis = gravity.** Fixes 2 of 3 rotational DoF. Estimable from **normal maps** (the
   dominant ground-normal) and, in sim, known exactly. On real: from surface normals /
   vanishing points (or an IMU if the corpus has one).
2. **Origin + metric scale = ground plane.** The ground plane (from **depth + normals**,
   which we already produce as dense heads) gives the origin height and, with known camera
   height/intrinsics, metric length. `‖g‖` gives a metric *time* scale.
3. **Yaw (last rotational DoF) = a horizontal reference.** Use the dominant horizontal
   scene axis — the **grid-line ground texture** (data-gen axis D,
   [../IMPLEMENTATION.md](../IMPLEMENTATION.md)) is literally a yaw reference — or, lacking
   that, the `t=0` camera forward projected onto the ground.

**Why this is the answer and not a hack:** every component is recoverable from signals the
pipeline already emits (depth, normals, grid ground), so the gauge transfers sim→real on
the *same* dense heads that are already the transfer bridge. In sim, supervise `G` against
GT; at inference, *estimate* `G` from depth+normals. **[Honest take]** Scale is the weak
leg: monocular metric scale on real data needs camera height or intrinsics; without them,
world-lines are recoverable only up to scale. Document the claim as scale-up-to-prior.

**Rejected alternatives:**
- *Learned canonicalization head* (predict `G` freely) — collapse-prone, and it re-opens
  the gauge it was meant to close. Only as a refinement on top of the gravity+ground gauge.
- *`W = C(0)` (first-frame camera)* — removes ambiguity but makes "up" clip-dependent and
  throws away the gravity prior that gives permanence its anchor. Acceptable only as a
  debugging baseline.

---

## 4. Differentiable encoding (the architecture)

Extends the object-query design from the prior discussion to world frame.

### 4.1 Backbone (unchanged main line)
Frozen encoder (V-JEPA 2) over the K-frame clip → spatiotemporal feature volume `F`. No
raw-pixel branch (that is the escape hatch, [../BRAINSTORM.md §2.3](../BRAINSTORM.md#23-alternative--escape-hatch-the-raw-pixel-two-stream-the-pdf-design)).
Which block(s) `F` taps is decided by **GATE B** ([../IMPLEMENTATION.md](../IMPLEMENTATION.md)).

### 4.2 Object queries → world-lines (DETR in space-*time*)
`N` learned queries `q_i` cross-attend `F` (over all K frames at once) → tokens `h_i`. One
query = one world-line. `N` is fixed (max objects) with a **no-object logit** for padding —
this is how variable object count stays differentiable. Each `h_i` emits:

```
h_i → [ z_inv_i ,  c_i^pos ,  c_i^rot ,  e_i(t) ,  δ_i(t) ,  occ_i ]
```

- `z_inv_i` — **invariant intrinsics**: mass, CoM-offset *magnitude*, restitution, shape
  scalars. Observer-independent. (Predict CoM *offset*, score lift over centroid — R7.)
- `c_i^pos` — **covariant** position-trajectory coefficients in **world frame**:
  `p_i^W(t) = B(t) · c_i^pos` for a fixed temporal basis `B` (§4.4).
- `c_i^rot` — orientation-trajectory coefficients on the Lie algebra `so(3)`:
  `R_i^W(t) = R_i^W(0) · exp([B(t) · c_i^rot]_×)`. Gives ω as `d/dt` of the curve.
- `e_i(t)` — **existence/visibility** logit over time (sigmoid). The world-line persists in
  latent space through occlusion even when `e_i(t)→0`. **This is the permanence channel.**
- `δ_i(t)` — **contact residual** (§5), the non-smooth correction to the basis curve.
- `occ_i` — the no-object logit.

### 4.3 Ego-line head
A separate head reads `F` → camera world-trajectory on `se(3)`:
`T_WC(t) = T_WC(0) · exp([B(t) · c_ego]_×)`, with `T_WC(0)` *fixed by the gauge `G`* (§3),
not free. The ego-line is **shared across all objects** — a strong, parameter-cheap
constraint (every object sees the same ego-motion), which is exactly what helps on a small,
low-variety corpus.

### 4.4 The temporal basis (the "compact formula")
`B(t)` is a fixed, differentiable basis matrix — **B-spline or Chebyshev, degree 2–4**.
The coefficients `c` *are* the compact numerical form / "formula" that started this thread.
Properties bought for free: resolution-independence (sample `t` anywhere), a built-in
smoothness prior (good for small data), and interpretability (degree-2 part ≈ ballistic;
the residual above degree 2 carries drag/contact/asymmetry signal). Implement as a fixed
matrix; the head predicts only `c`.

### 4.5 Invariant⊕covariant, built in (not hoped for)
Covariance is a *correctness criterion*, not a synonym for "world-line." Enforce it
structurally: predict kinematics in the **canonical world frame**, then map to the observed
frame by the *known* `T_CW(t)`. The head never has to discover `SE(3)` — it is handed the
rotation. This is the cheapest route to the moving-camera robustness of §1.2 and matches
the minimum equivariance group in
[theoretical-paper-plan.md §3](theoretical-paper-plan.md) (`SE(3)_camera × stab(g)_world`).

---

## 5. Contact — the one part that resists a compact form

**[Honest take]** A smooth basis (§4.4) *blurs* contact: a hard bounce is a near-velocity
discontinuity, and a degree-4 spline rounds it off. Contact is precisely the signal we care
about (R6; [RoboScape](../../summaries/roboscape.md)'s keypoint-dynamics head carried most
of its gain). So the compact world-line is best in the free-flight regime (already easy) and
worst at the contact event (where the physics is). This is the recurring theme of the whole
project, localized to one term.

Options, ranked by trust on a ~50 GB low-variety corpus:
1. **Smooth base curve `+` learned residual `δ_i(t)`** (small MLP `t, h_i → ℝ³`), with an
   **L1/sparsity prior** so `δ_i` is ~0 except near contact. Keeps the interpretable
   ballistic part; lets the residual absorb the kink. *Start here.*
2. **Piecewise basis with learnable knots** — knot *locations* are ~non-differentiable;
   needs a soft/relaxed knot or a fixed dense knot grid. More capacity, fiddlier.
3. **Full Neural-ODE world-line** (token = `z₀`, learned vector field, integrate via
   adjoint). Represents contact naturally; data-hungry and is the Path-B research bet the
   ladder doc flags as week-scale on toys. Do **not** lead with it.

The contact term, not the smooth part, is what **GATE A should stress-test** (§8).

---

## 6. Losses (differentiable ledger)

Set-level supervision against the sim's world-frame GT.

- **Set matching.** Hungarian (`scipy.linear_sum_assignment`) between predicted world-lines
  and GT bodies on a cost mixing world-trajectory L2 + existence. Matching itself is
  discrete → compute it with current predictions (no gradient through the assignment),
  backprop only through matched pairs (DETR convention). Sinkhorn/OT is the soft alternative;
  start with DETR-style.
- `L_world = Σ_{i,t} ‖ p_i^W(t) − p_i^{W,GT}(t) ‖²` (matched, covariant kinematics).
- `L_rot` — geodesic `SO(3)` distance on `R_i^W(t)`; gives ω supervision.
- `L_ego = Σ_t d_SE(3)( T_WC(t), T_WC^{GT}(t) )` — ego-line geodesic loss.
- `L_proj` — consistency of the **derived** `p_i^C(t)` (or 2D reprojection `u_i(t)`) with GT
  camera-frame state. Couples the object and ego heads; partially **self-supervised** (the
  two heads must agree on what the camera sees) even without per-object labels.
- `L_inv` — intrinsics vs GT (mass, CoM-offset, restitution); CoM as offset, scored as lift
  over centroid (R7).
- `L_exist = Σ_{i,t} BCE( e_i(t), visible_i^{GT}(t) )` — permanence channel.
- `L_dense` — depth / normals / segmentation dense heads (the sim+real bridge **and** the
  gauge estimator of §3). These do double duty.
- `L_smooth` — curvature penalty on `p_i^W`, `T_WC`; `L1` sparsity on `δ_i` (§5).

### 6.1 R6/R8 cleanliness (do not skip)
Feed `z_inv_i` **only dynamics-derived features** (trajectory curvature, the contact
residual `δ_i`), **never** the raw appearance token. Otherwise the invariant block silently
becomes an appearance classifier — the R8 failure — and "mass" degenerates into "what does
it look like." This is the differentiable instantiation of "CoM/mass observable only through
dynamics" (R6). One uniform material per body upstream (R8) is assumed.

---

## 7. Disentanglement: why it is solvable here despite being ill-posed in general

**[Honest take]** Monocular separation of ego-motion from object-motion is fundamentally
ill-posed: "camera moved / world static" and "camera static / world moved" can produce
identical pixels, and "far+fast" ≈ "near+slow" (depth–speed ambiguity). It is solvable
*only* with anchors. Ours, all already in the pipeline:
- **Static background → ego-line.** The rigid scene (Grid-room backdrop, ground plane)
  pins camera motion (this is the SfM/SLAM core); residual motion = object world-lines.
  **This makes a usable static-background anchor a *data-design requirement*** — write it
  into the scenario spec before scale, or the decomposition is unrecoverable.
- **Depth + normals → gauge + scale** (§3). Available as dense heads, sim and (pseudo) real.
- **Shared ego-line across objects → rigidity.** All objects constrain the same `T_WC`.
- **Gravity → metric time-scale** via `‖g‖`.

Per the user's call, **corner cases are out of scope for now** (degenerate camera ∥ gravity,
fully dynamic scenes with no static background, transparent/zero-texture grounds). Revisit
only after the anchored case works.

---

## 8. How to test it cheaply (folds into GATE A)

GATE A data (Kubric MOVi-E / CLEVRER) already ships **GT camera pose + object trajectories +
intrinsics**, so the whole thing is testable on *perfect RGB* before any Isaac spend:

1. **Recoverability.** Can a small model recover the ego/object world-line split from perfect
   RGB with a static-background anchor? (world-line R², ego pose error.)
2. **The decisive ablation.** Does **object-world-line error stay flat as camera-motion
   magnitude increases**, while a camera-frame baseline degrades? This is the entire value
   proposition of going allocentric. If it does not hold on perfect data, the framing is dead
   before sim spend.
3. **Permanence.** Does `e_i(t)` + persistent latent recover an object's world-line *through*
   an occlusion interval better than a per-frame detector?
4. **Contact.** Does the `δ_i` residual (§5) capture bounce/settling, measured as world-line
   error in a window around GT contact events?

## 9. Kill criteria

- A static-scene SfM/SLAM baseline + per-object tracker already recovers ego/object
  world-lines to sim noise → the learned encoder adds nothing on the easy case; move to
  clutter/occlusion immediately.
- Object-world-line error grows with camera motion **as fast as** the camera-frame baseline
  → allocentric framing bought nothing (the disentanglement failed).
- A linear probe on the world-line latent predicts **camera-frame** coordinates better than
  **world-frame** coordinates → gauge not pinned / the model is still egocentric.
- `z_inv` (mass/CoM) is predictable from a single static frame → R8 leak; the "physics" is
  appearance.

## 10. Implementation order (code, written cautiously)

All Phase-0 work is **off Isaac**, on GATE-A data. Build and unit-test bottom-up.

1. **Data adapter** — extract per-object world-frame trajectories, camera poses, intrinsics,
   visibility flags from Kubric/CLEVRER. *Unit test:* round-trip `p^W → p^C → u` against the
   dataset's own 2D annotations.
2. **Gauge module `G`** — gravity (normals) + ground (depth) + yaw (grid/forward). *Unit
   test:* invertibility, and that a known `SE(3)` perturbation of the scene leaves world-lines
   invariant after re-gauging.
3. **Temporal basis `B`** — B-spline/Chebyshev, plus `se(3)`/`so(3)` exp/log. *Unit test:*
   fit known analytic curves (parabola, constant-ω spin) to tolerance.
4. **Object-query head** over frozen features; **ego-line head**.
5. **Set-matching loss** (Hungarian, gradient through matched pairs) + the loss ledger (§6).
6. **Train small model on perfect RGB**; run the four §8 tests + §9 kill checks.

Only after §8 passes does any of this touch the Isaac corpus.

---

## 11. Open questions (settle before code)

1. **Orientation basis** — `so(3)` Lie-algebra coefficients vs. per-frame quaternion +
   smoothness. Lie-algebra is cleaner for ω but watch wrap-around.
2. **Ego-line parametrization** — basis on `se(3)` (smooth, compact) vs. per-frame pose
   regression with a smoothness prior (more flexible for jerky cameras).
3. **`L_proj` weight** — too high and the world head collapses to whatever the ego head says;
   too low and the two heads decorrelate. Needs a warm-up like `λ_dom`.
4. **Scale on real** (§3) — accept scale-up-to-prior, or require camera height/intrinsics?
   Determines what the sim→real claim can say.
5. **Where this lands architecturally** — F-config head over frozen V-JEPA (main line), or a
   separate Path-B track? Decide after GATE B says whether the frozen features even retain
   object world-lines through occlusion (cause 1 vs cause 2 in the IntPhys 2 analysis).
6. **`N` (max objects)** and the no-object-logit calibration for our scene complexity.

## 12. Relationship to the rest of the plan

- **Does not replace** the frozen-first main line ([../BRAINSTORM.md §2](../BRAINSTORM.md#2-architecture)).
  This is a *target representation*, agnostic to whether GATE B keeps us frozen-first.
- **Consumes** the dense depth/normal/seg heads twice over: transfer bridge **and** gauge
  estimator (§3). Strengthens the case for the B-config dense heads.
- **Instantiates** the equivariance structure of
  [theoretical-paper-plan.md §3](theoretical-paper-plan.md) and the conservation framing of
  §5 (world-line derivatives ↔ momentum) in a concrete, differentiable head.
- **Gated by GATE A** (§8) before any sim spend, exactly like the rest of the pipeline.

---

## 13. Implementation — encoding sim data → world-lines (Phase E0)

Moved to its own file: **[world-lines-encoding.md](world-lines-encoding.md)** (data
contract, `RawClip` interface, the encode pipeline, the round-trip verification, the
synthetic generator, module layout, and what's tweakable). Code:
[`src/worldlines/`](../../src/worldlines/) — implemented, round-trip verified.

## 14. Camera & units at inference: what's recoverable (reference)
Not geometrically camera-agnostic; the camera is *predicted from the image under a learned
prior* (taught by intrinsics+pose DR in sim) and/or delegated to pretrained depth/normals
(inheriting R3). Sorted by what scale a target needs:

| Target | Needs metric length? | Recoverable, unknown real camera |
|---|---|---|
| ω (rad/s; time is metric from fps) | no | ✅ |
| restitution, mass **ratios**, `J/m`, `c/m` | no (dimensionless) | ✅ |
| CoM offset / object size (R7) | no (ratio) | ✅ |
| absolute velocity / mass / length | yes | ❌ (up-to-scale only) |

The survivable targets are exactly the dimensionless ones — and the ones that die are the
absolute-metric quantities that have no real GT anyway (R2), so unknown-camera and
unverifiability fail on the *same* set. One practical metric anchor: assuming Earth
`g≈9.81` and a fall/ballistic segment in-frame, `½g·t²` converts metric time → length.
No gravity cue → up-to-scale (§2.4, tier 2).
