# Physics-LoRA adapter — Rung 1.5 design (parked, seeded)

**Status:** design only, pre-experiment. Parked in [EXPERIMENTS.md](../EXPERIMENTS.md)
behind the active corpus-rebuild line; see the promotion condition there.
**CoM is abandoned (2026-07-27)** — it is not a target anywhere in this design;
the twin driver is repointed at friction/restitution (§3.1).
Implementation companion (verified V-JEPA hook points, LoRA class skeletons,
split-forward/caching mechanics, variant table):
[lora-implementation-notes.md](lora-implementation-notes.md).

**Question.** Can a small low-rank adapter on frozen V-JEPA's *upper blocks*,
trained with sim physics supervision, make latent physics (material parameters,
contact structure) more decodable — **without drifting off the real-video
manifold**? This is deliberately a different question from the TRUNK ("does the
frozen encoder *already* encode physics"): it asks whether the encoder can be
*cheaply steered*. It only becomes decisive once the frozen read is locked.

---

## 1. Position in the doctrine

Rung **1.5** on the capacity ladder ([BRAINSTORM.md §7](../BRAINSTORM.md)):
frozen + head (Rung 1) → **LoRA on upper blocks (1.5)** → trainable two-stream
geometry branch (Rung 2, escape hatch). Candidate **replacement** for Rung 2:
the R1 failure mode (parallel sim-trained branch silently ignores the frozen
stream — bridge dies, sim loss looks great) is structurally impossible for a
LoRA adapter, which is not a parallel path but a bounded low-rank perturbation
*of* the frozen stream. The failure degrades from "bridge silently ignored" to
"bridge slightly bent", and the bend is measurable (adapter norm, feature drift
on real clips). Not yet adopted into BRAINSTORM §2.3 — adopt only if promoted
and the first run survives its controls.

**What LoRA does and does not buy** (be precise, this was the motivating
discussion):

- **Buys: anti-forgetting.** Base weights bit-exact frozen; adapters are
  removable and α-scalable at inference (continuous interpolation frozen ↔
  steered). Empirically low-rank tuning stays close to the pretrained function
  ("LoRA learns less and forgets less", Biderman et al. 2024). The real-video
  manifold is recoverable *by construction* — full fine-tuning can never
  promise that.
- **Does NOT buy: shortcut immunity.** Rank bounds how *much* new circuitry,
  not *what kind* — a rank-4 adapter can gate on a renderer statistic. The
  shortcut half of domain memorization stays governed by data (R8, DR, twin
  construction) and by real-side evaluation, exactly as before. R2 is untouched.

**The genuinely new capability vs. a deeper head** (arm_2b already does
MLP+concat): LoRA can reshape **attention routing** — change what gets
aggregated into tokens — while any post-hoc head only reweights features the
frozen aggregation already pooled. The adapter tests the hypothesis that
physics evidence lives in fine spatiotemporal token relations the frozen
pooling discards.

## 2. Architecture

- Base: repo-native ViT-g/384 V-JEPA (bit-identical encoder path verified —
  [vjepa-embedding-structure.md](vjepa-embedding-structure.md)).
- **LoRA on attention Q,V of blocks ~24–40 only** (optionally MLP later).
  Rationale: GATE-B localized the kinematics signal at **L28–32/40**
  (`results/probe/fit_matrix`); the sim↔real gap is largest in low-level
  appearance statistics, so freezing the lower two-thirds structurally denies
  the adapter the ability to retune to sim pixel statistics at the input end.
- **Rank sweep r ∈ {4, 16, 64}** — the dose-response knob for domain
  memorization (see §5).
- **Compute plan:** cache activations at the cut layer (~L23) once for the
  whole corpus with the existing probe-extraction infra; training touches only
  the upper blocks + adapters + heads. Cost class: probe-scale, not
  encoder-scale. Real-anchor clips (§3.3) get the same cached treatment.

### 2.1 Embedding topology (decided 2026-07-18): explicit physics projection

Between the adapted vision embedding and the physics losses sits a **shallow
projection** (1–2 layer MLP, ~128–512 d) — the "physics embedding" `z_phys`,
the named product of the experiment. Routing rules (load-bearing):

- **On `z_phys` (post-projection):** twin-contrastive loss (its metric
  geometry lives here) + all sparse physics heads (material params, contact
  timing, impulses, outcomes).
- **On the vision embedding (pre-projection):** the real-video anchor (§3.3)
  — real clips have no physics code to anchor, so the leash holds the backbone
  function — and the dense depth/normals anchor heads, which must NOT pass
  through the physics bottleneck (they would flood it with geometry; geometry
  is anchor, physics is product).
- **Shallowness is a constraint, not a preference:** if the projection (or any
  head) is deep, it becomes the physics network and the adapter learns
  nothing. Adapter = routing, projection = compression, heads = readout.
- **SimCLR lesson, inverted:** contrastive projection heads are usually
  discarded because they absorb invariances; here `z_phys` is the deliverable,
  so supervised heads sit *after* it to keep it information-rich — and probing
  the same targets pre- vs post-projection is itself a diagnostic (how much
  physics the vision embedding carries vs. what only the projection distills).

## 3. Training signals (from the supervision-menu discussion)

### 3.1 Twin-contrastive (primary; the sim-unique, R8-proof signal)
[collect_pairs.py](../../src/data_gen/collect_pairs.py) already builds
controlled-intervention twins (same seed, frame-0-identical pixels). Its CoM
intervention mode is **retired with the CoM target**; what survives is the
mechanism — build once, re-simulate per variant, log GT divergence + fork frame.
**Repoint the driver at friction / restitution / density twins.** Loss structure
— *not* vanilla InfoNCE (twins are positives in appearance and negatives in
physics):

- **pre-fork windows** of a twin pair must map together (appearance + shared
  history invariance);
- **post-fork windows** must separate, and the separation direction must be
  consistent with the parameter delta across pairs (metric learning on
  parameter deltas, not just apart-ness);
- **cross-view pairs** (the rollout → n_cameras fan-out) supply the invariance
  positives for free.

Appearance signature of the intervention is zero by construction, so feature
divergence can only come from dynamics — this is the strongest anti-shortcut
signal Isaac can produce, and no real-video source can produce it at all.

*Later refinement (do not start here):* **active twin generation** — Isaac as
a query engine, generating twins in the (oracle-certified) scenario regions
where `z_phys` currently confuses parameters: hard-negative mining in scenario
space. Caveat: active generation biases the training distribution — the eval
corpus must be generated once from the certified scenarios and **frozen**
before any active loop touches collection.

### 3.1b Canonical target table (2026-07-27)

Density spans ~5 orders of magnitude — hence uncertainty weighting (§3.3), and
hence the routing split of §2.1. "Exists" = in the current corpus; the rebuild
regenerates all of it anyway, but *existing* means it can be probed **today**.

| Target | Labels/clip | Status | Attaches to |
|---|---|---|---|
| depth, normals | ~10⁵/frame | collected | vision emb (anchor, **pre**-projection) |
| lin/ang velocity | ~10²–10³ | logged (`log_state_frame`) | `z_phys` |
| future-state Δv, outcomes | ~10² | derivable from `state.json` | `z_phys` |
| **restitution, friction** | 3–8 | **logged already** ([body_factory.py:457](../../src/data_gen/body_factory.py)); **no target builder yet** — `build_targets_v2` emits only `v_clipmean`/`w_clipmean` (+ the now-dead `com_norm`) | `z_phys` |
| contact events | ~10–50 | **nothing exists** (only the coarse per-scenario `OBSERVABILITY_HINT` tag) | `z_phys` |
| gravity tilt | 1 → T′ | derived from extrinsics; **demoted** (dynamic-camera.md §0) | `z_phys` |
| egomotion | 0 → T′ | needs dynamic camera | `z_phys` |
| **mass / density ratio at collisions** | 1 per collision pair | **NEW (2026-07-27)**; needs body-body contact events | `z_phys` |
| **twin interventions** | — | **primary signal**; needs `collect_pairs` generalized | `z_phys` (§3.1) |

**★ Collisions open a latent-inertial target that CoM could not deliver.** In a
two-body collision, momentum conservation gives `m₁·Δv₁ = −m₂·Δv₂` — the mass
ratio is *exactly* readable from the velocity changes, with no separatrix and no
marginal-observability problem. This is the structurally-observable replacement
for what the CoM line was trying to buy.
- **R8 holds**, but state the target carefully. Mass ~ size correlates +0.72
  (the cube law — a DONE result), so *absolute* mass is partly visible. The
  ratio decomposes as `m₁/m₂ = (V₁/V₂)·(ρ₁/ρ₂)`: the volume ratio is visible,
  the **density ratio is not** (density ⟂ size, r≈0, 25–40× spread). So the
  well-posed target is the **density ratio**, or mass ratio with volume as a
  covariate — consistent with the standing "decode density, not total mass".
- **Two fixes needed before it works:**
  1. **Rebalance the mix.** `collide` is ~1.8% of the current corpus (63 of 3511
     bodies) — far too thin to learn from. It needs to be a first-class share.
  2. **A third event anchor.** `_event_features` detects *vertical* velocity sign
     flips (floor impacts); body-body collisions exchange mostly *horizontal*
     momentum and are invisible to it. This — not n=57 — is the better
     explanation of `collide`'s null in the restitution oracle. Needs
     `--anchor collision`, keyed on the pair's relative approach velocity, which
     is what body-body contact events from PhysX provide directly.

**Velocities are NOT sparse** — they are the densest physics label we have
(per body, per frame) and GATE-B-validated. The sparse-label problem belongs to
contacts (impulsive) and per-clip scalars (materials).

### 3.2 Sparse supervised heads (labels already exist or are free)
- **Restitution + friction** — sampled per body and already logged
  ([body_factory.py:457-459](../../src/data_gen/body_factory.py)). Best-posed
  latent material params in the project (a bounce is a loud, appearance-free
  readout). Caveat: observability regime — restitution 0–0.4 band may be thin;
  certify first (§4).
- **Future-state targets** (v/ω at t+Δ from frames ≤ t; esp. Δv across contact
  events) — derivable from `state.json`, retrofittable; the JEPA-AC-shaped
  anticipation form.
- **Outcome labels** (topples/recovers, time-to-rest) — derivable, retrofittable.
- Gravity tilt: **postponed** (decision 2026-07-18) — degenerate on the current
  rig (look-at cameras, zero roll, floor always visible ⇒ tilt ≡ readable
  camera elevation); needs camera-roll + floor-suppression knobs (non-
  retrofittable) before it is a physics target. Dense form when revisited:
  per-pixel `g·n` (derivable from stored normals + `R_camera_world`).
  Unblocking spec written (2026-07-24):
  [data-gen/dynamic-camera.md](../data-gen/dynamic-camera.md) — v1 flying
  camera; under it the label becomes a `(T', 3)` trajectory with a free
  temporal-equivariance consistency loss, and **egomotion twins** (same
  rollout, different camera path) join the twin taxonomy of §3.1.

### 3.3 Anchors (anti-drift, function-space)
- **Real-video feature anchor:** L2 to the frozen base's outputs (or the native
  JEPA objective) on real unlabeled clips, through the same adapter. LoRA
  constrains the update in *weight space*; this constrains it in *function
  space on the real domain*. Finally uses the vast real unlabeled video.
  Reference features are fixed → **precompute and cache** them per real clip;
  no second model at train time.
- **Full-depth anchor term (added 2026-07-27) — closes a real gap.** A tap-level
  anchor constrains only L28–32, but the canary (and any downstream use) reads
  the **final** layer, and blocks 33–39 are frozen *and unadapted*: they receive
  out-of-distribution input and 7 blocks of residual accumulation can amplify it.
  Nothing in training would defend that. So add
  `λ_deep · ‖f_adapted − f_frozen‖²` **at the final layer**, on real clips, every
  k-th step (the extra blocks cost forward *and* backward, since gradient must
  pass through the frozen tail to reach the adapters). Turns the canary from a
  post-hoc verdict into a defended objective. `λ_deep` is a genuine tradeoff, not
  free safety — pinning final features fights the steering being paid for.
- Optional: low-capacity depth/normals readouts as the sim+real thermometer
  (R2 mitigation) — anchors, **not** the physics claim; uncertainty-weighted so
  dense terms can't dominate the sparse physics heads.

## 4. Pre-flight gates (no GPU until passed)

1. **Observability oracle** — the `com_oracle` *harness* outlives its target
   (CoM is abandoned; the method is what was worth keeping) and is
   target-agnostic:
   GT-trajectory → ridge/MLP decodability of restitution/friction per scenario,
   from `state.json` alone. Scale nothing that fails. **Runs on the EXISTING
   corpus with no Isaac and no GPU** (labels are already logged), so it is the
   one pre-flight that can invalidate a regenerated corpus and must not be
   deferred — the CoM line already paid for "generate first, discover
   unobservability after GPU spend".
   - `restitution_range` (0–0.4) / `friction_range` (0.3–0.8) are
     **dynamics-changing knobs**: settle before the rebuild.
   - **Per-scenario ranges, not one global** (decided 2026-07-27): high
     restitution is the *observable* in drop/throw/collide (bounce ratio ≈ e²)
     but would wreck the tuned mechanisms of low-energy scenarios. A global
     widening is not free.
   - **★★ RESOLVED (2026-07-27): restitution IS observable, and the blocker was
     the WINDOW, not the parameter range.** Same heads, same folds, same labels
     — only the feature window changed:

     | features | dim | ridge R² | MLP R² |
     |---|---|---|---|
     | generic 32-frame trajectory summary | 391 | −0.003 ± 0.007 | −0.018 ± 0.015 |
     | **event-anchored (±4 frames at impact)** | **58** | **+0.129 ± 0.032** | **+0.217 ± 0.037** |

     Null → +0.22 with **7× FEWER features**. That is the signature of a
     misalignment problem, not a capacity problem. Passes in 8/9 scenarios
     (only `collide` nulls, n=57). **Consequences:**
     1. **`restitution_range` (0.0–0.4) stays as-is** — one fewer
        non-retrofittable knob in the rebuild. An earlier "widen the band"
        reading is **retracted**; it was drawn from the generic-window null.
     2. **The GATE-B lesson sharpens one level.** Window-aligned targets bought
        +0.29 for velocity; impulsive quantities need **event-aligned** ones.
        A clip-mean encoder probe will NOT see restitution no matter how good
        the encoder is — the target form is the experiment's load-bearing part.
     3. **Contact logging is doubly justified:** exact impulses *and* the event
        timestamps that make event-anchored windows constructible at all.
     4. **Method warning:** this oracle was NOT an upper bound here. For CoM the
        generic features were adequate (the offset shapes the whole
        trajectory); for impulsive quantities they are not. Any future null must
        be tested on the target's OWN event before it is believed — hence
        `--anchor {impact,slide}`.
   - **Earlier diagnostic pass (kept — it localized the problem):** the
     temporal-undersampling read on the v2 corpus (3770 bodies / 7257 bounce
     events):
     - Event coverage is *excellent* — **91.1%** of bodies bounce (85–95% in
       every scenario), 75% slide. The "scenarios never produce the event"
       hypothesis is dead.
     - But state is logged once per **rendered** frame, so gravity injects
       **g·Δt = 0.327 m/s** into vz between samples, while the corpus's median
       predicted rebound is **e·|vz_in| = 0.178 m/s — below the floor.** Only
       **33.8%** of bounces produce a rebound that survives to the next sample.
     - Consequence: `corr(e_est, e_true)` = **+0.081** pooled. Restricted to
       above-floor events it recovers to **+0.278** (+0.319 gravity-corrected),
       and `topple` — whose impacts are high-energy — reaches **+0.639**
       unrestricted. The physics writes restitution into the motion; our
       sampling erases it.
     - **Fix order:** (1) **contact-impulse logging from PhysX** — exact and
       sampling-independent, which promotes contact logging from "extra target"
       to *the* reliable restitution channel; (2) raise `e·|vz_in|` per scenario
       (widen restitution upward, increase drop height / impact speed), target
       `e·|vz_in| > ~3·g·Δt ≈ 1 m/s`; (3) optionally log state at physics rate
       (60 Hz), which only halves the floor — a mitigation, not a fix.
     - The `frac_bounce_above_floor` metric is now first-class in the oracle and
       drives its localization verdict.
2. **Frozen baseline** — Gate-B-style linear + MLP probes for
   restitution/friction on frozen features: the Rung-1 number the adapter must
   beat, with group-aware folds as always.
3. Twin divergence tuning — pick intervention magnitudes from the GT
   divergence diagnostic collect_pairs already logs (fork visible, not
   saturated), before GPU.

## 5. Evaluation, controls, kill criteria

- **Primary read:** pre-registered decodability lift on held-out groups vs. the
  frozen baseline (§4.2), same probe harness.
- **Controls that must not degrade:** the standard control probe (v_clipmean ≥
  frozen value − margin) and the **real-side drift canary** — the IntPhys2
  surprise pipeline run frozen vs. adapted. ("Canary" = sentinel/guard metric:
  never trained on, chosen to fail early and cheaply. The moment it becomes an
  optimization target it stops being an independent warning.) If sim-physics fit
  rises while the canary falls, the adapter is trading real physics for sim
  shortcuts: domain memorization, *measured*.
- **Calibrate the canary BEFORE trusting it** (cheap, no training): perturb block
  32's output with **random noise of matched norm** and measure the canary
  response. The predictor is 12 blocks deep and was trained on the original
  feature distribution, so it may be sensitive enough that a harmless shift moves
  the score a lot. This converts "does final-layer drift matter?" from assumption
  into a measured number, and tells you how much score movement is meaningless.
- **Final-layer drift is a verification risk, not a forgetting risk.** Base
  weights stay bit-exact and adapters are removable/α-scalable, so nothing is
  lost irreversibly; and `z_phys` reads L28–32, so tail drift does not damage the
  deliverable. What it damages is our ability to *check* the encoder is intact.
- **Dose-response read:** sweep rank; pick the knee where real-side canary
  peaks. Sim-fit rising with rank past that knee = memorization margin.
- **Matched-capacity confound control:** compare against head-only with the
  same added parameter count. The adapter's value must come from in-network
  routing (§1), not raw capacity.
- **Kill:** no lift at any rank over matched-capacity Rung-1 → the information
  is not cheaply steerable → the Rung-2 escape-hatch discussion returns with
  this as evidence.

## 6. Data prerequisites that can ride along with planned collection

All dynamics-neutral (do not trip the non-retrofittable-knob rule): PhysX
contact logging (+ the closed-form `impulse ≈ Δ(m·v)` check for
`verify_physics_labels.py`), twin-driver generalization, future-state/outcome
target builders. Exception: any `restitution_range` widening (§4.1) changes
dynamics — settle it first.

**Dynamic camera** ([data-gen/dynamic-camera.md](../data-gen/dynamic-camera.md))
is dynamics-neutral too (camera is render-side, so oracle certifications
survive) but it *is* a non-retrofittable **pixel/label-semantics** knob and
forms its own corpus slice: `lin_vel_camera` mixes object motion with
egomotion, so Gate-B targets are not comparable across slices without the
logged egomotion decomposition.
